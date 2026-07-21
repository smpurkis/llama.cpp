// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius

#include "llama-moe-residency.h"

#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline size_t page_align_down(size_t x) {
    return x & ~(size_t(getpagesize()) - 1);
}

static inline size_t page_align_up(size_t x) {
    return (x + size_t(getpagesize()) - 1) & ~(size_t(getpagesize()) - 1);
}

// madvise a region. Aligns to page boundaries so the kernel can act on it.
// Silently ignores invalid pointers (non-mmap'd regions).
static void safe_madvise(void * base, size_t len, int advice) {
    if (!base || len == 0) return;
    uintptr_t p = reinterpret_cast<uintptr_t>(base);
    uintptr_t page_start = p & ~(uintptr_t(getpagesize()) - 1);
    uintptr_t end = p + len;
    uintptr_t page_end = (end + uintptr_t(getpagesize()) - 1) & ~(uintptr_t(getpagesize()) - 1);
    size_t aligned_len = page_end - page_start;
    if (aligned_len == 0) return;
    (void) madvise(reinterpret_cast<void *>(page_start), aligned_len, advice);
}

template <typename Fn>
static void for_each_tensor(llama_moe_layer_residency_internal & lr, Fn fn) {
    struct entry { ggml_tensor * t; size_t stride; };
    entry entries[4] = {
        { lr.t_gate,    lr.gate_stride    },
        { lr.t_up,      lr.up_stride      },
        { lr.t_down,    lr.down_stride    },
        { lr.t_gate_up, lr.gate_up_stride },
    };
    for (auto & e : entries) {
        if (e.t && e.t->data && e.stride > 0) {
            fn(e.t->data, e.stride);
        }
    }
}

// ---------------------------------------------------------------------------
// build()
// ---------------------------------------------------------------------------

bool llama_moe_residency_build(
        const struct llama_model * model,
        struct llama_moe_residency_internal_cfg cfg,
        struct llama_moe_residency_state * out) {
    if (!out) return false;
    out->cfg = cfg;
    if (!cfg.enabled) return false;
    if (!model) return false;

    const auto & hparams = model->hparams;
    const int n_expert = hparams.n_expert;
    const int n_expert_used = hparams.n_expert_used;
    const int n_layer = hparams.n_layer();
    if (n_expert <= 0 || n_expert_used <= 0) {
        return false;
    }

    out->layers.clear();
    out->layers.reserve(n_layer);
    out->n_layers = n_layer;
    out->n_expert = n_expert;
    out->n_expert_used = n_expert_used;

    int layers_with_experts = 0;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model->layers[il];
        ggml_tensor * t_gate = layer.ffn_gate_exps;
        ggml_tensor * t_up   = layer.ffn_up_exps;
        ggml_tensor * t_down = layer.ffn_down_exps;
        ggml_tensor * t_gu   = layer.ffn_gate_up_exps;

        const bool has_any = (t_gate && t_up && t_down) || t_gu;
        if (!has_any) continue;

        llama_moe_layer_residency_internal lr;
        lr.model_layer = il;
        lr.n_expert = n_expert;
        lr.t_gate    = t_gate;
        lr.t_up      = t_up;
        lr.t_down    = t_down;
        lr.t_gate_up = t_gu;

        if (t_gate) lr.gate_stride    = t_gate->nb[2];
        if (t_up)   lr.up_stride      = t_up->nb[2];
        if (t_down) lr.down_stride    = t_down->nb[2];
        if (t_gu)   lr.gate_up_stride = t_gu->nb[2];

        out->layers.push_back(std::move(lr));
        layers_with_experts++;
    }

    if (layers_with_experts == 0) {
        out->layers.clear();
        out->n_layers = 0;
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// touch()
// ---------------------------------------------------------------------------

void llama_moe_residency_touch(
        struct llama_moe_residency_state * st,
        int layer_idx,
        int expert_id,
        bool * was_already_loaded) {
    if (!st || !st->cfg.enabled) return;
    if (layer_idx < 0 || layer_idx >= (int) st->layers.size()) return;
    if (expert_id < 0 || expert_id >= st->layers[layer_idx].n_expert) return;

    auto & lr = st->layers[layer_idx];

    bool hit = lr.loaded_set.count(expert_id) > 0;
    if (was_already_loaded) *was_already_loaded = hit;

    if (hit) {
        lr.lru.erase(std::remove(lr.lru.begin(), lr.lru.end(), expert_id), lr.lru.end());
        lr.lru.push_back(expert_id);
        lr.hits++;
        st->total_hits++;
        return;
    }

    lr.loaded_set.insert(expert_id);
    lr.lru.push_back(expert_id);
    lr.misses++;
    st->total_misses++;
    st->total_touched++;

    const size_t off = (size_t) expert_id;
    for_each_tensor(lr, [&](void * base, size_t stride) {
        safe_madvise((uint8_t *) base + off * stride, stride, MADV_WILLNEED);
    });

    while (lr.lru.size() > st->cfg.max_resident_per_layer) {
        int evict = lr.lru.front();
        lr.lru.pop_front();
        lr.loaded_set.erase(evict);

        const size_t eoff = (size_t) evict;
        for_each_tensor(lr, [&](void * base, size_t stride) {
            safe_madvise((uint8_t *) base + eoff * stride, stride, MADV_DONTNEED);
        });
        st->total_evicted++;
    }
}

void llama_moe_residency_touch_layer_selection(
        struct llama_moe_residency_state * st,
        int model_layer,
        const int32_t * expert_ids,
        int n_expert_ids) {
    if (!st || !st->cfg.enabled) return;
    if (n_expert_ids <= 0 || !expert_ids) return;

    int idx = -1;
    for (size_t i = 0; i < st->layers.size(); ++i) {
        if (st->layers[i].model_layer == model_layer) { idx = (int) i; break; }
    }
    if (idx < 0) return;

    for (int i = 0; i < n_expert_ids; ++i) {
        llama_moe_residency_touch(st, idx, expert_ids[i], nullptr);
    }
}

// ---------------------------------------------------------------------------
// prewarm()
// ---------------------------------------------------------------------------

void llama_moe_residency_prewarm(
        struct llama_moe_residency_state * st,
        const int * const * top_experts) {
    if (!st || !st->cfg.enabled) return;
    if (!st->cfg.prewarm_on_init) return;

    const int K = st->cfg.prewarm_top_k;
    if (K <= 0) return;

    for (size_t il = 0; il < st->layers.size(); ++il) {
        const int * layer_top = top_experts ? top_experts[il] : nullptr;
        for (int k = 0; k < K; ++k) {
            int expert_id;
            if (layer_top) {
                expert_id = layer_top[k];
            } else {
                expert_id = k;
            }
            if (expert_id < 0) continue;
            if (expert_id >= st->layers[il].n_expert) continue;
            auto & lr = st->layers[il];
            if (lr.loaded_set.count(expert_id) > 0) continue;
            lr.loaded_set.insert(expert_id);
            lr.lru.push_back(expert_id);
            st->total_touched++;

            const size_t off = (size_t) expert_id;
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise((uint8_t *) base + off * stride, stride, MADV_WILLNEED);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// release()
// ---------------------------------------------------------------------------

void llama_moe_residency_release(
        struct llama_moe_residency_state * st) {
    if (!st) return;
    for (auto & lr : st->layers) {
        for (int expert_id : lr.lru) {
            const size_t off = (size_t) expert_id;
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise((uint8_t *) base + off * stride, stride, MADV_DONTNEED);
            });
        }
        lr.loaded_set.clear();
        lr.lru.clear();
    }
    st->layers.clear();
}

// ---------------------------------------------------------------------------
// log_stats()
// ---------------------------------------------------------------------------

void llama_moe_residency_log_stats(
        const struct llama_moe_residency_state * st) {
    if (!st || !st->cfg.enabled) return;

    const uint64_t total = st->total_hits + st->total_misses;
    const double hit_rate = total > 0 ? double(st->total_hits) / double(total) : 0.0;

    LLAMA_LOG_INFO(
        "moe-residency: decodes=%llu touches=%llu hits=%llu misses=%llu evictions=%llu hit_rate=%.1f%%\n",
        (unsigned long long) st->decode_count,
        (unsigned long long) st->total_touched,
        (unsigned long long) st->total_hits,
        (unsigned long long) st->total_misses,
        (unsigned long long) st->total_evicted,
        hit_rate * 100.0);
}

// ---------------------------------------------------------------------------
// topk_from_stats()
// ---------------------------------------------------------------------------

bool llama_moe_residency_topk_from_stats(
        const struct llama_context * ctx,
        int k,
        std::vector<std::vector<int>> & out_top) {
    if (!ctx || k <= 0) return false;
    out_top.clear();

    const auto & model = ctx->get_model();
    const int n_layer = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;
    if (n_expert <= 0) return false;

    out_top.resize(n_layer);
    bool any_data = false;

    for (int il = 0; il < n_layer; ++il) {
        const auto * stats = ctx->get_expert_stats(il);
        if (!stats) continue;

        std::vector<std::pair<int, uint64_t>> ranked;
        ranked.reserve(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            ranked.emplace_back(e, stats->activation_count[e]);
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const auto & a, const auto & b) {
                return a.second > b.second;
            });

        std::vector<int> topk;
        topk.reserve(k);
        for (int i = 0; i < k && i < (int) ranked.size(); ++i) {
            if (ranked[i].second == 0) break;
            topk.push_back(ranked[i].first);
            any_data = true;
        }
        if (topk.empty()) {
            for (int i = 0; i < k; ++i) topk.push_back(i);
        }
        out_top[il] = std::move(topk);
    }

    return any_data;
}