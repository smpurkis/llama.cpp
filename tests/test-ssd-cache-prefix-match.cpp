#undef NDEBUG

#include "kv-ssd-cache.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <vector>

static kv_ssd_cache * create_cache(const std::filesystem::path & path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path, ec);
    assert(!ec);

    kv_ssd_config cfg;
    cfg.auto_size = false;
    cfg.no_fsync = true;

    kv_ssd_cache * cache = kv_ssd_init(path.string().c_str(), &cfg, 0x1234ULL);
    assert(cache != nullptr);
    return cache;
}

static void destroy_cache(kv_ssd_cache * cache, const std::filesystem::path & path) {
    kv_ssd_free(cache);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// A checkpoint store + query with a different first token should NOT match
// (LCP = 0).
static bool test_prefix_mismatch_is_rejected() {
    namespace fs = std::filesystem;

    const fs::path path = fs::temp_directory_path() / "test-ssd-cache-prefix-match-reject";
    kv_ssd_cache * cache = create_cache(path);

    constexpr size_t n_tokens = 8192;
    std::vector<uint32_t> stored_tokens(n_tokens);
    for (uint32_t i = 0; i < stored_tokens.size(); ++i) {
        stored_tokens[i] = i;
    }

    // Query differs at position 0: first token is 9999 vs stored token 0.
    std::vector<uint32_t> query_tokens = stored_tokens;
    query_tokens[0] = 9999;

    const std::array<uint8_t, 1> state = { 0 };

    const uint64_t checkpoint_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, (int32_t) n_tokens - 1,
        n_tokens, 0, stored_tokens.data(), stored_tokens.size());
    assert(checkpoint_id != 0);

    const uint64_t match = kv_ssd_find_match(
        cache, query_tokens.data(), query_tokens.size(), 0, query_tokens.size());

    destroy_cache(cache, path);
    return match == 0;
}

// A checkpoint whose prefix matches the query prefix should be found.
static bool test_prefix_match_is_found() {
    namespace fs = std::filesystem;

    const fs::path path = fs::temp_directory_path() / "test-ssd-cache-prefix-match-found";
    kv_ssd_cache * cache = create_cache(path);

    constexpr size_t n_tokens = KV_SSD_TOKEN_PREFIX_MAX + 100;
    std::vector<uint32_t> stored_tokens(n_tokens);
    for (uint32_t i = 0; i < stored_tokens.size(); ++i) {
        stored_tokens[i] = i;
    }

    std::vector<uint32_t> query_tokens = stored_tokens;

    const std::array<uint8_t, 1> state = { 0 };

    const uint64_t checkpoint_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, (int32_t) n_tokens - 1,
        n_tokens, 0, stored_tokens.data(), stored_tokens.size());
    assert(checkpoint_id != 0);

    const uint64_t match = kv_ssd_find_match(
        cache, query_tokens.data(), query_tokens.size(), 0, query_tokens.size());

    destroy_cache(cache, path);
    // Should find the checkpoint since the prefix matches fully
    return match == checkpoint_id;
}

// When two checkpoints exist, the one with the higher LCP wins.
// Checkpoint A: prefix tokens [0..99] match query prefix (LCP = 100)
// Checkpoint B: prefix tokens [0..4095] match query prefix (LCP = 4096)
// B should win on LCP.
static bool test_higher_lcp_wins() {
    namespace fs = std::filesystem;

    const fs::path path = fs::temp_directory_path() / "test-ssd-cache-prefix-match-higher";
    kv_ssd_cache * cache = create_cache(path);

    constexpr size_t query_len = KV_SSD_TOKEN_PREFIX_MAX;
    std::vector<uint32_t> query_tokens(query_len);
    for (uint32_t i = 0; i < query_len; ++i) {
        query_tokens[i] = i;
    }

    const std::array<uint8_t, 1> state = { 0 };

    // Checkpoint A: prefix matches first 100 tokens only (differs at 100).
    // n_tokens must be at least 100 + 1 to carry the mismatch.
    constexpr size_t a_count = KV_SSD_TOKEN_PREFIX_MAX;
    std::vector<uint32_t> a_tokens(a_count);
    for (uint32_t i = 0; i < 100; ++i) a_tokens[i] = i;
    // differs at position 100 -> LCP will be 100
    a_tokens[100] = 0xDEAD;
    for (uint32_t i = 101; i < a_count; ++i) a_tokens[i] = i;

    const uint64_t a_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, (int32_t) a_count - 1,
        a_count, 1, a_tokens.data(), a_tokens.size());
    assert(a_id != 0);

    // Checkpoint B: full prefix match (LCP = KV_SSD_TOKEN_PREFIX_MAX).
    constexpr size_t b_count = KV_SSD_TOKEN_PREFIX_MAX / 4;
    std::vector<uint32_t> b_tokens(b_count);
    for (uint32_t i = 0; i < b_count; ++i) b_tokens[i] = i;

    const uint64_t b_id = kv_ssd_store(
        cache, 0, state.data(), state.size(), 0, (int32_t) b_count - 1,
        b_count, 2, b_tokens.data(), b_tokens.size());
    assert(b_id != 0);

    const uint64_t match = kv_ssd_find_match(
        cache, query_tokens.data(), query_tokens.size(), 0, query_tokens.size());

    destroy_cache(cache, path);
    // B has higher LCP (b_count > 100), so B should win
    return match == b_id;
}

int main() {
    int failures = 0;

    if (!test_prefix_mismatch_is_rejected()) {
        fprintf(stderr, "prefix mismatch was not rejected\n");
        failures++;
    }

    if (!test_prefix_match_is_found()) {
        fprintf(stderr, "prefix match was not found\n");
        failures++;
    }

    if (!test_higher_lcp_wins()) {
        fprintf(stderr, "higher LCP checkpoint did not win\n");
        failures++;
    }

    return failures;
}
