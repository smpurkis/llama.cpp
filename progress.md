# progress.md

## Goal

Maintain a **personal CachyLlama-based fork for AMD Strix Halo (Ryzen AI MAX+ 395 / 8060S, `gfx1151`, RDNA3.5)** that:

- Uses `fewtarius/CachyLlama` as its base and periodically incorporates its updates.
- Also tracks `ggml-org/llama.cpp` `master` so upstream changes and conflicts can be evaluated.
- Carries Strix Halo ROCm/Vulkan optimizations and any ROCmFPX dtype changes needed for ROCmFP quantized models.
- Avoids unrelated NVIDIA-focused work and separate runtimes.

The repo currently clones upstream directly (`origin` = `ggml-org/llama.cpp`). CachyLlama is a
separate remote added as `cachyllama`. Updates are integrated from CachyLlama first, then checked
against current upstream `master`; ROCm-specific carry commits remain isolated above the CachyLlama base.

**What is already on `halo`** (verified in commit log, see "Integration status" below for details):
- GQA-dequant tile FA decode fix (Nathanw1014, `5cf8b7a5c`, `c4f21afe1`)
- Vulkan quantized-KV FA prefill dequant (CachyLLama #25494 backport, `7d191aa41`)
- Vulkan `nodes_per_submit` auto-lower for UMA (`19b33f2eb`)
- MMQ RDNA3_5 tuning (`db7ce487a`, `b6face667`)
- `__builtin_amdgcn_sudot4` for RDNA3.5 (upstream)
- `__float2int_rn` in quantize.cu (upstream)
- SSD cache `posix_fadvise` readahead (`864cadc49`)

**What remains blocked or not yet integrated:**
- ROCmFPX dtype changes (manual port required)
- ROCmFPX dtype changes (manual port required)
- q4_0-specialized FA tile loader (Nathanw1014-identified; ~15% q4_0 headroom)
- `__expf()` intrinsics for MoE routing
- MMQ per-quant tuning for gfx1151
- Vulkan `nodes_per_submit` value sweep
- Vulkan FA scratch safety margin tuning
- Vulkan vs ROCm runtime dispatch
- GTT/VRAM BIOS partitioning
- CPU ISA auto-detection for Vulkan builds

## Base and excluded projects

- **CachyLlama** — selected as the base to include its SSD KV cache, Vulkan FA scratch, `nodes_per_submit` auto-lower, and RDNA3.5 MMQ tuning.
- **ROCmFPX** — source for the ROCmFP dtype changes. Carry the required dtype/quantization patchset rather
  than treating ROCmFPX as the branch base.
- **beellama.cpp** — NVIDIA/RTX-3090 focused (KVarN KV-quant, spec decode). No gfx1151-specific code. Skip.
- **Lucebox** — a *separate* speculative-inference server (`dflash_server`) that vendors its own ggml.
  Not a llama.cpp fork; cannot be merged in. Run it side-by-side if spec-decode speedups are wanted.

## Source evaluation table

Patches considered for the fork, with provenance and upstream status (verified against current
`master` at commit `178a6c449`).

| Source | What it does | Useful? | Upstream status (verified) | Merge risk | On `halo`? |
|---|---|---|---|---|---|
| **PR #21344 / #21284 part 1** (gfx1151 MMQ tuning) | `mmq_x=48, mmq_y=64, nwarps=4` for RDNA3_5. +50–70% pp short ctx. | **High** | MMVQ table + `sudot4` already upstream; MMQ x/y/nwarps tuning NOT done | Low | ✅ via CachyLlama base |
| **#21284 part 2** (intrinsics) | `__expf`, `__float2int_rn`, `sudot4` | `sudot4` & `__float2int_rn` upstream (`common.cuh:699`, `quantize.cu:65`); `__expf` NOT present | n/a | None (skip sudot4/float2int); add `__expf` | Partial |
| **Nathanw1014/strix-halo-fa-fixes** (GQA dequant fix) | `tile` dequant-on-load for quantized-KV decode; removes `gqa_ratio`× redundant dequant. +129% tg@32k q8_0 | **Highest** | **NOT upstream** (`vec` still hardcodes `need_f16_K`, `fattn-vec.cuh:540`) | Med–high | ✅ `5cf8b7a5c`, `c4f21afe1` |
| **CachyLLama** (Vulkan FA scratch + `nodes_per_submit`) | backports Vulkan qKV FA fix (#25494) + RDNA3 iGPU tuning | Medium (Vulkan only) | Vulkan-side equivalent of Nathanw1014; included by selected base | Med (base updates) | ✅ `7d191aa41`, `19b33f2eb` |
| **CachyLLama** (persistent SSD KV cache) | disk-backed paged KV, survives restart, prefix cache | **High (wanted)** | NOT upstream; included by selected base | Med (base updates) | ✅ CachyLlama base |
| **PR #16827** (lhl rocWMMA tune) | fixes rocWMMA long-ctx tg regression | **High (required)** | WMMA kernel **still present** (not removed); used with `-DGGML_HIP_ROCWMMA_FATTN=ON` | High churn | ✅ Implemented — prefill-only pieces (`__launch_bounds__`, adaptive KQ stride, nwarps bump) cherry-picked from commit `a3c9d1d` |
| **ROCmFPX dtype changes** | adds ROCmFP3/4/6/8 weight dtypes and required ROCm support | **High (required)** | separate format ecosystem; models require matching fork changes | Very high | 🚫 Not integrated |
| **beellama.cpp** | KVarN KV-quant, spec-decode (NVIDIA-focused) | Low (off-target) | n/a | High | ❌ Skip |
| **Lucebox** | DFlash/DDTree/PFlash/KVFlash spec-decode + paged KV | Low for fork | separate runtime (vendored ggml, custom server) — not mergeable into llama.cpp | n/a (run side-by-side) | ❌ Skip |

## Planned branch contents

1. **CachyLlama base** ✅ — includes persistent SSD KV cache, Vulkan FA scratch, `nodes_per_submit` auto-lower, and RDNA3.5 MMQ tuning.
2. **MMQ RDNA3_5 tuning** ✅ — applied via CachyLlama base + manual MMQ config bump in CachyLlama commit `b6face667`.
3. **Nathanw1014 GQA-dequant fix** ✅ — cherry-picked as `22e192204` and `80d27d18a` from commits `5cf8b7a5c`/`c4f21afe1`.
4. **PR #16827 rocWMMA tuning** ✅ — prefill-only pieces ported (commit `a3c9d1d`: `__launch_bounds__` min 2 blocks/SM, adaptive KQ stride 128 for D≤128, nwarps bump for small D). Decode WMMA exclusion remains CachyLlama policy (tile path handles decode correctly). 
5. **ROCmFPX dtype changes** 🚫 — requires manual port. Integration plan in Integration status below.
6. **q4_0-specialized FA tile loader** 🚫 — identified in Nathanw1014 docs; ~15% q4_0 headroom. See Integration status.
7. **`__expf()` intrinsics for MoE routing** 🚫 — not present in codebase. See Integration status.

## Provenance / links

- PR #21344 (gfx1151 MMQ/MMVQ tuning, mainline-closed): https://github.com/ggml-org/llama.cpp/pull/21344
- Issue #21284 (inefficient gfx1151 defaults; gist patch): https://github.com/ggml-org/llama.cpp/issues/21284
  - gist patch: https://gist.github.com/pedapudi/183f337e687630a43eacb293e157c9bd
- Nathanw1014/strix-halo-fa-fixes (GQA dequant fix): https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes
  - docs: https://github.com/Nathanw1014/llama.cpp/tree/strix-halo-fa-fixes/docs/fa-quant-kv-gqa
- CachyLLama (fewtarius): https://github.com/fewtarius/CachyLLama
- PR #16827 (rocWMMA tune, mainline-closed): https://github.com/ggml-org/llama.cpp/pull/16827
- ROCmFPX (charlie12345): https://github.com/charlie12345/ROCmFPX
- beellama.cpp (Anbeeld): https://github.com/Anbeeld/beellama.cpp
- Lucebox (Luce-Org): https://github.com/Luce-Org/lucebox
- Upstream: https://github.com/ggml-org/llama.cpp

## Integration status (2026-07-23)

### ✅ On `halo` — verified in commit log

| # | Optimization | Commit(s) | What was done | Verifying |
|---|---|---|---|---|
| 1 | GQA-dequant tile FA decode fix | `5cf8b7a5c`, `c4f21afe1` | Cherry-picked from Nathanw1014/strix-halo-fa-fixes. `fattn-tile.cuh` now dequantizes KV on load into its SRAM tile using `dequantize_V_*<T,ne>` primitives; q4_0/q8_0 tile instantiations at head dims 64/128/256 added; `need_f16_K/V` derived from tensor types instead of hardcoded `true,true`; `tile` preferred over `vec` for quantized decode when `gqa_opt_applies`. | `test-backend-ops -o FLASH_ATTN_EXT` (6000/6000 cases); greedy generation byte-identical to stock; gfx1151 ROCm execution pending |
| 2 | Vulkan quantized-KV FA prefill dequant | `7d191aa41` | Backport of CachyLLama's memory-gated version of ggml-org#25494. The coopmat1 FA path dequantizes+transposes quantized K/V into per-head-contiguous f16 scratch once per layer before running coalesced f16 FA, instead of re-dequantizing inside every Q workgroup on every step. Gate uses `common::host_available_ram()` to fall back to slow path on memory-constrained UMA. | Verify Vulkan logs for scratch allocation message; test OOM margin on long-context prefills with q8_0 KV |
| 3 | Vulkan `nodes_per_submit` auto-lower | `19b33f2eb` | Defaults `max_nodes_per_submit=8` on UMA devices (iGPU/APU) instead of upstream's 100, which starves the shader engine on shared-memory gfx1151. Exposed via `GGML_VK_NODES_PER_SUBMIT=N` env override. | Check Vulkan logs for node submission count; if GPU stalls or timeouts on decode, lower value |
| 4 | MMQ RDNA3_5 tuning | `db7ce487a` | Applies gaetan-puleo's Strix Halo RDNA3.5 tuning: MMQ tile sizing, warp counts, and MMVQ parameter table tuned for gfx1151. | Benchmark pp512 on Qwen3.5-122B-A10B Q4_K_M — expect +30–70% vs untuned baseline |
| 5 | `sudot4` for RDNA3.5 | upstream `common.cuh:699` | `__builtin_amdgcn_sudot4` replaces scalar dp4a for RDNA3.5 integer dot product. | Verify `common.cuh:699` has `__builtin_amdgcn_sudot4`; affects MoE routing and FFN down projections |
| 6 | `__float2int_rn` in quantize.cu | upstream `quantize.cu:65` | Replaces `roundf()` with IEEE-congruent intrinsic for round-to-nearest in weight quantization. | Verify `quantize.cu:65` has `__float2int_rn`; affects all round-in-quant paths |
| 7 | SSD cache readahead | `864cadc49` + kv-ssd-cache | `posix_fadvise(POSIX_FADV_WILLNEED)` on Linux / `readahead()` on macOS overlaps SSD I/O with CPU work during checkpoint restore. | Check `kv-ssd-cache.cpp` lines 133–163, 474; verify `posix_fadvise` calls present |

### 🚫 Blocked — not yet integrated

#### PR #16827 rocWMMA tuning (decode fix still blocked; prefill pieces done)

**Why decode fix is blocked.** The original PR (lhl, 2025-10-28) WMMA decode selector conflicts semantically with CachyLlama's tile path: (a) CachyLlama's RDNA3.5 tile FA is a different kernel path than WMMA decode, (b) CachyLlama explicitly excludes gfx1151 from rocWMMA FA runtime selection, (c) the tile path already handles decode correctly without WMMA.

**What is done.** The prefill-only pieces (from `a3c9d1d`) are now implemented:
1. `__launch_bounds__(min_blocks_per_sm=2)` on HIP FA kernels → higher occupancy on gfx1151
2. Adaptive KQ stride — stride 128 for D≤128 → reduced LDS footprint on RDNA
3. Bump `nwarps` for small D in the WMMA launch config

**What remains blocked.** The decode-selector/TILE-pruning removal commits from PR #16827 are unnecessary with CachyLlama's tile path and are not being applied.

#### ROCmFPX dtype changes

**Source fork:** `https://github.com/charlie12345/ROCmFPX` (charlie12345/caf, Strix Halo author).
**Branch to target:** `main` — confirmed on gfx1151 with Strix Halo build script
(`scripts/build-strix-rocmfp4-mtp.sh`) and verified benchmarks. The `experimental-rocmfpx-branch`
is preserved for history but do NOT use it.

**Why blocked.** ROCmFPX has unrelated snapshot history, no coherent dtype commit series, and spans too many files to cleanly cherry-pick: `ggml/rocmfpx/`, `ggml/rocmfp4/`, `src/llama-model.cpp` (tensor type routing), `common/` (CPU quantization), `Python/generated/` (type IDs), `tests/` (new FAATTN_EXT cases), `scripts/quantize-rocmfpx-agent.sh`, `docs/ROCmFPX-HANDOFF.md`.

**What needs to be ported** (minimum set for gfx1151):
1. GGUF type IDs for ROCmFP3/4/6/8 and their quantized-KV variants
2. CPU reference dequant paths (for validation)
3. HIP/ROCm copy/dequant kernels — specifically the ROCmFP4 fast path
4. Vulkan dequant kernel for ROCmFP4
5. MMQ/MMVQ tuning entries for ROCmFP4 quant types in `mmq-config-rdna3_5.cuh`
6. `llama-quantize` integration for `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q4_0_ROCMFP4_COHERENT`
7. MTP speculative decoding support (server + CLI)
8. TurboQuant K/V cache types (turbo3/turbo4) — separate from model-weight ROCmFPX

**Integration plan.** Use `main` branch of ROCmFPX as the source revision. Port the required files as a single isolated commit set on `halo`. Keep ROCmFPX changes in a dedicated subdirectory to minimize merge conflicts with upstream CachyLlama. After porting, verify against the benchmarks in the ROCmFPX README (Section: "Verified MTP Results — Strix Halo, 2026-07-12").

**Verification.** Build with ROCmFPX support; run `llama-quantize` to convert a BF16 GGUF to ROCmFP4; benchmark decode at 16k/32k/64k/128k on gfx1151 vs Q4_K_M baseline. Expected: 8–15% decode speedup on weight-quantized models, 1.5–2.4× with MTP enabled.

#### q4_0-specialized FA tile loader

**Source analysis.** Nathanw1014 documents the bottleneck: `dequantize_V_q4_0` only supports `ne ∈ {2,4}` (4-byte int nibble decoding), so a 32-element tile chunk needs `32/2 = 16` calls instead of `32/8 = 4` for q8_0. Additionally, each `qs` byte is fetched twice (low nibble for element j, high nibble for element j+16) through the generic `dequantize_V` interface. This means q4_0 issues ~2× the unpack ALU and ~2× the byte-fetch overhead vs q8_0.

**Estimated gain.** Nathanw1014 documents: forcing q4_0 to `ne=2` (same bytes, 2× the chunks) costs −31% at 16k and −38% at 32k — confirming the loader is overhead-bound. A specialized loader that fetches each byte once and emits both nibble halves in a 32-element aligned tile could recover most of the remaining ~15% gap between q4_0's 85% and q8_0's 95% of bandwidth ceiling.

**Integration plan.** Add a new `dequantize_V_q4_0_tile/ne32` primitive to `fattn-tile.cuh` that loads 32 q4_0 elements in one pass (16 bytes → 32 nibbles), paired with the existing `epc=8` tile layout. Template q4_0 alongside q8_0 in the `type_K`/`type_V` dispatch. Verify correctness with `test-backend-ops -o FLASH_ATTN_EXT`.

**Verification.** Benchmark q4_0 decode at 16k/32k/64k on gfx1151 with and without specialized loader. Target: move q4_0 from 85% to ≥92% of q8_0 bandwidth.

#### `__expf()` intrinsics for MoE routing

**Files to modify.** `ggml/src/ggml-cuda/gated_delta_net.cu` (MoE routing in Qwen3.5/3.6 models), `ggml/src/ggml-cuda/ggml_cuda_op_silu_single.cu` (SiLU/F_silu activation).

**Change.** Replace `expf(x)` with `__expf(x)` in gated DeltaNet's expert gating and SiLU activation. These are single-precision float operations where the precision difference between `expf` (C library) and `__expf` (hardware intrinsic) is below the noise floor for MoE routing decisions.

**Verification.** Run the `test-backend-ops` test suite focusing on gated delta net and SiLU operators. Compare output bit-exactness against stock `expf` path. Run llama-bench on Qwen3.5-122B-A10B Q4_K_M pp512@4096 to confirm no quality regression (output should be identical with greedy sampling).

### Not yet integrated (cont.)

#### MMQ per-quant tuning for gfx1151

**Background.** The `mmq-config-rdna3_5.cuh` file has MMVQ parameter tables with RDNA3_5 entries (`71d1e8f2f` bumps I from 48 to 64 in all 232 MMQ cases to satisfy upstream's `static_assert((I_) % 32 == 0)`). However, MMQ x/y/nwarps are set once for ALL quantization types, when different qtypes (Q4_K vs Q8_0 vs F16) have different register pressure and block sizes.

**Potential approach.** Profile each qtype at common block sizes on gfx1151 and adjust MMQ tile dimensions per-quant. The I dimension (64) is already correct via the CachyLlama bump. The x/y/nwarps tuning could differ between Q4_K (4-block dequant), Q8_0 (8-block dequant), and F16 (no dequant).

**Verification.** Build with `DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`, run `llama-bench` with each qtype separately on Qwen3.5-122B-Q4_K_M and compare pp512/tg128.

#### Vulkan `nodes_per_submit` value sweep

**Background.** CachyLLama defaults `max_nodes_per_submit=8` for UMA to avoid the 2s amdgpu `lockup_timeout`. The upstream default is 100. The current code path auto-detects UMA and applies 8.

**To test on gfx1151.** For each value in {4, 8, 12, 16}, run:
```bash
GGML_VK_NODES_PER_SUBMIT=$N ./build-vulkan/bin/llama-bench \
  -m Qwen3.5-122B-A10B-Q4_K_M.gguf \
  -p 512 -n 64 -d 32768 -c 32768 \
  -ngl 99 -fa on -b 2048 -ub 2048 \
  -r 3 --no-warmup
```
Measure pp512 and tg32 at each value. Look for throughput plateau or degradation past the sweet spot. On 64 GB UMA the sweet spot may be higher than 8.

#### Vulkan FA scratch safety margin tuning

**Background.** The scratch grows with context: ~256 MiB at 128k for head_dim 128 (Qwen3-Coder-30B-A3B), ~512 MiB at 128k for head_dim 256 (Qwen3.6-35B-A3B). The default `GGML_VK_FA_SCRATCH_SAFETY_MB=1024` was conservative for 32 GB UMA APUs. On 64 GB Strix Halo, this margin may be unnecessarily large.

**To test.** Build with scratch transpose enabled, then run long-context prefills with progressively smaller `GGML_VK_FA_SCRATCH_SAFETY_MB` values:
```bash
# Test with decreasing safety margins
for MB in 1024 512 256 128 64; do
  GGML_VK_FA_SCRATCH_SAFETY_MB=$MB ./build-vulkan/bin/llama-bench \
    -m Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
    -d 32768 -p 512 -n 64 -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048
done
```
If the fast path activates at a lower margin (no warning message in logs), reduce the default or document the lower value as the recommended setting for 64 GB UMA hardware.

#### Vulkan vs ROCm runtime dispatch

**Background.** Nathanw1014's benchmarks show a crossover: Vulkan wins at short context (depth 0: 65.45 vs 55.14 t/s) but ROCm wins at long context (depth 128k: 19.77 vs 17.85 t/s). The CachyLlama base currently has no backend-selector — the user picks Vulkan or ROCm at build time.

**Integration concept.** A single build with both backends that selects at runtime based on expected context length. This would require: (a) building both HIP and Vulkan backends in the same binary, (b) adding a `--backend-preference` CLI option or server config, (c) implementing the context-depth heuristic (e.g., if `-c > 8192`, prefer ROCm; if `-c ≤ 8192`, prefer Vulkan).

**Verification.** Build with both backends, run the crossover benchmark (Qwen3-Coder-30B-A3B, q8_0 KV, depths 0/4k/16k/32k/64k/128k), confirm the crossover point, then test the heuristic selects the right backend at each depth.

#### GTT/VRAM BIOS partitioning

**Status.** System-level, not code. Strix Halo (gfx1151) has 16 GB VRAM and 48 GB GTT in a 64 GB UMA pool. The BIOS-configured split determines how much GPU-accessible memory is in the VRAM pool vs GTT. Larger VRAM allocation = faster access for model weights + KV cache; smaller VRAM = more total GPU-addressable memory but slower GTT access for overflow.

**Investigation needed.** (1) Check current BIOS VRAM/GTT split on the test machine (`amdmind` or sysfs). (2) Benchmark a model that's larger than VRAM (e.g., Qwen3.5-122B Q4_K_M at 71 GB with partial offload) at different splits to find the point where GTT access penalty outweighs the VRAM capacity increase. (3) Document the optimal split in `scripts/build-halo.sh` comments or a `README-GFX1151.md`.

#### CPU ISA auto-detection for Vulkan builds

**Background.** The upstream Vulkan build defaults to `GGML_NATIVE=OFF`, meaning AVX-512 code paths are compiled out even on Zen 4 CPUs that Strix Halo's x86_64 platform supports. This affects CPU-offloaded layers (layers not on GPU) and Vulkan host-side operations. CachyLlama's parent project reads `/proc/cpuinfo` to enable the right ISA — the `halo` build script does not.

**Integration.** Add CPU ISA detection to `scripts/build-halo.sh` in the `configure_and_build` function. Check `/proc/cpuinfo` for AVX-512, AVX2, or SSE support and pass the appropriate CMake flags (`-DGGML_NATIVE=ON -DGGML_AVX512=ON` for Zen 4).

**Verification.** Build with auto-detection, check CMake output for CPU flags enabled. Run `llama-bench` with CPU-only offload (large model, minimal GPU layers) and compare tok/s against a build without ISA flags. Expected: 5–15% gen speedup on Vulkan and 30–100% on CPU-offloaded layers.

## Notes

### Build commands

The `halo` branch includes `scripts/build-halo.sh`. It creates independent build trees and builds
`llama-cli` by default:

```bash
# Build ROCm, Vulkan, and rocWMMA variants.
JOBS=16 scripts/build-halo.sh all

# Build selected variants.
scripts/build-halo.sh rocm
scripts/build-halo.sh vulkan
scripts/build-halo.sh rocwmma

# Override the SDK, parallelism, or CMake target.
ROCM_PATH=/opt/rocm-7.2.4 JOBS=16 TARGET=llama-server scripts/build-halo.sh rocm
```

Outputs are written to `build-rocm`, `build-vulkan`, and `build-rocwmma`. All three `llama-cli`
variants were built and smoke-tested successfully with ROCm 7.2.4 on 2026-07-20.

Do not set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` for DeepSeek V4 Flash on gfx1151. A controlled test
with stock upstream llama.cpp `76f46ad29`, ROCm 7.2.4, Q8_0 KV, flash attention, and the model
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` isolated this variable:

| Unified memory | `--fit` | Result |
|---|---|---|
| unset | default | Correct output, 14.60 tok/s |
| unset | `off` | Correct output, 14.65 tok/s |
| `1` | default | Corrupt repeated `work` tokens, false 615.67 tok/s |
| `1` | `off` | Corrupt repeated `work` tokens, false 582.14 tok/s |

The deterministic prompt was `Reply with exactly: The quick brown fox jumps over the lazy dog.`
The same corruption occurred in Halo, stock CachyLlama, and stock upstream builds whenever unified
memory was enabled. Therefore `--fit off` is not the cause; omit `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
for this model. Treat benchmark results from a corrupt run as invalid.

## DeepSeek V4 Flash ROCm comparison at 32k depth

> **Note:** These results were collected on 2026-07-21 and may be out of date.

Benchmarked on 2026-07-21 with ROCm 7.2.4 and gfx1151. Each result is three repetitions from
`llama-bench`; tests ran sequentially with unified memory unset. All builds used full GPU offload,
flash attention, Q8_0 K/V cache, batch and ubatch size 2048, mmap disabled, 32,768-token depth,
512 prompt tokens, and 64 generated tokens.

```sh
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    ./build-rocm/bin/llama-bench \
    -m DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    -d 32768 -p 512 -n 64 -r 3 -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 -ub 2048 -mmp 0
```

| Build | Revision | pp512 samples (tok/s) | pp512 mean | tg64 samples (tok/s) | tg64 mean |
|---|---|---:|---:|---:|---:|
| Halo | `9034409a6` | 55.3941, 57.8626, 58.6506 | 57.3024 | 11.0857, 10.9751, 10.7813 | 10.9474 |
| Stock llama.cpp | `76f46ad29` | 46.4651, 48.7557, 48.7461 | 47.9890 | 10.9709, 11.1114, 11.0777 | 11.0533 |
| Stock CachyLlama | `0da40ed6c` | 55.6596, 58.7606, 58.8351 | 57.7518 | 10.6281, 11.1954, 11.0855 | 10.9697 |

Halo versus stock llama.cpp: pp512 is 19.41% faster and tg64 is 0.96% slower. Halo versus stock
CachyLlama: pp512 is 0.78% slower and tg64 is 0.20% slower. The prompt-processing gain over
upstream is real, but this test does not show an additional performance gain from Halo's two GQA
commits over the CachyLlama base. Generation performance is effectively tied across the CachyLlama
and Halo builds, with upstream about 1% faster.

Installed ROCm SDKs:

- `/opt/rocm-7.2.0` is retained as the fallback SDK.
- `/opt/rocm-7.2.3` is retained as an additional fallback SDK.
- `/opt/rocm-7.2.4` is the active SDK through `/opt/rocm`.
- The `halo` HIP backend was successfully compiled for `gfx1151` with ROCm 7.2.4 in
  `build-hip-724`. The compiler emits existing FA template division-by-zero and system `libxml2`
  version-information warnings, but completes and links `libggml-hip.so`.
- ROCm Core SDK 7.14.0 is newer, but it is a preview using the new TheRock-based
  `amdrocm-*` packaging. Do not replace the working 7.2.x stack on this Ubuntu 25.10 host without a
  separate migration and rollback plan.

Select and verify ROCm 7.2.4 explicitly:

```bash
export ROCM_PATH=/opt/rocm-7.2.4
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

hipcc --version
rocminfo | grep -A 2 gfx1151
```

HIP release build for gfx1151:

```bash
export ROCM_PATH=/opt/rocm-7.2.4
export PATH="$ROCM_PATH/bin:$PATH"

cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DLLAMA_HIP_UMA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j 16
```

Vulkan release build:

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -- -j 16
```

### rocWMMA

rocWMMA is a header-only C++ library for AMD matrix-core MMA operations. The matching 2.2.0 headers
are already installed by `rocwmma-dev7.2.4` under `/opt/rocm-7.2.4/include/rocwmma`. llama.cpp does
not need rocWMMA to be built as a separate library; enable its optional FA implementation at configure
time:

```bash
export ROCM_PATH=/opt/rocm-7.2.4
export PATH="$ROCM_PATH/bin:$PATH"

cmake -S . -B build-rocwmma \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_HIP_UMA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocwmma --config Release -- -j 16
```

Important: the current CachyLlama base explicitly excludes RDNA3.5/gfx1151 from selecting its
rocWMMA FA path. The command above verifies that the code compiles with rocWMMA available, but it does
not make gfx1151 use that kernel at runtime. PR #16827 prefill-only pieces ✅ are now implemented (cherry-picked from commit `a3c9d1d`). The decode-selector/TILE-pruning removal commits remain blocked pending a manual port and correctness testing against CachyLlama's tuned tile FA.

To build rocWMMA's own gfx1151 samples and validation tests from source:

```bash
git clone --branch rocm-7.2.4 --depth 1 https://github.com/ROCm/rocWMMA.git
cmake -S rocWMMA -B rocWMMA/build \
  -DCMAKE_C_COMPILER=/opt/rocm-7.2.4/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.4/bin/amdclang++ \
  -DGPU_TARGETS=gfx1151 \
  -DROCWMMA_BUILD_TESTS=ON \
  -DROCWMMA_BUILD_SAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build rocWMMA/build -- -j 16
ctest --test-dir rocWMMA/build --output-on-failure
```

The rocWMMA `rocm-7.2.4` release contains the 2.2.0 source rebuilt for the 7.2.4 stack. Use the
installed 7.2.4 headers for llama.cpp; use the source build only to inspect and
exercise rocWMMA samples/tests. Sources:

- https://github.com/ROCm/rocWMMA/releases/tag/rocm-7.2.4
- https://rocm.docs.amd.com/projects/rocWMMA/en/latest/install/installation.html
- https://github.com/ROCm/ROCm/releases/tag/rocm-7.14.0
- https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html

- The GQA-dequant patch (#3) is likely to conflict on rebase (touches `fattn-vec.cuh` /
  `fattn-tile.cuh` dispatch, which upstream changes often). Keep it isolated.
- The MMQ tuning (#2) is the most stable (only touches `mmq`/`mmvq` config tables).
- The ROCmFPX dtype patchset intentionally makes this branch format-divergent from stock llama.cpp.
  ROCmFPX models will not load in an unmodified stock build, and base updates may require coordinated
  changes across GGUF types, quantization code, model loading, and HIP kernels.
- Because CachyLlama is now the base, review its full update range before rebasing or merging; do not
  assume the branch has the low divergence of an upstream-based carry branch.
- Build flags used by the Strix Halo community: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
  -DLLAMA_HIP_UMA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`. ROCmFPX/CachyLLama references also use
  `GGML_HIP_ENABLE_UNIFIED_MEMORY`.

## Testing matrix

The following benchmark commands are the standard verification suite for gfx1151
performance changes. Run each with 3 repetitions (`-r 3`), page cache warmed, box idle.

### Core benchmark configuration

```bash
export ROCM_PATH=/opt/rocm-7.2.4
export PATH="$ROCM_PATH/bin:$PATH"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
```

### pp512 throughput sweep (ROCm backend)

```bash
# Qwen3.5-122B-A10B Q4_K_M — primary Strix Halo benchmark model
./build-rocm/bin/llama-bench \
  -m Qwen3.5-122B-A10B-Q4_K_M.gguf \
  -p 512 -n 64 -d 32768 -c 32768 \
  -ngl 99 -fa on -b 2048 -ub 2048 \
  -r 3 --no-warmup

# Qwen3.6-35B-A3B Q6_K — MoE model for GQA decode fix
./build-rocm/bin/llama-bench \
  -m Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
  -p 512 -n 64 -d 16384 -c 16384 \
  -ngl 99 -fa on -b 2048 -ub 2048 \
  -ctk q8_0 -ctv q8_0 -r 3 --no-warmup
```

### Depth sweep (ROCm backend) — detect crossover points

```bash
# Decode throughput at varying context depths (Qwen3-Coder-30B-A3B, q8_0 KV)
for depth in 4096 16384 32768 65536 131072; do
  echo "=== depth=$depth ==="
  ./build-rocm/bin/llama-bench \
    -m Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
    -p 512 -n 64 -d $depth -c $depth \
    -ngl 99 -fa on -b 2048 -ub 2048 \
    -ctk q8_0 -ctv q8_0 -r 3 --no-warmup
done
```

### Vulkan vs ROCm comparison

```bash
# Same model on both backends for the same depth range
./build-rocm/bin/llama-bench ...  # ROCm
./build-vulkan/bin/llama-bench ...  # Vulkan
```

### `nodes_per_submit` value sweep (Vulkan only)

```bash
for N in 4 8 12 16; do
  echo "=== nodes_per_submit=$N ==="
  GGML_VK_NODES_PER_SUBMIT=$N ./build-vulkan/bin/llama-bench \
    -m Qwen3.5-122B-A10B-Q4_K_M.gguf \
    -p 512 -n 64 -d 32768 -c 32768 \
    -ngl 99 -fa on -b 2048 -ub 2048 -r 3 -no-warmup
done
```

### Vulkan FA scratch safety margin sweep

```bash
# Test with decreasing safety margins on long-context prefill (needs 64 GB UMA)
for MB in 1024 512 256 128 64; do
  echo "=== FA_SCRATCH_SAFETY_MB=$MB ==="
  GGML_VK_FA_SCRATCH_SAFETY_MB=$MB ./build-vulkan/bin/llama-bench \
    -m Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
    -d 32768 -p 512 -n 64 -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -r 3 -no-warmup
done
```

### Correctness verification (run after each carry commit)

```bash
# Focused FA attention extension test suite
./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT
./build-vulkan/bin/test-backend-ops -o FLASH_ATTN_EXT

# Spot-check greedy generation byte-identical to stock
./build-rocm/bin/llama-cli -m model.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -p "Hello, world" -n 128 --temp 0
# Compare output to stock CachyLlama build; should be byte-identical
```

### rocWMMA build (compilation check, not runtime on gfx1151)

```bash
export ROCM_PATH=/opt/rocm-7.2.4
export PATH="$ROCM_PATH/bin:$PATH"
cmake -S . -B build-rocwmma \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_HIP_UMA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocwmma --config Release -- -j 16
```

Important: CachyLlama excludes gfx1151 from the rocWMMA FA runtime path. This build
compiles successfully but does NOT make gfx1151 use the WMMA kernel at runtime. The
WMMA kernel is compiled but dispatch falls through to the tile path.

### ROCmFPX integration verification (once ported)

```bash
# Quantize a BF16 model to ROCmFP4
./build-rocm/bin/llama-quantize model-BF16.gguf model-ROCMFP4_FAST.gguf Q4_0_ROCMFP4_FAST

# Run with ROCmFP4 model, Vulkan backend
./build-rocm/bin/llama-cli \
  -m model-ROCMFP4_FAST.gguf -dev Vulkan0 -ngl 99 -fa on --jinja \
  -c 4096 -b 512 -ub 512

# Run with MTP speculative decoding
./build-rocm/bin/llama-cli \
  -m model-with-MTP.gguf -dev Vulkan0 -ngl 99 -fa on --jinja \
  --temp 0 --spec-type draft-mtp \
  --spec-draft-n-max 4 --spec-draft-p-min 0.55

# Benchmark ROCmFP4 vs Q4_K_M decode at various depths
./build-rocm/bin/llama-bench \
  -m model-ROCMFP4_FAST.gguf -d 32768 -p 512 -n 64 \
  -ngl 99 -fa on -b 2048 -ub 2048 -r 3 --no-warmup
```

---

## Next steps and potential optimizations — ROCm only (2026-07-23)

All items below target Strix Halo (`gfx1151`, RDNA3.5, UMA) via the ROCm/HIP backend.
Vulkan-only optimizations, CPU-only optimizations, and system-level items are excluded;
they belong in a separate Vulkan-specific tracking document if needed.

### Done (already on `halo`)

| # | Optimization | Commit(s) | ROCm gain |
|---|---|---|---|
| 1 | GQA-dequant tile FA decode fix | `5cf8b7a5c`, `c4f21afe1` | **+127–322% tg@32–128k** q8_0 |
| 2 | **PR #16827 rocWMMA prefill tuning** | `a3c9d1d` | pp gains expected at 4096/16384/65536 |
| 3 | MMQ RDNA3_5 tuning (`mmq_x=48, mmq_y=64, nwarps=4`) | `db7ce487a`, `b6face667` | **+50–70% pp** short ctx |
| 4 | `sudot4` for RDNA3.5 (`__builtin_amdgcn_sudot4`) | upstream `common.cuh:699` | included |
| 5 | `__float2int_rn` in quantize.cu | upstream `quantize.cu:65` | included |
| 6 | SSD cache `posix_fadvise` readahead | `864cadc49` | included |

### Not yet integrated

#### #1 — ROCmFPX weight formats + MTP (charlie12345)

**Source fork:** `https://github.com/charlie12345/ROCmFPX` (author: `caf`, Strix Halo)
**Target branch:** `main` (do NOT use `experimental-rocmfpx-branch`)
**Strix Halo build script in ROCmFPX:** `scripts/build-strix-rocmfp4-mtp.sh`

**What it does:** Adds ROCmFP3/4/6/8 GGUF model-weight formats with native HIP/ROCm
kernels. ROCmFP4 is 4.25 bpw (speed-first) vs Q4_K_M's ~4.5 bpw — smaller and faster
on gfx1151. Includes MTP (speculative decoding via embedded draft head) for 1.5–2.4×
decode speedup.

**Verified gfx1151 results (ROCm0 backend):**

| Model | Q4_K_M decode | ROCmFP4 STRIX_LEAN decode | Speedup |
|-------|---------------|---------------------------|---------|
| Qwen3.6-27B | 11.74 t/s | 13.53 t/s | +15.2% |
| Qwen3.6-35B-A3B | 66.42 t/s | 106.2 t/s (MTP n4/p0.55) | 1.60× (MTP) |

**Files to port (minimum set for gfx1151 ROCmFP4):**
1. `ggml/rocmfpx/` and `ggml/rocmfp4/` — dequant kernels, reference paths
2. `src/llama-model.cpp` — tensor type routing, GGUF type IDs
3. `common/` — CPU quantization reference paths
4. `mmq-config-rdna3_5.cuh` — MMQ/MMVQ entries for ROCmFP4 quant types
5. `llama-quantize` integration for `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q4_0_ROCMFP4_COHERENT`
6. `scripts/quantize-rocmfpx-agent.sh`, `docs/ROCmFPX-HANDOFF.md`
7. MTP speculative decoding support (server + CLI hooks)
8. TurboQuant K/V cache types (`turbo3`/`turbo4`) — separate from model-weight ROCmFPX

**Integration plan:** Port from ROCmFPX `main` branch as one isolated commit set on `halo`.
Keep ROCmFPX changes in a dedicated subdirectory. After porting, benchmark ROCmFP4 decode
at 16k/32k/64k/128k on gfx1151 vs Q4_K_M baseline. Expected: 8–15% weight-quant speedup,
1.5–2.4× with MTP.

---

#### #2 — q4_0-specialized FA tile loader

**Source analysis** (from Nathanw1014 docs): The tile FA kernel's `dequantize_V_q4_0`
only supports `ne ∈ {2,4}` (4-byte int nibble decoding), so a 32-element tile chunk needs
16 calls instead of 4 for q8_0. Each `qs` byte is fetched twice (low nibble for element j,
high nibble for element j+16) through the generic interface → q4_0 issues ~2× the unpack
ALU and ~2× the byte-fetch overhead vs q8_0.

**Estimated ROCm gain:** q4_0 decode at ~85% vs q8_0's 95% of bandwidth ceiling. A
specialized loader fetching each byte once could recover ~15% gap on gfx1151.

**Integration plan:** Add `dequantize_V_q4_0_tile/ne32` primitive to `fattn-tile.cuh` that
loads 32 q4_0 elements in one pass (16 bytes → 32 nibbles). Template q4_0 alongside q8_0
in the `type_K`/`type_V` dispatch. Verify with `test-backend-ops -o FLASH_ATTN_EXT`.
**Benchmark target:** q4_0 decode at 16k/32k/64k on gfx1151, move from 85% to ≥92% of q8_0.

---

#### #3 — `__expf()` intrinsics for MoE routing (from #21284)

**Status:** NOT present in the codebase (`__expf` not found in any `.cu`/`.cuh` file).
**ROCm relevance:** High — Qwen3.5 MoE models are the primary gfx1151 workload.

**Files to modify:**
- `ggml/src/ggml-cuda/gated_delta_net.cu` — MoE expert gating (rocm path via HIP)
- `ggml/src/ggml-cuda/ggml_cuda_op_silu_single.cu` — SiLU/F_silu activation

**Change:** Replace `expf(x)` with `__expf(x)` in gated DeltaNet's expert routing and
SiLU activation. The precision difference between `expf` (C library) and `__expf`
(hardware intrinsic) is below the noise floor for MoE routing decisions on rocm.

**Verification:** Run `test-backend-ops` focusing on gated delta net and SiLU operators.
Run `llama-bench` on Qwen3.5-122B-A10B Q4_K_M pp512@4096 (ROCm) to confirm bit-identical
output with greedy sampling.

---

#### #4 — MMQ per-quant tuning for gfx1151

**Background:** The `mmq-config-rdna3_5.cuh` has RDNA3_5 entries (via CachyLlama commit
`b6face667` which bumped I from 48 to 64 for all 232 MMQ cases). However, MMQ x/y/nwarps
are one-size-fits-all across qtypes — different block sizes (Q4_K vs Q8_0 vs F16) have
different register pressure profiles on gfx1151.

**Work remaining:** Profile each qtype at common block sizes on gfx1151 and adjust MMQ
tile dimensions per-quant. The I dimension (64) is already correct.

**Verification:** Build with `DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_CUDA_FA_ALL_QUANTS=ON`,
run `llama-bench` with each qtype separately on Qwen3.5-122B Q4_K_M, compare pp512/tg128.

---

#### #5 — ROCm vs ROCm dispatch heuristic (optional, low priority)

**Background:** Nathanw1014 benchmarks show a crossover on gfx1151: Vulkan wins at short
context (depth 0: 65.45 vs 55.14 t/s) but ROCm wins at long context (depth 128k:
19.77 vs 17.85 t/s). The current `halo` branch requires choosing a backend at build time.

**Integration concept:** A single ROCm build that selects the HIP dispatch path at runtime
based on expected context length. This is a more complex change touching the FA dispatch
logic and CLI/server config.

**Verification:** Build with ROCm only, benchmark Qwen3-Coder-30B-A3B q8_0 decode at
depths 0/4k/16k/32k/64k/128k, confirm the crossover point, test runtime heuristic
selects the right path at each depth.

---

### Explicitly excluded (not ROCm-targeted)

| Item | Reason excluded |
|---|---|
| Vulkan quantized-KV FA prefill | Vulkan backend only (already on halo) |
| Vulkan `nodes_per_submit` auto-lower | Vulkan backend only (already on halo) |
| Vulkan `nodes_per_submit` value sweep | Vulkan-only profiling |
| Vulkan FA scratch safety margin tuning | Vulkan-only feature |
| Vulkan vs ROCm runtime dispatch | Vulkan-focused heuristic |
| CPU ISA auto-detection for Vulkan builds | CPU-side, not ROCm |
| GTT/VRAM BIOS partitioning | System-level, not backend-specific |
| `posix_fadvise` readahead for SSD | General, already on halo |
