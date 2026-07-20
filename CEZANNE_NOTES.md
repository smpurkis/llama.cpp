# Cezanne (gfx90c) Vulkan Notes

Last updated: 2026-07-20

## Goal

Document Vulkan-backend tuning findings for the Cezanne APU (gfx90c,
5800H/5700G with Radeon Vega iGPU). Companion to `RDNA3_NOTES.md`
(gfx1103 / 7840U) and `STRIX_HALO_NOTES.md` (gfx1151 / Ryzen AI Max+ 395).

## Environment

- CPU/GPU: AMD Ryzen 7 5800H (Minisforum UM580+ "zaphod"), Radeon Graphics (Cezanne)
- GPU arch: gfx90c (GCN5.1 / Vega)
- VRAM carveout: 512 MiB (BIOS default, `amdgpu.vis_vramlimit=0`)
- GTT: ~24 GiB effective (kernel `ttm.pages_limit=6291456`)
- System RAM: 30 GiB DDR4
- Mesa driver: RADV (open source)
- Build: CachyLLama `0da40ed6c` Vulkan backend
- Wave size: 64 (GCN, same as RDNA)
- Subgroup: KHR_coopmat not supported on gfx90c

## Tier classification

zaphod's 512 MiB VRAM carveout rounds to 0 GiB in integer division, which
would normally drop it into the `handheld` tier. The 2026-07-20 work adds
a fallback in `scripts/detect-gpu.sh`:

```
VRAM < 1 GiB AND system RAM >= 24 GiB  ->  standard tier
```

This catches zaphod (and other mini-PCs with tiny BIOS carveouts but
desktop-class RAM) and gives them the `standard` profile instead of being
misclassified as a Steam Deck. The 7840U is unaffected because its VRAM
(6 GiB) is well above the 1 GiB threshold.

## Vulkan nodes_per_submit sweep (nps)

Commit `1c19480da` lowered the default `max_nodes_per_submit` from 100 to 8
on UMA devices to avoid hitting the kernel's 2s `amdgpu.lockup_timeout`.
For the 7840U (`gfx1103`), nps=64 gave a reproducible +4.5% tg64 win
(see `RDNA3_NOTES.md`). The Cezanne sweep does NOT replicate that win.

**Models:** DeepSeek-R1-7B-Q4_K_M, qwen2.5-14b-Q4_K_M, Qwen3.6-27B-Q4_K_XL,
Qwen3.6-35B-A3B-UD-Q4_K_XL
**Settings:** `-ngl 99 --flash-attn 1 --mmap 0 --direct-io 1 -b 2048 -ub 2048`
**Repetitions:** 3 per run

| Model    | nps=8 pp1024 | nps=64 pp1024 | nps=8 tg64 | nps=64 tg64 |
|----------|-------------:|--------------:|-----------:|------------:|
| 7B       | 96.16        | 95.66         | 10.01      | 10.15       |
| 14B      | 46.14        | 46.08         | 5.20       | 5.28        |
| 27B      | 24.44        | 24.45         | 2.44       | 2.52        |
| 35B-A3B  | 132.75       | 133.89        | 12.74      | 12.90       |

7B full sweep (nps ∈ {1,4,8,16,32,64,100}): pp range 95.50-96.46 (1% spread),
tg range 9.78-10.15 (4% spread with one outlier). All within noise.

**Selected value: 8 (default).**

Reasoning:
- tg64 gain at nps=64 vs nps=8 across all models is +1.3% to +3.3% - mostly
  within noise
- pp is unchanged across the sweep
- The 7840U got +4.5% tg with nps=64 because its 12 RDNA3 CUs benefit from
  deeper submit queues. Cezanne's 8 Vega CUs at lower clocks are
  compute-bound, not submit-bound
- Adding an override for marginal benefit adds profile complexity for no
  real-world gain

**Where this is wired in:** `llama-run.sh` `setup_vulkan_env()` has a
comment block explaining why gfx90c is intentionally NOT in the nps
override. Manual override is still possible: `export
GGML_VK_NODES_PER_SUBMIT=N` before running `llama-server`.

## ubatch sweep (nps=8, q8_0 KV)

| Model | ub=256 pp | ub=512 pp | ub=1024 pp | ub=2048 pp | tg64 (all) |
|-------|----------:|----------:|-----------:|-----------:|-----------:|
| 14B   | **46.85** | 46.22     | 45.82      | 45.71      | 5.08-5.09  |
| 27B   | **25.17** | 24.66     | 24.57      | 24.53      | 2.44       |
| 35B-A3B | 90.42   | 112.08    | 132.68     | **133.58** | 12.82-12.84 |

**Findings:**
- For dense models (14B, 27B), smaller ubatch wins. ub=256 is +1-2.5%
  faster than ub=512 (the standard tier default).
- For MoE (35B-A3B), larger ubatch wins. ub=2048 is +48% faster than
  ub=256 (the handheld tier default). Active compute per token is only
  ~3B params, so compute buffers stay small and ubatch up to 2048 fits.
- The handheld tier's ub=256 for MoE is a hard-lock safety valve (see
  llama-run.sh comment in the MoE handheld branch: "ubatch 512 causes
  GPU hard-lock at ~3K tokens"). This was measured on Phoenix (gfx1103)
  with 6 GiB VRAM; not validated for Cezanne's 512 MiB VRAM + 24 GiB
  GTT setup. The standard tier's ub=512 is the conservative default that
  shipped with the tier change.

**Recommendation:** Standard tier (which zaphod now uses) keeps ub=512.
Cezanne-specific override to ub=256 for dense models would give +1-2% pp
but isn't worth a per-GPU branch in the profile. MoE on Cezanne could
go higher (1024-2048) but the hard-lock risk hasn't been measured past
pp1024. Document and leave for future validation.

## KV cache type sweep (27B and 35B-A3B)

| Model | KV type | pp1024 | tg64  |
|-------|---------|-------:|------:|
| 27B   | q4_0    | 24.22  | 2.44  |
| 27B   | q8_0    | 24.49  | 2.44  |
| 35B-A3B | q4_0  | 130.53 | 12.84 |
| 35B-A3B | q8_0  | 132.45 | 12.84 |

q8_0 is +1-1.5% pp at no tg cost. Standard tier already uses q8_0, so
this is just confirmation. The previous handheld large-dense profile
used q4_0 which is no longer the bottleneck.

## Tier-related fixes in this pass

1. **`scripts/detect-gpu.sh`**: added VRAM<1 GiB + RAM>=24 GiB -> standard
   tier fallback. See "Tier classification" above.
2. **`llama-run.sh`** MoE handheld (`moe-optimized` profile, `*` branch):
   floor `--cache-ram $(( LLAMA_APU_VRAM_GB * 1024 - 2048 ))` at 1024 MiB.
   The unfixed formula produces `-2048` when VRAM rounds to 0 GiB.
   zaphod no longer hits this path (standard tier) but other low-VRAM
   systems still might.
3. **`llama-run.sh`** large-dense handheld (`large-dense` profile, `*`
   branch): same cache-ram floor fix.
4. **`llama-run.sh`** `setup_vulkan_env()`: added comment explaining why
   gfx90c is NOT in the nps=64 override. No code change - just docs.

## Validation

Smoke test on zaphod with standard-tier profile (Qwen3.6-35B-A3B-Q4_K_XL):

| Phase  | Wall (ms) | pp_ms | gen_ms | Speedup |
|--------|----------:|------:|-------:|--------:|
| Cold   | 8790      | 2895  | 5845   | 1.0x    |
| Warm   | 6233      | 282   | 5888   | 1.41x   |

SSD cache restores in-memory state across restarts. Wall speedup is
modest on small prompts (118 tokens) because most of the 6.2s wall time
is generation. pp_ms drops 10x (2895 -> 282 ms) confirming the cache hit.

Full benchmark (small/medium/large cold+warm) for Qwen3.6-27B at the
prior zaphod run (2026-07-20) showed 4.22x TTFT speedup at 15K tokens,
which should hold or improve with the standard tier profile (q8_0 KV +
ub=512 + cache-ram 8192 instead of the previous q4_0 + ub=512 + broken
cache-ram formula).

## Files changed in this pass

- `scripts/detect-gpu.sh` - tier detection fallback
- `llama-run.sh` - cache-ram floor fix (2 sites), setup_vulkan_env comment
- `CachyLLama/CEZANNE_NOTES.md` - this file
