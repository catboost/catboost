# Sprint 18 — S18-09 Metal System Trace Verification

**Branch:** `mlx/sprint-18-hist-privhist-tile`
**Kernel under verification:** `histogram_one_byte_features` (L1a layout, `kernel_sources.h:100–266`)
**Status:** PARTIAL — static (source-derivable) facts verified; dynamic (xctrace-derived) facts BLOCKED on permission to run `xcrun xctrace`.

---

## Status summary

The agent does not have unrestricted Bash permission in this session. `xctrace`, `xcrun metal-profiler`, and the `csv_train_s18_fixed` launch were denied by the sandbox the moment the parallel batch attempted to invoke them. No `.trace` bundle was produced for Sprint 18, and on inspection no `mst_*.trace` exists for Sprint 17 either — the Sprint 17 `.cache/profiling/sprint17/` directory contains only bench JSON results (`baseline_*`, `after_*`, `s17_gate_after.json`), not Instruments traces. The Sprint 17 MST capture appears to never have actually been run.

The findings below split into:
- **Verified statically** from `kernel_sources.h` (the live MLX JIT source string at HEAD on this branch) — high confidence, deterministic from source.
- **Blocked on xctrace** — exact command + what the trace would tell us is listed at the bottom for Ramos to run inline.

---

## Static verification (from `catboost/mlx/kernels/kernel_sources.h`)

### (b) Threadgroup memory allocation — VERIFIED 32 KB

`kernel_sources.h:151`:
```
threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]; // 32 KB
```
Where `NUM_SIMD_GROUPS = BLOCK_SIZE / SIMD_SIZE = 256/32 = 8` and `HIST_PER_SIMD = FEATURES_PER_PACK * BINS_PER_BYTE = 4 * 256 = 1024` (header lines 28–34).
- Static size = `8 × 1024 × 4 B = 32768 B = 32 KB`.
- This is the only `threadgroup` declaration in the kernel body (verified by grep — no auxiliary scratch).
- Metal will report exactly 32 KB; this is at the Apple Silicon per-threadgroup ceiling (32,768 B for M1/M2/M3 GPU families).

### (d) Barrier count — VERIFIED 6

Counted directly in `kernel_sources.h`:
- Line 163: `threadgroup_barrier(...)` — barrier 1, post zero-init.
- Line 210: barrier 2, post stride-partition accumulation.
- Line 238: 1 barrier *inside* a `for (uint tile = 0u; tile < FEATURES_PER_PACK; tile++)` loop with `FEATURES_PER_PACK = 4` → barriers 3, 4, 5, 6 (one per tile of the cross-SIMD fold).

Total = **6 barriers per dispatch** (1 zero-init + 1 accumulation + 4 cross-SIMD fold), exactly matching the expectation. Down from Sprint 17's 9. The kernel comment at line 218–219 documents the count itself, and the source confirms it.

### (a) Register usage — INFERRED, NOT INSTRUMENTS-CONFIRMED

The L1a kernel removes all `private` arrays from the per-thread context. Only scalar locals survive in the accumulation loop: `tid, lane, simd_id, batch_start, d, valid, packed, stat, p_s, s_s, valid_s, src, f, bin, foldBase, partOffset, partSize, docsPerBlock, myDocStart, myDocEnd, myDocCount, featureColumnIdx`. None of these are arrays. The 4 KB per-thread `privHist[1024]` that drove the Sprint 17 spill is gone.

Lower-bound from source: there is no construct in the kernel body that requires more than ~40–60 live VGPRs at peak (the inner `simd_shuffle` loop over 32 sources × 4 features holds at most a handful of values live at once). Apple Silicon's M-series GPU advertises 32 VGPRs per thread as the boundary for full occupancy and supports up to 128 with throttled occupancy. With the spill gone, the kernel should comfortably sit in the ≤32 VGPR tier — i.e. ≥75% reduction vs Sprint 17's reported 128 VGPRs.

**Caveat:** the *actual* VGPR count is determined by the AGX compiler, not the Metal IR, and is only observable via Instruments / `metal-profiler` / pipeline statistics — see blocked commands below. Treat the "≤32 VGPR / ≥75% reduction" estimate as a source-grounded prediction, not a measured fact.

### (c) Occupancy — NOT VERIFIABLE FROM SOURCE

Threadgroup memory at 32 KB per threadgroup is at the Apple Silicon per-tg ceiling, so even if VGPRs allow more, occupancy is **tg-mem-limited to 1 threadgroup per SM** by construction. This is a structural ceiling, not an empirical one — Instruments would only confirm. There is no path to ≥2 tg/SM at this layout; the only "unexpected bonus" outcome would be the compiler somehow folding two dispatches into one SM via a feature I am not aware of, which I do not expect.

---

## Blocked dynamic verification

The following requires Bash permission. Each command is exactly what I would run; the second column lists what each would prove.

| Command | Validates |
|---|---|
| `xcrun xctrace record --template "Metal System Trace" --output .cache/profiling/sprint18/mst_s18.trace --target-stdout - --launch -- ./csv_train_s18_fixed --loss-function RMSE --depth 6 --iterations 50 --learn-set sample.csv --max-bin 128` | Captures the trace artifact for question 1–3 + 5. Replace `sample.csv` with the gate-config dataset (`csv_train_sprint16` or whichever 10k/RMSE/d6/128bins fixture the bench harness uses — confirm with `benchmarks/check_histogram_gate.py`). |
| `xcrun xctrace export --input .cache/profiling/sprint18/mst_s18.trace --xpath '/trace-toc/run[@number=1]/data/table[@schema="metal-pipeline-statistics"]'` | Pulls per-kernel VGPR count, occupancy, threadgroup memory bytes, instruction mix → answers (a), (b), (c) numerically. |
| `xcrun xctrace export --input .cache/profiling/sprint18/mst_s18.trace --xpath '/trace-toc/run[@number=1]/data/table[@schema="metal-gpu-counter-set-counters"]'` | Memory bandwidth (DRAM read/write GB/s), cache hit rates, ALU/MEM stall ratio → answers (e) Sprint 19 planning signals. |

**If the above commands run cleanly,** the parsed numbers should be appended to this file under "Measured" sub-sections for (a)/(b)/(c)/(e). The JSON exports themselves should land at `.cache/profiling/sprint18/mst_s18.{pipeline,counters}.xml`.

### What to flag if measurements diverge

- **VGPR > 64.** Spill returned somewhere unexpected (probably the inner `simd_shuffle` loop's 32× unroll — try `[[loop_unroll(8)]]` or a manual outer/inner split).
- **Threadgroup memory ≠ 32 KB exactly.** Compiler padded the array; recheck `static_assert` in `kernel_sources.cpp` and confirm `simdHist` is the only `threadgroup` decl.
- **Occupancy ≥2 tg/SM.** Bonus — would imply the GPU silently relaxes the 32 KB ceiling on this family, worth a short follow-up note in `docs/sprint18/results.md`.
- **Barrier count in trace ≠ 6.** Either the JIT inlined a helper that adds barriers, or `simd_shuffle` is being lowered to something that implies an extra fence on this driver. Worth digging into.

---

## (e) Sprint 19 planning signals (preliminary, source-grounded)

Even without the trace, the source structure already telegraphs likely Sprint 19 levers:

1. **DRAM bandwidth in the doc loop is unchanged.** L1a only restructures *threadgroup* and *register* memory. Each doc still pays one `compressedIndex[docIdx*lineSize+col]` and one `stats[stat*totalNumDocs+docIdx]` global load, gated by `docIndices[sortedPos]`. At N=50k × 100 features × 50 iters, this is the next-likely bottleneck. MST counters would confirm by reporting %-of-peak DRAM BW.
2. **The 32-iteration `simd_shuffle` unroll over 32 source lanes × 4 features = 128 inner ops** dominates the accumulation arithmetic. If pipeline statistics show "ALU bound, low memory stall," then Sprint 19 wants to either fuse FEATURES_PER_PACK=4 into a single packed-bin compare, or coalesce the foldCount predicate into a precomputed mask.
3. **Atomic writeback at the tail** still does one `atomic_fetch_add` per non-zero bin per partition. With BLOCK_SIZE=256 threads and ≤1024 bins per tile, this is fine for accuracy, but at small N (1k config), the writeback may dominate launch overhead. MST "kernel duration distribution" would tell us.
4. **Threadgroup memory is at the ceiling.** Any Sprint 19 idea that wants more (e.g., double-buffered accumulation) is structurally blocked at L1a. Sprint 19 either re-tiles down to 12 KB (L1b layout) and pays 4× DRAM re-reads, or finds a different lever entirely.

---

## Reproducibility

- Branch: `mlx/sprint-18-hist-privhist-tile`
- Source verified at: HEAD of branch (commit `dccb7ec0a2` on the surrounding tree; L1a kernel content is in `kernel_sources.h:100–266`).
- Hardware needed for capture: same Apple Silicon machine where Sprint 17/18 bench results were collected (any consistency between captures matters more than the specific chip).
- Tooling: Xcode Command Line Tools providing `xcrun xctrace`. `xcrun --find xctrace` was attempted to confirm but the call was denied; presence at `/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace` is normal on a developer machine.
