# Architecture & Design Decisions — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Sprint 16+ is source of truth.

## DEC-001: MLX over PyTorch MPS for GPU backend

**Sprint**: 0 (project inception)
**Problem**: Which framework to use for Apple Silicon GPU acceleration of CatBoost?
**Considered**: (a) PyTorch MPS backend, (b) Apple MLX framework, (c) Raw Metal
**Chosen**: MLX
**Rationale**: MLX provides direct Metal kernel dispatch via `mx::fast::metal_kernel`, unified memory model, lazy evaluation graph, and C++ API. PyTorch MPS adds a layer of abstraction that limits kernel customization. Raw Metal has no array abstraction. MLX is the sweet spot. [reconstructed from CLAUDE.md]
**Status**: Standing decision.

## DEC-002: `mx::fast::metal_kernel` for custom GBDT kernels

**Sprint**: 5–6 era
**Problem**: How to implement GBDT-specific GPU operations (histogram, scoring, leaf estimation)?
**Considered**: (a) High-level MLX ops only, (b) Custom Metal kernels via `mx::fast::metal_kernel`
**Chosen**: Custom Metal kernels
**Rationale**: GBDT histogram and scoring patterns don't map cleanly to standard linear algebra ops. Custom kernels allow optimal threadgroup sizing, shared memory usage, and GBDT-specific data layouts. [reconstructed from histogram.cpp, score_calcer.cpp]
**Status**: Standing decision.

## DEC-003: int32 accumulator in ComputePartitionLayout (16M row limit fix)

**Sprint**: 9
**Commit**: `227a6e6455`
**Problem**: float32 accumulator in partition layout overflows at ~16M rows (precision loss beyond 2^24).
**Chosen**: Use int32 accumulator for exact partition counts.
**Rationale**: int32 supports up to 2^31 rows. Avoids float precision trap. [from commit message]
**Status**: Resolved. Guard enforces 16M-row safety limit was added in Sprint 4 (`8717dddd5b`), int32 fix in Sprint 9.

## DEC-004: Sprint branches push to origin (RR-AMATOK) only

**Sprint**: 4
**Commit**: `e7c7c67d73`
**Problem**: Prevent accidental pushes to upstream `catboost/catboost`.
**Chosen**: All sprint branches push to `origin` (`RR-AMATOK/catboost-mlx`) only. Never push upstream.
**Rationale**: Explicit standing instruction from Ramos. Enforced by convention, not git hooks.
**Status**: Standing decision.

## DEC-005: Measurement before optimization (Operation Verstappen)

**Sprint**: 16
**Date**: 2026-04-15
**Problem**: MLX is 10–24x slower than CPU CatBoost. Six bottlenecks identified by code reading. Which to fix first?
**Considered**: (a) Fix all obvious bottlenecks immediately, (b) Fix one zero-risk free-move + measure everything, (c) Pure diagnosis sprint
**Chosen**: (b) Diagnosis + sync-storm fix
**Rationale**: Without per-stage attribution, we cannot sequence Sprints 17–24 by actual impact. Shipping three fixes simultaneously confounds attribution. The sync-storm fix (removing 18 EvalNow calls in pointwise_target.h) is provably correct by static inspection and does not conflict with profiler instrumentation. One free-move + measurement infrastructure is the optimal first step for a multi-sprint campaign.
**Status**: Active. Sprint 16 in progress.

## DEC-006: Dual bin count (32 + 128) for championship benchmarks

**Sprint**: 16
**Date**: 2026-04-15
**Problem**: Published CatBoost benchmarks use different bin counts. Which should we use for the dominance scoreboard?
**Considered**: (a) 128 only (CatBoost default), (b) 32 only (faster), (c) Both
**Chosen**: (c) Both, report separately
**Rationale**: Honest comparisons require matching published numbers. arXiv 1810.11363 reports both; szilard uses 128 (CatBoost default). Doubles benchmark runtime but keeps all comparisons defensible.
**Status**: Active. Benchmark harness will emit two columns.

## DEC-007: Small-N CPU fallback below 5k rows

**Sprint**: 16
**Date**: 2026-04-15
**Problem**: GPU launch overhead (~50-200us per kernel) makes MLX uncompetitive on tiny datasets.
**Considered**: (a) Optimize MLX for all N, (b) CPU fallback below threshold, (c) Hybrid dispatch
**Chosen**: (b) CPU fallback below ~5k rows
**Rationale**: GPU physics — sub-millisecond launch overhead floor means 1k-row datasets will never beat a tuned CPU path. Championship push focuses on N ≥ 10k where GPU parallelism has headroom. Runtime dispatch decision deferred to Sprint 22–23.
**Status**: Active. Threshold TBD based on Sprint 16 profiling.

## DEC-008: Parity tolerance envelope — RMSE/Logloss ulp≤4, MultiClass ulp≤8

**Sprint**: 17
**Date**: 2026-04-17
**Problem**: Reordered FP32 reductions (D1c tree reduction vs serial baseline) are not bit-exact by construction. What tolerance gates merge?
**Considered**: (a) Bit-exact mandatory (blocks all parallel reductions), (b) ulp≤4 universal (matches CatBoost CPU/CUDA parity conventions), (c) loss-specific tolerances derived from Higham γ_N analysis
**Chosen**: (c) Loss-specific, derived from Higham γ_8 bound on the 8-term cross-SIMD fold: **RMSE ulp≤4, Logloss ulp≤4, MultiClass ulp≤8**
**Rationale**: MultiClass loss composes three independent reductions (one per class dim) before the final loss aggregation; error compounds accordingly. RMSE and Logloss have single-reduction structure. Higham γ_8 at FP32 is ~1e-6 relative — well within 4 ulp on typical loss magnitudes.
**Scope**: Bounded to `approxDim ∈ {1, 3}`, `N ≤ 50k`, 50 iterations, depth 6. Outside this envelope the Σ|xᵢ| growth and fold-depth compounding may exceed these bounds; re-validation required before extending.
**Status**: Active, hard merge gate. Sprint 17 actual: 35/36 checkpoints ulp=0 (exceeds bound). See `docs/sprint17/parity_results.md`.

## DEC-009: Linear 8-term cross-SIMD fold (over 3-level butterfly)

**Sprint**: 17
**Date**: 2026-04-17
**Problem**: D1c's 8-way cross-SIMD reduction can use either a linear sum (`simdHist[0]+simdHist[1]+...+simdHist[7]`) or a 3-level balanced butterfly. Which to ship?
**Considered**: (a) 3-level balanced butterfly (tighter error, more code), (b) Linear 8-term sum (simpler, deterministic, matches ablation reference code in §1.2)
**Chosen**: (b) Linear 8-term sum
**Rationale**: Linear form is deterministic (fixed simd_id order), trivially readable, and measured bit-exactly against baseline on 35/36 parity checkpoints. The one transient 17-ulp drift at iter=10 of 10k/MultiClass/32 healed to 0 by iter=20 — within Higham γ_8 envelope, does not justify the balanced-tree complexity today.
**Future trigger**: If Sprint 18+ parity sweeps at `approxDim > 3` or `N > 50k` show drift exceeding DEC-008 bounds, revisit with 3-level butterfly.
**Status**: Active. Implementation in `catboost/mlx/kernels/kernel_sources.h:208–220`.

## DEC-010: Sprint 18 L1 lever — reduce `privHist[1024]` register pressure

**Sprint**: 17 → forward to 18
**Date**: 2026-04-17
**Problem**: After D1c, kernel is compute-latency-bound, not memory-bandwidth-bound. Steady-state `histogram_ms` is ~23 ms at N=10k/32-bin; theoretical memory floor is ~0.13 ms. What is the next ceiling?
**Chosen**: `privHist[1024]` register-array spill (per-thread 4KB working set). Sprint 18 headline lever is tiled accumulation (256-lane × 4-pass fold) that preserves D1c's SIMD-shuffle benefit while cutting per-thread register use ~4×.
**Rationale**: Observed steady-state has ~175× headroom to memory-bandwidth floor. Register pressure is the architecturally obvious next bottleneck (confirmed via Sprint 17 profile analysis in `docs/sprint17/results.md` §Surprises 4). Details in `docs/sprint18/plan_prior.md`.
**Status**: Resolved — Sprint 18 shipped L1a (DEC-011/DEC-012).

## DEC-011: L1a per-SIMD shared threadgroup histogram layout

**Sprint**: 18
**Date**: 2026-04-17
**Commit**: `19fa5ce6cc`
**Problem**: `float privHist[HIST_PER_SIMD]` allocates 4 KB per thread, 1 MB per threadgroup, spilling entirely to device memory. Every accumulation RMW is a device-memory round-trip.
**Chosen**: Replace with `threadgroup float simdHist[8][1024]` — one 1024-float histogram per SIMD group, held in threadgroup memory. 8 SIMD groups × 1024 bins × 4 B = 32 KB. Stride-partition assigns ownership of bin `b` to lane `b & 31` within each SIMD group, giving each bin a single writer and zero atomics during accumulation.
**Rationale**: Eliminates the 4 KB/thread device-memory spill (the plurality cost at 27% of steady-state after D1c). Threadgroup memory is on-chip; RMW latency drops by ~100×. Zero-init is implicit for threadgroup memory, removing the 4.0 ms zero-init loop. Buffer is 2.67× larger than S17's 12 KB D1c layout, but eliminates ~1 MB/threadgroup of device-memory spill traffic per accumulation phase.
**Trade-off**: 32 KB is the Apple Silicon hard threadgroup-memory ceiling. Forces exactly 1 threadgroup/SM (down from ≥2 with 12 KB). No headroom for Sprint 19+ geometry changes without redesigning the buffer layout. Sprint 19 re-negotiates if the 1-tg/SM occupancy creates scheduling pressure.
**Scope**: `approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations (DEC-008 envelope). Results: `histogram_ms` -66.8% on gate config (28.75 → 9.56 ms), -56.6% to -85.5% across 18 configs.
**Status**: Active. Shipped Sprint 18.

## DEC-013: Two-phase writeback reduction over batched-atomic

> **STATUS: SUPERSEDED by DEC-014 on 2026-04-17.** Rationale: S19-01 ground-truth attribution (commit `d7ea14e28c`) falsified the premise — writeback is 0.79 ms / 5%, not 15 ms / plurality. R8 failed at 1.02–1.04× e2e. Pivoted to accumulation redesign.

**Sprint**: 19
**Date**: 2026-04-17
**Status**: SUPERSEDED by DEC-014
**Problem**: Under L1a (`simdHist[8][1024]`, 32 KB on-chip), the writeback phase — copying per-SIMD-group histograms from threadgroup memory to the global accumulation buffer via atomic adds — floors N=50k `histogram_ms` at ~15 ms regardless of accumulation improvements. This is the Sprint 19 headline bottleneck (identified in S18-05b). Two candidate reduction strategies exist: (a) two-phase on-chip reduction before global writeback, (b) batched-atomic writeback with reduced atomic conflict window.
**Considered**:
- (a) Two-phase reduction — after barrier-6 (end of accumulation), fold the 8 per-SIMD histograms into a single threadgroup-scope sum using `simdHist[0..1023]` as on-chip staging, then write the result to global memory with one atomic-free store per bin; eliminates all cross-threadgroup atomic contention during writeback.
- (b) Batched-atomic writeback — group bins into batches, serialize atomic writes per batch; reduces contention window but does not eliminate atomics.
**Chosen**: (a) Two-phase on-chip reduction. Ramos chose robustness ("whatever is more robust") over the batched-atomic approach.
**Rationale**: Two-phase eliminates cross-threadgroup atomic contention entirely, gives deterministic reduction order (parity-friendly — consistent with DEC-009's linear fold preference), and removes the ~15 ms writeback floor rather than shaving it. Reuses `simdHist[0..1023]` post-barrier-6 as on-chip staging — no additional threadgroup memory required, preserves DEC-011's 32 KB ceiling. The one-atomic-free-store-per-bin write is optimal for both latency and parity guarantees.
**Trade-off**: Requires an additional intra-threadgroup reduction pass (folds 8 simd histograms to 1) before the global write; adds ~1 barrier vs the current single writeback. If the 8-fold intra-TG reduction is itself memory-bound (unlikely at 32 KB on-chip), gain may be partial — S19-01 attribution will confirm.
**Parity note**: Deterministic reduction order means results are bit-reproducible across runs — a stronger guarantee than batched-atomic, which can vary in accumulation order depending on Metal scheduler behavior.
**Scope**: DEC-008 envelope (`approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations). Projection gate config: 50k/RMSE/128b. Full ablation in `docs/sprint19/ablation.md` (S19-02 complete; see SUPERSEDED note above).

## DEC-014: Accumulation redesign — wider batch (BATCH_DOCS=64)

**Sprint**: 19
**Date**: 2026-04-17
**Branch**: `mlx/sprint-19-hist-writeback`
**Status**: **REJECTED (2026-04-19).** Toy micro-bench post-T1 (`microbench_algorithmic.cpp`) showed A1 vs T1 = −1.9% (3-run mean, noise-marginal: stdev ~1%). Production port of A1 at 50k/RMSE/d6/128b measured **+9.4% REGRESSION** (T1-only 31.7 ms mean vs T1+A1 34.7 ms mean, 3 runs each). Parity bit-exact (0.48047778) — the regression is pure performance. Root cause: register pressure from lo/hi slab state (packed_lo/hi, stat_lo/hi, d_lo/hi, valid_lo/hi) composes with existing production live-ness (statIdx loop, partition offsets, docIndices gather) and pushes allocation over the VGPR spill threshold. Halved outer-loop count does not offset the spill cost. Fourth analytical model falsified this sprint (after DEC-013 writeback plurality, DEC-014 original gather sub-phase, DEC-015 col-major). A1 variant retained in `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` as `kA1Source` for record. Empirical drop disposition: `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md`. Plan clause "if not reproducible, drop" honored — A1 not shipped.

---

**Original DRAFT retained below for audit trail.**
**Supersedes**: DEC-013 (writeback two-phase reduction; falsified premise).
**Problem**: S19-01 ground-truth attribution (`docs/sprint19/attribution.md`) falsified the writeback-as-plurality hypothesis. At the 50k/RMSE/128b gate, writeback = 0.79 ms (5%) and accumulation = 14.30 ms (93%) of `histogram_ms` 15.43 ms. The DEC-013 writeback rewrite addresses a 5%-share lever and cannot clear R8 (≥1.5× e2e). The accumulation phase — the 32-doc cooperative scatter loop in `kernel_sources.h:175–209` — is the bottleneck.
**Considered** (full ablation in `docs/sprint19/ablation_accumulation.md`, S19-02b):
- (A1) BATCH_DOCS=64 wider batch — each lane holds 2 docs in registers across the inner shuffle loop; outer batch stride doubles. Halves outer-loop iter count (3 → 2 at gate); enables load-shuffle latency overlap via two-slab issue. TG memory unchanged 32 KB. Higham γ_7 unchanged.
- (B)  TG-memory doc staging — coalesce-load 32-doc tile into TG memory; all SIMD groups read from tile. **BLOCKED**: 35–58 KB TG memory exceeds DEC-011 32 KB ceiling. Sprint 20+ candidate iff DEC-011 renegotiated.
- (C)  Per-feature kernel specialization — 4 dispatches per feature group instead of 1. **KILLED**: 75 extra dispatches/iter × ~30 µs = +2.25 ms overhead, eats 2.8× the per-TG kernel-work saving. Net +1.45 ms (regression).
- (D)  16-lane stride-partition — 2 lanes own each bin, joined by 1-level XOR shuffle reduction. **DEFERRED**: algebraic-risk pattern (BUG-S18-001 lesson). Higham γ_7 → γ_8 at boundary of DEC-008 RMSE/Logloss ulp ≤ 4. Sprint 20 candidate (A1)+(D) iff (A1) alone misses R8.
**Chosen**: **(A1) BATCH_DOCS=64 standalone.** Sprint 20 stacking pathway: (A1) + (D) if S19-03/04 measurement shows (A1) alone misses R8 midpoint.
**Rationale**:
- **Lowest-risk, highest-confidence single variant.** No layout change, no new TG memory, no new dispatches, no XOR reduction proof obligation. Bit-exact reduction order preserved.
- **Projected `histogram_ms` 9.5 ms ± 1.2 ms (−38% vs 15.43 ms baseline).** Outer-loop iter count halved (3 → 2 at gate) + load-shuffle latency overlap from two-slab issue (6 in-flight loads, AGX max ~4). Worst-case sub-phase ranking (shuffle ALU bound): 11.5 ms (−26%); best-case (load-latency bound): 9.0 ms (−42%). Robust across all four sub-phase rankings.
- **R8 verdict: MARGINAL.** Projects e2e 1.39× midpoint (15.10 ms `iter_total_ms`), 1.51× lower bound, 1.29× upper bound. Reaches 1.5× target only at the lower-bound projection (~16% probability under symmetric error model). Path to clear R8: ship (A1) Sprint 19; if measured ≥1.5× → cleared; if measured 1.3–1.5× → Sprint 20 ships (D) on top, projected (A1)+(D) midpoint = 1.49×, lower bound = 1.67×; if measured <1.3× → escalate.
- **DEC-011 32 KB ceiling preserved.** `simdHist[8][1024]` unchanged. Net new threadgroup memory: 0 KB.
- **DEC-008 envelope preserved.** γ_7 unchanged → ≈4.2e-7 FP32, well within RMSE/Logloss ulp ≤ 4 (≈4.77e-7) and MultiClass ulp ≤ 8 (3× factor for K=3 → ≈1.3e-6).
- **Register pressure delta: +6 VGPR/lane** (3 → 9 VGPR for doc state). Within typical AGX VGPR budget (~256/thread); S19-03 must verify no spill via Metal compiler register-allocation report.
**Trade-off**: ~+30 LOC in `kernel_sources.h` for slab-pair register state and 64-iter inner loop. Outer-loop iter count halved at gate. Bit-exact semantics preserved.
**Stacking pathway (Sprint 20+)**: (A1) + (D) projected to clear R8 at midpoint (1.49×) and lower bound (1.67×). DEC-015 (forthcoming if needed) would lock (D) with full algebraic re-derivation under (A1) composition + parity sweep at γ_8.
**Scope**: DEC-008 envelope (`approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations). Gate config: 50k/RMSE/128b. Full ablation: `docs/sprint19/ablation_accumulation.md`.

## DEC-012: Intra-SIMD butterfly removed under L1a layout

**Sprint**: 18
**Date**: 2026-04-17
**Commit**: `19fa5ce6cc`
**Problem**: D1c's `simd_shuffle_xor` butterfly (5 rounds, xor 16/8/4/2/1) was designed to reduce 32 distinct per-lane `privHist` partials into a single SIMD-group sum. Under L1a's `simdHist[g][bin]` layout, all 32 lanes in group `g` read the same shared slot — there are no per-lane partials to reduce. Applying the butterfly over a shared slot multiplies the accumulated value by 32 (BUG-S18-001).
**Chosen**: Remove the intra-SIMD butterfly entirely. Under stride-partition ownership, `simdHist[g][bin]` already holds the full per-SIMD-group sum after the accumulation loop. Only the cross-SIMD 8-term linear fold (DEC-009) runs downstream.
**Rationale**: The butterfly is algebraically redundant and structurally incorrect under the new layout. Its removal tightens the effective reduction depth from γ_12 (5 butterfly levels + 7 linear cross-SIMD levels) to γ_7 (7 linear cross-SIMD levels only). Higham bound improves from ~7.2e-7 to ~4.2e-7. Barriers per dispatch drop from 9 to 6 as a side-effect.
**Future trigger**: Any future kernel that accumulates into per-lane register state (not shared threadgroup memory) should re-introduce the intra-SIMD butterfly for that phase. The xor butterfly remains the correct pattern for per-lane-held partials; its removal here is specific to the per-SIMD-group shared layout.
**Status**: Active. Shipped Sprint 18.

## DEC-015: Col-major `compressedIndex` transposed view

**Sprint**: 19
**Date**: 2026-04-18
**Branch**: `mlx/sprint-19-hist-writeback` (WIP snapshot: `108c7a59d2`)
**Status**: **REJECTED.** Implementation complete and parity-clean (18/18 bit-exact, 100/100 deterministic). S19-01b attribution model projected 2.13× e2e. Direct measurement: ~0.98× (no improvement, essentially noise). S19-01c re-attribution probe D (production kernel with compressedIndex loads disabled) made the kernel ~2% SLOWER, empirically demonstrating AGX out-of-order + hardware prefetcher fully hides the row-major gather — gather cost is not the bottleneck. The compressedIndex access pattern is irrelevant to throughput under the current accumulation structure. Layout change NOT committed. Side-fix extracted separately (per-group `featureColumnIndices`+`numGroups` variable correction in `DispatchHistogramBatched`) and shipped as Commit 1 (`77db8b5631`). Full analysis: `docs/sprint19/reattribution.md` §3–5.
**Lesson**: Analytical models of AGX cache hierarchy are unreliable; the memory subsystem hides more latency than first-principles reasoning suggests. Future kernel layout decisions must have empirical micro-bench backing before implementation.

## DEC-016: T1 fuse-valid simd_shuffle reduction (MSB-sentinel)

**Sprint**: 19
**Date**: 2026-04-19
**Commit**: `92f3832169`
**Branch**: `mlx/sprint-19-hist-writeback`
**Status**: **ACTIVE (SHIPPED).**
**Problem**: Production L1a histogram kernel (`kernel_sources.h:181–215`) issues 3 `simd_shuffle` calls per src iteration inside the 32-iter inner broadcast loop: `packed`, `stat`, `valid`. S19-01c probe evidence (`docs/sprint19/reattribution.md`) identified the simd_shuffle serial chain as 86% of the accumulation phase and ~80% of `histogram_ms`. Reducing the chain length is the highest-leverage kernel-local change.
**Chosen**: Pack the valid flag into the MSB (bit 31) of the `packed` uint32 at load time. On the src broadcast, derive validity from `(p_s & VALID_BIT)` instead of an independent shuffle. Drops one shuffle per src iteration (3 → 2), reduces the shuffle chain depth by 1/3.
**Rationale**:
- **Safe when max fold count ≤ 127 (⇔ all bin values ≤ 127).** `packed` holds four 8-bit values per feature: slot-0 at bits 24–31, slot-1 at 16–23, slot-2 at 8–15, slot-3 at 0–7. Bit 31 is the MSB of slot-0's 8-bit field — it is zero on load only when slot-0's bin ≤ 127. A host-side `CB_ENSURE(maxFoldCount ≤ 127)` in `DispatchHistogramBatched` (and mirrored in `bench_boosting.cpp::DispatchHistogram`) enforces this loudly; out-of-envelope callers are rejected rather than silently corrupted. The kernel masks with `p_clean = p_s & 0x7FFFFFFFu` before bin extraction.
- **Parity bit-exact.** MSB-sentinel encoding does not alter reduction order for valid docs; invalid docs do not write (gated by sentinel check). Higham γ_7 unchanged.
- **Measured parity (3 configs, seed 42, depth 6, 128 bins):** 50k/RMSE 0.48047778; 10k/RMSE 0.48016092; 50k/MultiClass 0.94424933 — all bit-exact pre vs post edit.
- **Measured perf (50k/RMSE/d6/128b, 3-run warm mean):** pre-edit 32.47 ms, post-edit 31.73 ms — **−2.3% e2e.** Matches S19-01c probe projection (kernel ≈ 97.7% of iter time, so kernel-level shuffle reduction amplifies near 1:1 to e2e).
**Trade-off**: +6 LOC kernel-side, +1 constant (`VALID_BIT`), +1 masking op (`p_clean`), +1 host-side `CB_ENSURE`. No new TG memory, no new register state beyond the existing OR-in at load. DEC-011 ceiling preserved.
**Scope limit**: `maxFoldCount > 127` (e.g. default `MaxBins=255`, or `bins=128` with a NaN offset that pushes a feature's Folds to 128) is rejected at dispatch. Gate config (bins=128 synthetic no-NaN → Folds=127) is in-envelope. Wider envelope support is Sprint 20 work (DEC-017 redesign or dedicated valid buffer). Envelope guard added in S19-13 after S19-07 code review found the pre-guard commit (`92f3832169`) silently corrupted slot-0 bins 128..255.
**Follow-up**: T3b atomic-CAS (DEC-017, Sprint 20) eliminates the shuffle chain entirely; T1 is the bit-exact interim.

## DEC-017: T3b threadgroup-atomic-CAS no-shuffle accumulator (RETIRED — SUPERSEDED BY EMPIRICAL FALSIFICATION)

**Sprint**: 20 (attempted, abandoned)
**Date retired**: 2026-04-19
**Branch**: `mlx/sprint-20-hist-atomic-cas`
**Status**: **RETIRED — SUPERSEDED BY EMPIRICAL FALSIFICATION.** See `docs/sprint20/d2b_design.md §2` for the full empirical case.

### Post-mortem

**Projected**: −84.4% single-TG accumulation at gate config 50k/RMSE/d6/128b, toy-kernel isolation (Sprint 19 algorithmic ablation, `microbench_algorithmic.cpp`).

**Measured (Sprint 20 D2, commit `9079ad3873`)**: **+42.3% regression** at the same gate config in production dispatch. Parity was 18/18 bit-exact across the full DEC-008 envelope — the failure mode is pure dispatch-shape mismatch.

**Why it failed**: Toy-kernel ablation ran 1 TG × 256 threads × all 50k docs in a single partition (195 docs/thread, root depth). Production at depth 6 runs 1638 TGs × 256 threads × ~781 docs per partition (~3 docs/thread). T3b's fixed per-TG overhead is absolute, not proportional to per-TG work:

- Per TG: 1024-slot `atomic_uint` zero-init + 1024-slot writeback read = ~8 memory ops per thread
- At 195 docs/thread: 8 / (195 × 4 features) = **1.0%** of per-thread cost → amortized, T3b wins
- At 3 docs/thread: 8 / (3 × 4 features) = **67%** of per-thread cost → overhead dominates, T3b loses
- CAS atomics compound the problem: each CAS is a read-modify-write with conditional retry, cannot pipeline like `simd_shuffle` chains

**Standing warning for Sprint 21+**: Toy-kernel ablations at single-TG root dispatch shape **DO NOT predict production partition-fragmented dispatch** in this codebase. Future T* algorithmic campaigns MUST validate at production dispatch shape (multi-TG, depth-appropriate partition granularity) before committing to an R8 projection. This is now a standing rule for all Sprint 21+ lever evaluations.

**Fourth analytical model falsified** in the T3b campaign (DEC-013 writeback, DEC-014 original gather, DEC-015 col-major, DEC-017 T3b-as-drop-in).

**Pointer**: See `docs/sprint20/d2b_design.md §2` for the empirical case and §5 for the Sprint 21+ plan.

**Commits**:
- `9216f4941c` — Sprint 20 D1 parity sweep (18/18 bit-exact, 100/100 determinism — preserved as the isolated-shape measurement record)
- `9079ad3873` — Sprint 20 D2 falsification record (+42.3% regression at production shape)
- Kernel/host integration attempt reverted before commit per DEC-012; only the empirical record is in git history.

---

## DEC-018: TG-count reduction variant A — RETIRED (never activated)

**Sprint**: 21 (DRAFT-S21; never progressed past D0)
**Date**: 2026-04-20
**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Status**: **RETIRED — D0 kill-switch fired; decision never activated.**

### Post-mortem

**Projected mechanism**: Dispatch 26 TGs (13 feature groups × 2 stats) × 195 docs/thread instead of the production 1638 TGs × ~3 docs/thread, restoring a T3b-compatible docs/thread ratio at every depth by amortizing per-partition work across a single feature-group TG.

**D0 kill-switch result** (`docs/sprint21/d0_attribution.md`): Fixed per-TG overhead at depth 6 = **2.5% ± 1.3%** of `histogram_ms` (R²=0.9989 depth-sweep regression, 3 independent warm runs × 6 depths). Kill-switch threshold: ≥10%. Measured: **2.5% → FIRES**.

**Specification error (captured for campaign learning)**: The D0 kill-switch as written tested whether T1's fixed per-TG overhead was large enough to amortize via TG-count reduction. But variant A's actual savings mechanism was **T3b-accumulator-swap at the restored 195-docs/thread shape** — TG-count reduction is the shape-restoration *enabler*, not the savings source. T1's fixed overhead was already low (2.5%) because Sprint 18 L1a eliminated the dominant DRAM costs. The 40% "fixed overhead" figure in `docs/sprint20/d2b_design.md §2` referred to T3b's per-thread work ratio at production shape, not T1's. The gate tested a proxy (T1 amortization), not the lever's direct mechanism (T3b shape-restoration). See `docs/sprint21/d0_attribution.md §6.2`.

**Generalizable lesson (encoded in `feedback_ultrathink_task_planning.md`)**: Kill-switch gates must be direct tests of the lever's mechanism, not proxies. A proxy gate can fire or pass for reasons orthogonal to the lever's actual savings path.

**Decision**: Variant A is RETIRED without implementation. It never reached D1, D2, or any kernel commit. No production source was modified.

**Pointer**: `docs/sprint21/d0_attribution.md §6.2`; `docs/sprint21/README.md §Pivot`.

---

## DEC-019: L2 stats pre-permute — FALSIFIED

**Sprint**: 21 (D1-R1 direct mechanism test)
**Date**: 2026-04-20
**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Commit**: `fedf9d5348`
**Status**: **FALSIFIED — zero-gather upper bound shows no gain at production shape.**

### Mechanism tested

Pre-permute `gradients` and `hessians` arrays so the histogram kernel reads contiguous memory instead of scatter-gathering via `compressedIndex[doc]`. Hypothesis: the per-doc stats gather is a measurable fraction of `histogram_ms` at production multi-TG depth-6 shape, and eliminating it would save ≥10% of `histogram_ms`.

### Direct mechanism test

Built a kernel variant of the L1a accumulator with stats loads replaced by constants (`g = 1.0f`, `h = 1.0f` instead of `g = gradients[compressedIndex[doc]]`). This is the **maximum possible saving** L2 could ever deliver — zero gather cost, zero permutation cost. Measured `histogram_ms` reduction vs T1 baseline at production shape (50k/RMSE/d6/128b, 1664 TGs × ~3 docs/thread).

### Result

Zero-gather variant: **+2.61% slower** than T1 baseline (not ≥10% faster). Gate miss: **12.6 percentage points below the 10% floor**.

The upper bound is negative — real L2 integration (which costs an additional O(N) permute kernel per iteration) would be yet more expensive. L2 is falsified at production shape.

### Interpretation

The S19-01c probe D finding (global-memory loads = 0% of kernel cost at single-TG root depth) generalizes to multi-TG depth-6 shape. AGX out-of-order execution and the L2 hardware prefetcher fully hide the stats gather latency behind the simd_shuffle serial chain — confirming the pattern established in DEC-015 (col-major compressedIndex, Sprint 19): AGX memory subsystem hides more latency than first-principles reasoning predicts.

**This also rules out any future L2 variant** absent new evidence that the memory subsystem behavior changes (e.g., much larger N, different partition depth structure, or different kernel structure that changes the overlap).

**Pointer**: `docs/sprint21/d1r1_l2_attribution.md`; S19-01c probe D in `docs/sprint19/reattribution.md`.

---

## DEC-020: T2 sort-by-bin — SHIPPED / VALIDATED

**Sprint**: 21 (D1-R2 production-shape micro-bench) → **22 (Option III fix + full gate suite)**
**Date**: 2026-04-20
**Branch**: `mlx/sprint-21-hist-tg-reduction` → `mlx/sprint-22-t2-integration`
**Commits**: `13322feaca` (D1-R2 VIABLE) → `4333c82a7e` (D0 in-situ PASS) → `73baadf445` (D1–D6 gates PASS)
**Status**: **SHIPPED — VALIDATED. 4/4 exit gates PASS. Cumulative R8 = 1.90×. Verstappen ≥1.5× gate CLEARED by 40 pp.**

### Final numbers (commit `73baadf445`)

| Metric | Value |
|--------|-------|
| Gate config ratio (T2/T1 hist_ms) | **0.317×** cross-session (band 0.315–0.319×) |
| S22 e2e multiplier (T1/T2 iter_total_ms) | **1.778×** (33.958 ms → 19.098 ms) |
| Cumulative R8 (post-S22) | **1.07 × 1.778 = 1.90×** |
| Verstappen gate (≥1.5×) | **CLEARED +40 pp** |
| Parity | 18/18 ULP=0 bit-exact; 100/100 determinism runs |
| Code review | 0 blockers, 6 deferred nits (NIT-6 removed at audit) |
| Security audit | 0 CRITICAL, 0 HIGH; overflow structurally eliminated |

### Mechanism

T2 replaces the production T1 kernel's 32-iteration simd_shuffle broadcast chain with a two-kernel dispatch:
1. A counting-sort pre-pass that bin-partitions each partition's docs by feature-0 bin and emits `binOffsets[]`.
2. An accumulator that for feature-0 does a pure bin-range scan (no shuffle, single writer per bin → no contention) and for features 1–3 does a per-doc sorted scatter via global atomics.

The simd_shuffle serial chain — the measured plurality cost at production dispatch shape (~80% of `histogram_ms`) — is structurally eliminated for feature-0.

### D1-R2 measurement (production dispatch shape)

Harness measured sort+accumulation together (total `histogram_ms`-equivalent, not accumulation alone) at 1664-TG production shape (50k/RMSE/d6/128b, gate config).

| Metric | Value |
|---|---|
| T1 baseline | 1.479 ms (per-dispatch mean) |
| T2 sort+accum | 0.520 ms (cross-session mean) |
| Reduction | **64.8%** (band 63.6–66.7%, 2σ ±2.7–4.4%) |
| Gate threshold (≥50%) | **CLEARED by 28–34 pp** |

Gate B parity: max ULP 64 (well below 1024 accumulation-order bound); mass conservation 0 ULP across 812,800 bins.

### Projected e2e (from d1r4_synthesis.md §3)

Starting point: `iter_total_ms = 31.93 ms`, `histogram_ms = 21.57 ms`, non-hist = 10.36 ms.

| Scenario | T2/T1 ratio | New iter_total | e2e speedup |
|---|---|---|---|
| Optimistic | 0.33× | 17.48 ms | 1.83× |
| Midpoint | 0.36× | 18.13 ms | 1.76× |
| Conservative (in-harness gate) | 0.50× | 21.15 ms | 1.51× |
| Kill-switch boundary | 0.60× | 23.30 ms | 1.37× |

### Risks

- **Ratio-transfer (primary)**: D1-R2 used identity-permuted docs (synthetic). Production uses argsort-permuted docs from `structure_searcher.cpp`. The T2/T1 ratio should cancel locality artifacts to first order, but if the sort pre-pass is cache-sensitive under argsort-permuted access, the ratio could widen. Sprint 22 D0 tests this directly.
- **Parity at integration**: Per-bin D1-R2 ULP ≤ 64, but DEC-008 envelope requires end-to-end loss ULP ≤ 4. Sprint 22 D1 parity sweep (18-config DEC-008 envelope) must validate end-to-end. Kahan-compensated summation is the fallback if any config fails.
- **Multi-feature sort scope**: D1-R2 sorts on feature-0 bin only; features 1–3 use atomic scatter. Sprint 22 ships as-measured; multi-feature optimization deferred.

### Sprint 22 D0 kill-switch

Measure `ratio = hist_ms(T2) / hist_ms(T1)` at gate config via in-situ `DispatchHistogramT2` variant under real training-loop conditions (argsort-permuted `docIndices`).
- `ratio ≤ 0.45` — optimistic band holds; proceed to D1 parity.
- `0.45 < ratio ≤ 0.60` — conservative band; proceed to D1, but R8 projection drops to 1.37–1.51×; Ramos re-decides.
- `ratio > 0.60` — **T2 FALSIFIED at production shape**. Drop to RESEARCH. Sprint 22 pivots to tree-search restructure.

The 0.60 threshold is the point where cumulative e2e falls below the Verstappen 1.5× gate — below 0.60 there is no single-sprint stacking path.

**Pointer**: `docs/sprint21/d1r2_t2_microbench.md`; `docs/sprint21/d1r4_synthesis.md §3/§4`.

**Footnote (added 2026-04-20 during S23 D0)**: The S22 D3 parity sweep ran 1-run-per-config, which missed a bimodal ~50/50 distribution at config #8 (N=10000) — see DEC-023. S22 D3's "18/18 ULP=0" record is corrected to 17/18 ULP=0 + 1 latent bimodal; discovered during S23 D0 scratch→production promotion. 1.90× perf record at gate config (config #14) is unaffected — gate config is 100/100 deterministic. The bimodality is a features 1-3 atomic-float race, not a T2 structural bug; T2 itself ships as designed.

---

## DEC-021: Option III slab-by-partOffsets layout over Option I uniform-ceiling

**Sprint**: 22 (D1c root-cause + D2 implementation)
**Date**: 2026-04-20
**Branch**: `mlx/sprint-22-t2-integration`
**Commit**: `73baadf445`
**Status**: **ACTIVE (SHIPPED).**

### Problem

D0 T2 scratch kernel used `maxPartDocs = ceil(N/K)` as the per-TG `sortedDocs` slab stride. Under real argsort-permuted training-loop splits, partitions are heavily skewed — depth-1 on 50k yields sizes [442, 49558] vs `maxPartDocs = 25000`. The 49558-doc partition's counting-sort scatter overflowed 24558 docs into the neighboring TG's slot, corrupting histograms. This was the root cause of the 18/18 DEC-008 parity failure in D1 (`docs/sprint22/d1c_t2_troubleshoot.md`).

### Candidates

- **Option I** (one-line): `maxPartDocs = numDocs`. Structurally safe — `ceil(N/1)` always ≥ any partition size. Buffer at gate config: **333 MB** (`numTGs × numDocs × 4 B = 13 × 2 × 50000 × 50000 × 4 B`). Unacceptable worst-case.
- **Option II** (dynamic ceiling): `maxPartDocs = max(partSizes)` per dispatch. Safe. Buffer ≈ `numTGs × max(partSizes) × 4 B`. Requires an O(K) scan per dispatch on the CPU. Workable but fragile — depends on `max(partSizes)` being accurate before every dispatch; any future code path that skips the scan would reintroduce overflow.
- **Option III** (structural): Replace per-TG `maxPartDocs`-sized slots with per-(groupIdx, statIdx) slabs of size `numDocs` addressed by `partOffsets[partIdx]`. `sortedDocs` is `numGroups × numStats × numDocs`. Overflow is **structurally impossible** since `sum(partSizes) == numDocs` and `partOffsets` are prefix sums of `partSizes`.

### Chosen: Option III

**Rationale**: Option III makes overflow structurally impossible, not just unlikely for the current code path. Buffer at gate config = **5.2 MB** (`numGroups × numStats × numDocs × 4 B = 13 × 2 × 50000 × 4 B`), vs 333 MB worst-case for Option I. The `slotBase` formula is simpler than the old `partIdx * maxPartDocs` stride. No new correctness assumption is added — the slot-disjointness invariant follows trivially from `sum(partSizes) == totalNumDocs`.

**Trade-off**: Buffer size is now proportional to `numGroups × numStats × numDocs` regardless of tree depth (shallower depths no longer benefit from smaller `maxPartDocs`). At the gate config this is 5.2 MB. Beyond DEC-008 scope (e.g. N=500k / 200 features / K=4 MultiClass), the buffer would be ~300 MB — still fine on unified memory but warranting re-evaluation. See D5 NOTE-1.

**Perf impact**: Option III marginally improved ratio vs D0 (0.317× vs 0.328× cross-session). The slotBase formula simplification appears to reduce index-arithmetic overhead. 1.6 pp perf headroom preserved over D0.

**Pointer**: `docs/sprint22/d1c_t2_troubleshoot.md §6.1`; `docs/sprint22/d2_t2_fix_verified.md §2`; `docs/sprint22/d6_security_audit.md §3 Check 1`.

---

## DEC-022: Kahan/compensated-summation concern RETIRED (S21 D1-R4 §5 bug β does not exist)

**Sprint**: 22 (D1c determinism probe)
**Date**: 2026-04-20
**Branch**: `mlx/sprint-22-t2-integration`
**Commit**: `73baadf445`
**Status**: **CLOSED — concern FALSIFIED by evidence.**

### Background

`docs/sprint21/d1r4_synthesis.md §3` (Sprint 22 risks section) documented a parity risk:

> "Bug β atomic-scatter float drift — if end-to-end loss ULP > 4, Kahan-compensated summation on the per-doc scatter path is the fallback (+2–3 days)."

The concern was that per-doc global atomic scatter in T2-accum (features 1–3) could produce non-deterministic float accumulation across runs, requiring Kahan compensation to stay within DEC-008 ULP ≤ 4.

### Evidence retiring the concern

The D1c diagnostic arc, as a side-effect of Option III fix verification, produced:

- **10/10 determinism runs pre-Option-III fix** (at features=1/iters=2 minimum-reproducer): identical float outputs across all 10 runs on both T1 and (once fix applied) T2.
- **100/100 determinism runs post-Option-III fix** at gate config (50k/RMSE/d6/128b): single unique BENCH_FINAL_LOSS = 0.47740927 across all 100 runs.
- **D3 QA independent verification**: 18/18 ULP=0, 100/100 determinism, 5 edge-case configs all ULP=0.

The non-determinism observed in D0/D1 sweeps was entirely caused by the `maxPartDocs` buffer overflow (the 24558-doc neighbor-slot corruption). Once the overflow was removed by Option III, T2's per-doc atomic scatter is deterministic at the gate config — the Metal atomic scheduler resolves float scatter in a consistent order for this dispatch shape and data. No Kahan compensation is needed or appropriate.

**Lesson**: The anticipated "bug β" was a misidentified symptom of the structural overflow bug; it does not exist as an independent failure mode. S21 D1-R4 §3 fallback budget (+2–3 days for Kahan) is retired.

**Pointer**: `docs/sprint22/d1c_t2_troubleshoot.md §1` (false-positive retirement); `docs/sprint22/d3_parity_gate.md §1` (100/100 determinism table).

**Scope qualifier (added 2026-04-20 during S23 D0)**: Original evidence base (10/10 and 100/100 determinism) was AT GATE CONFIG ONLY (N=50000/RMSE/128b). Bug β is partially real at smaller N — see DEC-023. Features 1-3 atomic_fetch_add race fires at N=10000 (config #8). DEC-022 remains valid at gate config and for the S21 D1-R4 §5 framing (the D1 root cause was maxPartDocs, not atomic drift). Kahan concern is NOT fully re-opened; Options 1-2 in DEC-023 are the primary fix path.

---

---

## DEC-023: Features 1-3 atomic-float race in T2-accum — deterministic reduction required

**Status**: RESOLVED 2026-04-21 — close-commit `784f82a891` (v5: T2-accum rewritten to T1 accumulation topology for all features; T2-sort removed from dispatch)
**Discovered**: Sprint 23 D0 during scratch→production promotion parity sweep
**Scope**: T2-accum features 1-3 use atomic_fetch_add on float; FP non-associativity + non-deterministic thread scheduling produces 1-2 ULP drift in histogram bins, which can flip near-tie split decisions early in training and cascade to 105+ ULP in final loss

**Footprint** (S23 D0 measured, N=100 per config):
  Config #8 (N=10000/RMSE/128b/depth=6/iters=50): BIMODAL 50/50 at 0.48231599 vs 0.48231912 (105 ULP gap)
  Config #14 gate (N=50000/RMSE/128b/depth=6/iters=50): DETERMINISTIC 100/100 at 0.47740927
  Configs #1–#7, #9–#13, #15–#18: DETERMINISTIC (100/100 each)
  **Singleton footprint**: exactly 1 of 18 configs fires. All N=1000, all N=50000, and the other five N=10000 configs (varying loss / bins) are clean. Race is narrowly conditioned on (N=10000, RMSE, bins=128).

**Cascade mechanism**: 1-2 ULP bin drift → near-tie GAIN flip at iteration k → different tree at k → all subsequent iters diverge. **Cascade onset** between iters=40 (30/30 deterministic) and iters=45 (bimodal). At iters=50 spread is 105 ULP; at iters=100 spread narrows to 39 ULP (non-monotone — branches converge toward a common limit). Cascade-factor table: docs/sprint23/d0_bimodality_verification.md §D.

**Why it hides at gate — H1 SUPPORTED**: Gate-config seed-sweep (500 runs × 5 seeds at config #14) returned 100/100 deterministic on every seed. H2 (seed-coincidental determinism) refuted. H1 (structural: larger bin counts resolve additions in consistent order) is the operative explanation. The 1.90× R8 record at gate config is structurally robust across seed space. See docs/sprint23/d0_bimodality_verification.md §B.

**Sibling race — S-1 latent** (found during S23 D0 site inventory): `kHistOneByte` writeback in kernel_sources.h uses atomic-float and is RACY when `maxBlocksPerPart > 1`. Currently DEAD CODE PATH — NIT-4 enforces `maxBlocksPerPart == 1` via CB_ENSURE. Any future optimization that relaxes this constraint (e.g., per-partition multi-block dispatch) reactivates the race. Fix options for S-1 mirror the T2-accum options below; address alongside DEC-023 if multi-block dispatch is needed, or leave guarded by the CB_ENSURE otherwise.

**Feature-0 is clean**: bin-range scan over counting-sorted docs is ordered by sort order, no atomics; 100/100 deterministic. Only features 1-3 (atomic scatter path) are affected.

**Fix options for S24**:
  1. Threadgroup-local reduce + single-thread commit (mirrors feat-0 design; known-clean mechanism, preserves T2 perf envelope)
  2. Int-atomic fixed-point accumulation (CatBoost CPU uses uint64 fixed-point for exactly this reason; deterministic by construction; accuracy calibration required)
  3. Kahan/Neumaier compensated summation per bin (mitigates but does NOT eliminate non-determinism; probably not sufficient standalone)

**Budget**: S24 D0, 1-2 days. Kill-switch: if fix degrades gate-config ratio below 0.45× (optimistic band), escalate to structural redesign.

**Relation to prior work**:
  - Partially re-validates S21 D1-R4 §5 bug β concern (retired too broadly under DEC-022; see DEC-022 scope qualifier)
  - Kahan concern (DEC-022 body) is not resurrected as a standalone fix but may complement Options 1-2
  - R8 1.90× record at gate config is unaffected (gate is deterministic)

### S24 D0 resolution (appended 2026-04-21)

**Fix chosen**: v5 — all-feature T1-style SIMD-shuffle accumulation. All four features (0-3) in
T2-accum rewritten to use T1-style SIMD-shuffle + linear fold + writeback reading from
`docIndices`. T2-sort kernel removed from dispatch. Feature-0 no longer scans `sortedDocs`.

**Why Option 1 (TG-local reduce) and int-fixed-point (Option 2) were not sufficient**: All
Path 5 variants retaining feature-0's bin-range scan over `sortedDocs` pinned to Value B
(0.48231912, 105 ULP off T1 Value A). Root cause: reduction topology difference between T2's
sort-based scan and T1's SIMD fold. Determinism (Path 5 int-fixed-point) was achieved but did
not produce Value A. Only accumulation-topology matching achieves ULP=0 vs T1.

**Acceptance-criteria results** (all 4 gates PASS):

| Gate | Criterion | Measured | Verdict |
|------|-----------|----------|---------|
| S24-D0-G1 | Config #8: 10/10 deterministic | 10/10 at 0.48231599 (ULP=0) | PASS |
| S24-D0-G2 | 18/18 ULP=0, ≥5 runs per config | 18/18 ULP=0, all 5/5 det. | PASS |
| S24-D0-G3 | Gate config: 100/100 deterministic | 100/100 at 0.47740927 | PASS |
| S24-D0-G4 | hist_ms ratio ≥ 0.45× (kill-switch) | 0.959× | PASS |

**R8 consequence**: 1.90× → 1.01×. T2 v5 runs at T1 speed (0.959× hist_ms ratio). The 1.90×
record was predicated on T2's non-deterministic sort-based accumulation providing a 0.317×
hist_ms ratio. Making T2 deterministic requires matching T1's accumulation topology, which
eliminates T2's structural speed advantage. Verstappen ≥1.5× gate failed retroactively.

**Forward**: DEC-026 (below) opens the research track for recovering T2's speedup via
cascade-robust GAIN comparison in S25.

---

### Preserved for history (original DRAFT content below)

**Sprint**: 20 (flagship)
**Date**: 2026-04-19 (draft; will lock at Sprint 20 D1 parity sweep close)
**Branch**: TBD (`mlx/sprint-20-*`)
**Status**: **DRAFT.** Ships only if full DEC-008 envelope parity sweep passes. If parity fails: Kahan/Higham compensated summation + re-sweep.
**Problem**: Even with DEC-016 (T1), the simd_shuffle serial chain remains the dominant kernel cost (~83% of accumulation post-T1). The structural ceiling on shuffle-based broadcast is bounded by SIMD-width ALU latency — no incremental fuse is near the −40% R8 needs. A qualitative structural replacement of the shuffle broadcast is required to clear R8 in Sprint 20.
**Proposed**: Replace `simdHist[8][1024]` per-SIMD-group layout with `atomic_uint simdHistU[1024]` single-TG layout. Each lane processes its own doc directly (no `src` broadcast loop, no cross-lane communication). Accumulation is a float-CAS add on the shared per-bin slot. Cross-SIMD fold phase eliminated (T3b produces per-bin sum directly).
**Measurement (toy kernel isolation, `microbench_algorithmic.cpp`)**: T3b = 0.387 ms vs T0 = 2.485 ms → **−84.4% accumulation**. Contention sweep (`microbench_contention.cpp`): speedup holds at all bin counts from 128 down to 16 (worst ratio 0.218 at 16 bins, 4.6× still).
**Parity risk**: FP32 reduction-order drift. Atomic-CAS adds docs in arrival order (non-deterministic across TGs on reruns), compared to T0's fixed src-lane order per SIMD group. Higham worst-case error γ_N where N = docs/bin (at 50k/128b, N=390). γ_390 ≈ 2.3e-5 FP32 — **exceeds DEC-008 Logloss ulp ≤ 4 threshold (≈4.77e-7) by ~50×.** Parity is not given; must be measured across the full envelope.
**Integration cost**: 2–3 days kernel rewrite + 1–2 days parity sweep. Requires DEC-011 amendment (32 KB → 4 KB TG memory). Rewrite replaces `simdHist[8][1024]` with `simdHistU[1024]`; accumulation loop structure changes entirely; cross-SIMD fold phase removed.
**Scope scoping decision points (Sprint 20 D1)**:
1. Full DEC-008 envelope parity sweep (18 configs × 100 runs).
2. Full-grid scaling validation (toy measures 1 TG × 256 threads; production dispatches 1575 TGs at depth 5–6 — atomic contention under concurrent dispatch unmeasured).
3. MultiClass approxDim=3 parity (3 independent reductions compound drift).
**Fallback**: If parity fails, implement Kahan/Higham compensated summation (+1 uint32 per bin as running compensation term) and re-sweep. If Kahan still fails → T3b is structurally incompatible with DEC-008; alternative exploration.
**Commits placeholder**: `<TBD-sprint-20>`.

---

## DEC-024: S23-R1 EvalAtBoundary readback elimination — DEFERRED (harness gap, not falsified)

**Status**: DEFERRED pending harness extension (not falsified; not retired)
**Date**: 2026-04-20 (Sprint 23 R1)
**Author**: @ml-engineer
**Authority**: `docs/sprint23/r1_evalatboundary.md` (R1 scope report)

**Finding**: The three live `EvalAtBoundary` readback sites in `structure_searcher.cpp` (lines 290, 609, 705) are architecturally unreachable from the gate config (50k/RMSE/d6/128b).

- Site A (line 290): `SearchDepthwiseTreeStructure` only — requires `EGrowPolicy::Depthwise`.
- Site B (line 609): `SearchLossguideTreeStructure` evalLeaf — requires `EGrowPolicy::Lossguide`.
- Site C (line 705): `SearchLossguideTreeStructure` split-apply — requires `EGrowPolicy::Lossguide`.

Gate config runs `EGrowPolicy::SymmetricTree` (oblivious). `bench_boosting.cpp` uses its own inline oblivious-tree loop (`RunIteration`, lines 967–1243) and does not call `structure_searcher.cpp` at all. Production `mlx_boosting.cpp` at gate config dispatches to `SearchTreeStructure` which has no EvalAtBoundary calls.

**Site fate per kill-switch**: A=SKIP, B=SKIP, C=SKIP (all 3 sites; 0/3 replaceable at gate).

**Cost-estimate provenance**: S16 `sync_inventory.md` ~0.3 ms/iter was a theoretical cost-class-A projection, not a measured gate-config value. There is no measurable gate-config iter_total_ms reduction available via Sites A–C.

**Lever disposition**: DEFERRED, not RETIRED. The lever may still be valid for production workloads using `Depthwise` or `Lossguide` grow policies. It cannot be ranked or executed until one of:
  1. `bench_boosting` adds `--grow-policy {depthwise|lossguide}` flags, or
  2. A separate benchmark harness wraps `mlx_boosting.cpp` with non-default grow policy and timing instrumentation.

**Relation to prior decisions**: Complements DEC-017/DEC-018 (retired T3b/variant A — levers retired on evidence). DEC-024 is NOT a retirement; it is a gap diagnosis. If the harness extension lands, the lever re-enters the Verstappen candidate pool under standard gate criteria.

**R8 position unchanged**: Gate config `iter_total_ms = 19.098 ms` (S22 D4). Cumulative 1.90×.

**Commits**: None (no production changes in R1).

---

## DEC-025: S23-R2 dispatch inversion — FALSIFIED (structural algebraic blocker)

**Status**: FALSIFIED — do not re-enter without new mask-mechanism evidence
**Date**: 2026-04-20 (Sprint 23 R2)
**Author**: @research-scientist
**Authority**: `docs/sprint23/r2_dispatch_inversion_spike.md` (NO-GO verdict, 2-day timebox Day 1 closed)

**Finding**: The proposal (replace partition-fragmented 1664-TG dispatch with a single all-docs histogram over `(feature × stat × bin)`, recovering per-partition bin sums at scoring time via masks) cannot yield per-partition bin-sums `h_p[f][b]` from the collapsed global `H[f][b]`. The algebra `H[f][b] = Σ_p h_p[f][b]` is not invertible without a second per-doc pass equivalent to the work the inversion was meant to eliminate.

**Three mask mechanisms considered** (all fail):
  1. **Per-partition mask at scoring time**: equivalent to a second histogram pass; no net win.
  2. **Doc-level bin×partition tensor**: memory footprint 2–3× current histogram (50k docs × 64 parts × bins-worth ≥ 3× the current 3.2 MB).
  3. **Bit-packed doc→partition with per-bin gather at scoring**: gather-count `numParts × numFeatures × numBins = 64 × 50 × 128 = 409,600 gathers/iter` adds ≥4–6 ms at AGX bandwidth, consuming the entire optimistic 5.05 ms headroom.

**Parity regression risk**: Merging 64 partition atomic writers into a single contended per-bin slot raises atomic contention 64× over current T2. At the DEC-023 race footprint (N=10k config #8 already bimodal with 1664 TGs fragmentation), inversion strictly worsens the race — 26 TGs × 195 docs/thread contending on 128 bins × 4 features = 512 slots each receiving 64× more writers. Cannot ship within DEC-008 envelope without Kahan/fixed-point compensation (which itself has independent risks — see DEC-023 fix-option analysis).

**Relation to prior falsifications**:
  - DEC-017 (T3b retired): failed at production shape (+42.3%) because single-TG toy timing did not survive partition-fragmentation atomic contention.
  - DEC-018 (variant A retired): failed the 10% fixed-overhead kill-switch because gate tested T1 amortization proxy, not T3b shape-restoration mechanism.
  - **R2 is the non-atomic-CAS cousin of DEC-017**: both restore 195 docs/thread at gate by reducing TG count; R2 trades dispatch fragmentation for per-bin atomic fragmentation at scoring time — the same contention surface, relocated, plus a new reconstruction cost.

**Re-entry policy**: Do not re-enter this design space in S24+ without new evidence of a structurally different reconstruction mechanism (e.g., hardware segment-sum primitives unique to a future GPU, or a hybrid per-partition sparse overlay pattern not considered in Day 1). Timeboxed spike closed at end of Day 1 — Day 2 not exercised.

**R8 position unchanged**: Gate config `iter_total_ms = 19.098 ms` (S22 D4). Cumulative 1.90×.

**Commits**: None (research spike only, no production changes).

---

## DEC-026: Cascade-robust GAIN comparison — research track for T2 speedup recovery

**Status**: FALSIFIED (S25 G1, 2026-04-21)
**Date opened**: 2026-04-21
**Date falsified**: 2026-04-21
**Sprint**: 25 (research)
**Opened by**: S24 D0 close-out
**Falsified by**: S25 G1 empirical ε-threading sweep (`docs/sprint25/g1_epsilon_calibration.md`)

### Problem statement

DEC-023 v5 resolution established that T2-accum's sort-based feature-0 bin-range scan produces
Value B (0.48231912) while T1's SIMD accumulation produces Value A (0.48231599) — a 105 ULP gap
at iters=50 at config #8. The gap originates from 1-2 ULP/bin differences in the accumulation
topology (sort-based scan vs SIMD fold) and cascades to 105 ULP via a near-tie GAIN flip at
approximately iteration 20-40 of config #8's training run.

The cascade mechanism: 1-2 ULP bin-histogram difference → GAIN comparison at a near-tie split
flips to select a different tree node → all subsequent iterations diverge on different trajectories
→ ~70× ULP amplification by iters=50.

At 17/18 DEC-008 configs, the 1-2 ULP/bin difference does not reach a GAIN near-tie and the
cascade does not fire — T2's Value B stays within DEC-008 ULP tolerance (RMSE ≤ 4, etc.).
Config #8 is the sole exception where the near-tie GAIN flip exists at T2's Value B.

### Research question

A deterministic GAIN tiebreak — applied when `|GAIN_A - GAIN_B| < ε` for a calibrated ε —
could prevent the near-tie flip from selecting a different node. If T2 (with sort-based
accumulation producing Value B inputs) selects the same split as T1 (producing Value A inputs)
at every near-tie iteration in config #8's training run, the cascade is blocked and the end-to-end
loss converges to a value within DEC-008 ULP tolerance of T1's result (possibly Value A itself).

If the research succeeds, T2's structural speedup (Path 5 design: T2-sort + int-atomic
fixed-point accumulation for features 1-3) becomes shippable at R8 ≈ 1.85–1.90× (the pre-S24
measured position).

### Research questions (falsification-first order)

1. **Epsilon calibration study**: How often do near-tie GAIN comparisons occur at config #8?
   What ε separates genuine near-tie flips (where 1-2 ULP histogram difference changes the
   winner) from legitimate GAIN gaps (where even a 105 ULP histogram difference would not change
   the winner)? Is there a viable ε range, or is the gap between "too small to catch the flip"
   and "too large to avoid false-positive tiebreaks" empty?

2. **Model-quality validation**: Does the tiebreak change tree structure at any of the 18
   DEC-008 configs in a way that degrades AUC/RMSE? Even if the tiebreak blocks the cascade,
   it may select a different split (the lexicographic tiebreak winner) than T1 would select
   naturally — this is only acceptable if the quality impact is within the 0.5% tolerance.

3. **T2 rebuild**: Rebuild T2 with the Path 5 design (T2-sort + int-atomic fixed-point
   accumulation for features 1-3) on top of the tiebreak mechanism in the scoring kernel.
   Measure the resulting hist_ms ratio and verify 18/18 DEC-008 ULP ≤ 4 across ≥5 runs
   including config #8.

### Deliverable gates

| Gate | Criterion | Pass condition |
|------|-----------|----------------|
| DEC-026-G1 | Epsilon calibration study complete | Viable ε range identified; GAIN near-tie frequency at config #8 quantified |
| DEC-026-G2 | Tiebreak implemented in scoring kernel | Tiebreak fires only when `\|GAIN_A - GAIN_B\| < ε`; lexicographic (featureIdx, binIdx) ordering |
| DEC-026-G3 | T2 Path 5 rebuild complete | T2-sort + int-atomic fixed-point + tiebreak; compiles, parity sweep started |
| DEC-026-G4 | 18-config parity sweep + determinism | 18/18 DEC-008 ULP ≤ 4; ≥5 runs per config; config #8 10/10 deterministic |
| DEC-026-G5 | Model-quality validation | AUC/RMSE drop ≤ 0.5% at any of the 18 DEC-008 configs relative to T1 baseline |

**Success criterion**: T2 Path 5 design passes all 5 gates AND gate config hist_ms ratio ≤ 0.45×.
If all 5 gates pass, re-ship T2 at R8 ≈ 1.85–1.90×.

### Kill-switches (abandon DEC-026 if any fires)

- **Epsilon study shows no viable ε**: too small — cascade persists; too large — quality
  degrades at one or more DEC-008 configs. If no ε threads this needle, the cascade-robust
  GAIN approach is structurally infeasible. FALSIFY DEC-026.
- **Model quality degrades > 0.5%** at any DEC-008 config at any ε → abandon.
- **Tiebreak fires on legitimate GAIN gaps** (not just near-tie flips) at ≥2 of 18 configs
  → ε calibration failure; abandon or restart from G1.

### Budget and sprint classification

Research sprint. Not a guaranteed delivery. Estimated 1-2 weeks with falsification checkpoints
at each gate. The epsilon calibration (G1) is the highest-risk step; if it fails, G2–G5 are
not attempted.

**R8 target if research succeeds**: re-ship T2 Path 5 at R8 ≈ 1.85–1.90× (consistent with
pre-S24 measured T2 hist_ms ratio of 0.317×). If research fails, R8 stays at 1.01× honest
position.

**Pointer**: `docs/sprint25/README.md` for the sprint scaffold and detailed research plan.

### Falsification result (S25 G1, 2026-04-21)

G1 ε-calibration sweep: 18 configs × 5 runs × 2 kernels = 180 runs, 5 min 4 s wall, 180/180
deterministic, 35 flip events (all at config #8; zero at 17 non-#8 configs).

| Quantity | Value | Source |
|---|---|---|
| ε_min (required to gate all flips) | 2.200e-03 | config #8 iter 45 depth 0 |
| ε_max (incl. zero-gain ties) | 0.0 | configs 1/2/8/14 pure/terminal nodes |
| ε_max⁺ (positive-gap floor) | 1.043e-07 | config #1 iter 40 depth 3 |
| Safety ratio (positive) | 4.742e-05 | target ≥ 2.0 — 21,091× below threshold |

**Structural cause**: Path 5's flip gaps span **5.96e-08 to 2.2e-03** — the full range of
legitimate top-2 separations observed at non-#8 configs. No ε can simultaneously (a) gate the
2.2e-03 flip at config #8 iter 45 and (b) leave the 1.04e-07 legitimate separation at config #1
iter 40 depth 3 untouched. The iter-43 near-tie (5.96e-08) is itself a legitimate rank-0/rank-1
gap, so ε-gating cannot discriminate "ambiguous split" from "clear split" in that regime.

**Implication**: the cascade-robust GAIN approach is structurally infeasible under
DEC-008 ULP=0 discipline. R8 stays at 1.01× post-S24 position. Verstappen ≥1.5× gate remains
retroactively failed from S24 D0.

**Forward paths considered and deferred** (verdict doc §9):
1. Accept v5 as final, drop Path 5 — **taken** (S25 closes here; no production code changes).
2. SCALE widening to 2⁴⁰ — won't help; iter-45 flip is structural, not quantization.
3. Hybrid routing by (N, bins) — not recommended; ongoing maintenance overhead.
4. DEC-027 alternative accumulation (XGBoost-style per-feature deterministic radix-sum) —
   deferred to a future research sprint. Not opened as part of S25 closure.

---

## DEC-028: RandomStrength noise must scale by gradient RMS, not totalHessian / numPartitions

**Sprint**: 26 (D0)
**Date**: 2026-04-22
**Status**: Implemented (`24162e1006`). Full text in `docs/decisions.md §DEC-028`.

**Problem**: MLX `FindBestSplit` (csv_train.cpp) computed per-split-candidate noise scale as
`randomStrength × totalWeight / (numPartitions × K)`. For RMSE, `totalWeight = N`, producing
`noiseScale = N` — dimensionally wrong (scales with dataset size instead of gradient magnitude).
At N=10k the noise was ~16,895× larger than CPU; SNR vs root-split gain ~0.16; noise dominated
split selection → leaf magnitudes shrunk to 0.69× of CPU.

**Decision**: Replace with CPU's `CalcDerivativesStDevFromZeroPlainBoosting` formula:
`gradRms = sqrt( sum_{k,i} g_k[i]^2 / N )`, then `noiseScale = randomStrength × gradRms`.
`gradRms` threaded from `RunTraining` into `FindBestSplit`.

**Result**: Python-path SymmetricTree `pred_std_R` 0.69 → 1.00; G1 18-cell segmented parity
18/18 PASS; no impact on bench_boosting ULP=0 record.

**Risks carried forward**:
- Depthwise/Lossguide use `FindBestSplitPerPartition`, which has no noise path. RandomStrength
  has no effect there (was true before this fix). Tracked as S26-FU-2 for a separate sprint.
- `gradRms` computed via CPU readback loop. Minor per-iteration cost (not profiled as hot).

**S26-FU-2 extension (2026-04-22)**: Extended to `FindBestSplitPerPartition` (Depthwise and
Lossguide). CPU source audit confirmed global scalar `scoreStDev` is shared identically across
all three grow policies. No new design content — pure mirror of this decision in the
non-oblivious path. See `docs/sprint26/fu2/`.

**Authority**: `docs/decisions.md §DEC-028`; gate artifacts
`docs/sprint26/d0/g1-g3-g4-report.md`, `benchmarks/sprint26/d0/g1-results.md`.

---

## DEC-029: Non-oblivious tree SplitProps never populated → empty model JSON splits

**Sprint**: 26 (D0)
**Date**: 2026-04-22
**Status**: Implemented (`9bd980a37f` C++ + `06fa2a58ee` Python). Full text in `docs/decisions.md §DEC-029`.

**Problem**: After DEC-028, Depthwise/Lossguide grow policies still showed 561%/598% RMSE
delta vs CPU. Cause was NOT the noise formula (non-oblivious paths have no noise):
`TTreeRecord.SplitProps` was populated only in the SymmetricTree `else` branch; Depthwise
and Lossguide `if` branches pushed `cursor` updates but never pushed split descriptors.
`WriteModelJSON` serialized from `SplitProps.size() == 0` → `"splits": []` → Python
`compute_leaf_indices` iterated an empty list → every doc assigned to leaf 0 → constant
prediction at `leaf_values[0]`.

**Decision**:
- C++: Add `TTreeRecord.SplitBfsNodeIds`. Populate `SplitProps` and `SplitBfsNodeIds` in both
  Depthwise and Lossguide tree-build paths. `WriteModelJSON` emits `grow_policy` per tree,
  `bfs_node_index` per split, and `leaf_bfs_ids` inverse map for Lossguide.
  BFS node index for partition `p` at depth `d` is computed by walking bits 0..d-1 of `p`
  (bit k = direction at depth k), producing the correct 2n+1 / 2n+2 walk from root.
- Python: `compute_leaf_indices` dispatches on `grow_policy`. `_compute_leaf_indices_depthwise`
  uses `_bfs_traverse_bitpacked` (mirrors the C++ partition update `bits = updateBits << depth;
  partitions |= bits`). `_compute_leaf_indices_lossguide` uses `leaf_bfs_ids` for the inverse map.

**Result**: Depthwise rs=0 delta 561% → −0.64%; Lossguide rs=0 delta 598% → −1.01%.
SymmetricTree path unchanged (already BFS-ordered implicitly).

**Risks carried forward**:
- **S26-FU-1**: `ComputeLeafIndicesDepthwise` (C++ validation path) still returns
  `nodeIdx − numNodes` (BFS leaf order) instead of bit-packed partition order. Affects
  validation RMSE tracking during Depthwise training only; does not affect training
  correctness or Python predictions.
- Model JSON now has a new `bfs_node_index` field. Old SymmetricTree models (which never
  had this field) still work correctly via dispatch; old broken non-oblivious models continue
  to produce all-leaf-0 predictions (no worse than pre-fix behavior).

**Authority**: `docs/decisions.md §DEC-029`; verification artifact `docs/sprint26/d0/d0-8-verification.md`;
diagnostics `docs/sprint26/d0/depthwise-lossguide-root-cause.md`,
`docs/sprint26/d0/leaf-magnitude-code-diff.md`,
`benchmarks/sprint26/d0/one_tree_depthwise.py` + `one-tree-depthwise-instrumentation.txt`.

---

## DEC-030: Depthwise leaf-index encoding for validation-path inference

**Sprint**: 27 (S27-FU-1)
**Date**: 2026-04-22
**Status**: DRAFTED — pending T3 implementation and G1-FU1 gate
**Authored by**: S27-FU-1-T2 (@ml-engineer)
**Commits (T1 repro + call-site evidence)**: `34f62b32c9` (repro harness), `eca086e4dd` (call-site triage)
**Extends**: DEC-029 (non-oblivious tree serialization) — adds the evaluation-side encoding decision.

### Context

`ComputeLeafIndicesDepthwise` (`csv_train.cpp:1751`) recomputes leaf indices for validation
documents on each training iteration. It was introduced alongside DEC-029's split-population
fix but never validated against the training-path encoding. Two independent bugs were confirmed
by the S27-FU-1-T1 harness (`docs/sprint27/scratch/fu1-t1-repro.md`): 103/200 sample
mismatches at depth=3; 51.5% mismatch rate. The function is called only inside the
`if (valDocs > 0)` guard at `csv_train.cpp:4040` (sole call site — confirmed by grep).

### Bug A — Encoding mismatch (fires at depth >= 2)

**Symptom**: `leafVec[d] = nodeIdx − numNodes` returns the BFS-array-leaf-offset (0-based rank
among depth-`D` leaves in BFS left-to-right order). The `leafValues` array is indexed by the
bit-packed partition encoding established in the training path: `bit k = goRight at depth k`.

**Mapping mismatch at depth=2** (from T1 evidence table):

| Path | BFS nodeIdx | BFS-offset (buggy) | Bit-packed (correct) | Match? |
|------|-------------|-------------------|----------------------|--------|
| LL   | 3           | 0                  | 0b00 = 0             | YES    |
| LR   | 4           | 1                  | 0b10 = 2             | NO     |
| RL   | 5           | 2                  | 0b01 = 1             | NO     |
| RR   | 6           | 3                  | 0b11 = 3             | YES    |

Mixed-direction paths (LR, RL) are routed to the wrong `leafValues` entry.

### Bug B — Split-lookup mismatch (fires at depth >= 3)

**Symptom**: `nodeSplits[nodeIdx]` uses the BFS node index as a flat-array position. But `splits`
is built in partition order (bit-packed order per depth level): for depth level `d` with
`2^d` partitions, position `splits[prefix + p]` holds the split for partition `p`, whose
BFS node is computed as:
```
bfsNode = 0; for lvl in 0..d-1: bfsNode = 2*bfsNode + 1 + ((p >> lvl) & 1)
```
At depth=3, `splits[4]` holds partition `p=1` (BFS node 5) and `splits[5]` holds partition
`p=2` (BFS node 4) — the indexing is transposed. Any traversal that reaches BFS nodes 4 or 5
at depth level 2 evaluates the wrong feature and threshold.

### Decision

Use **bit-packed partition encoding** for leaf-index output and a **BFS-node-keyed map**
for split lookup, mirroring `ComputeLeafIndicesLossguide`.

**Fix for Bug B**: Build `std::unordered_map<ui32, TObliviousSplitLevel> nodeSplitMap` keyed by
BFS node index, populated from the parallel `splits` + `splitBfsNodeIds` arrays (the latter
already built at `csv_train.cpp:3644–3654` as part of DEC-029). Replace `nodeSplits[nodeIdx]`
with `nodeSplitMap.at(nodeIdx)`.

**Fix for Bug A**: Accumulate `partBits |= (goRight << lvl)` inside the depth-traversal loop.
Replace `leafVec[d] = nodeIdx − numNodes` with `leafVec[d] = partBits`.

**Pseudo-code for the corrected inner loop**:
```
// Pre-traversal: build BFS → split map once per call
nodeSplitMap = { splitBfsNodeIds[i] → splits[i]  for i in 0..len(splits)-1 }

// Per-doc traversal
for d in 0..numDocs-1:
    nodeIdx = 0; partBits = 0
    for lvl in 0..depth-1:
        ns = nodeSplitMap.at(nodeIdx)
        fv = extract_feature(dataPtr, d, ns)
        goRight = (ns.IsOneHot) ? (fv == ns.BinThreshold ? 1 : 0)
                                : (fv >  ns.BinThreshold ? 1 : 0)
        partBits |= (goRight << lvl)
        nodeIdx = 2 * nodeIdx + 1 + goRight
    leafVec[d] = partBits
```

The signature gains one parameter: `const std::vector<ui32>& splitBfsNodeIds`.

### Rationale

**CPU-source authority**: `catboost/libs/model/cpu/evaluator_impl.cpp:462–492`
(`CalcIndexesNonSymmetric`) confirms that CatBoost CPU **never** uses `nodeIdx − numNodes`
to resolve a leaf value. The CPU stores `NonSymmetricNodeIdToLeafId[nodeIndex]` — an
explicit per-node map. The `nodeIdx − numNodes` formula is an MLX-specific invention that
has no basis in either the CPU evaluator or the training-path partition accumulation.

**MLX canonical encoding**: `csv_train.cpp:3660–3683` establishes bit-packed partition as the
leaf-index encoding (comment at line 3991: "partitions is already the correct leaf index").
`csv_train.cpp:3644–3654` defines the partition → BFS node forward map (DEC-029). The inverse
(BFS node → partition) is needed only at inference time in `ComputeLeafIndicesDepthwise`.

**Structural mirror**: `ComputeLeafIndicesLossguide` already holds the correct pattern —
BFS-keyed `nodeSplitMap` for lookup, explicit inverse map for final resolution. For Depthwise
the bit-packed accumulation replaces the `bfsToLeafId` inverse map (the encoding is implicit
in the traversal itself, not requiring a separate map), but the BFS-keyed split lookup is
the same mechanism.

### Scope

**Validation path only.** The sole call site is `csv_train.cpp:4040` inside `if (valDocs > 0)`.
The training hot path (`csv_train.cpp:3990–4003`), approximation update, `CalcLeafValues`, and
`FindBestSplitPerPartition` are all unaffected — they use the `partitions` bit array directly
and never call `ComputeLeafIndicesDepthwise`. Confirmed by call-site triage commit `eca086e4dd`
and `docs/sprint27/scratch/fu1-call-site-triage.md`.

### Retires DEC-029 risk entry

DEC-029 Risks section included:
> "S26-FU-1: `ComputeLeafIndicesDepthwise` (C++ validation path) still returns `nodeIdx − numNodes`
> (BFS leaf order) instead of bit-packed partition order."

This entry is retired upon T3 landing and G1-FU1 gate passage.

### Test gate

**G1-FU1**: Depthwise validation RMSE (with eval set provided, `use_best_model=True`) matches
CPU CatBoost within `rs=0` tight band `ratio ∈ [0.98, 1.02]`, across 3 seeds × {N=10k, N=50k}.
Gate must confirm `val_loss` history is now monotone-decreasing (or near-so) across iterations,
consistent with correct leaf routing.

Kill-switch: if T3 fix causes any bench_boosting ULP != 0 (v5 kernel-output parity) or any
`tests/test_python_path_parity.py` failure, abort and keep DEC-029 risk entry open.

### Authority

- `docs/sprint27/scratch/fu1-t1-repro.md` — bug evidence, mismatch table, depth-conditional fire analysis
- `docs/sprint27/scratch/fu1-call-site-triage.md` — scope confirmation (validation-only)
- `docs/sprint27/scratch/fu1-t2-audit.md` — CPU source audit + fix specification (this decision's input)
- `catboost/libs/model/cpu/evaluator_impl.cpp:462–492` — CPU canonical traversal (no `nodeIdx−numNodes`)
- `catboost/mlx/tests/csv_train.cpp:3644–3683` — MLX canonical encoding (bit-packed partition)
