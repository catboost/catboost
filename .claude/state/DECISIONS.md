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
- ~~**S26-FU-1**: `ComputeLeafIndicesDepthwise` (C++ validation path) still returns
  `nodeIdx − numNodes` (BFS leaf order) instead of bit-packed partition order. Affects
  validation RMSE tracking during Depthwise training only; does not affect training
  correctness or Python predictions.~~ — RETIRED by G1-FU1 pass (commit `88cbe6d067`), DEC-030 implements correct encoding. **Superseded-by**: DEC-030.
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
**Status**: IMPLEMENTED — T3 `fb7eb59b5f`, G1-FU1 PASS 6/6 `88cbe6d067`
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

This entry is retired. T3 landed (`fb7eb59b5f`) and G1-FU1 passed 6/6 cells (`88cbe6d067`). DEC-029 Risks entry struck through (2026-04-22).

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

---

## DEC-031: Anchor hygiene protocol

**Sprint**: 27 (Track B, T5)
**Date**: 2026-04-22
**Status**: Adopted. Protocol applies to all future `mlx/` commits.
**Authored by**: S27-AA-T5 (@technical-writer)
**Source material**: S27-AA-T1 inventory (`d4e2d7cf88`), T2 re-run (`800fdc8fce`), T3 classification (`9be26b91c0`), T4 landings (`adce339b56`–`62f17df7a9`).

### Context

In 22 sprints this project hit the same failure mode twice:

- **Sprint 8, TODO-022**: AN-008 (`2.22267818`) was a CHANGELOG-only "canonical" value captured from a mismatched-param run. No live assertion enforced it. It was discovered stale and corrected to `1.78561831`.
- **Sprint 26, D0-9**: AN-001–005 (five live test assertions, atol=1e-3) had silently diverged from current output after DEC-028 landed. They were discovered only when CI red-flagged `test_rmse_final_loss_matches_sprint4_anchor`.

The S27 anchor audit (T1–T4) enumerated 18 committed numeric anchors and found:

- 4 had drifted > 1e-4 from current output (AN-006, AN-007, AN-008, AN-018)
- 3 were structurally dead — no live harness enforces them and no path exists to regenerate them (AN-013, AN-014, AN-015)
- AN-008 is on its **third numeric lifetime** (`2.22267818` → `1.78561831` → `1.85752499`)
- 9 of 18 anchors were docs-only (no automated enforcement)

The pattern is not coincidence. Numeric anchors accumulated under a then-correct configuration; when the underlying code path changed legitimately, the anchors did not follow. The root cause is structural: docs-only "canonical" values have no enforcement mechanism and no update trigger.

### The four drift classes

The following classification, established in T3, is the canonical taxonomy for all future anchor triage:

- **class-a** (stale-capture): anchor was captured under a then-correct configuration; the underlying code or config has since legitimately changed; the new value is correct. Standard T4 update applies.
- **class-b** (regression): anchor value was correct at capture; current code differs from it; **this is a code bug, not an anchor problem.** Do not update the anchor. Escalate to an engineer before any anchor change. T3 confirmed zero class-b anchors in S27.
- **class-c** (documented-supersession): anchor was superseded by a later committed value (e.g., an intermediate port-tip or reverted branch); update to the post-supersession value with a pointer to the superseding commit. AN-012 and AN-018 are examples.
- **class-d** (dead anchor): anchor is not asserted by any live test, is behind a broken fixture, or is a docs-only "canonical" value with no enforcement path. Structural debt beyond stale values. AN-013, AN-014, AN-015 are examples.

### The hygiene rules

**Rule 1 — No new docs-only canonical values.**
Every committed numeric anchor must be wired to at least one live assertion in a pytest (or equivalent automated test). CHANGELOG.md "canonical anchor" entries and bench-report "expected values" tables without matching assertions are prohibited going forward.
- **Why**: The S8/S26/S27 pattern is driven entirely by docs-only values. A live assertion means CI catches drift at the next run; a docs-only value means drift can age undetected for 17+ sprints (as AN-015 demonstrated).
- **How to apply**: Before committing a new numeric result as "canonical" or "expected" in any markdown file, first wire it to an assertion in `python/tests/`. Then commit both in the same atomic change.

**Rule 2 — Anchor-change-on-path-change requirement.**
Any commit that modifies a code path must include anchor re-runs for anchors generated from that path — either updating them atomically in the same commit, or opening a follow-up AA-style audit immediately (not at the next sprint close).
- **Why**: The AN-006/AN-007 drift accumulated across five distinct shipping commits (S18, S19 ×2, S22/S23, S24) — each individually correct, collectively untracked. No single commit felt responsible for updating the test anchors.
- **How to apply**: If a commit touches histogram, kernel, accumulation, leaf-update, or gain-score code, grep the anchor inventory for anchors generated from that path and re-run them. The anchor inventory lives in `docs/sprint27/scratch/aa-t1-anchor-inventory.md`; keep it updated.

**Rule 3 — Every sprint close gets an anchor-drift check.**
At sprint close, re-run the anchors touched by that sprint's code changes and confirm no unexpected drift. This is cheap (minutes for bench_boosting configs) and prevents accumulation of the "code changed, anchor didn't follow" debt class.
- **Why**: S27's oldest undetected drift was AN-015 (15+ sprint staleness) and AN-008 (S18 to S27 = ~9 sprints). Both would have been caught within one sprint under this rule.
- **How to apply**: Sprint close QA responsibility. Scope is limited to anchors whose generating harness intersects with that sprint's diffs — not a full 18-anchor sweep every sprint.

**Rule 4 — Dead anchors are removed or wired, not ignored.**
When an anchor is classified class-d: either wire it to a live automated test (preferred) or remove it from committed documentation. Leaving unreachable "canonical" values in docs is actively misleading — they imply enforcement that does not exist.
- **Why**: AN-013, AN-014, and AN-015 are examples. AN-015 went further: the test *appeared* to assert CI workflow values but always skipped due to a filename typo (`mlx_test.yaml` vs `mlx-test.yaml`), giving false confidence for 15+ sprints.
- **How to apply**: T4 landed DEAD markers on AN-013/AN-014 pending this DEC. AN-015 fixture path was corrected in T4; the CI-embed design was retired in favor of standalone assertions. Future class-d identifications should resolve within the sprint they are found.

**Rule 5 — Repeat-offender promotion clause.**
If an anchor has had its numeric value updated more than once (i.e., a second stale-capture event), the next update must also promote it to a live-asserted test. A value-only change is not sufficient.
- **Why**: AN-008's three lifetimes (`2.22267818` → `1.78561831` → `1.85752499`) indicate it sits on a code path that is actively re-tuned. Another value-only CHANGELOG update will be stale again at the next kernel change.
- **How to apply**: Any T4-style anchor-update PR must run `git log -p` on the anchor's source file or the constant definition. If more than one prior value change exists, the PR must also land the test — e.g., add K=10 multiclass to `test_qa_round10_sprint5_bench_and_scan.py` alongside the value update.

### Scope and enforcement

- This protocol applies to **all future commits** on `mlx/` branches from S27 forward. It is not retroactive except where T4 already landed updates (`adce339b56`–`62f17df7a9`).
- `.claude/state/` narrative files (HANDOFF, CHANGELOG-DEV, TODOS, MEMORY) are exempt — they are prose, not enforced anchors.
- CI should eventually gain a lint step that detects new numeric literals in CHANGELOG.md or `docs/` bench-report tables without a matching `assert` or `pytest.approx` call. This is out of S27 scope; flagged as a follow-up task.

### Supersedes

- Supersedes the implicit "canonical value in CHANGELOG.md" pattern used for TODO-022 (Sprint 8) and similar.
- Does not retire any prior DEC.

### Audit references

- 18-anchor T1 inventory: `docs/sprint27/scratch/aa-t1-anchor-inventory.md` (`d4e2d7cf88`)
- T2 re-run results: `docs/sprint27/scratch/aa-t2-rerun-results.md` (`800fdc8fce`)
- T3 classification: `docs/sprint27/scratch/aa-t3-classification.md` (`9be26b91c0`)
- T4 atomic anchor updates: `adce339b56` (AN-006) through `62f17df7a9` (AN-013/014 DEAD markers)

**Next audit due**: Sprint 31 (every 4 sprints), or upon any kernel or accumulation algorithm change, whichever is sooner.

---

## DEC-032: MLX DW gain function scope (L2-only, not parity-equivalent to CPU)

**Sprint**: 27 (S27-FU-3, T5) → **PARTIALLY CLOSED by DEC-033 (S28, 2026-04-23)**
**Date**: 2026-04-22
**Status**: PARTIALLY CLOSED by DEC-033. Python canonical surface fully dispatched and guarded. SA-H1 closed in S29: C++ nanobind entry (`train_api.cpp:TrainConfigToInternal`) and CLI entry (`csv_train.cpp:ParseArgs`) now reject forbidden combinations. DEC-034 resolved S29 (outcome A — shared compounding mechanism). `Cosine+Lossguide` and `Cosine+SymmetricTree` guards remain at all three layers until S30-COSINE-KAHAN lands and gates. Full closure (CLOSED) queued for S30-close once Kahan parity gates pass. See DEC-033, DEC-034, DEC-035.
**Authored by**: S27-FU-3-T3/T5 (@ml-product-owner + @technical-writer)
**Source commit**: `0931ad6e9c` (FU-3 T1 triage — per-partition gain instrumentation + score_function=L2 CPU forcing)
**Triage doc**: `docs/sprint27/scratch/fu3-t1-triage.md`

### Context

S27-FU-3 was opened to triage five DW parity cells failing at N=1000 with `pred_std_R` up to 1.10 (MLX consistently better than CPU at small N, a direction not explained by noise or leaf-magnitude shrinkage). The original Track C framing offered a three-way trichotomy for the verdict: (a) BUG — open a fix; (b) NOISE — tighten gate scope to N≥10k; (c) ACCEPTED — widen pred_std_R band with rationale.

FU-3 T1 instrumented `FindBestSplitPerPartition` at DW N=1000 depth-0, capturing per-partition `(gain_MLX, gain_CPU, chosen_split, gradRms)` across 3 seeds. The resulting evidence shows:

- MLX `FindBestSplitPerPartition` hardcodes **L2 Newton gain** throughout the DW (and LG) code path. There is no `score_function` dispatch, no Cosine implementation, no Newton-Cosine or NewtonL2 variant.
- CPU CatBoost DW defaults to **Cosine gain**. It accepts a `score_function` hyperparameter selecting among `{Cosine, L2, NewtonL2, NewtonCosine}`.
- Forcing `score_function='L2'` on the CPU side reproduces MLX per-partition gain decisions within ±0.11% across all 3 seeds. The five failing cells collapse to parity when both sides use L2.

The asymmetry is therefore not algorithmic noise, not a parity gate edge case, and not a bug in either codebase. It is a **configuration divergence where both sides implement correct but different algorithms**. This is a 4th class of finding not represented in the original (a)/(b)/(c) trichotomy.

At small N (≤1000, depth=6, 64 partitions × ~15 docs each), the per-partition gain difference between Cosine and L2 is visible and produces measurable split disagreement (gain ratio 0.82–0.86 across the failing cells). At larger N or in the SymmetricTree path (where the gain is computed once over the full fold and aggregation smooths per-partition variation), the two formulas happen to agree closely — a coincidental numerical agreement, not structural parity.

### Decision

1. **MLX `FindBestSplitPerPartition` implements L2 Newton gain only.** This is the current state and is documented as a deliberate scope boundary pending S28 work.

2. **This is NOT algorithmically equivalent to CPU CatBoost's default (Cosine).** They are different algorithms. "Both are correct implementations of different gain functions" does not make them parity-equivalent. A parity claim requires both sides to compute the same algorithm; they do not.

3. **DW (and LG) parity tests in MLX MUST set CPU `score_function='L2'` explicitly.** Any parity test that compares MLX DW/LG output against CPU CatBoost with default `score_function` (Cosine) is measuring algorithmic divergence, not parity. The test outcome is not interpretable as parity evidence in either direction.

4. **v5 `bench_boosting` ULP=0 record remains valid at kernel scope only.** The ULP=0 record covers histogram kernel output (`bench_boosting.cpp` harness). It does not involve `FindBestSplitPerPartition` gain computation. This finding does not affect the v5 record.

5. **Python-path parity harness and S26-FU-2 gate numbers (aggregate RMSE parity) are coincidental-not-structural.** Those measurements were made with CPU default `score_function` (Cosine). The numbers pass the gate because at N≥10k the Cosine/L2 difference is small enough to fall within the gate's tolerance bands — they are not evidence of structural algorithmic equivalence. Re-labeling and re-blessing required in S28.

6. **FU-3 T4 gate adjustment**: the DW parity harness's CPU side MUST add an explicit `score_function='L2'` argument. This is the correct gate-scope tightening. Do NOT widen N scope to ≥10k to hide the disagreement — that would be the anchor-drift pattern documented in DEC-031 Rule 3 applied to gate scope.

### Rationale

**(a) The evidence shows structural algorithmic difference, not noise.** Per-partition gain ratios of 0.82–0.86 across 3 seeds are consistent and reproducible. Forcing `score_function='L2'` on CPU eliminates the divergence to ±0.11%. This is a signal, not scatter.

**(b) Widening N scope to ≥10k would be the S8/S26 anchor-drift pattern that DEC-031 just codified against.** The DEC-031 anchor hygiene rules exist precisely to prevent "it passes at larger N by coincidence" from masquerading as "it is correct." The same logic applies to gate scope: a gate that passes only because aggregation smooths a structural difference is not a valid parity gate.

**(c) Honest scoping now enables S28 to do the port correctly.** If DEC-032 labeled this ACCEPTED or NOISE, the S28 work would be framed as an "improvement" rather than a "gap closure." The correct framing is: MLX currently implements one gain function (L2); CPU's default is a different function (Cosine); the port is incomplete until MLX implements and dispatches the full `score_function` enum.

### Scope of this DEC

Covers `FindBestSplitPerPartition` (DW and LG code paths) only. Does NOT cover `FindBestSplit` (SymmetricTree path) — that path may have the same scope gap but it is not confirmed by FU-3 instrumentation and is out of scope until S28-AUDIT confirms.

### Supersedes / retires

None. DEC-032 is new framing of a newly-discovered algorithmic scope debt. It does not supersede any prior DEC.

### Test gate action

S27-FU-3-T4 (@ml-engineer, running in parallel with this commit) will add an explicit `score_function='L2'` requirement on the CPU side of the DW parity harness. Gate scope tightening: DW N=1000 cells pass when both sides use L2. Do NOT widen N threshold.

### Follow-up

**S28 "Score function fidelity"** — audit `score_function` dispatch end-to-end for the MLX backend (Python binding → C++ entry → `FindBestSplitPerPartition`); implement Cosine gain (highest-impact missing function); make L2 explicit via enum/dispatch rather than hardcoded; re-bless all aggregate-scope parity claims with explicit score_function labeling; re-run FU-3's 5 failing cells with `score_function='Cosine'` on both sides as structural proof of gap closure. **Closed by S28 (see DEC-033) with partial scope.**

### Authority

- `0931ad6e9c` — FU-3 T1 triage commit (per-partition instrumentation + CPU score_function forcing)
- `docs/sprint27/scratch/fu3-t1-triage.md` — evidence document, per-partition gain table, ±0.11% reproduction result

---

## DEC-033: EScoreFunction dispatch across all MLX grow policies (S28 score-function fidelity)

**Sprint**: 28
**Date**: 2026-04-23
**Status**: ACTIVE — shipping decision. DEC-032 PARTIALLY CLOSED pending S29-CLI-GUARD.
**Tip commit**: `e0b0b1b527`
**Authored by**: S28-CLOSE (@technical-writer + @ml-engineer)
**Sprint record**: `docs/sprint28/sprint-close.md`

### Problem

DEC-032 established that MLX hardcodes L2 Newton gain throughout all three grow-policy paths
(`FindBestSplitPerPartition` for Depthwise/Lossguide; `FindBestSplit` for SymmetricTree) while
CPU CatBoost exposes a `score_function` enum selecting from `{L2, Cosine, NewtonL2, NewtonCosine}`.
S28 was opened to close this gap: implement the dispatch surface and port the Cosine function.

### Decision

1. **`EScoreFunction` enum** added to the MLX C++ backend (`catboost/mlx/tests/csv_train.cpp`).
   Values: `L2`, `Cosine`, `NewtonL2`, `NewtonCosine`. `ParseScoreFunction` converts string
   hyperparameter to enum; `NewtonL2` and `NewtonCosine` throw `std::invalid_argument`
   (explicit not-implemented rejection, not silent fallback).

2. **Four dispatch sites** — one per grow policy × entry point:
   - `FindBestSplitPerPartition` (Depthwise path)
   - `FindBestSplitPerPartition` (Lossguide path)
   - `FindBestSplit` (SymmetricTree path, commit `4083add248`)
   The dispatch selects `ComputeCosineGainKDim` (Cosine) or the existing L2 Newton computation
   based on the enum. All three paths share the same dispatch pattern.

3. **`ComputeCosineGainKDim` helper** — ported from CPU reference
   `catboost/private/libs/algo/score_calcers.cpp` (`TCosineScoreCalcer`). Implements the
   numerator `(Σg)² / (Σh + reg)` and joint-denominator Cosine structure. Commit `83f30c3677`.
   Dead scalar-signature helper `ComputeCosineGain` removed in `e0b0b1b527` (code-review CR-S1).

4. **Nanobind binding** exposes `score_function` string parameter from Python to C++. Commit
   `0ea86bde21`.

5. **Python-side validation** (`_validate_params` in `python/catboost_mlx/core.py`):
   - `NewtonL2` and `NewtonCosine` → `ValueError` (mirroring C++ rejection).
   - `Cosine + Lossguide` → `ValueError` (guarded; pending S29-LG-COSINE-RCA). Line 634.
   - `Cosine + SymmetricTree` → `ValueError` (guarded; pending S29-ST-COSINE-KAHAN). Line 644.

### Combination status

| `score_function` × grow policy | Status | Evidence |
|-------------------------------|--------|---------|
| L2 × any policy | Ships — no regression | 28/28 parity PASS at `b9577067ef` |
| Cosine × Depthwise | Ships — in-envelope | 1.6% drift at N=1000/50k/50-iter; gate G2a/G2b/G6a–d PASS |
| Cosine × Lossguide | Guarded — `ValueError` | ~unacceptable drift; LG priority-queue interaction; S29-LG-COSINE-RCA |
| Cosine × SymmetricTree | Guarded — `ValueError` | 0.77% @ 1 iter → ~47% @ 50 iter (float32 joint-denominator compounding); S29-ST-COSINE-KAHAN |
| NewtonL2 × any | Rejected — `ValueError` / `std::invalid_argument` | Not implemented; explicit not-implemented guard |
| NewtonCosine × any | Rejected — `ValueError` / `std::invalid_argument` | Not implemented; explicit not-implemented guard |

### Rationale

- **Enum model mirrors CPU CatBoost**: CPU exposes `EScoreFunction` with four values; the MLX
  port uses the same model, implementing the two viable variants (L2, Cosine) and explicitly
  rejecting the two unimplemented Newton variants.
- **Explicit guard over silent fallback**: `NewtonL2`/`NewtonCosine` raise loudly rather than
  silently using L2. Consistent with DEC-032's framing: wrong algorithm ≠ acceptable default.
- **Combination guards are evidence-driven**: LG+Cosine and ST+Cosine guards are not
  conservative blanket blocks — DW+Cosine ships without a guard because its drift is
  quantitatively in-envelope. The guards reflect measured drift evidence.

### S29 carry items

- **S29-CLI-GUARD** (SA-H1): Port combination rejections into `TrainConfigToInternal` and
  `csv_train.cpp:ParseArgs`. Currently only the Python surface enforces them.
- **S29-LG-COSINE-RCA**: Root-cause LG+Cosine unacceptable drift. Fix plan or defer to S30.
- **S29-ST-COSINE-KAHAN**: Port Kahan/Neumaier compensated summation to ST+Cosine denominator;
  gate: 50-iter drift ≤ 1% at N=50k.

### DEC-032 partial closure note

DEC-032's standing constraint ("DW and LG parity tests MUST set `score_function='L2'` explicitly
until S28 closes") is lifted for DW: DW+Cosine now ships in-envelope and all test cells carry
explicit `score_function` labels (S28-REBLESS, `c07e895f7c`). LG constraint retained: LG+Cosine
is guarded at Python API. DEC-032 status updated to PARTIALLY CLOSED by this decision.

### Authority

| Commit | Content |
|--------|---------|
| `da02da0259` | S28-AUDIT — zero-plumbing baseline |
| `83f30c3677` | S28-COSINE — `ComputeCosineGainKDim` helper |
| `0ea86bde21` | S28-L2-EXPLICIT — enum, dispatch, nanobind, Python validation |
| `4083add248` | S28-OBLIV-DISPATCH — ST dispatch |
| `c07e895f7c` | S28-REBLESS — parity cell labels |
| `dca62f0d72` | S28-FU3-REVALIDATE — DW force-L2 lifted |
| `b9577067ef` | S28-{LG,ST}-GUARD — combination `ValueError` guards |
| `e0b0b1b527` | S28-CR-S1 — dead helper removed |

Gate reports: `docs/sprint28/fu-cosine/t2-gate-report.md`,
`docs/sprint28/fu-l2-explicit/t3-gate-report.md`,
`docs/sprint28/fu-rebless/t4-rebless-report.md`,
`docs/sprint28/fu-fu3-revalidate/t5-gate-report.md`,
`docs/sprint28/fu-obliv-dispatch/t7-gate-report.md`,
`docs/sprint28/fu-cr/t6-cr-report.md`,
`docs/sprint28/fu-sa/t6-sa-report.md`.

---

## DEC-034: LG-Cosine mechanism resolution

**Sprint**: 29 (S29-LG-SPIKE-T1/T2, #84/#85; resolution checkpoint #86)
**Date**: 2026-04-23
**Status**: RESOLVED — outcome A (shared float32 joint-Cosine denominator compounding), moderate confidence. Verdict commit `64a8d9076b`. See `docs/sprint29/lg-mechanism-spike/verdict.md`.
**Owner**: @research-scientist (spike) → Ramos (T5 resolution at #86)
**Authored by**: S29-00 kickoff (@technical-writer)

### Context

LG+Cosine (`score_function='Cosine'` × Lossguide grow policy) shows unacceptable per-partition
gain drift vs CPU CatBoost. It was guarded at the Python API in S28 (`b9577067ef`, `core.py:634`)
to unblock S28 close. The guard is a `ValueError` on the Python surface; no equivalent guard
exists at the C++ or CLI entry (SA-H1, being closed by S29-CLI-GUARD #82/#83).

The mechanism driving the drift is unknown. Two hypotheses compete:

### Hypothesis A — Shared float32 compounding (outcome A)

The same `sqrt(Σden)` float32 joint-denominator accumulation that produces ~0.77% ST+Cosine
iter-1 drift also operates in the LG path. Compounding is a property of the `ComputeCosineGainKDim`
accumulator shared across grow policies, not of LG's priority-queue structure.

**Prediction**: LG iter-1 drift ≈1% (same order of magnitude as ST+Cosine 0.77%).

**Implication if confirmed**: Kahan/Neumaier compensated summation in `ComputeCosineGainKDim`
is a candidate fix for both LG and ST in a single pass. Opens S29-KAHAN-JOINT stretch or S30 work.

### Hypothesis B — Priority-queue ordering divergence (outcome B)

LG's priority-queue leaf-selection amplifies sub-ULP gain-ordering sensitivity. At each iteration
the queue pops the partition with the highest gain; a per-partition gain that is off by even
fractional ULP can change the queue's ordering, selecting a different partition to expand, which
in turn changes the data layout fed to subsequent iterations. This is a compounding amplification
that Kahan summation in the denominator accumulator would not address — the divergence is
architectural, not numerical.

**Prediction**: LG iter-1 drift ≥5% (substantially larger than ST+Cosine 0.77%).

**Implication if confirmed**: LG+Cosine requires algorithmic work (e.g., lexicographic gain
tiebreaking on the priority queue, or a separate LG-specific Cosine path) that is LG-specific
and likely higher-risk than Kahan. Does not generalize to ST.

### Discriminator

**Iter-1 drift measurement** at LG+Cosine vs the ST+Cosine iter-1 anchor (0.77%, from
`docs/sprint28/fu-obliv-dispatch/t7-gate-report.md`):

| Measured LG iter-1 drift | Outcome | Implication |
|--------------------------|---------|-------------|
| ≈1% (same order as ST) | A — shared compounding | Kahan candidate for both guards |
| ≥5% (substantially larger) | B — priority-queue divergence | Algorithmic work, LG-specific |
| Ambiguous (between) | C | Close S29 with spike doc; re-scope S30 |

### Decision tree

- **Outcome A**: Open S29-KAHAN-JOINT stretch (Ramos approval required at #86). If approved and
  landed in S29, both `core.py:634` (LG+Cosine) and `core.py:644` (ST+Cosine) guards may be
  lifted once the 50-iter drift ≤ 1% gate passes at N=50k.
- **Outcome B**: Close S29 with CLI-GUARD only. Open dedicated S30 LG-Cosine algorithmic work
  item. ST+Cosine Kahan work proceeds independently on its own timeline.
- **Outcome C**: Close S29 with CLI-GUARD + spike verdict doc. Re-scope S30 with additional
  discriminating measurements.

### Resolution path

1. #84 (S29-LG-SPIKE-T1) — instrument iter-1 drift measurement.
2. #85 (S29-LG-SPIKE-T2) — verdict doc with outcome A/B/C classification.
3. #86 (S29-BRANCH-DECISION) — Ramos decides stretch vs close. This DEC updates to RESOLVED
   at that checkpoint with the chosen outcome recorded.

### Authority

- `docs/sprint28/fu-obliv-dispatch/t7-gate-report.md` — ST+Cosine iter-1 anchor (0.77%)
- `python/catboost_mlx/core.py:634` — LG+Cosine Python guard (S28 `b9577067ef`)
- `python/catboost_mlx/core.py:644` — ST+Cosine Python guard (S28 `b9577067ef`)
- `docs/sprint29/lg-mechanism-spike/verdict.md` — S29 spike verdict (outcome A, `64a8d9076b`)
- Data artifacts: `docs/sprint29/lg-mechanism-spike/data/iter1_drift.json`, `iter_curve.csv`, `tree_structure_iter1.json`

### Outcome A summary

LG+Cosine iter-1 mean drift 0.0024% (3 seeds) vs ST+Cosine anchor 0.77% — same order of
magnitude (~300× smaller), same compounding direction. iter=1 BFS split sequences bit-identical
CPU vs MLX at seed=0. Confidence moderate: shallow cell only (depth=3, max_leaves=8). Deep
LG cells not exercised. Recommendation: apply Kahan/Neumaier to shared joint-Cosine denominator
once in S30 (DEC-035); re-open outcome B only if post-Kahan residual drift persists on deep LG.

---

## DEC-035: S30-COSINE-KAHAN — phased Kahan fix on Cosine accumulation

**Sprint**: 30 (active as of 2026-04-23; CLOSED 2026-04-24)
**Date**: 2026-04-23 (draft); elaborated 2026-04-23 after S30 kickoff ultrathink; closed 2026-04-24
**Status**: PARTIALLY CLOSED — precision fix class exhausted; K4 (fp64 cosDen/cosNum) + Fix 2 (fp64 gain/argmax) shipped but insufficient alone. ST+Cosine and LG+Cosine guards REMAIN in place. Dominant ST mechanism is structural (N-independent) and superseded to DEC-036.
**Authored by**: S29-CLOSE (@technical-writer); elaborated S30-00 kickoff (orchestrator); closure addendum S30-CLOSE 2026-04-24

### Context

DEC-034 outcome A (2026-04-23) establishes that `LG+Cosine` and `ST+Cosine` drift plausibly
share the same float32 joint-Cosine denominator compounding mechanism, with moderate confidence.
Two observations complicate a direct Kahan-then-remove-guards approach:

1. **300× magnitude gap**: ST anchor is 0.77% iter-1 drift; LG spike is 0.0024%. "Same
   mechanism, different cell" does not fully explain this — the gap is too large for cell
   geometry alone.
2. **iter=1 trees bit-identical** (LG spike): gain ordering is not diverging enough to flip
   splits. Therefore the observed 0.0024% drift must come from downstream of split selection
   — leaf values or approximation updates — not the `cosDen` accumulator itself.

If we Kahan-patch `cosDen` alone on the verdict's authority and the actual dominant error
source lives elsewhere (leaf-value sum, approx update), the fix will move the needle less
than expected and we'll have landed a "same mechanism" claim that doesn't hold numerically.

### Decision

Adopt a **phased T1→T4 approach** that instruments the drift *before* patching, verifying the
target accumulator is correct before committing to a fix. Each phase has its own gate;
kill-switches route around expected failure modes without requiring mid-sprint re-planning.

### Phased plan

| Task | Purpose | Primary gate |
|------|---------|--------------|
| **T1 — Instrument** (#90) | Per-stage float32 residual dump on ST anchor at iter-1 | Identifies dominant drift source; G1 "mechanism fingered" |
| **T2 — Kahan** (#91) | Kahan/Neumaier on the accumulator T1 fingers | G2: iter-1 drift reduces ≥10× on ST anchor |
| **T3 — Measure** (#92) | 2-tier post-Kahan parity (ST anchor + LG-Mid + LG-Stress) | G3a/b/c per cell (see below) |
| **T4a — ST remove** (#93) | Atomic ST+Cosine guard removal (3 languages + 2 tests) | G3a pass + 28/28 parity |
| **T4b — LG remove** (#94) | Atomic LG+Cosine guard removal (3 languages + 2 tests) | G3b AND G3c pass + 28/28 parity |

Secondary tracks (parallel):

| Task | Purpose |
|------|---------|
| **T5 — CLI exit wrap** (#95) | SA-I2-S29 — `csv_train:main()` try/catch for exit(1) |
| **T6 — Nits cleanup** (#96) | S29 CR items: N-1, N-2, N-3, SF-3 |

### Acceptance gates

| ID | Gate | Criterion | Rationale |
|----|------|-----------|-----------|
| G1 | Mechanism fingered | T1 triage identifies one accumulator with residual > 10⁻⁵ at iter-1 | Proves a non-trivial error source exists and is locatable |
| G2 | Mechanism targeted | iter-1 drift on ST anchor reduces ≥10× after Kahan applied | Tests the **lever's mechanism**, not aggregate drift proxy. Prevents DEC-028-style "kernel-ULP=0 ≠ full-path parity" trap |
| G3a | ST envelope | ST+Cosine aggregate drift < 2% at 50-iter on S28 anchor cell | API-layer guard rationale dissolves |
| G3b | LG-Mid envelope | LG+Cosine drift ratio ∈ [0.98, 1.02] at `N=1000, depth=6, max_leaves=31, 50-iter, seeds={42..46}` | t5-continuity for audit trail |
| G3c | LG-Stress envelope | LG+Cosine drift ratio ∈ [0.98, 1.02] at `N=2000, depth=7, max_leaves=64, 100-iter, seeds={0,1,2}` | **Genuinely stresses priority-queue divergence surface** — 8× the contested-split density of S29 spike. Rules out outcome-B residual for production `max_leaves` regimes |
| G4 | Parity | 28/28 `test_python_path_parity.py` hold after each Kahan + removal commit | No regression to L2 or DW+Cosine cells |
| G5 | Perf | < 5% regression on Cosine cells in bench_boosting | Kahan adds ~2 flops per accumulation; acceptable overhead |
| G6 | Guard-removal grep | `grep -rn 'TODO-S29-ST-COSINE'` → 0 (post T4a); `grep -rn 'TODO-S29-LG-COSINE'` → 0 (post T4b) | Single-point-of-removal invariant confirmed |

### Kill-switches (pre-authorized)

- **K1 (T1 mechanism miss)**: if T1 triage shows the dominant error source is NOT `cosDen` (e.g., leaf-value sum or approx update dominates), do not port Kahan to `cosDen` blindly. Update T2 plan to target the actual source; T2 must still pass G2. Branch plan change is expected to be a source-swap, not a sprint-scope expansion.

- **K2 (LG-Stress fails)**: if G3c fails but G3a + G3b pass, T4a lands (ST-only removal). T4b is skipped; LG guard stays; DEC-032 remains PARTIALLY CLOSED (LG side). File S31-LG-DEEP-RESIDUAL to re-examine outcome B. Sprint still ships ST closure as a real win.

- **K3 (perf regression)**: if G5 fails by >5%, consult @performance-engineer before merging. Do not remove guards under perf regression.

- **K4 (Metal auto-reassociation)**: if Metal compiler defeats the Kahan compensation term (auto-reassociation or `fast_math` aggressive optimization), fall back to double-precision for the denominator path. File DEC-036 documenting the CLAUDE.md "float32 for accumulation always" exception with full rationale. **Pre-authorized** — no user checkpoint required; just file DEC-036 at merge time.

### Atomicity policy

DEC-012 "one structural change per commit" interpreted as:
- T4a = one structural change = "remove ST+Cosine combination guards across all entry points"
- T4b = one structural change = "remove LG+Cosine combination guards across all entry points"

Within each removal commit, all three language layers (Python + C++ nanobind + CLI) + their
tests go together. Removing one language's guard without the others would recreate SA-H1
(the vulnerability S29-CLI-GUARD closed).

### Two-tier LG measurement rationale

S29 spike verdict explicitly flagged: "With only 8 leaves the queue makes few contested
choices; any latent ordering sensitivity would be more visible at 64+ leaves." LG-Stress
(`max_leaves=64`) is designed to exercise that surface. Passing LG-Stress is necessary — not
sufficient — to claim outcome-B residual is absent. Users running production LG with
`max_leaves > 64` remain on their own warranty; our evidence covers up to 64.

### Cleanup scope (post T4a+T4b)

Four guard sites removed atomically per-combo:
1. `python/catboost_mlx/core.py:628-647` (`_validate_params` ValueError blocks — 2 blocks, one per combo)
2. `catboost/mlx/train_api.cpp:25-51` (C++ nanobind entry guards — 2 blocks, one per combo)
3. `catboost/mlx/tests/csv_train.cpp:241-267` (CLI ParseArgs guards — 2 blocks, one per combo)
4. `tests/test_cli_guards.py` (4 test cases — 2 per combo; if both T4a+T4b land, file may be deleted entirely)

### Conditional S31 follow-up

Open S31-LG-DEEP-RESIDUAL only if K2 fires (G3c fails on `max_leaves=64` post-Kahan) or if
post-merge evidence surfaces LG-specific drift at production-deep configs. Do not pre-open.

### Authority

- DEC-034 outcome A: `docs/sprint29/lg-mechanism-spike/verdict.md`
- ST+Cosine anchor: `docs/sprint28/sprint-close.md` (0.77% iter-1, ~47% iter-50 aggregate drift)
- LG+Cosine spike: `docs/sprint29/lg-mechanism-spike/data/iter1_drift.json` (0.0024% mean at shallow cell)
- Phased-plan rationale: S30-00 kickoff commit (this commit)

### Closure addendum (2026-04-24)

S30 executed the full T1→T4 phased plan plus an extensive verification battery (D1/D2/D2-redux/D3/D4/V1/V2/V5/V6) after T3 failed G3a/G3b/G3c at the measurement layer. Key findings:

| Phase | Gate | Result | Meaning |
|-------|------|--------|---------|
| T1 (#90) | G1 — mechanism fingered | PASS | cosDen fingered; residual 4.067e-3 at iter-1 |
| T2 (#91) | G2 — ≥10× residual reduction | PASS (12.5×) | K4 fp64 widening applied at measurement layer |
| T3 (#92) | G3a ST < 2% @ 50-iter | **FAIL (53.30%)** | Measurement-layer fix did not reach trajectory layer |
| T3 (#92) | G3b LG-Mid ratio ∈ [0.98, 1.02] | FAIL (1.27–1.31) | LG did not converge either |
| T3 (#92) | G3c LG-Stress | FAIL (1.44–1.45) | K2 pre-authorized kill-switch fired |
| Fix 2 (#108) | Predicted ST drop toward DW floor | **FAIL (53.30% → 53.30%)** | L3/L4 gain cast + fp32 argmax not binding |
| V6 (#109) | L1 histogram N-scaling | **FALSIFIED (b ≈ 0.0 across 100× N)** | Precision fix class exhausted |

DEC-034 outcome A ("shared float32 joint-denominator compounding") is **partially falsified for ST** by V6's flat N-scaling (b=0.0017 across N ∈ {500, 1k, 5k, 10k, 25k, 50k}). A precision-compounding mechanism would produce scaling exponent b ≈ 1.0; b ≈ 0 indicates an N-independent structural divergence, not fp32 accumulation error.

**Ships from S30:**
- K4 fp64 widening of cosNum/cosDen accumulators (commits `108c7a59d2`-family)
- Fix 2 fp64 widening of totalGain/bestGain/TBestSplitProperties::Gain/perturbedGain/TLeafCandidate::Gain (commits `90a0cb4475` + `364d4ee962`)
- 13 verdict documents under `docs/sprint30/` (T1, T2, T3, D1, D2, D2-redux, D3, D4, V1, V2, V5, V6, Fix 2)
- Guards unchanged at all three layers (Python `_validate_params`, `train_api.cpp:TrainConfigToInternal`, `csv_train.cpp:ParseArgs`) — both ST+Cosine and LG+Cosine remain rejected.

Both K4 and Fix 2 are logically correct fixes that will become load-bearing once the structural mechanism is resolved; they remove precision floors that would otherwise re-surface. They are not wasted code.

**Status transitions:**
- DEC-035: ACTIVE → PARTIALLY CLOSED (precision fix class exhausted; atomicity clause and Kahan rationale preserved for reference)
- DEC-034: RESOLVED (outcome A) → PARTIALLY FALSIFIED for ST (V6 N-scaling rules out pure precision mechanism); LG outcome-B confirmed dominant for LG (D3 verdict)
- DEC-032: PARTIALLY CLOSED → STATUS UNCHANGED (LG and ST guards still in place)

**Forward pointer:** DEC-036 opens the structural divergence investigation. S31 targets iter=1 split-selection audit as T1.

---

## DEC-036: ST+Cosine structural divergence — iter=1 algorithmic audit

**Sprint**: 31 (kickoff 2026-04-24)
**Date**: 2026-04-24
**Status**: OPEN — investigation-phase; no mechanism identified yet
**Authored by**: S30-CLOSE (orchestrator)

### Context

S30 exhausted the precision fix class for the ST+Cosine 53% aggregate drift:

- **cosNum/cosDen accumulator** (K4): widened to fp64, measurement-layer 12.5× residual reduction, trajectory-layer no movement.
- **Gain scalar + argmax** (Fix 2): widened `totalGain`, `bestGain`, `TBestSplitProperties::Gain`, `perturbedGain`, `TLeafCandidate::Gain` to `double`. ST drift bit-identical before/after: 53.30% → 53.30%.
- **L0 histogram N-scaling** (V6): drift is N-independent (b ≈ 0 across 100× N range). Falsifies pure-precision class entirely. See `docs/sprint30/v6-n500-confirmer/verdict.md`.

V6 rules out any fix that would scale drift linearly with N. That forecloses: fp64 histogram accumulation, Kahan/Neumaier on any cross-partition accumulator, widening statistics containers, quantization-border precision fixes, and any other "more precision in accumulators" class of fix.

The remaining hypothesis class is **structural divergence**: CPU CatBoost and MLX compute different algorithms (different gain formula, different split enumeration, different candidate ordering, different argmax tie-break, different quantization borders, different basePred initialization, or different tree-construction nuance) that produce different split decisions regardless of precision.

### Decision

Open S31-ITER1-AUDIT with a **preflight-first** strategy:

**T1-PRE (source-level preflight, cheap, runs first)** — @research-scientist diffs CPU `TCosineScoreCalcer::CalcMetric` against MLX `ComputeCosineGainKDim` algebraically (side-by-side symbol mapping, regularization term, parent-gain subtraction, K-dim sum order). Simultaneously verifies that basePred initialization, feature quantization borders, and the iter=1 initial gradient vector match bit-for-bit between CPU and MLX. Three possible verdicts:
- **(i) FORMULA DIVERGENCE**: DEC-036 mechanism class named; skip to T2 fix design.
- **(ii) PRE-SPLIT DIVERGENCE**: fires K4 kill-switch; S31 re-scopes to pre-split fix.
- **(iii) CLEAN**: proceed to T1-AUDIT.

**T1-AUDIT (instrumented runtime comparison, contingent on T1-PRE verdict iii)** — @ml-engineer builds an iter=1 split-selection comparison harness at the S28 anchor cell (N=50k, ST, Cosine, rs=0, seeds 42/43/44). For each layer up to first divergence, dump from both CPU and MLX: parent aggregates `(Σg, Σh, W, leaf_count)`, top-K=5 candidates `(feature_idx, bin_idx, gain)`, and the winning tuple. Stop at the first diverging layer (deeper layers see stale assignment and are not comparable).

The **first diverging layer names the mechanism class**:

| First divergence at | Implied mechanism class |
|---------------------|-------------------------|
| Layer 0 (root split) | Cosine gain formula itself differs between CPU and MLX |
| Layer 1–5 with same feature, different bin | Split-candidate enumeration or bin-boundary difference |
| Layer 1–5 with different feature | Tie-break policy or gain ranking difference |
| Any layer with same (feature, bin) but different gain value | Gain computation scale/normalization difference |
| Top-1 matches, top-K=2..5 differ | Tie-break under near-equal gains |
| No divergence at iter=1, emerges later | Leaf-value estimation or approx update divergence (iter=2+ audit needed) |

### Gates

| ID | Gate | Criterion |
|----|------|-----------|
| G1-PRE | Source aligned | T1-PRE delivers side-by-side algebraic mapping of the two formulas + preflight checks recorded |
| G1 | Divergence localized | T1-AUDIT identifies first diverging layer with file:line pointers to both CPU and MLX implementations |
| G2 | Mechanism named | DEC-036 updated with specific mechanism class and CPU-vs-MLX algebraic difference |
| G3 | Fix proposed | Concrete fix proposal with parity gate and falsifiable prediction |

### Kill-switches (pre-authorized)

- **K1 (no iter=1 divergence)**: if CPU and MLX produce bit-identical iter=1 split sequences at all 3 seeds, the mechanism emerges at iter≥2 (leaf values or approx update). Expand audit to iter=2 leaf-value and approx-update comparison. **Pre-authorized**, no user checkpoint.
- **K2 (mechanism is an upstream gap)**: if the divergence is a missing MLX feature (e.g., MLX implements a different Cosine variant than CPU default), DEC-036 becomes scope for a **feature-port sprint**, not a precision sprint. Re-plan S31 as port work. Escalate to Ramos.
- **K3 (seed-independent false positive)**: if 0 of 3 seeds diverge at iter=1, the mechanism is not deterministic-structural and the premise of S31 is compromised. Revisit precision hypothesis with the new evidence (e.g., RandomStrength PRNG paths). Escalate to Ramos.
- **K4 (pre-split divergence)**: if T1-PRE finds basePred, quantization borders, or initial gradients differ between CPU and MLX, the mechanism is upstream of split selection. **Pre-authorized** re-scope S31 to a pre-split fix track; T1-AUDIT deferred. Trivial-class fix expected.
- **K5 (cross-cutting fix)**: if the mechanism is located but the fix requires changes across histogram kernels + score calcer + node aggregation, the scope exceeds a single structural sprint. Budget warning; escalate to Ramos before committing to implementation.

### Authority

- S30 closure addendum above (precision exhaustion rationale)
- V6 verdict `docs/sprint30/v6-n500-confirmer/verdict.md` (N-scaling falsification)
- Fix 2 verdict `docs/sprint30/fix2-fp64-gain/verdict.md` (L3/L4 exhaustion)
- D4 verdict `docs/sprint30/d4-joint-denom/verdict.md` (accumulator exhaustion, 2.42× not 64×)
- D1 audit `docs/sprint30/d1-cpu-audit/verdict.md` (CPU is fp64 end-to-end — bit-parity in fp32 MLX is unreachable; closest achievable is algorithmic equivalence modulo fp32 ULP)

### T1-PRE outcome (verdict ii — K4 fires)

**Commit**: `aed81c63d7` — `docs/sprint31/t1-pre/verdict.md`
**Date**: 2026-04-24
**Verdict**: (ii) PRE-SPLIT DIVERGENCE — kill-switch **K4 fires** (pre-authorized).

**Formula mapping (§2)** — 11 rows (F1–F11: avg formula, numerator term, denominator term, L2 regularization, parent-gain absence, leaf summation, K-dim aggregation, denominator guard, gain sign, stats container signs, weights) all confirmed ALIGNED between CPU `TCosineScoreCalcer` and MLX `ComputeCosineGainKDim`. The Cosine gain formula is CLEAN. No structural algebraic divergence at the split-scoring layer.

**Pre-split preflight (§3)** — P1–P7 checks:
- **P6 quantization borders (DIVERGENT by construction)**: CPU default is `GreedyLogSum` (from `library/cpp/grid_creator/binarization_options.h:16`); MLX uses a custom percentile-midpoint equal-frequency algorithm (`csv_train.cpp:816-889`). The two algorithms produce different bin edges for identical input distributions → different per-document bin indices → different per-bin histogram aggregates `(Σg_b, Σh_b, W_b)` → different split candidate enumeration and gain values. This is the primary suspected mechanism for the 53% ST+Cosine drift.
- **P1–P4, P7**: basePred initialization, iter=1 gradient vector, leaf count, tree topology all match. No upstream-of-quantization divergence.
- **P5 latent finding (secondary)**: MLX never calls `ScaleL2Reg`; CPU scales `L2RegLambda` by `sumAllWeights / docCount` before passing to score calcer. This is a systemic regularization-scale bug orthogonal to borders. Fix at `csv_train.cpp:4068, 4189`.

**Mechanism class named**: **P6 quantization border algorithm divergence**, with **P5 L2 regularization scaling** as a secondary orthogonal bug.

**Qualifier against trivial-class assumption**: The K4 kill-switch language anticipates a "trivial-class fix." Two pieces of prior evidence argue the border-port may not fully close the 53% drift:
- **S26-D0 P10 historical probe**: forcing CPU-computed borders into MLX yielded a **0.06% ratio gap at L2+RS=0+N=10k** — non-dominant for L2, but Cosine may amplify border sensitivity.
- **V6 N-scaling (b ≈ 0.0017)**: flat N-scaling does not cleanly predict a quantization-border mechanism, which would typically scale sub-linearly with N via bin-density convergence.

If the S31-T2 port of `GreedyLogSum` closes the drift to ≤ 2%, mechanism is confirmed. If residual drift remains > 2% after port + P5 fix, T1-AUDIT (now S31-T3b fallback) runs to hunt for a remaining structural layer mechanism.

### S31-T2-PORT-GREEDYLOGSUM (chosen path — "B")

**Decision date**: 2026-04-24
**Authority**: Ramos explicit "B" — port `GreedyLogSum` from `library/cpp/grid_creator/binarization.cpp` into MLX verbatim, replacing `csv_train.cpp:816-889`. Skip cheap falsification probe (forcing CPU borders into MLX) because the port is necessary cleanup regardless of drift outcome ("fix properly always").

**Scope**:
1. Port CPU `GreedyLogSum` (plus supporting `MakeBinarizer`) into MLX quantizer path; replace custom percentile-midpoint code at `csv_train.cpp:816-889`.
2. Fix P5: add `ScaleL2Reg` call at `csv_train.cpp:4068, 4189` → `L2RegLambda = L2RegLambda · (sumAllWeights / docCount)`.
3. P11 (hessian-vs-sampleWeight at `csv_train.cpp:3780, 3967`) → separate tracked task `S31-T-LATENT-P11`, out of scope for T2.

**Gates** (all hard):
- **G2a** Borders byte-match CPU — random 10 feature × 10 dataset probe
- **G2b** ST+Cosine drift ≤ 2% at S28 anchor (N=50k, rs=0, seeds 42/43/44)
- **G2c** bench_boosting v5 ULP=0 preserved (histogram kernel parity not regressed)
- **G2d** 18-config L2 non-regression (perf and parity)

**Fallback**: If G2b fails (residual drift > 2%), spawn S31-T3b = T1-AUDIT instrumented iter=1 split-selection harness per the original DEC-036 plan.

### T3b outcome (G1 PASS — GAIN-FORMULA mechanism)

**Commit**: `746d5090b5` — `docs/sprint31/t3b-audit/verdict.md`
**Date**: 2026-04-24
**Verdict**: G1 PASS. First diverging layer = depth=0 (seeds 42, 44) or depth=2 (seed 43). Mechanism class = **GAIN-FORMULA** per DEC-036 table.

**Evidence**: MLX Cosine gain ratio vs CPU = **0.946 stable across seeds and depths**.
- seed=42: CPU f0/b59 gain=89.616; MLX f0/b64 gain=84.777 (rdiff 5.40e-2)
- seed=44: CPU f0/b62 gain=89.098; MLX f0/b61 gain=84.140 (rdiff 5.56e-2)
- Partition `sumH` at depth=0 matches byte-exact — histograms are correct; only the score computation is wrong.

**Ultrathink reading of 0.946**: `0.946² ≈ 0.895`; `1/√1.117 ≈ 0.946`. A stable multiplicative ratio points to a single algebraic bias — either (i) missing numerator normalization, (ii) extra denominator term (~11.7%), or (iii) code-path skew (T1-PRE mapped `ComputeCosineGainKDim` but the live path at `S28-OBLIV-DISPATCH` may be inline).

**Forward pointer**: DEC-038 opens Sprint 32 term-level audit.

---

## DEC-037: Border-count off-by-one + greedy GreedyLogSumBestSplit restoration

**Sprint**: 31 (formalized post-hoc during S32 kickoff)
**Date**: 2026-04-24
**Commit**: `746d5090b5` (bundled into T3b-T1-AUDIT verdict commit)
**Status**: CLOSED — shipped.

### Problem

Two bugs in MLX quantization borders found during S31 T3b audit:

1. **Off-by-one**: `maxBordersCount = maxBins - 1 = 127` where CPU CatBoost uses `border_count = 128`. Caused systematic `MLX_bin = CPU_bin - 1` offset.
2. **DP with document-count weights**: T2 port (`768ee50abd`) replaced the greedy `GreedyLogSumBestSplit` with a dynamic-programming implementation using document-count weights. CPU's `TGreedyBinarizer` uses the unweighted `TFeatureBin` path — each unique value has weight 1, not the document count. Algorithmically wrong.

### Decision

Restore greedy priority-queue `GreedyLogSumBestSplit` (unweighted unique-value counts, per `library/cpp/grid_creator/binarization.cpp` `TGreedyBinarizer`), and fix `maxBordersCount = maxBins`.

### DEC-012 atomicity violation (flagged)

The fix was shipped INSIDE commit `746d5090b5` alongside the T3b verdict doc. DEC-012 requires one structural change per commit; this commit bundles a border-code change + a verdict doc. Flagged for post-hoc transparency. Not reverted — the fix is correct and the audit findings are still valid against the corrected port. Atomicity discipline reinforced for S32.

### Outcome

With DEC-037 applied, seeds 42 and 43 at depth=0 both select CPU's feature (f0), confirming border alignment is correct. The bin index divergence is entirely attributable to the gain formula — hence DEC-038.

---

## DEC-038: GreedyLogSumBestSplit operated on deduplicated values instead of all-docs

**Sprint**: 32 (identified T2-INSTRUMENT / T3-FIX 2026-04-24; formalized S32-T4-CLOSE 2026-04-24)
**Date**: 2026-04-24
**Commit**: `901bc760ac` (bundled with DEC-039 in T3-FIX commit — DEC-012 atomicity violation, see below)
**Status**: CLOSED — shipped.
**Authored by**: ml-engineer (T3-FIX); formalized by ml-engineer (T4-CLOSE)

### Problem

`QuantizeFeatures` in `catboost/mlx/tests/csv_train.cpp` was deduplicating the sorted
feature array before passing it to `GreedyLogSumBestSplit`. CatBoost CPU's
`TGreedyBinarizer<MaxSumLog>::BestSplit` initializes its `TFeatureBin` over
`features.Values` — the **full document array with duplicates** (N=50000 entries for
N=50000 docs). The penalty score function uses `BinEnd - BinStart` as the document count
in each bin. With the deduplicated input (49983 unique values for feature 0 vs 50000
total), the score landscape changed, causing a ~2-index border grid offset.

This was confirmed by a direct binary border dump (`CATBOOST_MLX_DUMP_BORDERS` build):
0 diffs at 1e-6 threshold across all 20 features at 128 borders after the fix.

### Decision

Pass `allVals` (sorted, with duplicates) to `GreedyLogSumBestSplit`, matching CPU's
`TFeatureBin` which is built over the full document array.

**File**: `catboost/mlx/tests/csv_train.cpp` — `QuantizeFeatures` function.

### Outcome

With DEC-038 applied: median gain ratio at depth=0 moves from 0.946 to 0.9999
(measured at seed=42, 127 bins). The residual 0.01% gap is from float32 vs float64
midpoint arithmetic in the border computation (see DEC-039 residual note).

### DEC-012 atomicity violation (flagged)

DEC-038 was shipped in the same commit (`901bc760ac`) as DEC-039. DEC-012 requires
one structural change per commit. Both bugs were discovered and fixed in the same
T3-FIX session. Flagged for post-hoc transparency — not reverted. S33 will enforce:
"if you find a second structural issue while fixing the first, STOP and commit the
first atomically before continuing."

### Authority

- T2-INSTRUMENT verdict: `docs/sprint32/t2-terms/verdict.md` (root cause = border grid divergence)
- T3-FIX verdict: `docs/sprint32/t3-fix/verdict.md` (fix description + verification)
- T4-VALIDATE G3a: `docs/sprint32/t4-validate/data/g3a_gain_ratio.csv` (3-seed ratio ≈ 1.000000)

---

## DEC-039: Histogram kernel VALID_BIT aliasing at fold_count=128 (T2_BIN_CAP violation)

**Sprint**: 32 (identified T3-FIX 2026-04-24; formalized S32-T4-CLOSE 2026-04-24)
**Date**: 2026-04-24
**Commit**: `901bc760ac` (bundled with DEC-038 in T3-FIX commit — DEC-012 atomicity violation, see DEC-038)
**Status**: CLOSED — shipped.
**Authored by**: ml-engineer (T3-FIX); formalized by ml-engineer (T4-CLOSE)

### Problem

The MLX histogram kernel uses `VALID_BIT = 0x80000000` (bit 31 of the packed 32-bit word)
to mark valid documents. Features at `posInWord=0` occupy bits 31..24 (shift=24). When
`fold_count=128` (i.e., 128 borders), `bin_value=128` (docs above all borders) sets bit 31
of the packed word at shift=24, which is the same bit as `VALID_BIT`.

The kernel strips bit 31 via `p_clean = p_s & 0x7FFFFFFF`, aliasing `bin_value=128` to
`bin_value=0`. The writeback loop skips slot 0 (reads `stagingHist[f*256 + bin + 1]`), so
these 391 documents were silently dropped from the histogram.

**Impact**: `wL = totalWeight - suffHess[b]` inflated by +391 for all bins of features
0, 4, 8, 12, 16 (the 5 `posInWord=0` features). This caused further split mismatch on
top of the DEC-038 border grid offset.

**Latent bug**: `kernel_sources.h:38` already documented this constraint as `T2_BIN_CAP`:
```
// Safe ONLY when every feature's fold count <= 127.
```
`csv_train.cpp` was violating this documented contract when `--bins 128` was passed,
producing `fold_count=128`. `bench_boosting` already respected the cap via `NumBins-1`.

### Decision

Cap `maxBordersCount = std::min(maxBins, 127u)` in `QuantizeFeatures`. With
`fold_count <= 127`, `bin_value <= 127` and bit 7 of the `posInWord=0` byte is never
set (bit 7 at shift=24 = bit 31 of the word = VALID_BIT), eliminating the collision.

**File**: `catboost/mlx/tests/csv_train.cpp` — `QuantizeFeatures` function.

### Outcome

With DEC-039 applied: wL delta for `posInWord=0` features drops from +391 to 25.
The residual ~25-doc delta is from float32 vs float64 midpoint arithmetic in
`GreedyLogSumBestSplit` producing 1-5 ULP border differences, reassigning ~25 docs at
physical split boundaries. This is a known limitation (not a structural bug).

The T2_BIN_CAP contract is now respected by `csv_train.cpp` as well as `bench_boosting`.

### G3c verification

`bench_boosting` was already computing `fold_count = NumBins - 1` (i.e., 127 for 128-bin
config) before DEC-039. The kernel sources are byte-identical to v5 (`784f82a891`).
`./bench_boosting_t4 --rows 10000 --features 50 --classes 1 --depth 6 --iters 50
--bins 128 --seed 42` → `BENCH_FINAL_LOSS=0.48231599` (ULP=0 vs AN-009 anchor).

### Authority

- T3-FIX verdict: `docs/sprint32/t3-fix/verdict.md` (bug description + fix verification)
- kernel_sources.h T2_BIN_CAP comment (line 38): pre-existing contract
- T4-VALIDATE G3c: bench_boosting ULP=0 confirmed; kernel sources md5=9edaef45b99b9db3e2717da93800e76f

---

### Note: DEC-038 original investigation scope (for archive)

The original DEC-038 entry described the S32 investigation scope (T1-T4 tasks, gates
G3a-G3d, kill-switches K6-K8). The investigation ran as planned. T1 confirmed SAME-PATH
(H1 eliminated). T2 identified gL divergence → border grid root cause. T3 fixed both
DEC-038 (allVals) and DEC-039 (fold_count cap) in a single commit. T4 validated
G3a PASS / G3b FAIL (52.6% drift, DEC-036 structural) / G3c PASS / G3d PASS. The
original kill-switches K6-K8 did not fire. DEC-036 remains OPEN for S33.

---

## DEC-040: S33 scope — L0-L4 SCAFFOLD for iter≥2 runaway divergence

**Sprint**: 33 (kickoff 2026-04-24)
**Date**: 2026-04-24
**Branch**: `mlx/sprint-33-iter2-scaffold` (cut from S32 tip `9fcc9827d9`)
**Status**: OPEN — investigation phase; implements DEC-036 closure.
**Authored by**: ml-product-owner (ultrathink kickoff)

### Problem

After S30 (precision class) and S32 (DEC-038 allVals + DEC-039 fold_count cap):
- **iter=1 residual**: 0.75% loss-relative drift (depth=0 gain ratio 0.9999, 200× tighter than spec).
- **iter=50 drift**: 52.6% — **unchanged** by every iter=1 fix.
- **Implied per-iter compounding**: 0.0075 × (1+r)^49 ≈ 0.526 → r ≈ 9% per-iter divergence growth.
  This is **runaway**, not gentle compounding. A 12× super-amplification factor between iter=1
  and iter=50 cannot be explained by float32 noise propagation alone.

DEC-036 is **reframed**: the structural divergence is no longer at iter=1 split selection
(closed by DEC-038/039). It is in the iter≥2 trajectory itself — leaf value computation,
approx update, gradient recomputation, fold-permutation/RNG, or trajectory chaos.

### Three-frame hypothesis ranking (priors)

| Frame | Mechanism | Prior | Falsifier |
|-------|-----------|-------|-----------|
| A | Trajectory lock-in cascade — chaotic GBDT search; tiny iter=1 ε amplifies via greedy argmax flips | 25% | L2 GRAFT: forcing iter=1 tree identical → iter=50 drift drops dramatically |
| B | Per-iter persistent mechanism — leaf value, approx update, gradient recomputation has its own bug | 30% | L2 GRAFT: drift unchanged after grafting iter=1 |
| C | Config/RNG mismatch — bootstrap_type, bagging_temperature, sampling_unit, leaf_estimation_method, langevin, fold_permutation_block, etc. | 30% | L0/L1: field-by-field config diff + deterministic config remeasure |

Remaining 15% = unknown / interaction effects.

### Decision: L0-L4 layered SCAFFOLD (cost-ordered falsification)

Falsify in cost order. Cheapest first. Stop at first frame closed.

| Layer | Task | Owner | Effort | Falsifies | Gate |
|-------|------|-------|--------|-----------|------|
| **L0** | CONFIG AUDIT — dump CPU + MLX effective config; field-by-field diff | @ml-engineer | ~45 min | Frame C-config | L0-PASS: no HARD-DIFF, OR drift unchanged after re-config |
| **L1** | DETERMINISM SHIFT — disable Bayesian bootstrap, set has_time, fix RNG seeds | @ml-engineer | ~2 hours | Frame C-RNG | L1-PASS: drift unchanged under deterministic config |
| **L2** | GRAFT EXPERIMENT — inject CPU iter=1 tree into MLX, run 49 more MLX iterations | @ml-engineer | ~3 hours | Frame A vs B | L2-DECIDE: drift drops ≥80% → Frame A; drift unchanged → Frame B |
| **L3** | ITER=2 INSTRUMENTATION (conditional, only if L2 → Frame B) — per-leaf, per-doc dump at iter=2 | @ml-engineer | ~1-2 days | Frame B sub-mechanism | L3-PASS: identify exact term causing per-iter drift |
| **L4** | FIX + FORMAL GATES — implement fix; ship | @ml-engineer + @qa-engineer | ~1-3 days | (closes DEC-036) | G4a iter=1 ≤0.1%, G4b iter=50 ≤2%, G4c v5 ULP=0, G4d 18-config L2 [0.98, 1.02], G4e DW sanity |

### Kill-switches (carried from DEC-038)

- **K6** — L1 deterministic config closes drift to ≤2% → S33 Frame C, no kernel changes needed.
- **K7** — L2 GRAFT closes drift via Frame A → revisit at iter=1 ratio target (1.000 ± 1e-6 instead of 1e-4).
- **K8** — L3 instrumentation surfaces a single per-iter term — fix it; if multi-term, escalate to architect.

### Hard rule (S33-only — DEC-012 reinforcement)

**If you find a second structural change while fixing the first, STOP and commit the first
atomically before continuing.** S31 (`746d5090b5`) and S32 (`901bc760ac`, `1aaf92497b`) each
violated DEC-012 atomicity — three sprints in a row. Self-flagging in sprint-close is not
sufficient deterrent. S33 enforces hard stop.

### Anchor

- **Anchor config**: N=50k, ST grow_policy, Cosine score, RMSE loss, depth=6, bins=128,
  iter ∈ {1, 50}, seeds 42/43/44, rs=0.
- **iter=1 floor**: depth=0 gain ratio 0.9999 (G3a-PASS at S32 close).
- **iter=50 ceiling**: 52.6% drift (G3b-FAIL at S32 close — target).
- **Production kernel**: v5 (`784f82a891`) — must remain ULP=0 across all S33 commits.

### Authority

- S32 sprint-close: `docs/sprint32/sprint-close.md`
- S33 ultrathink reasoning: this entry; subsequent verdict docs in `docs/sprint33/{l0,l1,l2,l3,l4}/`
