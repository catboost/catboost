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
- **Safe by construction at ≤128 bins.** `packed` holds four 8-bit values in bits 24–30 (feature 0), 16–22 (feature 1), 8–14 (feature 2), 0–6 (feature 3). Bit 31 is always zero on load — available as sentinel. `p_clean = p_s & 0x7FFFFFFFu` restores the bin byte field before extraction.
- **Parity bit-exact.** MSB-sentinel encoding does not alter reduction order for valid docs; invalid docs do not write (gated by sentinel check). Higham γ_7 unchanged.
- **Measured parity (3 configs, seed 42, depth 6, 128 bins):** 50k/RMSE 0.48047778; 10k/RMSE 0.48016092; 50k/MultiClass 0.94424933 — all bit-exact pre vs post edit.
- **Measured perf (50k/RMSE/d6/128b, 3-run warm mean):** pre-edit 32.47 ms, post-edit 31.73 ms — **−2.3% e2e.** Matches S19-01c probe projection (kernel ≈ 97.7% of iter time, so kernel-level shuffle reduction amplifies near 1:1 to e2e).
**Trade-off**: +6 LOC, +1 constant (`VALID_BIT`), +1 masking op (`p_clean`). No new TG memory, no new register state beyond the existing OR-in at load. DEC-011 ceiling preserved.
**Scope limit**: `>128 bin` configs (outside DEC-008 envelope) would need a different sentinel location. Current kernel is called exclusively under `bins ≤ 128` by the gate config and the DEC-008 envelope; out-of-envelope configs are unaffected by Sprint 19.
**Follow-up**: T3b atomic-CAS (DEC-017, Sprint 20) eliminates the shuffle chain entirely; T1 is the bit-exact interim.

## DEC-017: T3b threadgroup-atomic-CAS no-shuffle accumulator (DRAFT — Sprint 20)

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
