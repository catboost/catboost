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

## DEC-012: Intra-SIMD butterfly removed under L1a layout

**Sprint**: 18
**Date**: 2026-04-17
**Commit**: `19fa5ce6cc`
**Problem**: D1c's `simd_shuffle_xor` butterfly (5 rounds, xor 16/8/4/2/1) was designed to reduce 32 distinct per-lane `privHist` partials into a single SIMD-group sum. Under L1a's `simdHist[g][bin]` layout, all 32 lanes in group `g` read the same shared slot — there are no per-lane partials to reduce. Applying the butterfly over a shared slot multiplies the accumulated value by 32 (BUG-S18-001).
**Chosen**: Remove the intra-SIMD butterfly entirely. Under stride-partition ownership, `simdHist[g][bin]` already holds the full per-SIMD-group sum after the accumulation loop. Only the cross-SIMD 8-term linear fold (DEC-009) runs downstream.
**Rationale**: The butterfly is algebraically redundant and structurally incorrect under the new layout. Its removal tightens the effective reduction depth from γ_12 (5 butterfly levels + 7 linear cross-SIMD levels) to γ_7 (7 linear cross-SIMD levels only). Higham bound improves from ~7.2e-7 to ~4.2e-7. Barriers per dispatch drop from 9 to 6 as a side-effect.
**Future trigger**: Any future kernel that accumulates into per-lane register state (not shared threadgroup memory) should re-introduce the intra-SIMD butterfly for that phase. The xor butterfly remains the correct pattern for per-lane-held partials; its removal here is specific to the per-SIMD-group shared layout.
**Status**: Active. Shipped Sprint 18.
