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
