# Handoff — CatBoost-MLX

> Last updated: 2026-04-16 by orchestrator (Sprint 16 stage profiler attribution complete)

## Current state

- **Branch**: `mlx/sprint-16-perf-diagnosis`
- **Last commit**: `dccb7ec0a2` (sync-storm fix validation + bench bug fix + baseline update)
- **Working tree**: stage profiler instrumentation in csv_train.cpp + structure_searcher + mlx_boosting + stage_profiler.h (uncommitted)
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24)

## What just happened

**Stage profiler attribution complete (10k × 50f, RMSE, depth 6, 128 bins, 100 iters):**

| stage | mean ms | % of iter |
|-------|--------:|----------:|
| histogram_ms | **310.99** | **97.7%** |
| suffix_scoring_ms | 2.29 | 0.7% |
| partition_layout_ms | 1.71 | 0.5% |
| derivatives_ms | 0.29 | 0.1% |
| leaf_sums_ms | 0.26 | 0.1% |
| loss_eval_ms | 0.26 | 0.1% |
| leaf_values_ms | 0.24 | 0.1% |
| tree_apply_ms | 0.24 | 0.1% |
| cpu_readback_ms | 0.13 | 0.0% |
| init_partitions_ms | 0.00 | 0.0% |
| **sum_of_stages** | **316.40** | **99.4%** |
| iter_total_ms | 318.34 | 100.0% |

**Key findings:**
- Histogram phase = **97.7% of iter time** — dominant bottleneck
- Per-depth growth: 45ms (d0) → 24/24/39/74/114ms (d1–d5) — sub-linear in partition count
- `cpu_readback_ms` hypothesis (the suspected 313ms gap) **falsified** at 0.13ms
- Root cause of attribution gap: `DispatchHistogram()` at csv_train.cpp:953 calls `mx::eval(histogram)` per call — the "Phase 1: build lazy graph (no GPU sync)" comment at csv_train.cpp:3178 is wrong. Each per-dim/per-depth call drains the GPU
- Stage 4 timer was extended to span Phase 1 + Phase 2 to capture the full histogram cost (it previously only wrapped Phase 2's eval and missed 309ms)

**Sync-storm fix (S16-07) previously validated:**
- All 18 `EvalNow` calls removed from `pointwise_target.h`
- 3 per-depth `EvalNow` removed from `structure_searcher.cpp`
- Numerical parity bit-exact across RMSE/Logloss/MultiClass × 1k/10k/50k
- Zero perf regression

**Baseline:** MLX 100–300x slower than CPU CatBoost; per-iter cost barely scales with N (~300ms at 1k, ~323ms at 10k, ~487ms at 50k for 50 features).

**Sprint 17 implication:** The originally-planned target was `maxBlocksPerPart=1` in histogram.cpp:105. That value is already gone — csv_train.cpp:891-894 now computes `maxBlocksPerPart = clamp(ceil(avgDocsPerPart/4096), 1, 8)`. The real fix must target the histogram kernel itself (algorithm, threadgroup layout, per-feature-group dispatch overhead).

## Active Sprint 16 work

| Agent | Task | Status |
|-------|------|--------|
| @mlops-engineer | S16-05/06: bench harness extension + CI perf gate | DONE |
| @ml-engineer | S16-07: sync-storm elimination | DONE — validated, committed |
| @qa-engineer | S16-08: numerical parity validation | DONE — bit-exact across all 9 combos |
| @performance-engineer | S16-01: stage profiler implementation | DONE — uncommitted |
| @ml-engineer | S16-01b: wire profiler into csv_train.cpp production path | DONE — uncommitted |
| @performance-engineer | S16-02: capture baseline profile (10k RMSE d6 128b) | DONE — `.cache/profiling/sprint16/baseline_10k_rmse_d6_128bins.json` |
| @performance-engineer | S16-04: sync inventory | DONE — `docs/sprint16/sync_inventory.md` |
| @devops-engineer | CMake `CATBOOST_MLX_STAGE_PROFILE` option | DONE — committed |
| @technical-writer | S16-11: docs/sprint16/ skeleton + ARCHITECTURE.md | DONE — committed |
| @research-scientist | S16-03: Metal System Trace overnight capture | Pending dispatch |

## Blockers

None.

## Next action

1. Commit stage profiler instrumentation + the CpuReadback enum + Phase 1 expansion
2. Extend baseline capture to {1k, 50k} × {RMSE, Logloss, MultiClass} × {32, 128 bins} (S16-02 acceptance)
3. Dispatch @research-scientist for Metal System Trace + histogram kernel teardown — Sprint 17 plan needs to target the kernel itself, NOT `maxBlocksPerPart`
4. Final Sprint 16 PR
