# Session Handoff

<!-- This file is OVERWRITTEN each session. It captures current state only. -->
<!-- The last active agent updates this at the end of every session. -->
<!-- The first agent in the next session reads this first. -->

**Last Updated:** 2026-04-11
**Last Active Agent:** technical-writer (Sprint 10 state files and architecture docs)

## Completed This Session

### Sprint 10 — branch `mlx/sprint-10-lossguide-packaging-hardening`

Five commits delivered:

| Commit | Description | TODO |
|--------|-------------|------|
| `eececbf4d2` | Lossguide grow policy — priority queue, sparse node map, `max_leaves` param | TODO-012 |
| `7d03e4fe6e` | Model format versioning — `format_version=2` in save/load | TODO-032 |
| `ef215e5fe8` | Benchmark script — `python/benchmarks/benchmark_vs_catboost.py` | TODO-033 |
| `7e8fe31075` | PyPI packaging v0.3.0 — `--cov-fail-under=70`, Depthwise CI regression step | TODO-034 |
| `959e20c66e` | `python/README.md` — complete user-facing documentation | TODO-035 |

#### Documentation updated this session

- `.claude/state/TODOS.md` — TODO-012 marked Done (Lossguide complete); TODO-032 through TODO-035 added Done
- `.claude/state/CHANGELOG-DEV.md` — Sprint 10 section prepended
- `.claude/state/DECISIONS.md` — DEC-006 (Lossguide design) added
- `.claude/state/HANDOFF.md` — this file
- `catboost/mlx/ARCHITECTURE.md` — Lossguide section added; model versioning note added; table of contents updated

---

### Sprint 9 — branch `mlx/sprint-9-pybind-depth-policies-infra`

Six features across 7 commits. Branch was complete and pending merge at end of last session.

| Item | Description | TODO |
|------|-------------|------|
| E | 16M row fix — int32 scatter_add in ComputePartitionLayout (resolves DEC-003) | TODO-026 |
| H | CI bench regression check — two new steps in mlx_test.yaml | TODO-027 |
| F | MLflow integration — `mlflow_logging` + `mlflow_run_name` in Python fit() | TODO-028 |
| G | Histogram EvalNow deferral — removed 2 CPU-GPU syncs from histogram.cpp | TODO-029 |
| B | max_depth > 6 — chunked multi-pass leaf accumulation kernel (depth 7-10) | TODO-030 |
| D | Depthwise grow policy — SearchDepthwiseTreeStructure, ApplyDepthwiseTree, EGrowPolicy enum | TODO-031 |

## Current State

- **Branch:** `mlx/sprint-10-lossguide-packaging-hardening`
- **Master:** Sprint 8 merged (Sprint 9 and Sprint 10 pending merge)
- **Status:** Sprint 10 complete; all 3 grow policies implemented and shipped in v0.3.0 package

## Reference Baselines

### bench_boosting CI baselines (Sprint 9, from mlx_test.yaml)

| Configuration | BENCH_FINAL_LOSS | Tolerance |
|---------------|-----------------|-----------|
| Binary 1k, 10 features, depth 4, 20 iters, bins 32, seed 42 | **0.59795737** | 1e-4 |
| Multiclass K=3, 1k, 10 features, depth 4, 20 iters, bins 32, seed 42 | **0.95248461** | 1e-4 |

### bench_boosting performance baselines (Sprint 8, library path)

| Configuration | Warm mean | BENCH_FINAL_LOSS |
|---------------|-----------|-----------------|
| Binary 100k, 50 features, depth 6, 100 iters | 180.9 ms | **0.11909308** |
| Multiclass K=3, 20k, 30 features, depth 5, 50 iters | 101.3 ms | **0.63507235** |
| Multiclass K=10, 20k, 30 features, depth 5, 50 iters | 115.4 ms | **1.78561831** |

### Sprint 9 depth-8 baseline

| Configuration | BENCH_FINAL_LOSS |
|---------------|-----------------|
| Binary 1k, 10 features, depth 8, 20 iters, bins 32, seed 42 | **0.54883069** |

## In Progress

Nothing in flight — Sprint 10 complete.

## Blocked

Nothing blocked.

## Next Steps

1. **Merge Sprint 9 branch to master** — QA sign-off is the remaining gate (grow_policy, mlflow_logging, depth 8-10 tests)
2. **Merge Sprint 10 branch to master** — after Sprint 9 is merged
3. **Item A (deferred from Sprint 9)** — nanobind direct Python-C++ bindings (no subprocess)
4. **Inference performance** — `ComputeLeafIndicesLossguide` is a CPU-side BFS traversal; a GPU kernel can replace it if inference latency becomes a bottleneck (not urgent at Sprint 10 scale)
5. **Sprint 11 scope** — TBD (nanobind bindings is the leading candidate)

## Notes

- **Three grow policies complete as of Sprint 10:** SymmetricTree (default, Sprint 1), Depthwise (Sprint 9), Lossguide (Sprint 10).
- **Package version:** v0.3.0 ships all three. PyPI-ready structure in `python/pyproject.toml`.
- **Model format:** JSON files saved by Sprint 10+ carry `"format_version": 2`. Pre-Sprint-10 files (format_version absent or 1) load without error. Files from future versions (format_version > 2) raise ValueError with an upgrade hint.
- **Two code paths:** Python bindings call `csv_train` via subprocess. Library path (`bench_boosting`, `mlx_boosting.cpp`) is separate. Changes to `methods/` do NOT automatically apply to `csv_train.cpp`. Always verify both paths when touching kernel dispatch logic.
- **Apple Silicon threadgroup memory:** NOT zeroed between dispatches. Always pass `init_value=0.0f` to `mx::fast::metal_kernel()`.
- **Sprint branch rule (DEC-002):** Push to `origin` (RR-AMATOK) only; never to `upstream` (catboost/catboost).
