# Session Handoff

<!-- This file is OVERWRITTEN each session. It captures current state only. -->
<!-- The last active agent updates this at the end of every session. -->
<!-- The first agent in the next session reads this first. -->

**Last Updated:** 2026-04-11
**Last Active Agent:** technical-writer (Sprint 10 user documentation)

## Completed This Session

### Sprint 10 — branch `mlx/sprint-10-lossguide-packaging-hardening`

#### Documentation updated this session

- `python/README.md` — Complete rewrite of the user-facing Python package README. Now accurate to Sprint 9 state: depth 1-10 (was incorrectly capped at 6), Depthwise grow policy documented, 16M row limit removed (int32 fix), `mlflow_logging`/`grow_policy`/`random_strength` parameters added to reference table, Known Limitations corrected. Informal tone and Big Bang Theory references removed. Lossguide listed as "backlog" per TODOS.md TODO-012.

---

### Sprint 9 — branch `mlx/sprint-9-pybind-depth-policies-infra`

Six features delivered across 7 commits (branch is 7 commits ahead of master, pending merge).

| Item | Description | TODO |
|------|-------------|------|
| E | 16M row fix — int32 scatter_add in ComputePartitionLayout (resolves DEC-003) | TODO-026 |
| H | CI bench regression check — two new steps in mlx_test.yaml | TODO-027 |
| F | MLflow integration — `mlflow_logging` + `mlflow_run_name` in Python fit() | TODO-028 |
| G | Histogram EvalNow deferral — removed 2 CPU-GPU syncs from histogram.cpp | TODO-029 |
| B | max_depth > 6 — chunked multi-pass leaf accumulation kernel (depth 7-10) | TODO-030 |
| D | Depthwise grow policy — SearchDepthwiseTreeStructure, ApplyDepthwiseTree Metal kernel, EGrowPolicy enum, CLI + Python params | TODO-031 |

#### Documentation updated this session

- `.claude/state/TODOS.md` — Sprint 9 items (TODO-026 through TODO-031) added Done; TODO-012 updated (Depthwise done, Lossguide backlog); TODO-010 updated Done
- `.claude/state/CHANGELOG-DEV.md` — Sprint 9 section prepended
- `.claude/state/DECISIONS.md` — DEC-003 commit reference corrected; DEC-004 (Depthwise policy) and DEC-005 (multi-pass leaf accumulation) added
- `.claude/state/HANDOFF.md` — this file
- `catboost/mlx/ARCHITECTURE.md` — new sections: "Depthwise Grow Policy" under Metal Kernels; "Multi-Pass Leaf Accumulation (depth > 6)" under Metal Kernels; CPU-GPU sync table updated

## Current State

- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Master:** Sprint 8 merged
- **Status:** Sprint 9 complete, pending QA sign-off and merge PR

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

### New Sprint 9 baselines

| Configuration | BENCH_FINAL_LOSS |
|---------------|-----------------|
| Binary 1k, 10 features, depth 8, 20 iters, bins 32, seed 42 | **0.54883069** |

## In Progress

- QA round for Sprint 9 (grow_policy, mlflow_logging, depth 8-10 correctness) — not yet committed

## Blocked

Nothing blocked.

## Next Steps

1. **QA sign-off** — Write and pass tests for Sprint 9 features (Depthwise policy, max_depth 7-10, MLflow)
2. **Merge Sprint 9 branch to master** — after QA and MLOps sign-off
3. **Item A (deferred)** — nanobind direct Python-C++ bindings (no subprocess), was deferred from Sprint 9
4. **TODO-012 remainder** — Lossguide grow policy (leaf-priority BFS expansion), still Backlog
5. **Item J** — (assigned next sprint, details TBD)

## Notes

- **Two code paths:** Python bindings call `csv_train` via subprocess. Changes to `methods/` and `targets/` files (library path) are NOT exercised by the Python test suite — only by `bench_boosting` and `build_verify_test`. Always verify both paths when touching kernel dispatch logic.
- **Depthwise path:** `ApplyDepthwiseTree` issues `EvalNow` after every tree application (same as `ApplyObliviousTree`). This is the correct sync point for partition correctness.
- **Loss parity:** Both `csv_train` and the library path support the same 10 losses. Future loss additions must be added to both paths separately.
- **Sprint branch rule (DEC-002):** Push to `origin` (RR-AMATOK) only; never to `upstream` (catboost/catboost).
- **Apple Silicon threadgroup memory:** NOT zeroed between dispatches. Always pass `init_value=0.0f` to `mx::fast::metal_kernel()` when the kernel reads from threadgroup storage that may not be fully written by every thread.
