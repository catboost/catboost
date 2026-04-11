# Session Handoff

<!-- This file is OVERWRITTEN each session. It captures current state only. -->
<!-- The last active agent updates this at the end of every session. -->
<!-- The first agent in the next session reads this first. -->

**Last Updated:** 2026-04-10
**Last Active Agent:** qa-engineer (Sprint 6, TODO-007 sign-off)

## Completed This Session

### Sprint 5 — merged to master (`0d2e97f914`)

All 5 Sprint 5 TODOs (008/009/013/014/015) complete and merged.

| SHA | Description |
|-----|-------------|
| `3e764cc` | bench_boosting library-path benchmark harness (TODO-014) |
| `f8be378` | Parallel SIMD scan for suffix_sum_histogram (TODO-008) |
| `1232f98` | Delete dead CPU FindBestSplit paths (TODO-009) |
| `3ca05c3` | Fix build_verify_test kernel param names (TODO-013) |
| `dbfcf07` | Document 16M-row float32 limit in DECISIONS.md (TODO-015) |

### BUG-001 fixed (`acecd9cbbf`)

`suffixTG=(32,1,1)` left `scanBuf[32..255]` uninitialized — threadgroup memory is not zeroed between dispatches on Apple Silicon. Fixed by changing to `(256,1,1)` in both `FindBestSplitGPU` overloads and adding `init_value=0.0f`. 10-run determinism at 10k rows confirmed (bins 32,33,34,48,65,96,255 all bit-for-bit identical).

### Ruff lint cleanup (commits `32b5419` + `ee61752`)

36 lint errors → 0.

### TODO-007 sign-off (this session)

Ran `pytest python/tests/ -k "multiclass" -x -q`: 28/28 passed. Python path (csv_train subprocess) unaffected by structure_searcher.cpp changes. Last acceptance criterion checked; TODO-007 status set to Done.

## Current State

- **Test suite:** 684/684 passing (up from 622 pre-BUG-001-fix)
- **Branch:** `mlx/sprint-6-ci-and-infra`
- **Master:** Sprint 5 merged (`0d2e97f914`); all Sprint 1–5 TODOs Done

## In Progress

Sprint 6 on branch `mlx/sprint-6-ci-and-infra`. Focus areas:

1. **TODO-016** — CI blind spot fix: ensure pytest exercises the C++ library path (not only csv_train subprocess), so regressions in structure_searcher.cpp / mlx_boosting.cpp are caught automatically
2. **bench_boosting `--onehot` flag** — expose one-hot encoding in the benchmark harness CLI
3. **tree_applier Metal kernel** — GPU-side tree application to replace CPU loop in inference path

## Blocked

Nothing blocked.

## Next Steps (after Sprint 6)

1. **TODO-010** — MLflow integration via ITrainingCallbacks
2. **TODO-011** — Additional loss functions: Poisson, Tweedie, MAPE
3. **TODO-012** — Grow policies: Lossguide and Depthwise (depends on TODO-007, now Done)

## Notes

- **CRITICAL architecture clarification:** Python bindings call `csv_train` C++ binary via subprocess. Changes to `structure_searcher.cpp` / `mlx_boosting.cpp` / `kernel_sources.h` are NOT exercised by the Python test suite — they affect only the C++ library API (tested via bench_boosting and build_verify_test).
- Sprint branch rule (DEC-002): push to `origin` (RR-AMATOK) only; never to `upstream` (catboost/catboost).
- bench_boosting reference losses: BENCH_FINAL_LOSS=0.69314516 (binary 100k×50 bins=32), 0.69313669 (bins=255).
- BUG-001 root cause: Apple Silicon does NOT zero threadgroup memory between dispatches. Always initialize threadgroup buffers explicitly.
