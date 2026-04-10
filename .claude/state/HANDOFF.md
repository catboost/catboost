# Session Handoff

<!-- This file is OVERWRITTEN each session. It captures current state only. -->
<!-- The last active agent updates this at the end of every session. -->
<!-- The first agent in the next session reads this first. -->

**Last Updated:** 2026-04-09
**Last Active Agent:** ml-engineer (Sprint 5, TODO-008/009/013/014/015)

## Completed This Session

Sprint 5 (TODO-008/009/013/014/015). All 5 items complete on branch `mlx/sprint-5-parallel-scan-benchmark-harness`.

| SHA | Description |
|-----|-------------|
| `3e764cc` | bench_boosting library-path benchmark harness (TODO-014) |
| `f8be378` | Parallel SIMD scan for suffix_sum_histogram (TODO-008) |
| `1232f98` | Delete dead CPU FindBestSplit paths (TODO-009) |
| `3ca05c3` | Fix build_verify_test kernel param names (TODO-013) |
| `dbfcf07` | Document 16M-row float32 limit in DECISIONS.md (TODO-015) |

Test suite: 622/622. bench_boosting final loss identical before/after parallel scan change.

## In Progress

Sprint 5 branch awaiting QA + MLOps sign-off before merge to master.

## Blocked

Nothing blocked.

## Next Steps

1. **QA/MLOps sign-off on Sprint 5 branch** — Key things to verify:
   - bench_boosting: BENCH_FINAL_LOSS=0.69314516 (binary 100k×50 bins=32) and 0.69313669 (bins=255) stable
   - The parallel scan uses SIMD hardware tree-reduction so addition order differs from serial; mathematically equivalent but not bit-for-bit identical at individual bin level. Test by checking final loss rather than individual histogram bins.
2. **TODO-010** — MLflow integration via ITrainingCallbacks
3. **TODO-011** — Additional loss functions: Poisson, Tweedie, MAPE
4. **TODO-012** — Grow policies: Lossguide and Depthwise (depends on TODO-007 merged)

## Notes

- **CRITICAL architecture clarification:** Python bindings call `csv_train` C++ binary via subprocess. Changes to `structure_searcher.cpp` / `mlx_boosting.cpp` / `kernel_sources.h` are NOT exercised by the Python benchmark — they affect only the C++ library API (tested via bench_boosting).
- `build_verify_test.cpp` now correctly matches all production kernel input names. Previous mismatch caused numGroups guard to receive garbage from wrong buffer slot.
- Sprint branch rule (DEC-002): push to `origin` (RR-AMATOK) only; never to `upstream` (catboost/catboost).
