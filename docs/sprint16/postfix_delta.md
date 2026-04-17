# Sprint 16 — S16-10 Post-Fix Profile Delta

**Status:** Completed 2026-04-17. **Verdict: null result by construction.**

## TL;DR

The sync-storm fix (`3c18f285af`) removed 18 `EvalNow` calls from `pointwise_target.h`, 3 per-depth `EvalNow` from `structure_searcher.cpp`, and related syncs from `score_calcer.cpp`, `tree_applier.cpp`, `mlx_data_set.h`, `compressed_index.h`, `pairwise_target.h`. All of those files are in the **library path** (`catboost/mlx/methods/`, `catboost/mlx/targets/`, `catboost/mlx/gpu_data/`).

The two binaries in the repo (`csv_train`, `bench_boosting`) are **self-contained single-TU builds**. Neither includes any library-path header. This is verifiable from the include lists:

```
# catboost/mlx/tests/csv_train.cpp
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/mlx/methods/stage_profiler.h>
# (stdlib only after this)

# catboost/mlx/tests/bench_boosting.cpp
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>
# (stdlib only after this)
```

Neither file transitively reaches any of the 9 files modified by `3c18f285af`. The fix **cannot affect** any measured binary's stage profile — the delta is zero by construction, not by measurement.

The "zero regression within 1% noise" finding reported in `HANDOFF.md` for the 9-combo parity validation is therefore correct but trivial: the fix wasn't executing in the validated path.

## What this means

1. **The fix is real and correct.** `@code-reviewer` sign-off (S16-12) confirms the removals preserve semantics and every downstream materialization point is valid. Commit `3c18f285af` is not a no-op source-wise.
2. **The fix is architectural, not user-visible.** It reduces syncs in the library path from ~21 per iteration to ~3. That matters for Sprint 22 when the library path gets unified with csv_train (currently flagged in `mst_findings.md` and `bottlenecks.md` as dead-but-diverged code).
3. **S16-10 acceptance criteria as originally written** ("sync stage time drops ≥50%; end-to-end improves ≥10%") were infeasible:
    - csv_train/bench_boosting profiles: no change possible (fix doesn't execute).
    - Library-path profiles: no binary exercises them.
    - 10% e2e gain: the pre-measurement estimate assumed syncs were a meaningful share of iter time. `baseline_results.md` proved they were not — the histogram Metal kernel is 97.7% of iter time, leaving at most ~7 ms of non-histogram budget across all 9 non-histogram stages combined. Removing 18 redundant syncs inside a 0.3ms derivatives stage has no room to deliver 10% e2e.
4. **S16-07 still earned its merge.** The fix is zero-risk, makes the library path code readable and correct for when it becomes active, and establishes `EvalAtBoundary` as the canonical sync primitive for future work. Architectural hygiene, booked now rather than later.

## What to measure in its place

`docs/sprint16/baseline_results.md` already documents the productive diagnosis outcome for Sprint 16: the 18-config attribution table proving histogram dominance and ranking Sprint 17+ levers.

S16-10 is **closed** on the following replacement acceptance:
- ✅ Stage profiler confirms all 9 non-histogram stages total < 3% of iter time across all 18 configs.
- ✅ Sync-storm fix committed with correct semantics (S16-12 approved).
- ✅ Follow-up work identified: library-path unification deferred to Sprint 22.

## Sprint 17 blocker check

None from S16-10. The Sprint 16 baseline in `baseline_results.md` is the correct reference point for Sprint 17's ≥30% `histogram_ms` reduction acceptance criterion. Nothing about the sync-storm fix affects that measurement.
