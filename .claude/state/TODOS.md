# Task Tracker

<!-- @product-owner creates tasks. All agents update status. -->
<!-- IDs: TODO-NNN sequential. Status: Backlog → In Progress → In Review → Done / Blocked -->
<!-- Move completed tasks to Done section — don't delete. -->

## Active

### TODO-007 — GPU partition layout (MLOps #1)
- **Assigned to:** unassigned
- **Priority:** High
- **Status:** Backlog
- **Depends on:** none
- **Acceptance Criteria:**
  - `ComputePartitionLayout` runs on GPU via a Metal segmented-sort kernel
  - CPU-GPU sync count in `SearchTreeStructure` drops by at least one `EvalNow` call per depth level
  - Existing test suite (604 passing) remains green
  - Benchmark: multiclass K=3 micro-benchmark shows measurable regression against pre-Sprint-4 baseline

### TODO-008 — Parallel SIMD scan for suffix_sum_histogram (MLOps #3)
- **Assigned to:** unassigned
- **Priority:** High
- **Status:** Backlog
- **Depends on:** none
- **Acceptance Criteria:**
  - `suffix_sum_histogram` replaced with parallel Metal scan kernel (work-efficient or Blelloch)
  - No correctness regression versus serial reference on both binary and multiclass targets
  - Measured speedup >= 2x on a dataset with >= 256 bins

### TODO-009 — Dead code removal: CPU FindBestSplit paths
- **Assigned to:** unassigned
- **Priority:** Low
- **Status:** Backlog
- **Depends on:** none
- **Acceptance Criteria:**
  - `FindBestSplit` and `FindBestSplitMultiDim` removed from `score_calcer.cpp`
  - Build succeeds with no references remaining
  - No test regression

### TODO-010 — MLflow integration via ITrainingCallbacks
- **Assigned to:** unassigned
- **Priority:** Medium
- **Status:** Backlog
- **Depends on:** none
- **Acceptance Criteria:**
  - Python `CatBoostMLXRegressor`/`CatBoostMLXClassifier` accept an `mlflow_run_name` parameter (or similar)
  - Loss and iteration metrics logged to MLflow on each iteration via `ITrainingCallbacks`
  - No C++ changes required
  - Documented in Python docstring with a usage example

### TODO-011 — Additional loss functions: Poisson, Tweedie, MAPE
- **Assigned to:** unassigned
- **Priority:** Medium
- **Status:** Backlog
- **Depends on:** none
- **Acceptance Criteria:**
  - `pointwise_target.h` extended with Poisson, Tweedie (power param), MAPE targets
  - Wired into `train_lib/train.cpp` dispatch (same pattern as MAE/Quantile/Huber)
  - Python validator updated to accept new loss names and their named params
  - QA validation tests added (follow `test_qa_round8_sprint3_losses.py` pattern)

### TODO-012 — Grow policies: Lossguide and Depthwise
- **Assigned to:** unassigned
- **Priority:** Low
- **Status:** Backlog
- **Depends on:** TODO-007 (GPU partition layout helps here too)
- **Acceptance Criteria:**
  - `SearchTreeStructure` supports `GrowPolicy::Lossguide` and `GrowPolicy::Depthwise`
  - Policy selectable via Python interface
  - Correctness verified against CatBoost CPU reference on a held-out dataset

## Blocked

## Done (Sprint 1–3)

### TODO-001 — Metal histogram kernel (Sprint 1)
- **Status:** Done
- **Notes:** Core histogram kernel implemented and dispatched.

### TODO-002 — Score splits kernel (Sprint 1)
- **Status:** Done

### TODO-003 — Leaf estimation kernel (Sprint 1)
- **Status:** Done

### TODO-004 — Python bindings + CatBoostMLXRegressor/Classifier (Sprint 2)
- **Status:** Done

### TODO-005 — Wire MAE, Quantile, Huber losses into dispatch (Sprint 3)
- **Status:** Done
- **Commits:** `38f963c`, `e5b4204`
- **Notes:** BUG-001 (case sensitivity) and BUG-002 (named-param syntax) fixed in `e5b4204`.

### TODO-006 — Sprint 3 performance optimizations OPT-1 and OPT-2 (Sprint 3)
- **Status:** Done
- **Commits:** `b9314b2` (OPT-1: fuse leaf sum dispatch), `54038f2` (OPT-2: bin-to-feature lookup precompute)
- **Notes:** ~8% speedup on multiclass K=3 micro-benchmark. `928c7ff` added MAX_LEAVES=64 runtime guard.
