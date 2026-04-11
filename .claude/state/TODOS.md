# Task Tracker

<!-- @product-owner creates tasks. All agents update status. -->
<!-- IDs: TODO-NNN sequential. Status: Backlog → In Progress → In Review → Done / Blocked -->
<!-- Move completed tasks to Done section — don't delete. -->

## Active

### TODO-007 — GPU partition layout (MLOps #1)
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-4-gpu-partition-layout`
- **Commit:** `19d24ec`
- **Depends on:** none
- **Acceptance Criteria:**
  - [x] `ComputePartitionLayout` runs on GPU via MLX argsort + scatter_add + cumsum primitives
  - [x] CPU-GPU sync count in `SearchTreeStructure` drops by 2 EvalNow calls per depth level (3→1)
  - [x] Existing test suite (604 passing) remains green
  - [x] csv_train regression: final loss matches pre-sprint baseline (0.481507 = 0.481507, exact match)
  - [x] Multiclass K=3 Python benchmark unaffected (Python path uses csv_train binary, not structure_searcher.cpp)
- **Notes:** Python bindings invoke csv_train binary via subprocess — they do NOT exercise structure_searcher.cpp. The optimization applies to the C++ library API (mlx_boosting.cpp → SearchTreeStructure). csv_train.cpp already had its own GPU partition layout before this sprint.

### TODO-008 — Parallel SIMD scan for suffix_sum_histogram (MLOps #3)
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
- **Commits:** `f8be378` (initial), `acecd9cbbf` (BUG-001 fix)
- **Acceptance Criteria:**
  - [x] `suffix_sum_histogram` replaced with SIMD-group parallel scan (Option A: simd_prefix_inclusive_sum)
  - [x] No correctness regression: BENCH_FINAL_LOSS identical before/after at bins=32 and bins=255
  - [x] pytest 684/684 passing (post BUG-001 fix; was 622 before QA round 10 tests added)
  - [x] Threadgroup updated (1,1,1)→(32,1,1)→(256,1,1) in score_calcer.cpp
  - Note: Overall iter wall time is similar because suffix-sum is not the bottleneck at 50-feature scale. Cold-start compile time dropped 344→109ms.
- **BUG-001 (FIXED `acecd9cbbf` 2026-04-10):** `suffixTG=(32,1,1)` left `scanBuf[32..255]` uninitialized — threadgroup memory is not zeroed between dispatches on Apple Silicon. Changed to `(256,1,1)` in both `FindBestSplitGPU` overloads; added `init_value=0.0f`. 10-run determinism at 10k rows confirmed (bins 32,33,34,48,65,96,255 all bit-for-bit identical). 100k reference losses unchanged.

### TODO-009 — Dead code removal: CPU FindBestSplit paths
- **Assigned to:** ml-engineer
- **Priority:** Low
- **Status:** Done
- **Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
- **Commits:** `1232f98`
- **Acceptance Criteria:**
  - [x] `FindBestSplit` and `FindBestSplitMultiDim` removed from `score_calcer.cpp` and `score_calcer.h`
  - [x] grep shows only FindBestSplitGPU and csv_train's internal reimplementation
  - [x] Build clean, pytest 622/622

### TODO-016 — Add bench_boosting compile step to CI workflow
- **Assigned to:** mlops-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-6-ci-and-infra`
- **Commits:** `1a7b9b7`
- **Acceptance Criteria:**
  - [x] `mlx_test.yaml` compiles bench_boosting and build_verify_test alongside csv_train/csv_predict
  - [x] bench_boosting copied to `/tmp/bench_boosting` so test_qa_round10 tests execute (not silently skip)
  - [x] Verify binaries step checks all 4 executables
  - [x] ruff check passes, pytest 684/684

### TODO-017 — Add --onehot flag to bench_boosting
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-6-ci-and-infra`
- **Commits:** `abdd659`
- **Acceptance Criteria:**
  - [x] `--onehot N` CLI flag marks first N features as one-hot with small bin counts (2-10)
  - [x] `--onehot 0` preserves existing behavior (default)
  - [x] OneHot features exercise the one-hot skip/equality branches in kSuffixSumSource and tree applier
  - [x] Verified: `--onehot 5` produces finite BENCH_FINAL_LOSS, deterministic across 3 runs

### TODO-018 — Port tree_applier to Metal kernel dispatch
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-6-ci-and-infra`
- **Commits:** `caf4552`
- **Acceptance Criteria:**
  - [x] `kTreeApplySource` Metal kernel: one thread per doc, computes leaf index, updates predictions
  - [x] Handles binary/regression (approxDim=1) and multiclass (approxDim>1)
  - [x] OneHot (equality) vs ordinal (greater-than) split comparison preserved
  - [x] BENCH_FINAL_LOSS matches pre-change reference: 0.69314516 (binary 100k), 1.09757149 (multiclass 20k)
  - [x] Deterministic across 3 runs
  - [x] pytest 684/684 passing

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

### TODO-013 — Fix kernel param names in build_verify_test.cpp
- **Assigned to:** ml-engineer
- **Priority:** Low
- **Status:** Done
- **Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
- **Commits:** `3ca05c3`
- **Acceptance Criteria:**
  - [x] `featureColumnIndices` (1-element array), `foldCountsFlat`, `firstFoldIndicesFlat`, `numGroups` — all names match kHistOneByteSource
  - [x] build_verify_test: ALL TESTS PASSED

### TODO-014 — Add library-path C++ benchmark harness
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
- **Commits:** `3e764cc`
- **Acceptance Criteria:**
  - [x] `bench_boosting` standalone binary — exercises production Metal kernels, no subprocess
  - [x] CLI: --rows --features --classes --depth --iters --bins --lr --l2 --seed
  - [x] Binary 100k×50×depth6×100iters: completes, 38.5 ms/iter warm mean
  - [x] Multiclass K=3 20k×30×depth5×50iters: completes, 11.4 ms/iter warm mean
  - [x] Prints BENCH_FINAL_LOSS for regression testing

### TODO-015 — Document 16M-row float32 limit in DECISIONS.md
- **Assigned to:** ml-engineer
- **Priority:** Low
- **Status:** Done
- **Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
- **Commits:** `dbfcf07`
- **Acceptance Criteria:**
  - [x] DEC-003 added to DECISIONS.md with rationale, alternatives, and commit references

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
