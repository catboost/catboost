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

### TODO-019 — Fuse multiclass leaf computation, eliminate K EvalNow calls
- **Assigned to:** ml-engineer
- **Priority:** Critical
- **Status:** Done
- **Branch:** `mlx/sprint-7-multiclass-fuse-partition-output`
- **Commits:** `2908a84`
- **Acceptance Criteria:**
  - [x] Single `ComputeLeafValues` call over full `[approxDim * numLeaves]` arrays (no per-k loop)
  - [x] `EvalNow` removed from `ComputeLeafValues` — returns lazy MLX array
  - [x] Multiclass K=3 BENCH_FINAL_LOSS: 0.63507235 (new baseline after BUG-002 fix)
  - [x] K=10 multiclass: 1.78561831 (finite, deterministic; params: 20k×30×cls10×d5×50i×bins32×seed42)
  - [x] Binary BENCH_FINAL_LOSS: 0.11909308 (new baseline after BUG-002 fix)
  - [x] pytest 684/684 passing

### TODO-020 — Output partition indices from tree_applier Metal kernel
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-7-multiclass-fuse-partition-output`
- **Commits:** `5ef25eb`
- **Acceptance Criteria:**
  - [x] `kTreeApplySource` has two outputs: `cursorOut` (float32) and `partitionsOut` (uint32)
  - [x] O(depth) MLX op loop in `tree_applier.cpp` deleted (−28 lines)
  - [x] BENCH_FINAL_LOSS unchanged for binary and multiclass configs
  - [x] Deterministic: 3-run match at 10k rows
  - [x] pytest 684/684 passing

### TODO-021 — BUG-002 fix: bench_boosting threshold comparison
- **Assigned to:** ml-engineer
- **Priority:** Low
- **Status:** Done
- **Branch:** `mlx/sprint-7-multiclass-fuse-partition-output`
- **Commits:** `6969280`
- **Acceptance Criteria:**
  - [x] `> binThreshold + 1` changed to `> binThreshold` in bench_boosting.cpp
  - [x] New reference baselines recorded: 0.11909308 (binary 100k), 0.63507235 (multiclass 20k K=3)
  - [x] pytest 684/684 passing

### TODO-010 — MLflow integration
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Acceptance Criteria:**
  - [x] `mlflow_logging` bool param and `mlflow_run_name` str param added to `fit()` (Python only — no C++ changes)
  - [x] Lazy `import mlflow` inside `_log_to_mlflow()` — remains optional dependency
  - [x] Respects active runs: logs into existing run if open; otherwise starts and ends its own
  - [x] Hyperparameters, per-iteration loss, and final metrics logged
  - [x] Documented in Python docstring with usage example

### TODO-011 — Additional loss functions: Poisson, Tweedie, MAPE (library path)
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-8-housekeeping-library-losses`
- **Commits:** `a8bdf798e9` (impl), `67eb9b42ce` (QA tests), `3d350c7166` (BUG-004 fix)
- **Depends on:** none
- **Acceptance Criteria:**
  - [x] `pointwise_target.h` extended with `TPoissonTarget`, `TTweedieTarget` (power param), `TMAPETarget`
  - [x] Wired into `train_lib/train.cpp` dispatch (same pattern as MAE/Quantile/Huber)
  - [x] Python validator already accepts these losses — BUG-004 fixed (variance_power= prefix strip)
  - [x] `csv_train.cpp` already has full implementations — library path formulas matched
  - [x] QA: 39 tests in `test_qa_round11_sprint8_library_losses.py` (723 total pass)
- **BUG-004 (FIXED `3d350c7166`):** `Tweedie:variance_power=1.5` raised ValueError because `_validate_params` and `_normalize_loss_str` only stripped `alpha=` and `delta=` prefixes. Fixed by adding `variance_power=` to both prefix lists.

### TODO-022 — Gitignore compiled binaries and fix ruff I001
- **Assigned to:** mlops-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-8-housekeeping-library-losses`
- **Acceptance Criteria:**
  - [x] `.gitignore` updated: bench_boosting*, csv_train_phase_c* patterns added
  - [x] Pre-existing ruff I001 in `python/benchmarks/benchmark.py` fixed
  - [x] `python -m ruff check python/` passes clean

### TODO-023 — Resolve K=10 baseline discrepancy
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-8-housekeeping-library-losses`
- **Acceptance Criteria:**
  - [x] Re-run bench_boosting with exact documented params (20k×30×cls10×d5×50i×bins32×seed42)
  - [x] Measured: 1.78561831 (not 2.22267818 — old value was from different params)
  - [x] Updated: TODOS.md, HANDOFF.md, CHANGELOG-DEV.md, CHANGELOG.md

### TODO-024 — Record Sprint 7 runtime benchmarks
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-8-housekeeping-library-losses`
- **Acceptance Criteria:**
  - [x] Binary 100k: 180.9 ms warm mean, loss 0.11909308
  - [x] K=3 20k: 101.3 ms warm mean, loss 0.63507235
  - [x] K=10 20k: 115.4 ms warm mean, loss 1.78561831
  - [x] Recorded in HANDOFF.md performance table

### TODO-025 — Update HANDOFF.md merge status
- **Assigned to:** technical-writer
- **Priority:** Low
- **Status:** Done
- **Branch:** `mlx/sprint-8-housekeeping-library-losses`
- **Acceptance Criteria:**
  - [x] "Sprint 7 on branch (not yet merged)" corrected to "Sprint 7 merged (7b483ad631)"

### TODO-026 — Item E: 16M row fix — int32 scatter_add in ComputePartitionLayout
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Resolves:** DEC-003
- **Acceptance Criteria:**
  - [x] `scatter_add_axis` accumulator in `ComputePartitionLayout` switched from float32 to int32
  - [x] `CB_ENSURE(numDocs < 2^24)` guard removed — int32 exact up to ~2.1B docs
  - [x] DEC-003 status updated to Resolved in DECISIONS.md
  - [x] Existing test suite green; binary/multiclass regression losses unchanged

### TODO-027 — Item H: CI bench regression check
- **Assigned to:** mlops-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Acceptance Criteria:**
  - [x] `mlx_test.yaml` gains "Bench regression check (binary)" step: 1k×10×cls2×d4×20i×bins32×seed42, expected `0.59795737`, tolerance 1e-4
  - [x] `mlx_test.yaml` gains "Bench regression check (multiclass K=3)" step: 1k×10×cls3×d4×20i×bins32×seed42, expected `0.95248461`, tolerance 1e-4
  - [x] CI will fail if a kernel change silently shifts final loss

### TODO-028 — Item F: MLflow integration (Python fit())
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **See:** TODO-010 (same feature — merged into this entry)
- **Acceptance Criteria:**
  - [x] `mlflow_logging` bool param and `mlflow_run_name` str param in `__init__` and `fit()`
  - [x] Lazy `import mlflow` inside `_log_to_mlflow()` — mlflow remains optional
  - [x] Run scoping: logs into active run if present; starts/ends its own run otherwise
  - [x] Documented in Python docstring

### TODO-029 — Item G: Histogram EvalNow deferral
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Acceptance Criteria:**
  - [x] `EvalNow` removed from `ComputeHistogramsImpl` after group-dispatch accumulation — histogram returned as lazy MLX array
  - [x] `EvalNow` removed from `CreateZeroHistogram`
  - [x] `FindBestSplitGPU` in `score_calcer.cpp` consumes lazy histogram directly; MLX folds group graph into same command buffer as suffix-sum dispatch
  - [x] Net: removes 2 CPU-GPU syncs per depth level from histogram.cpp
  - [x] Regression losses and determinism unaffected

### TODO-030 — Item B: max_depth > 6 — chunked multi-pass leaf accumulation
- **Assigned to:** ml-engineer
- **Priority:** High
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Acceptance Criteria:**
  - [x] `kLeafAccumChunkedSource` Metal kernel added to `kernel_sources.h`: same private-accumulator + fixed-order-reduction design as `kLeafAccumSource`, but processes leaf slice `[chunkBase, chunkBase+chunkSize)` per dispatch
  - [x] `ComputeLeafSumsGPUMultiPass` in `leaf_estimator.cpp`: issues `ceil(numLeaves/64)` dispatches, each with `chunkBase` stepped by 64; concatenates chunk outputs into full `[approxDim * numLeaves]` arrays
  - [x] `LEAF_PRIV_SIZE = MAX_APPROX_DIM * LEAF_CHUNK_SIZE * 2 = 1280 floats = 5 KB` — no register spill at any supported depth
  - [x] Old `CB_ENSURE(numLeaves <= 64)` guard replaced by `CB_ENSURE(numLeaves >= 2 && numLeaves <= 1024)` (depth 1-10)
  - [x] Depth 8 bench_boosting baseline: `0.54883069` (1k×10×cls2×d8×20i×bins32×seed42)
  - [x] See DEC-005

### TODO-031 — Item D: Depthwise grow policy
- **Assigned to:** ml-engineer
- **Priority:** Medium
- **Status:** Done
- **Branch:** `mlx/sprint-9-pybind-depth-policies-infra`
- **Acceptance Criteria:**
  - [x] `EGrowPolicy` enum in `mlx_boosting.h`: `SymmetricTree`, `Depthwise`
  - [x] `TDepthwiseTreeStructure` struct: `TVector<TObliviousSplitLevel> NodeSplits` in BFS order, `ui32 Depth`
  - [x] `SearchDepthwiseTreeStructure` in `structure_searcher.cpp`: iterates depth levels; at each level calls `FindBestSplitGPU` per live node (2^d calls at level d)
  - [x] `kTreeApplyDepthwiseSource` Metal kernel in `kernel_sources.h`: BFS traversal (left child = 2n+1, right child = 2n+2); produces `cursorOut` + `partitionsOut`
  - [x] `ApplyDepthwiseTree` in `tree_applier.cpp`: dispatches `kTreeApplyDepthwiseSource`
  - [x] `--grow-policy` CLI flag in `csv_train.cpp`; `grow_policy` Python param in `core.py`
  - [x] See DEC-004

### TODO-012 — Grow policies: Lossguide and Depthwise
- **Assigned to:** ml-engineer (Depthwise done Sprint 9; Lossguide backlog)
- **Priority:** Low
- **Status:** Partially Done
- **Depends on:** TODO-007 (GPU partition layout)
- **Acceptance Criteria:**
  - [x] `EGrowPolicy` enum: `SymmetricTree`, `Depthwise` (Lossguide not yet implemented)
  - [x] `SearchDepthwiseTreeStructure` in `structure_searcher.cpp` — per-leaf `FindBestSplitGPU` at each depth level, BFS node ordering
  - [x] `ApplyDepthwiseTree` in `tree_applier.cpp` — `kTreeApplyDepthwiseSource` Metal kernel traverses BFS node array
  - [x] `--grow-policy` CLI flag in `csv_train.cpp`; `grow_policy` Python param in `core.py`
  - [ ] `GrowPolicy::Lossguide` — leaf-priority (best-leaf-first) BFS expansion (Backlog)
  - [x] `grow_policy` documented in Python docstring with description of SymmetricTree vs Depthwise semantics

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
