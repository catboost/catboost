# Developer Changelog

<!-- Agents append entries after completing work. Chronological, most recent first. -->
<!-- Format: Date header → Agent → What changed → Related TODOs/DECs -->

---

## 2026-04-10 — Sprint 6: CI infra, bench_boosting --onehot, tree applier Metal kernel

**Agents:** mlops-engineer, qa-engineer, ml-engineer
**Branch:** `mlx/sprint-6-ci-and-infra`

### TODO-016: CI blind spot fix
- Added bench_boosting and build_verify_test compile steps to `.github/workflows/mlx_test.yaml`
- bench_boosting copied to `/tmp/bench_boosting` so 62+ library-path tests (test_qa_round10) actually run in CI instead of silently skipping
- Verify binaries step now checks all 4 executables
- Commit: `1a7b9b7`

### TODO-007: Sign-off closed
- Ran 28 multiclass Python tests — all pass. Last acceptance criterion confirmed.
- HANDOFF.md updated for Sprint 6 state (684 tests, BUG-001 fixed, Sprint 5 merged).
- Commit: `89a768d`

### TODO-017: bench_boosting --onehot flag
- Added `--onehot N` CLI flag: marks first N features as one-hot with small bin counts (2-10)
- Exercises one-hot skip branch in kSuffixSumSource and equality comparison in tree applier
- Verified: `--onehot 5` produces finite, deterministic BENCH_FINAL_LOSS
- Commit: `abdd659`

### TODO-018: Tree applier Metal kernel port
- Added `kTreeApplySource` Metal kernel to `kernel_sources.h`: one thread per doc, computes leaf index via split loop, updates predictions
- Replaced CPU MLX op loop in `tree_applier.cpp` with single `mx::fast::metal_kernel()` dispatch
- Handles binary/regression and multiclass, OneHot equality vs ordinal greater-than
- BENCH_FINAL_LOSS: 0.69314516 (binary 100k, exact match), 1.09757149 (multiclass 20k, exact match)
- Performance: ~107ms warm mean (statistically equivalent to pre-port ~104ms — tree apply is not the bottleneck)
- Commit: `caf4552`

### Ruff lint cleanup (pre-Sprint-6, merged with Sprint 5)
- Fixed 36 ruff errors across python/catboost_mlx/ and python/tests/ (F401, F541, F841, E741, I001)
- Commits: `32b5419`, `ee61752`

### Verification
- pytest: 684 passed, 5 skipped, 4 xfailed
- ruff: 0 errors
- bench_boosting reference losses: exact match at binary 100k and multiclass 20k

---

## 2026-04-10 — BUG-001 fix: deterministic suffix-sum scan

**Agent:** ml-engineer
**Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`
**Commit:** `acecd9cbbf` `[mlx] kernels: deterministic suffix-sum scan (BUG-001 fix)`

### Root cause
`kSuffixSumSource` declares `threadgroup float scanBuf[256]` and runs a Hillis-Steele
inclusive scan (8 stride rounds, strides 1..128) requiring 256 active threads.
Both `FindBestSplitGPU` overloads in `score_calcer.cpp` dispatched with
`suffixTG=(32,1,1)`, leaving `scanBuf[32..255]` uninitialized. Metal threadgroup
memory is NOT zeroed between dispatches; garbage values propagated into suffix sums,
produced non-deterministic split decisions, and caused final loss variance of ~3e-3
at 10k rows for bins {33, 34, 48, 65, 96}. At 100k rows the large histogram values
dominated and splits were stable despite corrupted scan — masking the bug.

### Fix (score_calcer.cpp only — 4 lines changed)
- `suffixTG` changed from `(32,1,1)` → `(256,1,1)` in both `FindBestSplitGPU` overloads.
- `init_value` changed from `std::nullopt` → `0.0f` so one-hot bins and the skipped
  last ordinal bin read as 0.
- `bench_boosting.cpp` already used `(256,1,1)` correctly (explains why 10-run manual
  tests before the sprint showed determinism — they used the bench binary, not the library).

### Also included in commit (already written, not yet committed)
- `kernel_sources.h`: full redesign of `kHistOneByteSource` and `kLeafAccumSource`
  from CAS float atomics to per-thread private histograms + fixed-order sequential
  threadgroup reduction (eliminates all float-add ordering races).
- `bench_boosting.cpp`: `maxBlocksPerPart=1` (eliminates cross-threadgroup atomic races
  in global histogram output buffer).
- `test_qa_round10_sprint5_bench_and_scan.py`: multiclass anchor updated from
  `1.07820153` → `1.09757149` (old value was captured with the buggy 32-thread kernel
  which happened to zero scanBuf[32..255] on first dispatch).

### Verification
- 10x bench_boosting --rows 10000 --features 20 --classes 2 --depth 4 --iters 30 --bins 96 --seed 42: all `BENCH_FINAL_LOSS=0.69310117`
- 100k bins=32: `BENCH_FINAL_LOSS=0.69314516` (exact pre-fix reference match)
- 100k bins=255: `BENCH_FINAL_LOSS=0.69313669` (exact pre-fix reference match)
- pytest python/tests/: 684 passed, 5 skipped, 4 xfailed

---

## 2026-04-09 — Sprint 5: Parallel scan, benchmark harness, dead-code cleanup

**Agent:** ml-engineer
**Branch:** `mlx/sprint-5-parallel-scan-benchmark-harness`

### TODO-014: bench_boosting library-path benchmark harness
- New standalone C++ binary `catboost/mlx/tests/bench_boosting.cpp` that synthesizes in-memory data and exercises the production Metal kernels (histogram, suffix-sum, split scoring, leaf accumulation, tree application) directly — no subprocess, no CatBoost headers.
- CLI: `--rows`, `--features`, `--classes`, `--depth`, `--iters`, `--bins`, `--lr`, `--l2`, `--seed`.
- Supports regression (classes=1), binary (classes=2), multiclass (classes>=3).
- Prints iter-0 cold-start vs warm average, final loss for regression testing.
- Commit: `3e764cc` `[mlx] tests: add bench_boosting library-path benchmark harness (TODO-014)`

### TODO-008: Parallel SIMD scan for suffix_sum_histogram
- Replaced serial 1-thread-per-feature scan with 32-lane SIMD group per (feature, partition, stat) triple using `simd_prefix_inclusive_sum`.
- Single-pass for folds <= 32 (common case); right-to-left chunked with carry for folds > 32 (up to 255 bins).
- Serial skip of h'[folds-1] preserved.
- Threadgroup updated from (1,1,1) → (32,1,1) in score_calcer.cpp (both FindBestSplitGPU overloads).
- Determinism: BENCH_FINAL_LOSS=0.69314516 (bins=32) and 0.69313669 (bins=255) identical before/after.
- Cold-start kernel compile time dropped 344 ms → 109 ms.
- Commit: `f8be378` `[mlx] kernels: parallel SIMD scan for suffix_sum_histogram (TODO-008)`

### TODO-009: Delete dead CPU FindBestSplit paths
- Removed `FindBestSplit` and `FindBestSplitMultiDim` from `score_calcer.cpp` and `score_calcer.h`.
- Only `FindBestSplitGPU` remains. `csv_train.cpp`'s own standalone reimplementation is unrelated.
- Commit: `1232f98` `[mlx] score_calcer: delete dead CPU FindBestSplit paths (TODO-009)`

### TODO-013: Fix build_verify_test kernel param names
- Fixed three mismatches vs kHistOneByteSource: `featureColumnIdx` (scalar) → `featureColumnIndices` (1-element array), `foldCounts` → `foldCountsFlat`, `firstFoldIndices` → `firstFoldIndicesFlat`, added missing `numGroups=1`.
- build_verify_test: ALL TESTS PASSED.
- Commit: `3ca05c3` `[mlx] tests: fix build_verify_test kernel param names (TODO-013)`

### TODO-015: Document 16M-row float32 limit in DECISIONS.md
- Added DEC-003 entry explaining float32 scatter_add ceiling (2^24) and CB_ENSURE guard.
- Commit: `dbfcf07` `[mlx] state: document 16M-row float32 limit in DECISIONS.md (TODO-015)`

### Verification
- Python test suite: 622/622 passing.
- bench_boosting: binary 100k×50×depth6×100iters, 38.5 ms/iter warm mean; multiclass K=3 20k×30×depth5×50iters, 11.4 ms/iter warm mean.
- grep FindBestSplit catboost/mlx/ → only FindBestSplitGPU and csv_train internal.

---

## 2026-04-09 — Sprint 4: GPU Partition Layout (TODO-007)

**Agent:** ml-engineer
**Branch:** `mlx/sprint-4-gpu-partition-layout`

### GPU partition layout (ml-engineer)
- Rewrote `ComputePartitionLayout` in `catboost/mlx/methods/structure_searcher.cpp` to run entirely on GPU using MLX primitives: `argsort` (O(n) radix sort) + `scatter_add_axis` + `cumsum`. No CPU-GPU sync inside the function.
- Removed `PartSizesHost` and `PartOffsetsHost` CPU-mirror vectors from `TPartitionLayout` struct. No downstream consumers in production code.
- Updated `build_verify_test.cpp` to match new struct layout.
- EvalNow calls in `structure_searcher.cpp`: 3 → 1 (reduction of 2 per depth level).

### Verification
- Python test suite: 604/604 passing.
- csv_train regression: final loss on smoke test = 0.481507 (identical to pre-sprint baseline).
- Build: clean with clang++ on Apple Silicon.
- Commit: `19d24ec` `[mlx] methods: port ComputePartitionLayout to GPU (TODO-007)`

### Architecture note
- Python bindings invoke `csv_train` C++ binary via subprocess. `csv_train.cpp` has its own `ComputePartitionLayout` (already GPU-resident via argsort, written independently). The Sprint 4 change applies to the C++ library path: `mlx_boosting.cpp → SearchTreeStructure → ComputePartitionLayout`.
- Python benchmark timings (K=3 50-iter) are unaffected because they route through csv_train, not structure_searcher.cpp.

---

## 2026-04-09 — Sprint 3 close + follow-up cycle

**Agents:** ml-engineer, qa-engineer, mlops-engineer, technical-writer

### Loss function wiring (ml-engineer)
- Extended `catboost/mlx/train_lib/train.cpp` with dispatch cases for MAE, Quantile (alpha, default 0.5), and Huber (delta, mandatory). +29 lines. (TODO-005, commit `38f963c`)

### QA validation (qa-engineer)
- Added `python/tests/test_qa_round8_sprint3_losses.py`: 36 tests covering MAE, Quantile, and Huber loss functions. (commit `1cc4870`)
- QA identified BUG-001 and BUG-002 during this pass.

### Bug fixes (ml-engineer)
- **BUG-001** — `CatBoostMLXRegressor(loss='MAE')` (uppercase) caused SIGABRT. Root cause: Python `_validate_params` lowercased for its own check but passed original unmodified string to `csv_train` binary, which had no case folding. Fixed in both Python `_build_train_args` and C++ `ParseLossType`. (commit `e5b4204`)
- **BUG-002** — CatBoost canonical `Quantile:alpha=0.7` / `Huber:delta=1.0` syntax was rejected by Python validator because `_validate_params` called `float()` on the `alpha=0.7` string. Fixed by stripping the `alpha=`/`delta=` prefix before `float()`, then normalizing to positional form before the binary call. (commit `e5b4204`)

### Performance optimizations (ml-engineer)
- **OPT-1** — Fused leaf sum dispatch and removed partition-stats CPU-GPU round trip (`catboost/mlx/methods/`). Eliminates `approxDim` syncs per tree depth level. (commit `b9314b2`)
- **OPT-2** — Precomputed bin-to-feature lookup table for `score_splits` kernel (`catboost/mlx/kernels/`). (commit `54038f2`)
- **Safety guard** — Added runtime `CB_ENSURE(max_depth <= 6)` in `leaf_estimator` to guard `kLeafAccumSource` compile-time `MAX_LEAVES=64` limit. Previously silent wrong results above this depth. (commit `928c7ff`)

### MLOps audit findings (mlops-engineer)
- Static audit of `EvalNow` call sites across `catboost/mlx/methods/`: 13 total. Per `SearchTreeStructure` depth level: binary = 6 syncs, multiclass K = 4 + 2K syncs.
- Top 5 optimization opportunities identified. OPT-1 (#2 on list) and OPT-2 (#4 on list) addressed this sprint.
- Deferred to Sprint 4: GPU partition layout (#1), parallel suffix_sum_histogram scan (#3), MLflow integration (#5).
- No profiling numbers collected this sprint (static audit only).

### Verification
- Python test suite: 604 passing (up from 558 pre-Sprint-3, +46 net).
- csv_train regression smoke test: identical final loss (0.596), identical timing (0.30s). No regression.
- Multiclass K=3 micro-benchmark: 1.07s → 0.98s (~8% speedup from OPT-1+OPT-2).

### Process change
- Sprint branches policy adopted: starting Sprint 4, all sprint work goes to `mlx/sprint-<N>-<short-topic>` branch, merged to master only after QA/MLOps sign-off. (DEC-002)
