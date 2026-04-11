# Changelog

All notable user-facing changes to CatBoost-MLX are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions.

---

## [0.2.0] - 2026-04-10

### Added

**Core GPU backend (Sprints 1–2)**
- Metal compute kernels for histogram accumulation, split scoring, leaf value estimation, and tree application
- `CatBoostMLXRegressor` and `CatBoostMLXClassifier` Python classes with scikit-learn 1.0+ compatibility
- `CatBoostMLX` base class for direct loss control (regression, classification, ranking)
- `Pool` data container for bundling features, labels, categorical column metadata, and sample weights
- Standalone `csv_train` CLI tool: train GBDT models from CSV files using the Metal GPU
- Standalone `csv_predict` CLI tool: apply a saved JSON model to new CSV data
- JSON model save/load (`save_model` / `load_model` / `CatBoostMLXRegressor.load`)
- Feature importance (gain-based), returned as dict or normalized sklearn `feature_importances_` array
- SHAP values via TreeSHAP (`get_shap_values` Python, `--shap` CLI)
- Export to ONNX (`export_onnx`) and CoreML (`export_coreml`) formats
- `staged_predict` and `staged_predict_proba` for learning curve analysis
- `apply()` for leaf index extraction (useful for stacking / embeddings)
- `cross_validate` method and `--cv N` CLI flag (stratified for classification, group-aware for ranking)
- Snapshot save/resume (`snapshot_path`, `snapshot_interval`, `--snapshot-path`, `--snapshot-interval`)
- Real-time verbose training progress (`verbose=True`, `--verbose`)
- `plot_feature_importance()` terminal bar chart
- `get_trees()` and `get_model_info()` model inspection API
- `build_binaries.py` build script to compile C++ binaries from source

**Loss functions (Sprints 1–3)**
- RMSE, Logloss, MultiClass (softmax cross-entropy)
- MAE, Quantile (configurable alpha), Huber (configurable delta) — Sprint 3
- Poisson, Tweedie, MAPE — Sprint 3
- PairLogit and YetiRank pairwise ranking losses (with `--group-col` / `group_id`)
- Loss auto-detection from target column values

**Regularization and data handling**
- L2 regularization (`l2_reg_lambda` / `--l2`)
- Row subsampling (`subsample`) and feature subsampling per tree (`colsample_bytree`)
- Bootstrap types: Bayesian, Bernoulli, MVS
- Early stopping with validation loss (`eval_fraction`, `early_stopping_rounds`, explicit `eval_set`)
- Min data in leaf (`min_data_in_leaf`)
- Monotone constraints per feature
- Missing value handling (bin 0 assignment or error on NaN)
- OneHot encoding for categorical features; CTR target encoding for high-cardinality categoricals
- Auto class weights (`Balanced`, `SqrtBalanced`) for imbalanced classification
- Sample weights (`sample_weight` / `--weight-col`)
- Group/query support for ranking losses (`group_id` / `--group-col`)

**sklearn compatibility**
- `get_params()` / `set_params()` for pipeline use
- `score()` (R² for regression, accuracy for classification)
- `clone()` support (all 27+ parameters preserved correctly)
- `check_is_fitted()` via `__sklearn_is_fitted__()`
- `feature_names_in_`, `n_features_in_`, `n_outputs_`, `classes_` attributes
- `sklearn.model_selection.cross_val_score` and `Pipeline` integration

**GPU infrastructure improvements (Sprints 4–6)**
- `ComputePartitionLayout` ported to GPU: MLX `argsort` + `scatter_add_axis` + `cumsum`; eliminates 2 CPU-GPU syncs per depth level (Sprint 4, commit `19d24ec`)
- Suffix-sum histogram scan replaced with 32-lane SIMD group (`simd_prefix_inclusive_sum`); cold-start kernel compile time reduced from 344 ms to 109 ms (Sprint 5, commit `f8be378`)
- Tree applier ported to Metal kernel (`kTreeApplySource`): one thread per document, handles binary/multiclass and OneHot/ordinal splits (Sprint 6, commit `caf4552`)
- `bench_boosting` library-path C++ benchmark harness: exercises production Metal kernels without subprocess overhead (Sprint 5, commit `3e764cc`)
- CI (GitHub Actions) compiles and tests all 4 binaries (`csv_train`, `csv_predict`, `bench_boosting`, `build_verify_test`) on every push (Sprint 6, commit `1a7b9b7`)
- `bench_boosting --onehot N` flag to exercise one-hot branches in scan and tree apply kernels (Sprint 6, commit `abdd659`)

### Fixed

- **BUG-001** (Sprint 3): `CatBoostMLXRegressor(loss='MAE')` — uppercase loss names caused `SIGABRT`. Fixed by normalizing case before the binary call in both Python (`_build_train_args`) and C++ (`ParseLossType`). Commit `e5b4204`.
- **BUG-001** (Sprint 5, re-opened): `kSuffixSumSource` threadgroup buffer `scanBuf[32..255]` was left uninitialized when `suffixTG=(32,1,1)`. Apple Silicon does NOT zero threadgroup memory between dispatches. Fixed by changing to `(256,1,1)` and setting `init_value=0.0f`. Affected determinism at bins > 32 on small datasets. Commit `acecd9cbbf`.
- **BUG-002** (Sprint 3): CatBoost canonical loss syntax `Quantile:alpha=0.7` / `Huber:delta=1.0` was rejected by Python validator. Fixed by stripping `param=` prefix before `float()` parsing. Commit `e5b4204`.
- Histogram kernel (`kHistOneByteSource`) and leaf accumulation kernel (`kLeafAccumSource`) redesigned from CAS float atomics to per-thread private histograms with fixed-order sequential threadgroup reduction, eliminating float-add ordering races.

### Changed

- Dead CPU `FindBestSplit` and `FindBestSplitMultiDim` paths removed from `score_calcer.cpp`; only `FindBestSplitGPU` remains (Sprint 5, commit `1232f98`)
- Leaf sum dispatch fused and partition-stats CPU-GPU round trip eliminated (OPT-1, Sprint 3, commit `b9314b2`)
- Bin-to-feature lookup table precomputed for `score_splits` kernel (OPT-2, Sprint 3, commit `54038f2`)

### Known Issues

See the [Known Limitations](python/README.md#known-limitations) section in the Python README for full details.

- **max_depth capped at 6**: `kLeafAccumSource` uses a compile-time `MAX_LEAVES=64`. Depths above 6 produce a runtime error.
- **16M row limit**: `ComputePartitionLayout` uses float32 accumulators; datasets above 16,777,216 rows are rejected with a `CB_ENSURE` guard (DEC-003).
- **Apple Silicon only**: Requires Metal GPU (M1/M2/M3/M4, macOS 14+).
- **Subprocess overhead**: Python `fit()`/`predict()` add ~50 ms per call via subprocess.
- **Feature combinations (crosses) not implemented**.
- **Grow policies (Lossguide/Depthwise) not implemented** (TODO-012).

---

## [0.1.0] - 2026-03-29

Initial working implementation. Sprints 1–2: core Metal histogram kernel, basic tree search, Newton leaf estimation, Python bindings, and CSV training/prediction tooling.
