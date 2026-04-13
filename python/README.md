# CatBoost-MLX

**GPU-accelerated gradient boosted decision trees on Apple Silicon**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![macOS 13+](https://img.shields.io/badge/macOS-13%2B-lightgrey)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black)](https://www.apple.com/mac/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](../LICENSE)
[![C++ Build](https://github.com/RR-AMATOK/catboost-mlx/actions/workflows/mlx-build.yaml/badge.svg)](https://github.com/RR-AMATOK/catboost-mlx/actions/workflows/mlx-build.yaml)
[![Python Tests](https://github.com/RR-AMATOK/catboost-mlx/actions/workflows/mlx-test.yaml/badge.svg)](https://github.com/RR-AMATOK/catboost-mlx/actions/workflows/mlx-test.yaml)

CatBoost-MLX is a gradient boosted decision tree (GBDT) library that runs natively on Apple Silicon GPU via Apple's Metal framework. It provides a scikit-learn-compatible Python API for training, predicting, and exporting models — with no CUDA dependency, no cloud required, and no Intel fallback. If you have a Mac with an M-series chip, the GPU is your training device.

---

## Features

- **12 loss functions** — RMSE, MAE, Logloss, CrossEntropy, MultiClass, Quantile, Huber, Poisson, Tweedie, MAPE, PairLogit, YetiRank
- **3 grow policies** — SymmetricTree (default, oblivious), Depthwise (node-level splits), Lossguide (best-first leaf expansion)
- **Tree depth 1-10** — GPU-accelerated at all depths via multi-pass leaf accumulation
- **Categorical feature support** — one-hot encoding and CTR (target encoding) for high-cardinality categories
- **Early stopping** — automatic halt on validation plateau
- **Feature importance** — gain-based, plus TreeSHAP values
- **Model export** — JSON (portable), ONNX, CoreML
- **MLflow integration** — log hyperparameters, per-iteration loss, and final metrics automatically
- **scikit-learn compatible** — works in `Pipeline`, `cross_val_score`, `clone`, `GridSearchCV`

---

## Prerequisites

- **macOS 13+** (Ventura or later)
- **Apple Silicon Mac** (M1, M2, M3, M4, or any variant)
- **Python 3.9+**
- **MLX C++ library:**
  ```bash
  brew install mlx
  ```
- **Xcode Command Line Tools** (for building the C++ binaries):
  ```bash
  xcode-select --install
  ```
- **CMake 3.27+** and **nanobind** (for the in-process nanobind extension):
  ```bash
  brew install cmake
  pip install nanobind
  ```

> **Not supported:** Intel Macs, Linux, Windows.

---

## Installation

```bash
git clone https://github.com/RR-AMATOK/catboost-mlx.git
cd catboost-mlx

# Install the Python package with the nanobind in-process extension (recommended)
# This compiles the _core nanobind module and links it against MLX directly —
# no subprocess spawning, no temp files, faster startup.
cd python && pip install -e . --no-build-isolation

# Alternative: subprocess-only install (no CMake/nanobind required)
# pip install -e python/
```

The `--no-build-isolation` flag is required so CMake can locate the MLX installation provided by the active environment. If `mlx.extension` is not available, the install falls back to the pure-Python subprocess backend automatically.

### Build the standalone CLI binaries (optional)

The Python package can also drive training via standalone `csv_train` / `csv_predict` binaries. Build them if you want the CLI tools or if you skip the nanobind extension:

```bash
# From repo root
python3 python/build_binaries.py
```

### Optional extras

```bash
pip install -e "python/[sklearn]"   # scikit-learn integration
pip install -e "python/[onnx]"      # ONNX export
pip install -e "python/[coreml]"    # CoreML export
pip install -e "python/[all]"       # All of the above
```

### Verify

```bash
python3 -c "import catboost_mlx; print(catboost_mlx.__version__)"
# Check whether the nanobind in-process extension is active:
python3 -c "from catboost_mlx.core import _HAS_NANOBIND; print('nanobind:', _HAS_NANOBIND)"
```

---

## Quick Start

### Regression

```python
import numpy as np
from catboost_mlx import CatBoostMLXRegressor

X_train = np.random.rand(1000, 10)
y_train = X_train[:, 0] * 3 + X_train[:, 1] * -1.5 + np.random.randn(1000) * 0.1

model = CatBoostMLXRegressor(iterations=200, depth=6, learning_rate=0.1)
model.fit(X_train, y_train, feature_names=[f"feat_{i}" for i in range(10)])

predictions = model.predict(X_train)
print(f"R2: {model.score(X_train, y_train):.4f}")

# Feature importance
model.plot_feature_importance()
```

### Binary Classification

```python
from catboost_mlx import CatBoostMLXClassifier

X_train = np.random.rand(1000, 8)
y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).astype(float)

clf = CatBoostMLXClassifier(
    iterations=100,
    depth=5,
    eval_fraction=0.2,
    early_stopping_rounds=20,
)
clf.fit(X_train, y_train)

labels = clf.predict(X_train)          # array of 0 or 1
probs  = clf.predict_proba(X_train)    # shape (n_samples, 2)
```

### With early stopping and validation

```python
from catboost_mlx import CatBoostMLXRegressor

model = CatBoostMLXRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_fraction=0.15,       # reserve 15% for validation
    early_stopping_rounds=30, # stop after 30 rounds with no improvement
    verbose=True,
)
model.fit(X_train, y_train)
print(model.train_loss_history[-1])
print(model.eval_loss_history[-1])
```

### Save and load

```python
# Save to JSON
model.save_model("model.json")

# Load into a new instance
from catboost_mlx import CatBoostMLXRegressor
loaded = CatBoostMLXRegressor.load("model.json")
predictions = loaded.predict(X_test)
```

### scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", CatBoostMLXRegressor(iterations=100)),
])
pipe.fit(X_train, y_train)
```

---

## Parameters Reference

All parameters are set in the constructor and apply to `CatBoostMLX`, `CatBoostMLXRegressor`, and `CatBoostMLXClassifier`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | int | `100` | Number of boosting iterations (trees). Range: 1–100000. |
| `depth` | int | `6` | Maximum tree depth. Range: 1–10. Depths 7–10 use multi-pass leaf accumulation. |
| `learning_rate` | float | `0.1` | Step-size shrinkage applied to each tree's leaf values. |
| `l2_reg_lambda` | float | `3.0` | L2 regularization on leaf values. Higher values reduce overfitting. |
| `loss` | str | `"auto"` | Loss function. See [Loss Functions](#loss-functions) table below. |
| `bins` | int | `255` | Maximum quantization bins per feature. Range: 2–255. |
| `cat_features` | list[int] | `None` | Column indices of categorical features. |
| `eval_fraction` | float | `0.0` | Fraction of training data reserved for validation. Range: [0, 1). |
| `early_stopping_rounds` | int | `0` | Stop training if validation loss does not improve for this many rounds. 0 = disabled. |
| `subsample` | float | `1.0` | Row subsampling fraction per iteration. Range: (0, 1]. |
| `colsample_bytree` | float | `1.0` | Feature subsampling fraction per tree. Range: (0, 1]. |
| `random_seed` | int | `42` | Random seed for reproducibility. |
| `nan_mode` | str | `"min"` | Missing value handling. `"min"` assigns NaN to the smallest bin; `"forbidden"` raises an error. |
| `ctr` | bool | `False` | Enable CTR (target encoding) for high-cardinality categorical features. |
| `ctr_prior` | float | `0.5` | Bayesian smoothing prior for CTR encoding. Must be > 0. |
| `max_onehot_size` | int | `10` | Features with at most this many categories use one-hot; larger cardinality uses CTR. |
| `bootstrap_type` | str | `"no"` | Sampling scheme: `"no"`, `"bayesian"`, `"bernoulli"`, `"mvs"`. |
| `bagging_temperature` | float | `1.0` | Temperature for Bayesian bootstrap. Only used when `bootstrap_type="bayesian"`. |
| `mvs_reg` | float | `0.0` | Regularization for MVS bootstrap. Only used when `bootstrap_type="mvs"`. |
| `min_data_in_leaf` | int | `1` | Minimum number of training samples required in a leaf node. |
| `random_strength` | float | `1.0` | Score perturbation magnitude for tree structure randomization. |
| `monotone_constraints` | list[int] | `None` | Per-feature monotonicity constraints. Each value must be `0` (none), `1` (increasing), or `-1` (decreasing). Length must equal number of features. |
| `snapshot_path` | str | `None` | File path for training snapshots (resume interrupted training). |
| `snapshot_interval` | int | `1` | Save a snapshot every N iterations. Requires `snapshot_path`. |
| `auto_class_weights` | str | `None` | Automatic class weight balancing: `"Balanced"` or `"SqrtBalanced"`. Classification only. |
| `grow_policy` | str | `"SymmetricTree"` | Tree grow policy. See [Grow Policies](#grow-policies) below. |
| `max_leaves` | int | `31` | Maximum number of leaves for Lossguide grow policy. Only used when `grow_policy="Lossguide"`. |
| `mlflow_logging` | bool | `False` | Log hyperparameters and per-iteration loss to MLflow after training. Requires `pip install mlflow`. |
| `mlflow_run_name` | str | `None` | Run name for MLflow. Only used when `mlflow_logging=True` starts a new run. |
| `verbose` | bool | `False` | Print per-iteration training loss to stdout. |
| `binary_path` | str | `None` | Path to directory containing `csv_train`/`csv_predict`, or the path to `csv_train` directly. |
| `train_timeout` | float | `None` | Maximum wall-clock seconds allowed for training. `None` = no limit. |
| `predict_timeout` | float | `None` | Maximum wall-clock seconds allowed for prediction. `None` = no limit. |

---

## Grow Policies

| Policy | `grow_policy` value | Description | When to use |
|--------|--------------------|--------------|----|
| SymmetricTree | `"SymmetricTree"` (default) | Oblivious trees: every node at the same depth uses the same split condition. All `2^depth` leaves are grown simultaneously. Highly regular, fast on GPU. | Default choice. Best throughput, strong regularization. |
| Depthwise | `"Depthwise"` | Non-symmetric trees: each node at a given depth level gets its own best split independently (equivalent to XGBoost `grow_policy=depthwise`). More expressive than SymmetricTree at the same depth. | More complex decision boundaries at the cost of slightly higher per-iteration time. |
| Lossguide | `"Lossguide"` | Best-first BFS expansion: grow the leaf with the highest gain first, regardless of depth level. Produces unbalanced trees; depth is not a meaningful parameter. Use `max_leaves` to control tree size. | Best accuracy per leaf for high-variance targets; use when SymmetricTree and Depthwise underfit. |

---

## Loss Functions

| Loss | `loss` syntax | Task | Notes |
|------|--------------|------|-------|
| RMSE | `"rmse"` | Regression | Default for `CatBoostMLXRegressor`. L2 loss. |
| MAE | `"mae"` | Regression | L1 loss; less sensitive to outliers than RMSE. |
| Quantile | `"quantile"` or `"quantile:0.9"` | Regression | Asymmetric L1. Predicts the alpha-th quantile. Default alpha=0.5 (median). |
| Huber | `"huber"` or `"huber:1.5"` | Regression | Smooth L1; combines MAE and RMSE. Default delta=1.0. |
| Poisson | `"poisson"` | Regression | For non-negative count data. Uses log link. |
| Tweedie | `"tweedie"` or `"tweedie:1.5"` | Regression | For zero-inflated non-negative data. Variance power p in (1, 2). Default p=1.5. |
| MAPE | `"mape"` | Regression | Mean absolute percentage error. For relative error. |
| Logloss | `"logloss"` | Binary classification | Sigmoid link. Default for `CatBoostMLXClassifier` on binary targets. |
| CrossEntropy | `"crossentropy"` | Binary classification | Alias for Logloss. |
| MultiClass | `"multiclass"` | Multi-class classification | Softmax. Auto-selected by `"auto"` on targets with 3+ classes. |
| PairLogit | `"pairlogit"` | Pairwise ranking | Logistic loss over all (winner, loser) pairs within each group. Requires `group_id` in `fit()`. |
| YetiRank | `"yetirank"` | Pairwise ranking (stochastic) | Position-weighted pairwise loss with random permutations. Requires `group_id` in `fit()`. |
| Auto | `"auto"` | Any | Selects Logloss (binary) or MultiClass (multi-class) based on target. Default for `CatBoostMLXClassifier`. |

Parameterized losses use colon syntax: `"quantile:0.75"`, `"huber:2.0"`, `"tweedie:1.8"`. Named-parameter syntax is also accepted: `"quantile:alpha=0.75"`.

Ranking losses (`pairlogit`, `yetirank`) require group IDs passed via the `group_id` parameter of `fit()`. Predictions are raw ranking scores — higher score means higher predicted relevance.

---

## API Reference

### Classes

| Class | Default loss | Use case |
|-------|-------------|---------|
| `CatBoostMLX` | `"auto"` | Base class. Use directly when you need full control over the loss. |
| `CatBoostMLXRegressor` | `"rmse"` | Regression tasks. Provides `score()` returning R². |
| `CatBoostMLXClassifier` | `"auto"` | Classification tasks. Provides `predict_proba()` and `classes_` attribute. |
| `Pool` | — | Data container. Bundles feature matrix, labels, feature names, and categorical feature indices. |

### fit()

```python
model.fit(
    X_or_pool,           # array-like (n_samples, n_features), DataFrame, or Pool
    y=None,              # array-like (n_samples,); omit when passing a Pool with labels
    eval_set=None,       # (X_val, y_val) tuple or Pool; mutually exclusive with eval_fraction
    feature_names=None,  # list of str; auto-extracted from DataFrame columns
    group_id=None,       # array-like (n_samples,); for ranking losses
    sample_weight=None,  # array-like (n_samples,); per-sample weights
)
```

### predict()

```python
predictions = model.predict(X)
# Regression: array of float
# Binary classification: array of int (0 or 1)
# Multi-class classification: array of int (class index)
```

### predict_proba()

```python
probs = clf.predict_proba(X)
# Binary: shape (n_samples, 2) — columns are P(class=0), P(class=1)
# Multiclass: shape (n_samples, n_classes)
```

### Other methods

| Method | Returns | Description |
|--------|---------|-------------|
| `save_model(path)` | None | Save model to JSON. |
| `load_model(path)` | self | Load model from JSON (in-place). |
| `load(path)` | new instance | Classmethod: create and load in one step. |
| `export_onnx(path)` | None | Export to ONNX. Requires `pip install onnx>=1.14`. |
| `export_coreml(path)` | None | Export to CoreML. Requires `pip install coremltools>=7.0`. |
| `get_feature_importance()` | dict | Gain-based importance as `{name: gain}`. |
| `get_shap_values(X)` | dict | TreeSHAP values, expected value, and feature names. |
| `cross_validate(X, y, n_folds=5)` | dict | N-fold CV using the C++ binary's built-in CV mode. Returns `fold_metrics`, `mean`, `std`. |
| `staged_predict(X, eval_period=1)` | generator | Predictions at each boosting checkpoint. |
| `staged_predict_proba(X, eval_period=1)` | generator | Probabilities at each boosting checkpoint. |
| `apply(X)` | ndarray (n, n_trees) | Leaf indices for each sample in each tree. |
| `plot_feature_importance(max_features=20)` | None | Print text bar chart to stdout. |
| `get_trees()` | list[dict] | Tree structure with real-valued split thresholds. |
| `get_model_info()` | dict | Model metadata: loss, tree count, feature count, approx dimension. |
| `score(X, y)` | float | R² (regressor) or accuracy (classifier). |
| `get_params()` | dict | All hyperparameter values (scikit-learn compatible). |
| `set_params(**kw)` | self | Set hyperparameters (scikit-learn compatible). |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tree_count_` | int | Number of trees in the fitted model. |
| `feature_names_` | list[str] | Feature names used at training time. |
| `feature_importances_` | ndarray | Normalized gain importance, shape `(n_features,)`, sums to 1.0. |
| `train_loss_history` | list[float] | Per-iteration training loss. |
| `eval_loss_history` | list[float] | Per-iteration validation loss (empty if no validation data). |
| `n_features_in_` | int | Feature count at training time (scikit-learn). |
| `feature_names_in_` | ndarray | Feature names as object array (scikit-learn 1.2+). |
| `classes_` | ndarray | Unique class labels (classifier only). |

---

## MLflow Integration

```python
import mlflow

with mlflow.start_run(run_name="my-experiment"):
    model = CatBoostMLXRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        eval_fraction=0.2,
        mlflow_logging=True,    # log into the active run
    )
    model.fit(X_train, y_train)
    # hyperparameters, train_loss, eval_loss, and trees_built are logged automatically
```

Without an active run, `mlflow_logging=True` opens and closes its own run. Set `mlflow_run_name` to name it.

---

## Model Export

```python
# JSON — portable, human-readable
model.save_model("model.json")

# ONNX — for cross-platform inference
model.export_onnx("model.onnx")       # pip install onnx>=1.14

# CoreML — for iOS/macOS on-device inference
model.export_coreml("model.mlmodel")  # pip install coremltools>=7.0

# Pickle / joblib — for Python-only workflows
import joblib
joblib.dump(model, "model.joblib")
model2 = joblib.load("model.joblib")
```

---

## Comparison with catboost

CatBoost-MLX is **not** a drop-in replacement for the `catboost` Python package. Key differences:

| | catboost | catboost-mlx |
|--|---------|-------------|
| Hardware | CPU + CUDA GPU | Apple Silicon GPU only |
| Platform | Linux, Windows, macOS | macOS (Apple Silicon) only |
| API | CatBoostRegressor, CatBoostClassifier | CatBoostMLXRegressor, CatBoostMLXClassifier |
| Feature combinations | Yes | No |
| Lossguide grow policy | Yes | Yes |
| Ranking losses (PairLogit, YetiRank) | Yes | Yes (PairLogit, YetiRank) |
| Full CatBoost model format | Yes | JSON only |
| MLX/Metal | No | Yes |

If you need full CatBoost feature parity or need to run on Linux/CUDA, use the official `catboost` package.

---

## Performance

Benchmarks on MacBook Pro M3 Max (128 GB unified memory), 100 iterations, depth=6, learning_rate=0.1:

| Dataset   | Loss    | CatBoost CPU | CatBoost-MLX | CPU iter/s | MLX iter/s |
|-----------|---------|-------------|-------------|------------|------------|
| 10k × 50  | RMSE    |       0.20s |      32.73s |      506.8 |        3.1 |
| 100k × 50 | RMSE    |       0.41s |      70.43s |      244.3 |        1.4 |
| 500k × 50 | RMSE    |       1.18s |     175.97s |       84.5 |        0.6 |
| 10k × 50  | Logloss |       0.30s |      32.08s |      332.5 |        3.1 |
| 100k × 50 | Logloss |       0.70s |      69.55s |      142.7 |        1.4 |
| 500k × 50 | Logloss |       1.73s |     173.38s |       57.9 |        0.6 |

CatBoost CPU is extremely SIMD-optimized. The MLX backend is currently slower on small/medium datasets due to per-iteration Metal kernel dispatch overhead. This is consistent with GPU GBDT in general — the CUDA backend similarly requires large datasets to outperform CPU. Performance optimization (kernel fusion, batched dispatch, async overlap) is the next development phase.

Run benchmarks yourself:
```bash
python benchmarks/bench_mlx_vs_cpu.py --hardware "Your Mac" --output benchmarks/results/your_mac.md
```

---

## Troubleshooting

**"Cannot find 'csv_train' binary"**
The GPU training binary is not built or not on PATH.
```bash
python3 python/build_binaries.py
# or
model = CatBoostMLXRegressor(binary_path="/path/to/directory")
```

**"MLX not found" during build**
```bash
brew install mlx
```

**"predict_proba is not supported for loss 'rmse'"**
`predict_proba` is only valid for classification losses (`logloss`, `multiclass`). Use `predict()` for regression.

**"eval_set and eval_fraction are mutually exclusive"**
Pass either `eval_set` to `fit()` or set `eval_fraction > 0` in the constructor, not both.

**Slow first iteration**
Metal shaders compile on first use (~100–150 ms). This is a one-time cost per process; subsequent iterations are fast.

**Tests skip with "Compiled csv_train/csv_predict binaries not found"**
```bash
python3 python/build_binaries.py
cp python/catboost_mlx/bin/csv_train .
cp python/catboost_mlx/bin/csv_predict .
```

---

## Running Tests

```bash
# From repo root
python3 -m pytest python/tests/ -v

# With coverage
python3 -m pytest python/tests/ -v --cov=catboost_mlx --cov-report=term-missing

# From python/ directory
cd python
make test
make lint
make coverage
```

---

## License

Apache 2.0 — see [LICENSE](../LICENSE) at the repo root.
