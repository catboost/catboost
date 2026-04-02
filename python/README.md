# CatBoost-MLX Python Package

GPU-accelerated gradient boosted decision trees on Apple Silicon, with a scikit-learn-compatible Python API.

## What is CatBoost-MLX? (The Simple Version)

Imagine you want to teach a computer to predict things -- like house prices, or whether an email is spam.

**Decision trees** are like flowcharts of yes/no questions: "Is the house bigger than 1500 sqft? Yes -> Is it in a good neighborhood? Yes -> Predict $500K." One tree is okay, but not great.

**Gradient boosting** builds *many* small trees (typically 100-1000), where each tree focuses on fixing the mistakes of the ones before it. The final prediction combines all the trees together. This technique consistently wins machine learning competitions.

**The GPU part**: Building all those trees involves a LOT of math. Your Mac has a powerful GPU (Graphics Processing Unit) that can do this math much faster than the CPU alone. CatBoost-MLX uses Apple's **Metal** GPU framework (via the **MLX** library) to speed up training.

**In short**: CatBoost-MLX lets you train powerful prediction models on your Mac's GPU with just a few lines of Python code.

## Who Is This For?

- **Data scientists on Macs** who want GPU-accelerated gradient boosting
- **Anyone familiar with scikit-learn** -- CatBoost-MLX is a drop-in replacement
- **ML engineers** who want to export models to CoreML (iOS/macOS apps) or ONNX (cross-platform)
- **Researchers** exploring gradient boosting on Apple Silicon hardware

## Prerequisites

### Hardware
- **Apple Silicon Mac** (M1, M2, M3, M4 -- any variant)
- **macOS 14+** (Sonoma or later)

### Software (Step by Step)

**1. Python 3.9 or later**
```bash
python3 --version   # Check your version
```
If you don't have Python 3.9+, download it from [python.org](https://www.python.org/downloads/).

**2. Xcode Command Line Tools** (provides the C++ compiler)
```bash
xcode-select --install
```

**3. MLX C++ library** (Apple's GPU computation framework)
```bash
brew install mlx
brew info mlx   # Verify: should show version and install path
```

**4. numpy** (installed automatically with the package)

## Installation (Step by Step)

### Step 1: Clone the repository

```bash
git clone https://github.com/RR-AMATOK/catboost-mlx.git
cd catboost-mlx
```

### Step 2: Build the C++ binaries

The Python package delegates the heavy GPU computation to two compiled C++ programs. Build them with:

```bash
# Check that all prerequisites are installed
python3 python/build_binaries.py --check

# Compile the binaries (takes ~30 seconds)
python3 python/build_binaries.py
```

This creates two binaries in `python/catboost_mlx/bin/`:
- **csv_train** -- trains a model from CSV data using the Metal GPU
- **csv_predict** -- loads a trained model and makes predictions

### Step 3: Install the Python package

```bash
pip install -e python/
```

### Step 4: Verify the installation

```python
python3 -c "import catboost_mlx; print(catboost_mlx.__version__)"
# Should print: 0.1.0
```

### Optional dependencies

Install extras for additional features:

```bash
pip install -e "python/[sklearn]"   # scikit-learn integration (pipelines, cross_val_score)
pip install -e "python/[onnx]"      # Export models to ONNX format
pip install -e "python/[coreml]"    # Export models to CoreML format
pip install -e "python/[all]"       # All of the above
pip install -e "python/[dev]"       # Development tools (pytest, pandas)
```

## Quick Start

### Example 1: Predict house prices (Regression)

```python
import numpy as np
from catboost_mlx import CatBoostMLXRegressor

# Create some example data (100 houses, 3 features each)
np.random.seed(42)
X_train = np.random.rand(100, 3)           # features: size, bedrooms, age
y_train = X_train @ [50, 10, -5] + 100     # prices: linear combination + offset

X_test = np.random.rand(20, 3)
y_test = X_test @ [50, 10, -5] + 100

# Train a model
model = CatBoostMLXRegressor(iterations=100, depth=4, learning_rate=0.1)
model.fit(X_train, y_train, feature_names=["size", "bedrooms", "age"])

# Make predictions
predictions = model.predict(X_test)
print(f"RMSE: {np.sqrt(np.mean((predictions - y_test) ** 2)):.4f}")

# Check feature importance
print(model.get_feature_importance())
```

### Example 2: Classify spam vs not-spam (Binary Classification)

```python
from catboost_mlx import CatBoostMLXClassifier

# Synthetic binary data
X_train = np.random.rand(200, 5)
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(float)  # 0 or 1

clf = CatBoostMLXClassifier(iterations=100, depth=4)
clf.fit(X_train, y_train)

# Class labels (0 or 1)
predictions = clf.predict(X_test)

# Probabilities (shape: n_samples x 2)
probabilities = clf.predict_proba(X_test)
print(f"P(spam) for first sample: {probabilities[0, 1]:.3f}")
```

### Example 3: Using pandas DataFrames with Pool

```python
import pandas as pd
from catboost_mlx import Pool, CatBoostMLXClassifier

# DataFrame with mixed types
df = pd.DataFrame({
    "color": ["red", "blue", "red", "green", "blue"] * 20,
    "size": np.random.rand(100),
    "weight": np.random.rand(100),
})
labels = (df["size"] > 0.5).astype(float)

# Pool auto-detects "color" as categorical and extracts feature names
pool = Pool(df, y=labels)
print(pool)  # Pool(100 samples, 3 features, 1 categorical, with labels)

clf = CatBoostMLXClassifier(iterations=50, depth=4)
clf.fit(pool)
```

## How It Works

CatBoost-MLX has a simple architecture: Python handles the user-facing API, while compiled C++ binaries do the heavy computation on the GPU.

```
     Your Python Code
           |
           v
  ┌─────────────────┐
  │   catboost_mlx   │  <-- Python package (this code)
  │  (core.py, etc.) │
  └────────┬─────────┘
           |  writes CSV + reads JSON
           v
  ┌─────────────────┐
  │   csv_train /    │  <-- Compiled C++ binaries
  │   csv_predict    │
  └────────┬─────────┘
           |  Metal GPU kernels
           v
  ┌─────────────────┐
  │   Apple Silicon  │  <-- Your Mac's GPU
  │   Metal GPU      │
  └─────────────────┘
```

**Training flow**: Python writes your data to a temporary CSV file, calls the `csv_train` binary (which trains on the GPU using Metal), then reads back the resulting model from a JSON file.

**Prediction flow**: Python writes the model JSON and data CSV to temp files, calls `csv_predict`, and parses the output CSV of predictions.

## File Structure

```
python/
├── README.md                       # You are here!
├── pyproject.toml                  # Package config (name, version, dependencies)
├── build_binaries.py               # Compiles the C++ GPU binaries
│
├── catboost_mlx/                   # The Python package
│   ├── __init__.py                 # Entry point -- re-exports main classes
│   ├── core.py                     # Main classes: CatBoostMLX, Regressor, Classifier
│   ├── pool.py                     # Pool data container (bundles features + metadata)
│   ├── _predict_utils.py           # Python tree evaluation (for staged_predict/apply)
│   ├── _tree_utils.py              # Tree format conversion (for export)
│   ├── export_onnx.py              # Export to ONNX format
│   ├── export_coreml.py            # Export to CoreML format
│   └── bin/                        # Compiled C++ binaries (created by build_binaries.py)
│       ├── csv_train               #   GPU training binary
│       └── csv_predict             #   GPU prediction binary
│
├── tests/
│   └── test_basic.py               # 111 tests covering all functionality
│
└── benchmarks/
    ├── benchmark.py                # Speed/accuracy comparison tool
    └── README.md                   # How to run benchmarks
```

### How the files relate

```
Users import from:
  __init__.py ──────── re-exports classes from ──► core.py
                       re-exports Pool from ────► pool.py

core.py (main logic):
  ├── Uses pool.py .............. to unpack Pool objects in fit()
  ├── Uses _predict_utils.py .... for staged_predict() and apply()
  ├── Uses _tree_utils.py ....... for get_trees()
  ├── Uses export_coreml.py ..... for export_coreml()
  └── Uses export_onnx.py ...... for export_onnx()

export_coreml.py ─── uses ──► _tree_utils.py (unfold oblivious trees)
export_onnx.py ───── uses ──► _tree_utils.py (unfold oblivious trees)

Standalone (not imported by the package):
  build_binaries.py ............. compiles C++ binaries
  benchmarks/benchmark.py ....... performance comparison
  tests/test_basic.py ........... test suite
```

## API Reference

### Classes

| Class | Use Case | Default Loss |
|-------|----------|-------------|
| `CatBoostMLX` | General-purpose (any loss) | `auto` |
| `CatBoostMLXRegressor` | Predicting numbers | `rmse` |
| `CatBoostMLXClassifier` | Predicting categories | `auto` (detects binary vs multiclass) |
| `Pool` | Bundling data + metadata | -- |

### Key Methods

| Method | What it does | Returns |
|--------|-------------|---------|
| `fit(X, y)` | Train the model on data | `self` |
| `predict(X)` | Get predictions (values or class labels) | numpy array |
| `predict_proba(X)` | Get class probabilities | numpy array (n, k) |
| `staged_predict(X)` | Predictions at each boosting step | generator |
| `staged_predict_proba(X)` | Probabilities at each boosting step | generator |
| `apply(X)` | Get leaf indices for each tree | numpy array (n, n_trees) |
| `save_model(path)` | Save model to JSON file | None |
| `load_model(path)` | Load model from JSON file | self |
| `load(path)` | **Classmethod**: create + load in one step | new instance |
| `export_onnx(path)` | Export to ONNX format | None |
| `export_coreml(path)` | Export to CoreML format | None |
| `get_feature_importance()` | Feature importance (dict) | dict |
| `get_shap_values(X)` | TreeSHAP explanations | dict |
| `cross_validate(X, y)` | N-fold cross-validation | dict |
| `score(X, y)` | R² (regression) or accuracy (classification) | float |
| `get_params()` | Get all hyperparameters (sklearn) | dict |
| `set_params(**kw)` | Set hyperparameters (sklearn) | self |
| `get_trees()` | Structured tree details | list of dicts |
| `get_model_info()` | Model metadata | dict |
| `plot_feature_importance()` | Print bar chart to terminal | None |

### Properties

| Property | Description |
|----------|-------------|
| `tree_count_` | Number of trees in the model |
| `feature_names_` | Feature names used during training |
| `feature_importances_` | Normalized importance array (sums to 1.0, sklearn-compatible) |
| `train_loss_history` | Training loss at each iteration |
| `eval_loss_history` | Validation loss at each iteration |
| `n_features_in_` | Number of features at training time (sklearn) |
| `feature_names_in_` | Feature names as numpy array (sklearn 1.2+) |
| `n_outputs_` | Number of outputs (always 1) |
| `classes_` | Unique class labels (classifier only) |

### Supported Loss Functions

| Loss | Task | When to use |
|------|------|-------------|
| `rmse` | Regression | Predicting continuous numbers (default) |
| `mae` | Regression | When outliers are common (less sensitive) |
| `quantile` or `quantile:0.9` | Regression | Predict a specific percentile |
| `huber` or `huber:1.5` | Regression | Balanced between RMSE and MAE |
| `poisson` | Regression | Count data (always positive) |
| `tweedie` or `tweedie:1.5` | Regression | Zero-inflated continuous data |
| `mape` | Regression | When relative error matters |
| `logloss` | Classification | Binary (spam/not-spam, yes/no) |
| `multiclass` | Classification | Multiple categories (cat/dog/bird) |
| `pairlogit` | Ranking | Order items by relevance |
| `yetirank` | Ranking | Order items (stochastic, better quality) |
| `auto` | Any | Auto-detects from your target values |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 100 | Number of trees to build |
| `depth` | 6 | Max depth of each tree (1-16) |
| `learning_rate` | 0.1 | How much each tree contributes (smaller = more conservative) |
| `l2_reg_lambda` | 3.0 | Regularization strength (prevents overfitting) |
| `loss` | "auto" | Loss function (see table above) |
| `bins` | 255 | Max quantization bins per feature (2-255) |
| `cat_features` | None | Which columns are categorical (list of indices) |
| `eval_fraction` | 0.0 | Fraction of data for validation (0 = no split) |
| `early_stopping_rounds` | 0 | Stop if no improvement for N rounds |
| `subsample` | 1.0 | Row sampling fraction (regularization) |
| `colsample_bytree` | 1.0 | Feature sampling fraction per tree |
| `random_seed` | 42 | Random seed for reproducibility |
| `nan_mode` | "min" | How to handle missing values ("min" or "forbidden") |
| `ctr` | False | Enable target encoding for high-cardinality categoricals |
| `ctr_prior` | 0.5 | Bayesian prior for CTR encoding |
| `max_onehot_size` | 10 | Categories with <= N values use OneHot; > N use CTR |
| `bootstrap_type` | "no" | Bootstrap method: "no", "bayesian", "bernoulli", "mvs" |
| `bagging_temperature` | 1.0 | Temperature for Bayesian bootstrap |
| `mvs_reg` | 0.0 | Regularization for MVS bootstrap |
| `min_data_in_leaf` | 1 | Min samples per leaf (prevents tiny leaves) |
| `monotone_constraints` | None | Per-feature monotone constraints (0, 1, -1) |
| `snapshot_path` | None | Save/resume training checkpoints |
| `snapshot_interval` | 1 | Save snapshot every N iterations |
| `auto_class_weights` | None | "Balanced" or "SqrtBalanced" for imbalanced data |
| `verbose` | False | Print per-iteration progress during training |
| `binary_path` | None | Custom path to compiled binaries |
| `train_timeout` | 600.0 | Max seconds for training subprocess (None = no limit) |
| `predict_timeout` | 60.0 | Max seconds for prediction subprocess (None = no limit) |

### CLI to Python Parameter Mapping

Some CLI flags use different names than the Python parameters:

| CLI Flag | Python Parameter |
|----------|-----------------|
| `--lr` | `learning_rate` |
| `--l2` | `l2_reg_lambda` |
| `--early-stopping` | `early_stopping_rounds` |
| `--target-col` | (auto-detected from data) |
| `--cat-features` | `cat_features` |
| `--weight-col` | `sample_weight` (in fit()) |
| `--group-col` | `group_id` (in fit()) |

## Troubleshooting

### "Cannot find 'csv_train' binary"
**Cause**: The compiled C++ binaries are not on your system PATH or in the package.
**Fix**:
```bash
python3 python/build_binaries.py   # Compile them
# OR
model = CatBoostMLXRegressor(binary_path="/path/to/directory")  # Point to them
```

### "MLX not found"
**Cause**: The MLX C++ library is not installed.
**Fix**:
```bash
brew install mlx
```

### "predict_proba is not supported for loss 'rmse'"
**Cause**: You called `predict_proba()` on a regression model.
**Fix**: Use `predict()` for regression, or switch to `CatBoostMLXClassifier` for classification.

### "Model is not fitted. Call fit() first."
**Cause**: You tried to predict or export before training.
**Fix**: Call `model.fit(X, y)` first.

### "eval_set and eval_fraction are mutually exclusive"
**Cause**: You set both `eval_fraction > 0` in the constructor and passed `eval_set` to `fit()`.
**Fix**: Use one or the other.

### Slow first iteration
**Cause**: Metal shader compilation happens on the first GPU kernel dispatch.
**Not a bug** -- subsequent iterations will be much faster.

### Tests skip with "Compiled csv_train/csv_predict binaries not found"
**Cause**: Binaries are not compiled or not at the expected location.
**Fix**: Run `python3 python/build_binaries.py` first, then copy binaries to repo root:
```bash
cp python/catboost_mlx/bin/csv_train .
cp python/catboost_mlx/bin/csv_predict .
```

## Running Tests

```bash
# All tests (146 tests)
python3 -m pytest python/tests/ -v

# A specific test class
python3 -m pytest python/tests/test_basic.py::TestRegression -v

# With short tracebacks on failure
python3 -m pytest python/tests/ -v --tb=short

# With coverage report
python3 -m pytest python/tests/ -v --tb=short --cov=catboost_mlx --cov-report=term-missing
```

Or from the `python/` directory using the Makefile:

```bash
cd python
make test        # Run tests
make lint        # Run ruff linter
make coverage    # Tests with coverage report
```

## Benchmarks

Compare CatBoost-MLX against XGBoost, LightGBM, and CatBoost:

```bash
python3 python/benchmarks/benchmark.py
python3 python/benchmarks/benchmark.py --sizes 1000 10000 50000 --output results.json
```

Frameworks not installed are automatically skipped. See [benchmarks/README.md](benchmarks/README.md) for details.

## sklearn Integration

CatBoost-MLX is fully compatible with scikit-learn 1.8+:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Pipelines
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
pipe.fit(X_train, y_train)

# Clone (preserves all parameters)
model2 = clone(model)

# check_is_fitted
from sklearn.utils.validation import check_is_fitted
check_is_fitted(model)  # raises if not fitted
```

## Serialization

Models can be saved and loaded in multiple ways:

```python
# JSON (human-readable, portable)
model.save_model("model.json")
model.load_model("model.json")

# Classmethod (create + load in one step)
loaded = CatBoostMLXRegressor.load("model.json", binary_path="/path/to/binaries")

# Pickle / joblib (includes all state: model, loss history, parameters)
import pickle
data = pickle.dumps(model)
model2 = pickle.loads(data)

import joblib
joblib.dump(model, "model.joblib")
model2 = joblib.load("model.joblib")

# Export to ONNX or CoreML for cross-platform inference
model.export_onnx("model.onnx")      # pip install onnx>=1.14
model.export_coreml("model.mlmodel")  # pip install coremltools>=7.0
```

## Contributing

### Development setup

```bash
git clone https://github.com/RR-AMATOK/catboost-mlx.git
cd catboost-mlx
python3 python/build_binaries.py       # Compile C++ binaries
cp python/catboost_mlx/bin/* .         # Copy to repo root for tests
pip install -e "python/[dev]"          # Install in editable mode with dev deps
python3 -m pytest python/tests/ -v     # Run tests
```

Optional: set up pre-commit hooks to auto-lint on each commit:

```bash
pip install pre-commit
cd python && pre-commit install
```

### Code style
- Python 3.9+ compatible
- Type hints on public API methods
- Every file has a header comment explaining what it does
- Tests for every new feature
- Linted with [ruff](https://docs.astral.sh/ruff/) (`ruff check catboost_mlx/ tests/`)

### Making changes
1. Create a branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run lint: `cd python && make lint`
4. Add or update tests in `python/tests/`
5. Run tests: `cd python && make test`
6. Submit a pull request

## License

Apache 2.0 (see LICENSE in the repo root)
