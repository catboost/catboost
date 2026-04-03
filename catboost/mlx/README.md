# CatBoost-MLX: GPU Training on Apple Silicon

CatBoost-MLX replaces the CUDA GPU backend with Apple's Metal via the [MLX](https://github.com/ml-explore/mlx) framework, enabling gradient boosted decision tree training on Apple Silicon Macs (M1/M2/M3/M4).

## Feature Status

| Category | Feature | Status |
|----------|---------|--------|
| **Loss Functions** | RMSE (L2) | Done |
| | MAE (L1) | Done |
| | Quantile (asymmetric L1) | Done |
| | Huber (robust) | Done |
| | Logloss (binary classification) | Done |
| | MultiClass (softmax cross-entropy) | Done |
| | Poisson (count regression) | Done |
| | Tweedie (zero-inflated continuous) | Done |
| | MAPE (relative error) | Done |
| | Ranking losses (YetiRank, PairLogit, NDCG) | Done |
| **Tree Building** | Oblivious (symmetric) tree search | Done |
| | Histogram-based split scoring | Done |
| | OneHot categorical splits (equality) | Done |
| | Ordinal splits (threshold) | Done |
| | Newton leaf estimation | Done |
| | Non-symmetric / Lossguide grow policies | Not started |
| **Regularization** | L2 regularization (lambda) | Done |
| | Row subsampling (bagging) | Done |
| | Feature subsampling per tree | Done |
| | Early stopping + validation loss | Done |
| | Bootstrap types (Bayesian, Bernoulli, MVS) | Done |
| | Min data in leaf | Done |
| | Monotone constraints | Done |
| **Data Handling** | CSV loading + auto-detection | Done |
| | Feature quantization (equal-frequency) | Done |
| | NaN/missing value handling (bin 0) | Done |
| | Categorical auto-detection + OneHot encoding | Done |
| | Target encoding (CTR) | Done |
| | Feature combinations (crosses) | Not started |
| | Group/query support (ranking) | Done |
| | Sample weights | Done |
| **Model I/O** | Export to CatBoost native TFullModel | Done |
| | JSON model save/load | Done |
| | Standalone csv_predict CLI tool | Done |
| | ONNX export | Done |
| | CoreML export | Done |
| **Explainability** | Feature importance (gain-based) | Done |
| | SHAP values (TreeSHAP) | Done |
| **Infrastructure** | Metal GPU histogram kernel | Done |
| | Standalone csv_train CLI tool | Done |
| | Python bindings | Done |
| | Cross-validation | Done |
| | Snapshot save/resume | Done |
| | CI (GitHub Actions) | Done |
| | Full sklearn 1.8+ compatibility | Done |
| | Parameter validation | Done |
| | Explicit eval_set | Done |
| | Pool data container + Pandas integration | Done |
| | Auto class weights (Balanced/SqrtBalanced) | Done |
| | Model inspection API (tree_count_, get_trees, etc.) | Done |
| | feature_importances_ (sklearn ndarray) | Done |
| | Real-time verbose training progress | Done |
| | staged_predict / staged_predict_proba | Done |
| | apply() leaf index extraction | Done |
| | Build script for binary compilation | Done |
| | Benchmark suite | Done |

**Estimated CUDA parity: ~90%**

## Prerequisites

| Requirement | Minimum |
|------------|---------|
| **macOS** | 14.0 (Sonoma) |
| **Hardware** | Apple Silicon (M1, M2, M3, M4) |
| **Xcode** | 15.0+ (for Metal compiler) |
| **MLX** | 0.22+ (C++ library) |

## Installing MLX

MLX is Apple's open-source array framework for Metal GPU computation. It is the only external dependency.

### Option 1: Homebrew (recommended)

```bash
brew install mlx
```

Verify installation:
```bash
brew info mlx
# Should show: mlx: stable X.Y.Z, installed at /opt/homebrew/Cellar/mlx/X.Y.Z
```

### Option 2: Build from source

```bash
git clone https://github.com/ml-explore/mlx.git
cd mlx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Finding MLX paths

After installation, locate include and library paths:

```bash
# Homebrew (Apple Silicon)
MLX_INCLUDE=$(brew --prefix mlx)/include
MLX_LIB=$(brew --prefix mlx)/lib

# Verify
ls $MLX_INCLUDE/mlx/mlx.h
ls $MLX_LIB/libmlx.*
```

## Building the Standalone CSV Training Tool

The standalone `csv_train` binary can train GBDT models from any CSV file using the Metal GPU.

```bash
cd catboost-mlx

# Auto-detect MLX paths
MLX_PREFIX=$(brew --prefix mlx)

clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train
```

### Quick test

```bash
# Create a small test file
cat > test.csv << 'EOF'
x1,x2,target
0.1,0.2,0
0.3,0.1,0
0.6,0.7,1
0.8,0.9,1
0.9,0.8,1
0.2,0.3,0
EOF

./csv_train test.csv --loss logloss --iterations 50 --verbose
```

## Usage

```
./csv_train <file.csv> [options]

Options:
  --iterations N         Number of boosting iterations (default: 100)
  --depth D              Max tree depth (default: 6)
  --lr RATE              Learning rate (default: 0.1)
  --l2 LAMBDA            L2 regularization (default: 3.0)
  --loss TYPE            Loss function (default: auto)
  --bins B               Max quantization bins per feature (default: 255)
  --target-col N         0-based column index for target (default: last column)
  --cat-features L       Comma-separated 0-based column indices for categorical features
  --eval-fraction F      Fraction of data for validation (default: 0 = no split)
  --eval-file PATH       External validation CSV file (mutually exclusive with --eval-fraction)
  --early-stopping N     Stop after N iters with no validation improvement (default: 0)
  --subsample F          Row subsampling fraction per iteration (default: 1.0)
  --colsample-bytree F   Feature subsampling fraction per tree (default: 1.0)
  --bootstrap-type TYPE  Bootstrap method: no, bayesian, bernoulli, mvs (default: no)
  --bagging-temperature T  Temperature for Bayesian bootstrap (default: 1.0)
  --mvs-reg R            Regularization for MVS bootstrap (default: 0.0)
  --group-col N          0-based column index for group/query ID (ranking losses)
  --weight-col N         0-based column index for sample weights
  --min-data-in-leaf N   Minimum docs in a leaf to allow a split (default: 1)
  --monotone-constraints C  Comma-separated constraints per feature: 1=increasing, -1=decreasing, 0=none
  --snapshot-path PATH   Save/resume training snapshots to this file
  --snapshot-interval N  Save snapshot every N iterations (default: 1)
  --seed N               Random seed for subsampling (default: 42)
  --nan-mode MODE        NaN handling: min (default), forbidden
  --output PATH          Save trained model to JSON file
  --feature-importance   Print gain-based feature importance after training
  --cv N                 N-fold cross-validation (default: 0 = disabled)
  --ctr                  Enable CTR target encoding for high-cardinality categoricals
  --ctr-prior F          CTR prior (default: 0.5)
  --max-onehot-size N    Max categories for OneHot; above this uses CTR (default: 10)
  --verbose              Print per-iteration loss
```

### Loss functions

| Loss | Type | Syntax |
|------|------|--------|
| RMSE (L2) | Regression | `--loss rmse` |
| MAE (L1) | Regression | `--loss mae` |
| Quantile | Regression | `--loss quantile` or `--loss quantile:0.75` |
| Huber | Regression | `--loss huber` or `--loss huber:1.5` |
| Poisson | Count regression | `--loss poisson` |
| Tweedie | Zero-inflated regression | `--loss tweedie` or `--loss tweedie:1.5` |
| MAPE | Relative error regression | `--loss mape` |
| Logloss | Binary classification | `--loss logloss` |
| MultiClass | Multi-class classification | `--loss multiclass` |
| PairLogit | Pairwise ranking | `--loss pairlogit --group-col N` |
| YetiRank | Pairwise ranking (stochastic) | `--loss yetirank --group-col N` |
| Auto-detect | Any | `--loss auto` (default) |

### Loss auto-detection

If `--loss auto` (default), the tool detects the task type from the target column:
- **{0, 1}** targets → `logloss` (binary classification)
- **{0, 1, ..., K-1}** with K > 2 → `multiclass`
- **continuous** → `rmse` (regression)

### Missing values (NaN)

Missing values (`NaN`, `nan`, `NA`, `na`, `N/A`, `?`, empty cells) are handled automatically:
- **`--nan-mode min`** (default): NaN values are assigned to bin 0 (treated as the smallest value). The tree can split NaN docs away from non-NaN docs naturally.
- **`--nan-mode forbidden`**: Error if any NaN/missing value is found.

### Early stopping

Use `--eval-fraction` and `--early-stopping` together to prevent overfitting:

```bash
./csv_train data.csv --loss rmse --eval-fraction 0.2 --early-stopping 20 --verbose
```

This holds out 20% of data for validation and stops training if validation loss doesn't improve for 20 consecutive iterations.

### Subsampling (regularization)

Row and feature subsampling reduce overfitting:

```bash
./csv_train data.csv --subsample 0.8 --colsample-bytree 0.7 --seed 42
```

- `--subsample 0.8`: Use 80% of rows per iteration (random without replacement)
- `--colsample-bytree 0.7`: Consider 70% of features per tree

### Bootstrap types

Bootstrap controls how training samples are weighted each iteration:

```bash
# Bayesian bootstrap (Dirichlet-distributed weights)
./csv_train data.csv --bootstrap-type bayesian --bagging-temperature 1.0

# Bernoulli bootstrap (random include/exclude)
./csv_train data.csv --bootstrap-type bernoulli --subsample 0.8

# MVS (Minimum Variance Sampling — gradient-based importance sampling)
./csv_train data.csv --bootstrap-type mvs --subsample 0.8 --mvs-reg 0.0
```

| Type | Method | Key parameter |
|------|--------|--------------|
| Bayesian | Exponential random weights (Dirichlet) | `--bagging-temperature` (higher = more random) |
| Bernoulli | Independent coin flip per sample | `--subsample` (inclusion probability) |
| MVS | Top samples by gradient magnitude get w=1; rest proportional | `--subsample` (fraction), `--mvs-reg` |

If `--subsample < 1.0` is set without an explicit `--bootstrap-type`, Bernoulli is used automatically.

### Ranking losses

Ranking losses learn to order documents within groups/queries. Requires a group column (`--group-col`):

```bash
# PairLogit: all pairwise comparisons within each group
./csv_train data.csv --loss pairlogit --group-col 0 --target-col 3 --iterations 200

# YetiRank: stochastic pairwise with position-weighted gradients
./csv_train data.csv --loss yetirank --group-col 0 --target-col 3 --iterations 200
```

The group column contains query/group IDs (strings or integers). All documents with the same group ID are treated as belonging to the same query. Target values are relevance labels (e.g., 0-3).

Per-iteration output includes pairwise loss and NDCG metric. Validation split and cross-validation are group-aware (no group is split across train/test).

Prediction outputs raw ranking scores — higher score means higher predicted relevance.

### Sample weights

Assign per-sample weights to emphasize certain training examples:

```bash
# Weight column is column 2 (0-based)
./csv_train data.csv --loss rmse --weight-col 2 --target-col 3 --iterations 200
```

Weights multiply into both gradients and hessians before bootstrap, so they compose naturally with all bootstrap types.

### Min data in leaf

Prevent overfitting by requiring a minimum number of training documents in each leaf:

```bash
./csv_train data.csv --loss rmse --min-data-in-leaf 10 --iterations 200
```

Splits that would create a child with fewer than N documents are rejected.

### Monotone constraints

Enforce monotonic relationships between features and predictions:

```bash
# Feature 0 must be increasing, feature 1 decreasing, feature 2 unconstrained
./csv_train data.csv --loss rmse --monotone-constraints 1,-1,0 --iterations 200
```

| Value | Meaning |
|-------|---------|
| `1` | Prediction must increase as feature increases |
| `-1` | Prediction must decrease as feature increases |
| `0` | No constraint (default) |

Constraints are enforced both at split selection time and via post-tree isotonic adjustment.

### Snapshot save/resume

Save training checkpoints to resume long-running training:

```bash
# Train 500 iterations with snapshots every 50 iterations
./csv_train data.csv --loss rmse --iterations 500 --snapshot-path snap.json --snapshot-interval 50

# Resume training to 1000 iterations (automatically loads snapshot)
./csv_train data.csv --loss rmse --iterations 1000 --snapshot-path snap.json --snapshot-interval 50
```

The snapshot file stores all training state: trees, cursor arrays, early stopping state, and RNG state. Resume automatically skips already-completed iterations.

### Categorical features

Columns with non-numeric values are auto-detected as categorical. You can also specify them explicitly:

```bash
# Auto-detect (non-numeric columns treated as categorical)
./csv_train titanic.csv --loss logloss --iterations 200

# Explicit: columns 0 and 2 are categorical
./csv_train data.csv --cat-features 0,2 --loss logloss
```

Categorical features use OneHot encoding: each unique category value becomes a split candidate via equality comparison (`value == category`), matching CatBoost's CUDA behavior.

### CTR target encoding

For high-cardinality categorical features (many unique values), OneHot encoding creates too many split candidates. CTR (Click-Through Rate / target encoding) replaces the categorical with a numeric feature using target statistics:

```
ctr(category) = (count_in_class + prior) / (total_count + 1)
```

CatBoost-MLX uses **ordered (online) CTR** to prevent target leakage: each sample's CTR is computed using only statistics from samples that appear *before* it in a random permutation.

```bash
# Enable CTR for categoricals with >10 unique values (default threshold)
./csv_train data.csv --loss logloss --ctr --iterations 200

# Custom prior and threshold
./csv_train data.csv --loss logloss --ctr --ctr-prior 1.0 --max-onehot-size 5

# Multiclass: creates one CTR feature per class
./csv_train data.csv --loss multiclass --ctr --output model.json
```

- `--ctr`: Enable CTR encoding
- `--ctr-prior F`: Bayesian smoothing prior (default: 0.5). Higher values shrink CTR towards the prior for rare categories
- `--max-onehot-size N`: Categories with ≤N unique values use OneHot; >N use CTR (default: 10)

CTR statistics are saved in the JSON model, so `csv_predict` automatically applies the correct CTR values during prediction. Unknown categories get the default CTR value (= prior).

### Cross-validation

Use `--cv N` for N-fold cross-validation:

```bash
# 5-fold CV
./csv_train data.csv --loss logloss --cv 5

# With all options
./csv_train data.csv --loss rmse --cv 10 --iterations 500 --depth 4 --lr 0.05 --ctr
```

- Stratified splitting for classification (preserves class distribution across folds)
- Random splitting for regression
- Group-aware splitting for ranking (all docs in a group stay in the same fold)
- Reports per-fold and aggregate metrics (mean ± stddev)
- Quantization borders computed once from all data, reused across folds
- `--cv` and `--eval-fraction` are mutually exclusive

### Examples

```bash
# Regression
./csv_train housing.csv --loss rmse --depth 4 --lr 0.05 --iterations 500

# Robust regression (less sensitive to outliers)
./csv_train housing.csv --loss huber:1.0 --iterations 300

# Quantile regression (predict 90th percentile)
./csv_train housing.csv --loss quantile:0.9 --iterations 300

# Binary classification
./csv_train fraud.csv --loss logloss --iterations 200 --verbose

# Multiclass (e.g., Iris dataset)
./csv_train iris.csv --loss multiclass --iterations 300

# Mixed categorical + numeric
./csv_train customer_churn.csv --cat-features 1,3,5 --loss logloss --iterations 200

# With early stopping and subsampling
./csv_train large_data.csv --loss rmse --eval-fraction 0.2 --early-stopping 50 \
  --subsample 0.8 --colsample-bytree 0.7 --iterations 1000 --verbose

# Handle data with missing values
./csv_train messy_data.csv --loss mae --nan-mode min --iterations 200

# High-cardinality categoricals with CTR encoding
./csv_train clickstream.csv --loss logloss --ctr --iterations 500 --output model.json

# Cross-validation
./csv_train data.csv --loss rmse --cv 5 --iterations 200

# CTR + cross-validation
./csv_train data.csv --loss logloss --ctr --cv 5 --iterations 200

# Ranking with PairLogit
./csv_train search_results.csv --loss pairlogit --group-col 0 --target-col 3 \
  --iterations 500 --depth 4 --output ranking_model.json

# Ranking with YetiRank + validation
./csv_train search_results.csv --loss yetirank --group-col 0 --target-col 3 \
  --eval-fraction 0.2 --early-stopping 50 --iterations 1000 --verbose

# Bayesian bootstrap
./csv_train data.csv --loss rmse --bootstrap-type bayesian --bagging-temperature 0.5

# Sample weights (column 2 contains weights)
./csv_train data.csv --loss rmse --weight-col 2 --target-col 3 --iterations 200

# Min data in leaf (regularization)
./csv_train data.csv --loss rmse --min-data-in-leaf 10 --depth 6 --iterations 200

# Monotone constraints (feature 0 increasing, feature 1 unconstrained)
./csv_train data.csv --loss rmse --monotone-constraints 1,0 --iterations 200

# Snapshot save/resume
./csv_train data.csv --loss rmse --iterations 500 --snapshot-path snap.json --snapshot-interval 50
./csv_train data.csv --loss rmse --iterations 1000 --snapshot-path snap.json  # resumes from 500

# Predict on ranking model (exclude group column from features)
./csv_predict ranking_model.json test_data.csv --group-col 0 --target-col 3
```

## Prediction Tool

The `csv_predict` tool loads a trained model (JSON format) and applies it to new CSV data.

### Building csv_predict

```bash
MLX_PREFIX=$(brew --prefix mlx)

clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/csv_predict.cpp -o csv_predict
```

### Usage

```
./csv_predict <model.json> <data.csv> [options]

Options:
  --output PATH          Write predictions to CSV file (default: stdout)
  --target-col N         0-based column index for target (for evaluation)
  --group-col N          0-based column index for group/query ID (excluded from features)
  --shap                 Compute and output TreeSHAP values
  --verbose              Print detailed info including unknown category warnings
```

### Train → Save → Predict workflow

```bash
# Step 1: Train and save model
./csv_train train.csv --loss logloss --iterations 200 --output model.json --feature-importance

# Step 2: Predict on new data
./csv_predict model.json new_data.csv --output predictions.csv

# Step 3: Evaluate on labeled test data
./csv_predict model.json test.csv --target-col 3
```

### Prediction output format

| Loss type | Output columns |
|-----------|---------------|
| Regression (rmse, mae, quantile, huber, poisson, tweedie, mape) | `prediction` (raw value) |
| Binary classification (logloss) | `probability`, `predicted_class` |
| Multi-class (multiclass) | `predicted_class`, `prob_class_0`, `prob_class_1`, ... |
| Ranking (pairlogit, yetirank) | `prediction` (raw ranking score) |

### Feature importance

Use `--feature-importance` with csv_train to see gain-based feature importance after training:

```bash
./csv_train data.csv --loss rmse --iterations 200 --feature-importance
```

Output:
```
Feature Importance (Gain-based):
Rank  Feature                     Gain       %
----  --------------------  ----------  ------
1     petal_width              45.2300   52.3%
2     petal_length             25.1000   29.0%
3     sepal_length             16.1700   18.7%
```

Feature importance is also included in the saved JSON model file.

### SHAP values

Use `--shap` with csv_predict to compute TreeSHAP values for per-prediction feature explanations:

```bash
./csv_predict model.json data.csv --output predictions.csv --shap
```

This writes a separate `predictions_shap.csv` file with columns:
```
feature1_shap,feature2_shap,...,expected_value,prediction
```

**Sum property:** For every row, `sum(shap_values) + expected_value == prediction` (within floating point precision). SHAP values are in raw prediction space (log-odds for logloss).

Supports regression, binary classification (approxDim=1), and multiclass (approxDim=K-1). For multiclass, SHAP values have shape `(n_samples, n_features, approxDim)`.

## Python Bindings

> **For full Python documentation, see [python/README.md](../../python/README.md).**
> It includes beginner-friendly explanations, installation steps, file structure maps,
> API reference, troubleshooting, and more.

The `catboost_mlx` Python package wraps the compiled CLI binaries.

### Installation

```bash
# Build the binaries first (see above)
# Then install the Python package
cd catboost-mlx
pip install -e python/

# With all optional dependencies (sklearn, ONNX, CoreML)
pip install -e "python/[all]"
```

### Quick start

```python
from catboost_mlx import CatBoostMLXRegressor, CatBoostMLXClassifier
import numpy as np

# Regression
model = CatBoostMLXRegressor(iterations=200, depth=6, learning_rate=0.1)
model.fit(X_train, y_train, feature_names=["f1", "f2", "f3"])
predictions = model.predict(X_test)

# Binary classification
clf = CatBoostMLXClassifier(iterations=200, depth=6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)       # class labels
probabilities = clf.predict_proba(X_test)  # shape (n, 2)

# Save / load
model.save_model("model.json")
model2 = CatBoostMLXRegressor()
model2.load_model("model.json")

# Feature importance
fi = model.get_feature_importance()  # {name: gain_value}

# SHAP values
shap = model.get_shap_values(X_test)
# shap["shap_values"]    → array (n_samples, n_features)
# shap["expected_value"] → float
# shap["feature_names"]  → list of str

# Cross-validation
cv = model.cross_validate(X, y, n_folds=5)
print(f"CV: {cv['mean']:.4f} +/- {cv['std']:.4f}")

# Ranking
from catboost_mlx import CatBoostMLX
ranker = CatBoostMLX(loss="pairlogit", iterations=200, depth=4)
ranker.fit(X_train, y_relevance, group_id=query_ids)
scores = ranker.predict(X_test)  # raw ranking scores

# Bootstrap
model = CatBoostMLXRegressor(
    iterations=200, bootstrap_type="bayesian", bagging_temperature=1.0
)

# Sample weights
model.fit(X_train, y_train, sample_weight=weights)

# Monotone constraints (feature 0 increasing, feature 1 unconstrained)
model = CatBoostMLXRegressor(
    iterations=200, monotone_constraints=[1, 0]
)

# Min data in leaf
model = CatBoostMLXRegressor(iterations=200, min_data_in_leaf=10)

# Snapshot save/resume
model = CatBoostMLXRegressor(
    iterations=500, snapshot_path="snap.json", snapshot_interval=50
)

# Explicit eval_set (external validation data)
model = CatBoostMLXRegressor(iterations=200)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Export to ONNX (pip install onnx)
model.export_onnx("model.onnx")

# Export to CoreML (pip install coremltools)
model.export_coreml("model.mlmodel")

# sklearn compatibility (pip install scikit-learn)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(model.get_params())

# Full sklearn 1.8+ support
from sklearn.utils.validation import check_is_fitted
check_is_fitted(model)                    # uses __sklearn_is_fitted__()
print(model.feature_names_in_)            # numpy array of feature names
print(model.n_features_in_)               # number of features at fit time
print(model.n_outputs_)                   # always 1 (single-output)

# Feature name validation (warns on mismatch)
import pandas as pd
df_test = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
model.predict(df_test)  # warns if training used different feature names

# sklearn clone / pipeline integration
from sklearn.base import clone
model2 = clone(model)  # preserves all 27+ parameters correctly

# Pool data container
from catboost_mlx import Pool
import pandas as pd
df = pd.DataFrame({"color": ["red", "blue", ...], "size": [1.0, 2.0, ...]})
pool = Pool(df, y=labels)  # auto-detects categorical columns and feature names
model.fit(pool)

# Auto class weights for imbalanced classification
clf = CatBoostMLXClassifier(iterations=200, auto_class_weights="Balanced")
clf.fit(X_train, y_train)  # automatically computes balanced sample weights

# Model inspection
print(model.tree_count_)         # number of trees
print(model.feature_names_)      # feature names from training
print(model.get_model_info())    # loss, dimensions, tree count
trees = model.get_trees()        # structured tree dicts with real thresholds
model.plot_feature_importance()  # text bar chart to terminal

# sklearn-compatible feature importances (normalized array)
fi_array = model.feature_importances_  # shape (n_features,), sums to 1.0

# Staged predictions (learning curve analysis)
for preds in model.staged_predict(X_test, eval_period=10):
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"RMSE: {rmse:.4f}")

# Staged probabilities (classification)
for probs in clf.staged_predict_proba(X_test, eval_period=10):
    print(f"Mean P(class=1): {probs[:, 1].mean():.4f}")

# Leaf index extraction (for stacking / embeddings)
leaf_indices = model.apply(X_test)  # shape (n_samples, n_trees)

# Verbose real-time training progress
model = CatBoostMLXRegressor(iterations=200, verbose=True)
model.fit(X_train, y_train)  # prints per-iteration loss in real-time
```

### Build script

Automate compilation of the binaries:

```bash
# Check prerequisites
python python/build_binaries.py --check

# Build and install binaries into the package
python python/build_binaries.py

# Build to custom directory
python python/build_binaries.py --output /usr/local/bin
```

### Benchmarks

Compare CatBoost-MLX against other GBDT frameworks:

```bash
python python/benchmarks/benchmark.py
python python/benchmarks/benchmark.py --sizes 1000 10000 50000 --output results.json
```

### Binary path

The Python bindings need to find the compiled `csv_train` and `csv_predict` binaries. They are located automatically if:
1. They are on your `PATH`
2. They are bundled in the package (`python/catboost_mlx/bin/`, built via `build_binaries.py`)
3. They are in the current working directory
4. They are in the package directory

Otherwise, specify explicitly:
```python
model = CatBoostMLXRegressor(binary_path="/path/to/directory")
```

## Building the Full CatBoost-MLX Library

To build CatBoost with the MLX GPU backend (instead of CUDA):

```bash
cd catboost-mlx
mkdir build && cd build
cmake .. -DUSE_MLX=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

This requires the full CatBoost build system (ya make or CMake) and is more involved than the standalone tool.

## Architecture

```
catboost/mlx/
├── kernels/             # Metal Shading Language GPU kernels
│   ├── kernel_sources.h    # Histogram kernel (one-byte features)
│   ├── hist.metal          # Additional histogram kernels
│   ├── leaves.metal        # Leaf value computation
│   └── scores.metal        # Split scoring
├── gpu_data/            # GPU data structures
│   ├── gpu_structures.h    # TCFeature, split descriptors
│   ├── mlx_data_set.h      # GPU-resident dataset (targets, weights, cursor, partitions)
│   ├── mlx_device.h        # MLX device wrapper
│   ├── compressed_index.h  # Bit-packed feature storage
│   └── data_set_builder.cpp
├── methods/             # Training algorithms
│   ├── mlx_boosting.h/cpp  # Main boosting loop (validation, early stopping)
│   ├── score_calcer.h/cpp  # Split scoring (OneHot + ordinal, suffix-sum)
│   ├── structure_searcher.h/cpp  # Greedy tree search with feature masking
│   ├── tree_applier.h/cpp  # Tree evaluation (OneHot + ordinal comparison)
│   ├── histogram.h/cpp     # GPU histogram dispatch
│   └── leaves/
│       └── leaf_estimator.h/cpp  # Newton leaf value computation
├── targets/             # Loss functions
│   ├── target_func.h       # IMLXTargetFunc interface
│   └── pointwise_target.h  # RMSE, MAE, Quantile, Huber, Logloss, MultiClass
├── train_lib/           # Training loop orchestration
│   ├── train.h/cpp         # Trainer registration
│   └── model_exporter.h/cpp  # TFullModel export
└── tests/               # Test files
    ├── csv_train.cpp       # Standalone CSV training tool (all features)
    ├── csv_predict.cpp     # Standalone CSV prediction tool
    ├── classification_test.cpp
    ├── model_export_test.cpp
    ├── mlx_histogram_test.cpp
    ├── standalone_kernel_test.cpp
    └── build_verify_test.cpp
```

### Python Package Structure

For full details, see [python/README.md](../../python/README.md).

```
python/
├── catboost_mlx/               # Python package
│   ├── __init__.py             # Entry point -- re-exports main classes
│   ├── core.py                 # CatBoostMLX, Regressor, Classifier
│   ├── pool.py                 # Pool data container
│   ├── _predict_utils.py       # Python-side tree evaluation
│   ├── _tree_utils.py          # Tree format conversion
│   ├── export_onnx.py          # ONNX export
│   ├── export_coreml.py        # CoreML export
│   └── bin/                    # Compiled binaries (csv_train, csv_predict)
├── build_binaries.py           # Build script
├── tests/test_basic.py         # Test suite (111 tests)
└── benchmarks/benchmark.py     # Performance comparison
```

### CLI to Python Parameter Mapping

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

### "Cannot find mlx/mlx.h"

MLX is not installed or not on the include path. Install via `brew install mlx` and verify:
```bash
ls $(brew --prefix mlx)/include/mlx/mlx.h
```

### "Library not found for -lmlx"

MLX library not on the library path. Check:
```bash
ls $(brew --prefix mlx)/lib/libmlx*
```

If using a custom MLX build, set paths manually:
```bash
clang++ ... -I/path/to/mlx/include -L/path/to/mlx/lib -lmlx ...
```

### "Undefined symbols for architecture arm64"

Ensure you're compiling on Apple Silicon (not Intel via Rosetta):
```bash
uname -m  # should print: arm64
```

### Runtime: "No Metal device found"

Requires a Mac with Apple Silicon. Verify:
```bash
system_profiler SPDisplaysDataType | grep "Metal Support"
```

### Slow first iteration

Metal shader compilation happens on the first kernel dispatch. Subsequent iterations are faster.
