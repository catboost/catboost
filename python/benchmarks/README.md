# CatBoost-MLX Benchmarks

Compare CatBoost-MLX (Apple Silicon Metal GPU) against other GBDT frameworks.

## Quick Start

```bash
python benchmark.py
```

## Options

```
--sizes 1000 10000 50000    Dataset sizes (default: 1000 10000 50000)
--iterations 200            Boosting iterations (default: 200)
--depth 6                   Tree depth (default: 6)
--output results.json       Save results to JSON
```

## Frameworks Compared

| Framework | Source | Notes |
|-----------|--------|-------|
| CatBoost-MLX | This project | Metal GPU, Apple Silicon |
| XGBoost | `pip install xgboost` | CPU hist method |
| LightGBM | `pip install lightgbm` | CPU |
| CatBoost | `pip install catboost` | CPU (no CUDA on macOS) |

Frameworks not installed are automatically skipped.

## Tasks

- **Regression**: 50 features, linear + noise
- **Binary classification**: 50 features, logistic decision boundary
- **Multiclass** (5 classes): 50 features, sum-based partitioning

Each task uses 80/20 train/test split.

## Metrics

- **Train time**: Wall-clock seconds for `fit()`
- **Predict time**: Wall-clock seconds for `predict()`
- **RMSE** (regression) or **Accuracy** (classification) on test set
