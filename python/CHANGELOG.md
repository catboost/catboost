# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CatBoostMLX.load()` classmethod for convenient model loading
- `predict()`, `predict_proba()`, and other predict methods now accept Pool objects
- Pickle / joblib serialization support (`__getstate__` / `__setstate__`)
- `conftest.py` with shared test fixtures
- `py.typed` marker for PEP 561 compliance
- `logging` module usage for non-user-facing output
- Ruff and pytest configuration in `pyproject.toml`
- CI test matrix expanded to Python 3.10, 3.12, 3.13

### Changed
- `_HAS_SKLEARN` removed from `__all__` (still accessible via `catboost_mlx.core._HAS_SKLEARN`)
- `__version__` now reads from package metadata instead of hardcoded duplication
- `cross_validate()` reuses `_build_train_args()` to reduce code duplication
- `_to_numpy()` consolidated into `_utils.py` (imported by both `core.py` and `pool.py`)

### Known Limitations
- Binary bundling: `pip install` does not include pre-compiled csv_train/csv_predict.
  Users must compile binaries separately via `python build_binaries.py`.

## [0.1.0] - 2026-04-02

### Added
- Initial release
- CatBoostMLX base class with 27 hyperparameters
- CatBoostMLXRegressor and CatBoostMLXClassifier subclasses
- Pool data container with pandas DataFrame auto-detection
- Loss functions: RMSE, MAE, Quantile, Huber, Poisson, Tweedie, MAPE, Logloss, Multiclass, PairLogit, YetiRank
- Staged predict and staged predict_proba for learning curves
- TreeSHAP values via csv_predict --shap
- Feature importance (gain-based) with text bar chart visualization
- Model save/load (JSON format)
- Export to CoreML and ONNX formats
- N-fold cross-validation
- Bootstrap types: Bayesian, Bernoulli, MVS
- Monotone constraints, min_data_in_leaf, snapshot resume
- Auto class weights (Balanced, SqrtBalanced)
- scikit-learn 1.8+ compatibility (fit/predict/score, get_params/set_params, clone, pipelines)
- 120 tests across 27 test classes
