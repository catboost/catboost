# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `cross_validate()`: CV output parser now matches actual binary output format (`Fold N: test_loss=...` and `Test  loss: ... +/- ...`)
- `CatBoostMLXClassifier.fit()`: `y` parameter now defaults to `None` so `fit(pool)` works without passing `y` explicitly
- `load_model()`: restores `self.loss` from model JSON so the instance reflects the trained loss, not the constructor default
- `_array_to_csv`: return type annotation tightened from `tuple` to `Tuple[int, int, int]`

### Changed
- Added `Tuple` to typing imports; added return type hint on `_unpack_predict_input`
- `MANIFEST.in`: added `include LICENSE` for proper sdist packaging

## [0.2.0] - 2026-04-02

### Added
- `CatBoostMLX.load()` classmethod for convenient model loading
- `predict()`, `predict_proba()`, and other predict methods now accept Pool objects
- Pickle / joblib serialization support (`__getstate__` / `__setstate__`)
- `conftest.py` with shared test fixtures
- `py.typed` marker for PEP 561 compliance
- `logging` module usage for non-user-facing output
- Ruff and pytest configuration in `pyproject.toml`
- CI test matrix expanded to Python 3.10, 3.11, 3.12, 3.13
- `train_timeout` / `predict_timeout` parameters to prevent subprocess hangs
- CI ruff linting and pytest-cov coverage reporting
- CI concurrency groups to cancel redundant workflow runs
- Error path tests (corrupted JSON, missing files, empty datasets, timeout validation)
- Input validation: `cat_features` bounds check, `monotone_constraints` length, `eval_period >= 1`
- `load_model()` validates required JSON keys (`model_info`, `trees`, `features`)
- `cross_validate()` now calls `_validate_params()` and `_validate_fit_inputs()` before running
- Feature name sanitization (rejects commas, newlines, null bytes)
- Executable bit check in binary discovery with `chmod +x` hint
- Model JSON serialization cache for faster repeated `predict()` calls
- NaN handling tests, MAE/Huber/Quantile loss tests
- sklearn integration tests (`cross_val_score`, `Pipeline`)
- Multiclass staged_predict and classes_ attribute tests
- Makefile with `test`, `lint`, `coverage`, `build-binaries`, `install` targets
- `.pre-commit-config.yaml` for ruff hooks
- `MANIFEST.in` for proper sdist packaging

### Fixed
- `_array_to_csv`: `isinstance(val, float)` now also matches `np.floating` (numpy 2.x compat)
- `_array_to_csv`: numeric NaN check uses `float(val)` cast to avoid `TypeError` on non-float dtypes
- `quantize_features`: `np.clip(bins, 0, 255)` prevents silent uint8 overflow at 256 bins
- `fit(y=None)` on raw arrays now raises clear `ValueError` instead of cryptic `IndexError`
- `get_shap_values()`: checks SHAP output file exists before reading
- `group_col` mutation in `fit()` now wrapped in `try/finally` to prevent state leakage on error
- `CatBoostMLXClassifier.fit(pool)`: correctly extracts `classes_` from Pool labels instead of `None`
- `staged_predict`: uses `_get_loss_type()` to split parameterized loss strings (e.g. `tweedie:1.5` → `tweedie`) so `apply_link` applies the correct transform
- `load_model()`: restores `n_features_in_` and `feature_names_in_` from model JSON so `feature_importances_` and sklearn validation work after loading
- PTY verbose mode: reads stderr before `proc.wait()` to prevent deadlock on large error output
- Validation for `bagging_temperature`, `mvs_reg`, `max_onehot_size`, `ctr_prior` parameters

### Changed
- `_HAS_SKLEARN` removed from `__all__` (still accessible via `catboost_mlx.core._HAS_SKLEARN`)
- `__version__` now reads from package metadata instead of hardcoded duplication
- `cross_validate()` reuses `_build_train_args()` to reduce code duplication
- `_to_numpy()` consolidated into `_utils.py` (imported by both `core.py` and `pool.py`)
- `Pool` no longer copies data unnecessarily (uses `np.ascontiguousarray` instead of `.copy()`)
- `cross_validate()` docstring expanded with full parameter and return value documentation
- `pyproject.toml`: ruff target-version aligned to `py39`, added Python 3.9/3.11 classifiers, `scikit-learn` added to dev deps
- Fixed 36 ruff lint violations across all Python modules
- Disabled inherited upstream CatBoost CI workflows that always fail in this fork

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
