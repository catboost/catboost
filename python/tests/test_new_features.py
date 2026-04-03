"""
test_new_features.py -- Tests for new features added in the production readiness update.

Tests: load classmethod, Pool in predict methods, pickle/joblib support,
metadata (__version__, __all__), and _utils.py deduplication.
"""

import json
import os
import pickle
import tempfile

import numpy as np
import pytest

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor, Pool

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT


def _check_binaries():
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled binaries not found")


# ── Metadata ─────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_version_exists(self):
        import catboost_mlx
        assert isinstance(catboost_mlx.__version__, str)
        assert len(catboost_mlx.__version__) > 0

    def test_has_sklearn_not_in_all(self):
        import catboost_mlx
        assert "_HAS_SKLEARN" not in catboost_mlx.__all__

    def test_public_exports(self):
        import catboost_mlx
        expected = {"CatBoostMLX", "CatBoostMLXRegressor", "CatBoostMLXClassifier", "Pool"}
        assert set(catboost_mlx.__all__) == expected


# ── Load Classmethod ─────────────────────────────────────────────────────────

class TestLoadClassmethod:
    def test_load_regressor(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            loaded = CatBoostMLXRegressor.load(path, binary_path=BINARY_PATH)
            assert loaded._is_fitted
            preds_after = loaded.predict(X)
            np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
        finally:
            os.unlink(path)

    def test_load_classifier(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = (X[:, 0] > 0.5).astype(float)
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            loaded = CatBoostMLXClassifier.load(path, binary_path=BINARY_PATH)
            assert isinstance(loaded, CatBoostMLXClassifier)
            assert loaded._is_fitted
        finally:
            os.unlink(path)

    def test_load_base_class(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = rng.rand(20)
        model = CatBoostMLX(iterations=5, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            loaded = CatBoostMLX.load(path, binary_path=BINARY_PATH)
            assert isinstance(loaded, CatBoostMLX)
        finally:
            os.unlink(path)


# ── Pool in Predict ──────────────────────────────────────────────────────────

class TestPoolPredict:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(40, 3)
        self.y = self.X[:, 0] * 2 + rng.normal(0, 0.1, 40)
        self.model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        self.model.fit(self.X, self.y)

    def test_predict_with_pool(self):
        pool = Pool(self.X)
        preds_pool = self.model.predict(pool)
        preds_array = self.model.predict(self.X)
        np.testing.assert_allclose(preds_pool, preds_array, atol=1e-6)

    def test_apply_with_pool(self):
        pool = Pool(self.X)
        result_pool = self.model.apply(pool)
        result_array = self.model.apply(self.X)
        np.testing.assert_array_equal(result_pool, result_array)

    def test_staged_predict_with_pool(self):
        pool = Pool(self.X)
        staged_pool = list(self.model.staged_predict(pool))
        staged_array = list(self.model.staged_predict(self.X))
        np.testing.assert_allclose(staged_pool[-1], staged_array[-1], atol=1e-6)


class TestPoolPredictClassification:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(40, 2)
        self.y = (self.X[:, 0] > 0.5).astype(float)
        self.model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        self.model.fit(self.X, self.y)

    def test_predict_proba_with_pool(self):
        pool = Pool(self.X)
        proba_pool = self.model.predict_proba(pool)
        proba_array = self.model.predict_proba(self.X)
        np.testing.assert_allclose(proba_pool, proba_array, atol=1e-6)

    def test_staged_predict_proba_with_pool(self):
        pool = Pool(self.X)
        staged_pool = list(self.model.staged_predict_proba(pool))
        staged_array = list(self.model.staged_predict_proba(self.X))
        np.testing.assert_allclose(staged_pool[-1], staged_array[-1], atol=1e-6)


# ── Pickle / Joblib ──────────────────────────────────────────────────────────

class TestPickle:
    def test_pickle_roundtrip_regression(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_before = model.predict(X)

        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2._is_fitted
        assert model2._model_path is None
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_pickle_roundtrip_classifier(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = (X[:, 0] > 0.5).astype(float)
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        preds_before = model.predict(X)

        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        preds_after = model2.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_pickle_unfitted(self):
        model = CatBoostMLXRegressor(iterations=50, depth=4)
        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert not model2._is_fitted
        assert model2.iterations == 50
        assert model2.depth == 4

    def test_pickle_preserves_loss_history(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        history_before = model.train_loss_history

        data = pickle.dumps(model)
        model2 = pickle.loads(data)
        assert model2.train_loss_history == history_before


# ── _utils.py ────────────────────────────────────────────────────────────────

class TestUtils:
    def test_to_numpy_from_utils(self):
        from catboost_mlx._utils import _to_numpy
        arr = _to_numpy([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_to_numpy_passthrough(self):
        from catboost_mlx._utils import _to_numpy
        arr = np.array([1.0, 2.0])
        result = _to_numpy(arr)
        assert result is arr


# ── Error Paths ─────────────────────────────────────────────────────────────

class TestErrorPaths:
    """Tests for error conditions and edge cases."""

    def test_corrupted_model_json(self):
        """Loading a corrupted model JSON should raise a meaningful error."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{corrupted json content!!")
            path = f.name
        try:
            model = CatBoostMLX()
            with pytest.raises(json.JSONDecodeError):
                model.load_model(path)
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        """Loading from a nonexistent file should raise FileNotFoundError."""
        model = CatBoostMLX()
        with pytest.raises(FileNotFoundError):
            model.load_model("/tmp/catboost_mlx_nonexistent_model_xyz.json")

    def test_predict_not_fitted(self):
        """Predict before fit should raise RuntimeError."""
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.zeros((5, 2)))

    def test_save_model_not_fitted(self):
        """Saving an unfitted model should raise RuntimeError."""
        model = CatBoostMLXRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save_model("/tmp/catboost_mlx_test_save.json")

    def test_save_model_unwritable_path(self):
        """Saving to an unwritable path should raise an OS error."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises((OSError, PermissionError)):
            model.save_model("/nonexistent_dir_xyz/model.json")

    def test_empty_dataset(self):
        """Training with an empty dataset should raise an error."""
        _check_binaries()
        X = np.zeros((0, 3))
        y = np.zeros(0)
        model = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X, y)

    def test_train_timeout_validation(self):
        """Invalid train_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="train_timeout"):
            model = CatBoostMLX(iterations=1, train_timeout=-1)
            model._validate_params()

    def test_predict_timeout_validation(self):
        """Invalid predict_timeout should raise ValueError."""
        with pytest.raises(ValueError, match="predict_timeout"):
            model = CatBoostMLX(iterations=1, predict_timeout=0)
            model._validate_params()

    def test_pool_no_unnecessary_copy(self):
        """Pool should not copy data when input is already a C-contiguous numpy array."""
        arr = np.random.rand(10, 3)
        pool = Pool(arr)
        assert pool.X is arr or pool.X.base is arr

    def test_load_model_invalid_keys(self):
        """Loading valid JSON without required model keys should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"hello": "world"}, f)
            path = f.name
        try:
            model = CatBoostMLX()
            with pytest.raises(ValueError, match="missing required keys"):
                model.load_model(path)
        finally:
            os.unlink(path)

    def test_fit_y_none_raises(self):
        """fit() with y=None on raw arrays should raise ValueError."""
        model = CatBoostMLXRegressor(iterations=5)
        with pytest.raises(ValueError, match="y is required"):
            model.fit(np.random.rand(10, 3))


# ── NaN Handling ────────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_nan_in_numeric_features(self):
        """Training with NaN in numeric features should work with nan_mode='min'."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = rng.rand(50)
        X[0, 0] = np.nan
        X[5, 1] = np.nan
        X[10, 2] = np.nan
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, nan_mode="min", binary_path=BINARY_PATH
        )
        model.fit(X, y)
        # Predict on clean data to verify model works
        X_clean = rng.rand(20, 3)
        preds = model.predict(X_clean)
        assert preds.shape == (20,)
        assert not np.any(np.isnan(preds))

    def test_nan_mode_forbidden_raises(self):
        """nan_mode='forbidden' should cause an error when NaN values exist."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        X[0, 0] = np.nan
        y = rng.rand(20)
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, nan_mode="forbidden", binary_path=BINARY_PATH
        )
        with pytest.raises(RuntimeError):
            model.fit(X, y)

    def test_nan_predict(self):
        """Predicting on data with NaN should work when model was trained with nan_mode='min'."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X_train = rng.rand(50, 3)
        y_train = X_train[:, 0] + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, nan_mode="min", binary_path=BINARY_PATH
        )
        model.fit(X_train, y_train)
        X_test = rng.rand(10, 3)
        X_test[0, 0] = np.nan
        preds = model.predict(X_test)
        assert preds.shape == (10,)
        assert not np.any(np.isnan(preds))


# ── Additional Losses ───────────────────────────────────────────────────────

class TestAdditionalLosses:
    def test_mae_loss(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(
            loss="mae", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_huber_loss(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(
            loss="huber:1.5", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_quantile_loss(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(
            loss="quantile:0.5", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)


# ── Sklearn Integration ─────────────────────────────────────────────────────

class TestSklearnIntegration:
    def test_cross_val_score(self):
        _check_binaries()
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        from sklearn.model_selection import cross_val_score
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 60)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_pipeline(self):
        _check_binaries()
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 60)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (60,)


# ── Multiclass Extended ─────────────────────────────────────────────────────

class TestMulticlassExtended:
    def test_staged_predict_multiclass(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = (X[:, 0] * 3).astype(int).astype(float)  # classes 0, 1, 2
        model = CatBoostMLXClassifier(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        stages = list(model.staged_predict(X))
        assert len(stages) == 10
        assert stages[-1].shape == (60,)
        assert set(stages[-1]).issubset({0.0, 1.0, 2.0})

    def test_multiclass_classifier_classes(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = (X[:, 0] * 3).astype(int).astype(float)
        model = CatBoostMLXClassifier(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert hasattr(model, "classes_")
        np.testing.assert_array_equal(np.sort(model.classes_), [0.0, 1.0, 2.0])


# ── Input Validation ────────────────────────────────────────────────────────

class TestValidation:
    def test_cat_features_out_of_bounds(self):
        """cat_features index beyond feature count should raise ValueError."""
        _check_binaries()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, cat_features=[999], binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="out of bounds"):
            model.fit(X, y)

    def test_monotone_constraints_wrong_length(self):
        """monotone_constraints with wrong length should raise ValueError."""
        _check_binaries()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, monotone_constraints=[1, -1],
            binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="monotone_constraints"):
            model.fit(X, y)

    def test_eval_period_zero(self):
        """eval_period=0 should raise ValueError."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = rng.rand(30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises(ValueError, match="eval_period"):
            list(model.staged_predict(X, eval_period=0))

    def test_cross_validate_validates_params(self):
        """cross_validate should validate params before running."""
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(
            iterations=5, depth=0, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="depth"):
            model.cross_validate(X, y, n_folds=3)

    def test_feature_name_with_comma(self):
        """Feature names with commas should raise ValueError."""
        _check_binaries()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=3, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="invalid characters"):
            model.fit(X, y, feature_names=["price,usd", "size", "age"])

    def test_invalid_bagging_temperature(self):
        """Negative bagging_temperature should raise ValueError."""
        with pytest.raises(ValueError, match="bagging_temperature"):
            model = CatBoostMLX(iterations=1, bagging_temperature=-1)
            model._validate_params()

    def test_invalid_ctr_prior(self):
        """Non-positive ctr_prior should raise ValueError."""
        with pytest.raises(ValueError, match="ctr_prior"):
            model = CatBoostMLX(iterations=1, ctr_prior=0)
            model._validate_params()


# ── Classifier + Pool ───────────────────────────────────────────────────────

class TestClassifierPool:
    def test_classifier_fit_pool_classes(self):
        """Classifier.fit(pool) should correctly set classes_ from Pool labels."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = (X[:, 0] > 0.5).astype(float)
        pool = Pool(X, y=y)
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(pool)
        assert hasattr(model, "classes_")
        np.testing.assert_array_equal(np.sort(model.classes_), [0.0, 1.0])

    def test_classifier_fit_pool_predict_proba(self):
        """Classifier.fit(pool) then predict_proba should work."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = (X[:, 0] > 0.5).astype(float)
        pool = Pool(X, y=y)
        model = CatBoostMLXClassifier(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(pool)
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2)


# ── Load Model Attributes ──────────────────────────────────────────────────

class TestLoadModelAttributes:
    def test_feature_importances_after_load(self):
        """feature_importances_ should work after load_model()."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y)
        fi_before = model.feature_importances_

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            loaded = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            loaded.load_model(path)
            assert hasattr(loaded, "n_features_in_")
            assert loaded.n_features_in_ == 3
            fi_after = loaded.feature_importances_
            np.testing.assert_allclose(fi_before, fi_after, atol=1e-6)
        finally:
            os.unlink(path)

    def test_feature_names_in_after_load(self):
        """feature_names_in_ should be populated after load_model()."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=5, depth=3, binary_path=BINARY_PATH)
        model.fit(X, y, feature_names=["a", "b"])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            loaded = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            loaded.load_model(path)
            assert hasattr(loaded, "feature_names_in_")
            assert list(loaded.feature_names_in_) == ["a", "b"]
        finally:
            os.unlink(path)


# ── Staged Predict Link Function ────────────────────────────────────────────

class TestStagedPredictLink:
    def test_staged_predict_poisson_matches_predict(self):
        """staged_predict final output should match predict for Poisson loss."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = np.abs(rng.rand(50) * 10) + 0.1  # positive targets for Poisson
        model = CatBoostMLXRegressor(
            loss="poisson", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        staged = list(model.staged_predict(X))
        np.testing.assert_allclose(staged[-1], preds, atol=1e-4)

    def test_staged_predict_tweedie_matches_predict(self):
        """staged_predict final output should match predict for Tweedie loss."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = np.abs(rng.rand(50) * 10) + 0.1
        model = CatBoostMLXRegressor(
            loss="tweedie:1.5", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        preds = model.predict(X)
        staged = list(model.staged_predict(X))
        np.testing.assert_allclose(staged[-1], preds, atol=1e-4)


# ── Cross-Validate Robustness ──────────────────────────────────────────────

class TestCrossValidateRobust:
    def test_cv_returns_correct_fold_count(self):
        """cross_validate should return exactly n_folds fold metrics."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 60)
        model = CatBoostMLXRegressor(iterations=10, depth=3, binary_path=BINARY_PATH)
        result = model.cross_validate(X, y, n_folds=3)
        assert len(result["fold_metrics"]) == 3
        assert result["mean"] > 0
        assert result["std"] >= 0
