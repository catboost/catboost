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
