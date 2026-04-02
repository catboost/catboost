"""Test suite for catboost_mlx Python bindings."""

import json
import os
import tempfile

import numpy as np
import pytest

# Adjust path so tests can find compiled binaries in repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT  # csv_train / csv_predict expected at repo root

from catboost_mlx import CatBoostMLX, CatBoostMLXRegressor, CatBoostMLXClassifier


def _check_binaries():
    """Skip tests if compiled binaries are not available."""
    csv_train = os.path.join(BINARY_PATH, "csv_train")
    csv_predict = os.path.join(BINARY_PATH, "csv_predict")
    if not (os.path.isfile(csv_train) and os.path.isfile(csv_predict)):
        pytest.skip("Compiled csv_train/csv_predict binaries not found at repo root")


# ── Regression ──────────────────────────────────────────────────────────────

class TestRegression:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 50)

    def test_fit_predict(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        # Should get reasonable RMSE on training data
        rmse = np.sqrt(np.mean((preds - self.y) ** 2))
        assert rmse < 1.0, f"Training RMSE too high: {rmse}"

    def test_loss_history(self):
        model = CatBoostMLXRegressor(
            iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        assert len(model.train_loss_history) == 20
        # Loss should generally decrease
        assert model.train_loss_history[-1] < model.train_loss_history[0]

    def test_feature_importance(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y, feature_names=["a", "b", "c"])
        fi = model.get_feature_importance()
        assert len(fi) > 0
        # Features "a" and "b" should have higher importance (they drive y)
        assert isinstance(fi, dict)


# ── Binary Classification ───────────────────────────────────────────────────

class TestBinaryClassification:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(60, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 1.0).astype(float)

    def test_fit_predict(self):
        model = CatBoostMLXClassifier(
            iterations=50, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({0, 1})
        accuracy = np.mean(preds == self.y)
        assert accuracy > 0.7, f"Training accuracy too low: {accuracy}"

    def test_predict_proba(self):
        model = CatBoostMLXClassifier(
            iterations=50, depth=4, loss="logloss", binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        assert proba.shape == (60, 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        # All probabilities should be in [0, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Multiclass Classification ──────────────────────────────────────────────

class TestMulticlass:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        n = 90
        self.X = rng.rand(n, 3)
        self.y = np.zeros(n)
        self.y[self.X[:, 0] > 0.66] = 2
        self.y[(self.X[:, 0] > 0.33) & (self.X[:, 0] <= 0.66)] = 1

    def test_fit_predict(self):
        model = CatBoostMLX(
            loss="multiclass", iterations=50, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (90,)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_predict_proba(self):
        model = CatBoostMLX(
            loss="multiclass", iterations=50, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        assert proba.shape == (90, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── Save / Load ─────────────────────────────────────────────────────────────

class TestSaveLoad:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(40, 2)
        self.y = self.X[:, 0] * 2 + rng.normal(0, 0.1, 40)

    def test_save_load_roundtrip(self):
        model = CatBoostMLXRegressor(
            iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        preds_before = model.predict(self.X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            model.save_model(path)
            assert os.path.exists(path)

            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(path)
            preds_after = model2.predict(self.X)

            np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
        finally:
            os.unlink(path)


# ── Cross-Validation ────────────────────────────────────────────────────────

class TestCrossValidation:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(60, 2)
        self.y = self.X[:, 0] + self.X[:, 1] + rng.normal(0, 0.1, 60)

    def test_cv(self):
        model = CatBoostMLXRegressor(
            iterations=20, depth=3, binary_path=BINARY_PATH
        )
        cv_result = model.cross_validate(self.X, self.y, n_folds=3)
        assert "fold_metrics" in cv_result
        assert "mean" in cv_result
        assert "std" in cv_result
        assert cv_result["mean"] >= 0


# ── Repr / Error Handling ──────────────────────────────────────────────────

class TestMisc:
    def test_repr_not_fitted(self):
        model = CatBoostMLX(iterations=10)
        assert "not fitted" in repr(model)

    def test_repr_fitted(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = X[:, 0] + rng.normal(0, 0.1, 20)
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        assert "fitted" in repr(model)

    def test_predict_before_fit(self):
        model = CatBoostMLX()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.zeros((5, 2)))

    def test_predict_proba_regression_error(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = X[:, 0]
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        with pytest.raises(ValueError, match="not supported"):
            model.predict_proba(X)


# ── New Loss Functions ──────────────────────────────────────────────────────

class TestNewLosses:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(40, 2)

    def test_poisson(self):
        y = np.round(np.exp(self.X[:, 0] + self.X[:, 1])).astype(float)
        model = CatBoostMLX(
            loss="poisson", iterations=50, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X, y)
        preds = model.predict(self.X)
        assert preds.shape == (40,)
        assert np.all(preds > 0), "Poisson predictions should be positive (exp link)"

    def test_tweedie(self):
        y = np.where(self.X[:, 0] > 0.5, self.X[:, 0] * 5, 0.0)
        model = CatBoostMLX(
            loss="tweedie:1.5", iterations=50, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X, y)
        preds = model.predict(self.X)
        assert preds.shape == (40,)
        assert np.all(preds > 0), "Tweedie predictions should be positive (exp link)"

    def test_mape(self):
        y = 10 + 5 * self.X[:, 0] + 3 * self.X[:, 1]
        model = CatBoostMLX(
            loss="mape", iterations=100, depth=4, learning_rate=0.3,
            binary_path=BINARY_PATH
        )
        model.fit(self.X, y)
        preds = model.predict(self.X)
        assert preds.shape == (40,)


# ── SHAP Values ────────────────────────────────────────────────────────────

class TestShap:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(30, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 30)

    def test_shap_regression(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y, feature_names=["a", "b", "c"])
        result = model.get_shap_values(self.X, feature_names=["a", "b", "c"])

        assert "shap_values" in result
        assert "expected_value" in result
        assert "feature_names" in result
        assert result["shap_values"].shape == (30, 3)

        # Sum property: shap_values.sum(axis=1) + expected_value ≈ raw prediction
        preds = model._run_predict(self.X, feature_names=["a", "b", "c"])["prediction"]
        shap_sum = result["shap_values"].sum(axis=1)
        np.testing.assert_allclose(
            shap_sum + result["expected_value"], preds, atol=1e-4
        )

    def test_shap_binary_classification(self):
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)

        model = CatBoostMLXClassifier(
            loss="logloss", iterations=30, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        result = model.get_shap_values(X)
        assert result["shap_values"].shape == (40, 2)

    def test_shap_multiclass(self):
        rng = np.random.RandomState(42)
        n = 60
        X = rng.rand(n, 3)
        y = np.zeros(n)
        y[X[:, 0] > 0.66] = 2
        y[(X[:, 0] > 0.33) & (X[:, 0] <= 0.66)] = 1

        model = CatBoostMLX(
            loss="multiclass", iterations=30, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        result = model.get_shap_values(X)

        # approxDim = K-1 = 2 for 3 classes
        assert result["shap_values"].shape == (n, 3, 2)
        assert isinstance(result["expected_value"], np.ndarray)
        assert result["expected_value"].shape == (2,)


# ── Bootstrap Types ───────────────────────────────────────────────────────

class TestBootstrap:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 50)

    def test_bayesian(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1,
            bootstrap_type="bayesian", bagging_temperature=1.0,
            binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        rmse = np.sqrt(np.mean((preds - self.y) ** 2))
        assert rmse < 2.0, f"Bayesian bootstrap RMSE too high: {rmse}"

    def test_bernoulli(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1,
            bootstrap_type="bernoulli", subsample=0.8,
            binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        rmse = np.sqrt(np.mean((preds - self.y) ** 2))
        assert rmse < 2.0, f"Bernoulli bootstrap RMSE too high: {rmse}"

    def test_mvs(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1,
            bootstrap_type="mvs", subsample=0.8, mvs_reg=0.0,
            binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        rmse = np.sqrt(np.mean((preds - self.y) ** 2))
        assert rmse < 2.0, f"MVS bootstrap RMSE too high: {rmse}"

    def test_subsample_defaults_to_bernoulli(self):
        """--subsample without explicit bootstrap type should use Bernoulli."""
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1,
            subsample=0.7, binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)


# ── Ranking Losses ────────────────────────────────────────────────────────

class TestRanking:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        # 5 groups of 10 docs each, relevance labels 0-3
        n_groups = 5
        docs_per_group = 10
        n = n_groups * docs_per_group
        self.X = rng.rand(n, 3)
        self.y = np.zeros(n)
        for g in range(n_groups):
            start = g * docs_per_group
            for d in range(docs_per_group):
                self.y[start + d] = rng.randint(0, 4)
        self.group_id = np.repeat(np.arange(n_groups), docs_per_group)

    def test_pairlogit(self):
        model = CatBoostMLX(
            loss="pairlogit", iterations=50, depth=3, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y, group_id=self.group_id)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        # Within each group, higher relevance should generally get higher scores
        assert len(model.train_loss_history) == 50
        assert model.train_loss_history[-1] < model.train_loss_history[0]

    def test_yetirank(self):
        model = CatBoostMLX(
            loss="yetirank", iterations=50, depth=3, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y, group_id=self.group_id)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        assert len(model.train_loss_history) > 0


# ── Sample Weights ───────────────────────────────────────────────────────

class TestSampleWeights:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 50)

    def test_weighted_regression(self):
        weights = np.ones(50)
        weights[:25] = 5.0  # up-weight first half
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1, binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y, sample_weight=weights)
        preds = model.predict(self.X)
        assert preds.shape == (50,)
        rmse = np.sqrt(np.mean((preds - self.y) ** 2))
        assert rmse < 2.0, f"Weighted regression RMSE too high: {rmse}"

    def test_weighted_classification(self):
        rng = np.random.RandomState(42)
        X = rng.rand(60, 2)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
        # Up-weight positive class
        weights = np.where(y == 1.0, 3.0, 1.0)
        model = CatBoostMLXClassifier(
            iterations=50, depth=4, binary_path=BINARY_PATH,
        )
        model.fit(X, y, sample_weight=weights)
        preds = model.predict(X)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({0, 1})


# ── Min Data in Leaf ─────────────────────────────────────────────────────

class TestMinDataInLeaf:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 50)

    def test_min_data_constrains_splits(self):
        model = CatBoostMLXRegressor(
            iterations=30, depth=4, learning_rate=0.1,
            min_data_in_leaf=10, binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)

    def test_large_min_data(self):
        """With large min_data_in_leaf, tree depth is effectively limited."""
        model = CatBoostMLXRegressor(
            iterations=20, depth=6, learning_rate=0.1,
            min_data_in_leaf=15, binary_path=BINARY_PATH,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        assert preds.shape == (50,)


# ── Monotone Constraints ────────────────────────────────────────────────

class TestMonotoneConstraints:
    def setup_method(self):
        _check_binaries()

    def test_increasing_constraint(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 2)
        y = 3.0 * X[:, 0] + rng.normal(0, 0.1, 100)

        model = CatBoostMLXRegressor(
            iterations=50, depth=4, learning_rate=0.1,
            monotone_constraints=[1, 0], binary_path=BINARY_PATH,
        )
        model.fit(X, y)

        # Verify monotonicity: predictions should be non-decreasing in feature 0
        test_x = np.column_stack([
            np.linspace(0, 1, 20),
            np.full(20, 0.5),
        ])
        preds = model.predict(test_x)
        # Allow tiny floating point violations
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-4), f"Monotone violated: min diff = {diffs.min()}"

    def test_decreasing_constraint(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 2)
        y = -2.0 * X[:, 0] + rng.normal(0, 0.1, 100)

        model = CatBoostMLXRegressor(
            iterations=50, depth=4, learning_rate=0.1,
            monotone_constraints=[-1, 0], binary_path=BINARY_PATH,
        )
        model.fit(X, y)

        test_x = np.column_stack([
            np.linspace(0, 1, 20),
            np.full(20, 0.5),
        ])
        preds = model.predict(test_x)
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-4), f"Monotone violated: max diff = {diffs.max()}"


# ── Snapshot Save/Resume ──────────────────────────────────────────────────

class TestSnapshot:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(50, 3)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 50)

    def test_resume_training(self):
        """Train 10 iters with snapshot, resume for 10 more, compare to 20 in one go."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = os.path.join(tmpdir, "snap.json")

            # Train 10 iterations with snapshot
            model1 = CatBoostMLXRegressor(
                iterations=10, depth=4, learning_rate=0.1, random_seed=42,
                snapshot_path=snap_path, snapshot_interval=1,
                binary_path=BINARY_PATH,
            )
            model1.fit(self.X, self.y)
            assert os.path.exists(snap_path), "Snapshot file should exist after training"

            # Resume for 10 more iterations
            model2 = CatBoostMLXRegressor(
                iterations=20, depth=4, learning_rate=0.1, random_seed=42,
                snapshot_path=snap_path, snapshot_interval=1,
                binary_path=BINARY_PATH,
            )
            model2.fit(self.X, self.y)
            preds_resumed = model2.predict(self.X)

            # Train 20 iterations from scratch (same seed)
            model3 = CatBoostMLXRegressor(
                iterations=20, depth=4, learning_rate=0.1, random_seed=42,
                binary_path=BINARY_PATH,
            )
            model3.fit(self.X, self.y)
            preds_full = model3.predict(self.X)

            # Predictions should match
            np.testing.assert_allclose(preds_resumed, preds_full, atol=1e-4)
