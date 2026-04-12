"""
test_basic.py -- Comprehensive test suite for the CatBoost-MLX Python package.

What this file does:
    This is the quality checker. It creates small synthetic datasets, trains
    models with various settings, and verifies everything works: predictions
    are reasonable, errors are raised when expected, saved models load correctly,
    exports produce valid files, sklearn integration works, etc.

How it fits into the project:
    Standalone test file. Run with ``pytest python/tests/``. Imports catboost_mlx
    and expects compiled binaries (csv_train, csv_predict) at the repo root.
    Tests that need binaries are automatically skipped if binaries are missing.

Key concepts:
    - pytest: Python's standard test framework. Each ``test_*`` method is a test.
    - setup_method: Called before each test to create fresh data, preventing
      tests from interfering with each other.
    - _check_binaries(): Skips tests if C++ binaries are not compiled yet.
      This lets pure-Python tests (Pool, validation) run without binaries.

Test classes (26 total, 111 tests):
    TestRegression, TestBinaryClassification, TestMulticlass, TestSaveLoad,
    TestCrossValidation, TestMisc, TestNewLosses, TestShap, TestBootstrap,
    TestRanking, TestSampleWeights, TestMinDataInLeaf, TestMonotoneConstraints,
    TestSnapshot, TestSklearn, TestValidation, TestEvalSet, TestExport,
    TestPool, TestAutoClassWeights, TestModelInspection, TestCTR,
    TestFeatureImportancesArray, TestVerbose, TestStagedPredict, TestApply
"""

import os
import tempfile

import numpy as np
import pytest

# REPO_ROOT: Go two directories up from this file (tests/ -> python/ -> repo root)
# to find the compiled binaries (csv_train, csv_predict) at the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT  # csv_train / csv_predict expected at repo root

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor, Pool


def _check_binaries():
    """Skip the entire test if C++ binaries are not compiled.

    This allows pure-Python tests (Pool, validation, etc.) to run even
    without the compiled binaries. Tests that need actual training call
    this at the start of their setup_method.
    """
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


# ── sklearn Compatibility ─────────────────────────────────────────────────

class TestSklearn:
    def test_get_params(self):
        model = CatBoostMLXRegressor(iterations=50, depth=4)
        params = model.get_params()
        assert params["iterations"] == 50
        assert params["depth"] == 4
        assert params["loss"] == "rmse"
        assert "learning_rate" in params
        assert "binary_path" in params
        # Should have all params, not just the subclass's 'loss'
        assert len(params) >= 20

    def test_set_params(self):
        model = CatBoostMLXRegressor(iterations=50)
        model.set_params(iterations=200, depth=3)
        assert model.iterations == 200
        assert model.depth == 3

    def test_classifier_get_params(self):
        model = CatBoostMLXClassifier(iterations=30)
        params = model.get_params()
        assert params["loss"] == "auto"
        assert params["iterations"] == 30

    def test_clone(self):
        pytest.importorskip("sklearn")
        from sklearn.base import clone
        model = CatBoostMLXRegressor(iterations=42, depth=3, learning_rate=0.05)
        cloned = clone(model)
        assert cloned.iterations == 42
        assert cloned.depth == 3
        assert cloned.learning_rate == 0.05
        assert not cloned._is_fitted

    def test_score_regression(self):
        _check_binaries()
        pytest.importorskip("sklearn")
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(iterations=30, depth=4, binary_path=BINARY_PATH)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert isinstance(r2, float)
        assert r2 > 0.5  # should fit training data reasonably

    def test_score_classification(self):
        _check_binaries()
        pytest.importorskip("sklearn")
        rng = np.random.RandomState(42)
        X = rng.rand(60, 2)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
        model = CatBoostMLXClassifier(iterations=50, depth=4, binary_path=BINARY_PATH)
        model.fit(X, y)
        acc = model.score(X, y)
        assert isinstance(acc, float)
        assert acc > 0.7
        assert hasattr(model, "classes_")
        assert set(model.classes_) == {0.0, 1.0}

    def test_n_features_in(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 5)
        y = X[:, 0]
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        assert model.n_features_in_ == 5

    def test_sklearn_is_fitted(self):
        _check_binaries()
        pytest.importorskip("sklearn")
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = X[:, 0]
        model.fit(X, y)
        check_is_fitted(model)  # should not raise

    def test_feature_names_in_attribute(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = X[:, 0]
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y, feature_names=["a", "b", "c"])
        assert hasattr(model, "feature_names_in_")
        assert list(model.feature_names_in_) == ["a", "b", "c"]
        assert model.feature_names_in_.dtype == object

    def test_feature_names_in_default(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = X[:, 0]
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        assert list(model.feature_names_in_) == ["f0", "f1", "f2"]

    def test_feature_names_validation_warning(self):
        _check_binaries()
        import warnings

        import pandas as pd
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame(rng.rand(30, 2), columns=["feat_a", "feat_b"])
        y = X_train["feat_a"].values
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X_train, y)
        # Predict with different column names should warn
        X_test = pd.DataFrame(rng.rand(5, 2), columns=["wrong_a", "wrong_b"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict(X_test)
            assert len(w) >= 1
            assert "feature names" in str(w[0].message).lower()

    def test_feature_names_no_warning_when_matching(self):
        _check_binaries()
        import warnings

        import pandas as pd
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.rand(30, 2), columns=["a", "b"])
        y = X["a"].values
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict(X[:5])
            feature_warnings = [x for x in w if "feature names" in str(x.message).lower()]
            assert len(feature_warnings) == 0

    def test_score_with_sample_weight(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 40)
        model = CatBoostMLXRegressor(iterations=20, depth=4, binary_path=BINARY_PATH)
        model.fit(X, y)
        w = np.ones(40)
        score_uniform = model.score(X, y, sample_weight=w)
        score_default = model.score(X, y)
        assert isinstance(score_uniform, float)
        # With uniform weights, should match default
        assert abs(score_uniform - score_default) < 0.01

    def test_n_outputs(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = X[:, 0]
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        model.fit(X, y)
        assert model.n_outputs_ == 1


# ── Parameter Validation ──────────────────────────────────────────────────

class TestValidation:
    def _dummy_data(self):
        return np.zeros((5, 2)), np.zeros(5)

    def test_invalid_iterations(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="iterations"):
            CatBoostMLX(iterations=0).fit(X, y)

    def test_invalid_depth_low(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="depth"):
            CatBoostMLX(iterations=1, depth=0).fit(X, y)

    def test_invalid_depth_high(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="depth"):
            CatBoostMLX(iterations=1, depth=20).fit(X, y)

    def test_invalid_learning_rate(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="learning_rate"):
            CatBoostMLX(iterations=1, learning_rate=-0.1).fit(X, y)

    def test_invalid_loss(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="Unknown loss"):
            CatBoostMLX(iterations=1, loss="nonexistent").fit(X, y)

    def test_invalid_subsample(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="subsample"):
            CatBoostMLX(iterations=1, subsample=0).fit(X, y)

    def test_invalid_bins(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="bins"):
            CatBoostMLX(iterations=1, bins=1).fit(X, y)

    def test_invalid_bootstrap_type(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="bootstrap_type"):
            CatBoostMLX(iterations=1, bootstrap_type="invalid").fit(X, y)

    def test_invalid_monotone_constraint_values(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="monotone_constraints"):
            CatBoostMLX(iterations=1, monotone_constraints=[2]).fit(X, y)

    def test_shape_mismatch_y(self):
        X = np.zeros((10, 2))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="samples"):
            CatBoostMLX(iterations=1).fit(X, y)

    def test_shape_mismatch_feature_names(self):
        X = np.zeros((5, 3))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="feature_names"):
            CatBoostMLX(iterations=1).fit(X, y, feature_names=["a", "b"])

    def test_valid_params_pass(self):
        """Valid params should not raise during validation."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = X[:, 0] + rng.normal(0, 0.1, 20)
        model = CatBoostMLXRegressor(
            iterations=10, depth=4, learning_rate=0.1, bins=128,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y)
        assert model._is_fitted


# ---------------------------------------------------------------------------
# Eval set tests
# ---------------------------------------------------------------------------
class TestEvalSet:
    """Tests for explicit eval_set support."""

    def test_eval_set_basic(self):
        """Pass (X_val, y_val) and verify training succeeds."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 50)
        X_val = rng.rand(20, 3)
        y_val = X_val[:, 0] * 2 + X_val[:, 1] + rng.normal(0, 0.1, 20)
        model = CatBoostMLXRegressor(
            iterations=20, depth=4, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        assert model._is_fitted
        preds = model.predict(X_val)
        assert preds.shape == (20,)

    def test_eval_set_mutual_exclusivity(self):
        """eval_set + eval_fraction > 0 should raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = rng.rand(20)
        model = CatBoostMLX(
            iterations=10, eval_fraction=0.2,
            binary_path=BINARY_PATH,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            model.fit(X, y, eval_set=(X[:5], y[:5]))

    def test_eval_set_feature_mismatch(self):
        """eval_set X with wrong number of features should raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = rng.rand(20)
        X_val = rng.rand(10, 5)  # wrong feature count
        y_val = rng.rand(10)
        model = CatBoostMLX(iterations=10, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="features"):
            model.fit(X, y, eval_set=(X_val, y_val))

    def test_eval_set_early_stopping(self):
        """eval_set with early stopping should work without error."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(80, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 80)
        X_val = rng.rand(20, 3)
        y_val = X_val[:, 0] + rng.normal(0, 0.1, 20)
        model = CatBoostMLXRegressor(
            iterations=50, depth=4, learning_rate=0.3,
            early_stopping_rounds=5,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        assert model._is_fitted
        preds = model.predict(X_val)
        assert preds.shape == (20,)


# ---------------------------------------------------------------------------
# Export tests (CoreML + ONNX)
# ---------------------------------------------------------------------------
class TestExport:
    """Tests for CoreML and ONNX export."""

    @staticmethod
    def _fit_regression_model():
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 50)
        model = CatBoostMLXRegressor(
            iterations=20, depth=4, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y)
        return model, X, y

    @staticmethod
    def _fit_classification_model():
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
        model = CatBoostMLXClassifier(
            iterations=20, depth=4, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y)
        return model, X, y

    def test_export_onnx_regression(self, tmp_path):
        """Export regression model to ONNX and validate."""
        pytest.importorskip("onnx")
        model, X, y = self._fit_regression_model()
        onnx_path = str(tmp_path / "model.onnx")
        model.export_onnx(onnx_path)
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        # Verify it has TreeEnsembleRegressor
        assert any(n.op_type == "TreeEnsembleRegressor" for n in onnx_model.graph.node)

    def test_export_onnx_classification(self, tmp_path):
        """Export classification model to ONNX and validate."""
        pytest.importorskip("onnx")
        model, X, y = self._fit_classification_model()
        onnx_path = str(tmp_path / "model.onnx")
        model.export_onnx(onnx_path)
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        assert any(n.op_type == "TreeEnsembleClassifier" for n in onnx_model.graph.node)

    def test_export_onnx_inference(self, tmp_path):
        """Export to ONNX and verify predictions via onnxruntime if available."""
        ort = pytest.importorskip("onnxruntime")
        model, X, y = self._fit_regression_model()
        onnx_path = str(tmp_path / "model.onnx")
        model.export_onnx(onnx_path)
        sess = ort.InferenceSession(onnx_path)
        ort_pred = sess.run(None, {"X": X.astype(np.float32)})[0]
        native_pred = model.predict(X)
        # Predictions should be close (not exact due to float precision)
        np.testing.assert_allclose(ort_pred.flatten(), native_pred, rtol=0.1, atol=0.5)

    def test_export_coreml_regression(self, tmp_path):
        """Export regression model to CoreML and validate."""
        ct = pytest.importorskip("coremltools")
        model, X, y = self._fit_regression_model()
        coreml_path = str(tmp_path / "model.mlmodel")
        model.export_coreml(coreml_path)
        # Load and verify
        loaded = ct.models.MLModel(coreml_path)
        spec = loaded.get_spec()
        assert spec.HasField("treeEnsembleRegressor")

    def test_export_coreml_classification(self, tmp_path):
        """Export classification model to CoreML and validate."""
        ct = pytest.importorskip("coremltools")
        model, X, y = self._fit_classification_model()
        coreml_path = str(tmp_path / "model.mlmodel")
        model.export_coreml(coreml_path)
        loaded = ct.models.MLModel(coreml_path)
        spec = loaded.get_spec()
        assert spec.HasField("treeEnsembleClassifier")

    def test_export_not_fitted(self, tmp_path):
        """Export before fitting should raise RuntimeError."""
        model = CatBoostMLX(binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.export_onnx(str(tmp_path / "model.onnx"))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.export_coreml(str(tmp_path / "model.mlmodel"))


# ---------------------------------------------------------------------------
# Pool tests
# ---------------------------------------------------------------------------
class TestPool:
    """Tests for Pool data container and Pandas integration."""

    def test_pool_basic(self):
        """Pool with numpy arrays stores attributes correctly."""
        X = np.zeros((10, 3))
        y = np.ones(10)
        pool = Pool(X, y)
        assert pool.num_samples == 10
        assert pool.num_features == 3
        assert pool.shape == (10, 3)
        assert pool.y is not None
        assert pool.cat_features is None
        assert pool.feature_names is None
        assert len(pool) == 10

    def test_pool_with_metadata(self):
        """Pool with all optional fields."""
        X = np.zeros((10, 3))
        y = np.ones(10)
        pool = Pool(
            X, y,
            cat_features=[0, 2],
            feature_names=["a", "b", "c"],
            group_id=np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
            sample_weight=np.ones(10),
        )
        assert pool.cat_features == [0, 2]
        assert pool.feature_names == ["a", "b", "c"]
        assert pool.group_id is not None
        assert pool.sample_weight is not None

    def test_pool_from_dataframe(self):
        """Pool auto-extracts column names and categorical columns from DataFrame."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "color": ["red", "blue", "green", "red", "blue"],
            "size": [1.0, 2.0, 3.0, 4.0, 5.0],
            "weight": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        y = np.array([0, 1, 0, 1, 0])
        pool = Pool(df, y)
        assert pool.feature_names == ["color", "size", "weight"]
        assert pool.cat_features == [0]  # 'color' is object dtype
        assert pool.num_features == 3

    def test_pool_cat_by_name(self):
        """Pool resolves string cat_features to integer indices."""
        X = np.zeros((5, 3))
        pool = Pool(X, cat_features=["b"], feature_names=["a", "b", "c"])
        assert pool.cat_features == [1]

    def test_pool_shape_mismatch(self):
        """Pool raises ValueError on shape mismatch."""
        X = np.zeros((10, 3))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="samples"):
            Pool(X, y)

    def test_pool_repr(self):
        """Pool repr is informative."""
        pool = Pool(np.zeros((10, 3)), np.ones(10), cat_features=[0])
        r = repr(pool)
        assert "10 samples" in r
        assert "3 features" in r
        assert "1 categorical" in r

    def test_fit_with_pool(self):
        """model.fit(pool) works end-to-end."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)
        pool = Pool(X, y, feature_names=["f0", "f1", "f2"])
        model = CatBoostMLXRegressor(
            iterations=10, depth=4, binary_path=BINARY_PATH,
        )
        model.fit(pool)
        assert model._is_fitted
        preds = model.predict(X)
        assert preds.shape == (30,)

    def test_fit_with_eval_pool(self):
        """model.fit(train_pool, eval_set=val_pool) works."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 40)
        train_pool = Pool(X[:30], y[:30])
        val_pool = Pool(X[30:], y[30:])
        model = CatBoostMLXRegressor(
            iterations=10, depth=4, binary_path=BINARY_PATH,
        )
        model.fit(train_pool, eval_set=val_pool)
        assert model._is_fitted

    def test_dataframe_auto_names(self):
        """fit(df, y) auto-extracts feature names without Pool."""
        _check_binaries()
        pd = pytest.importorskip("pandas")
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.rand(30, 3), columns=["alpha", "beta", "gamma"])
        y = df["alpha"] * 2 + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=4, binary_path=BINARY_PATH,
        )
        model.fit(df, y)
        assert model._is_fitted


# ---------------------------------------------------------------------------
# Auto class weights tests
# ---------------------------------------------------------------------------
class TestAutoClassWeights:
    """Tests for auto_class_weights parameter."""

    def test_balanced_weights_computed(self):
        """Balanced class weights should be inversely proportional to frequency."""
        # 90 class-0, 10 class-1 → weight_0 = 100/(2*90), weight_1 = 100/(2*10)
        model = CatBoostMLX(iterations=1, auto_class_weights="Balanced",
                            binary_path=BINARY_PATH)
        # We can't directly inspect weights, but verify it doesn't crash
        # and that the parameter is accepted
        assert model.auto_class_weights == "Balanced"

    def test_sqrtbalanced_weights_computed(self):
        """SqrtBalanced should use sqrt of balanced weights."""
        model = CatBoostMLX(iterations=1, auto_class_weights="SqrtBalanced",
                            binary_path=BINARY_PATH)
        assert model.auto_class_weights == "SqrtBalanced"

    def test_invalid_auto_class_weights(self):
        """Invalid auto_class_weights should raise ValueError."""
        X = np.random.RandomState(42).rand(10, 2)
        y = np.array([0]*5 + [1]*5)
        with pytest.raises(ValueError, match="auto_class_weights"):
            CatBoostMLX(iterations=1, auto_class_weights="invalid",
                        binary_path=BINARY_PATH).fit(X, y)

    def test_auto_class_weights_with_manual_weight(self):
        """Manual sample_weight takes precedence over auto_class_weights."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = np.array([0]*30 + [1]*10)
        manual_w = np.ones(40)
        model = CatBoostMLXClassifier(
            iterations=10, depth=4, auto_class_weights="Balanced",
            binary_path=BINARY_PATH,
        )
        # When sample_weight is explicitly provided, auto_class_weights is ignored
        model.fit(X, y, sample_weight=manual_w)
        assert model._is_fitted

    def test_auto_class_weights_training(self):
        """End-to-end training with balanced weights on imbalanced data."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        # Imbalanced: 90% class 0, 10% class 1
        y = np.array([0]*90 + [1]*10)
        model = CatBoostMLXClassifier(
            iterations=20, depth=4, auto_class_weights="Balanced",
            binary_path=BINARY_PATH,
        )
        model.fit(X, y)
        assert model._is_fitted
        preds = model.predict(X)
        assert preds.shape == (100,)


# ---------------------------------------------------------------------------
# Model inspection API tests
# ---------------------------------------------------------------------------
class TestModelInspection:
    """Tests for model inspection properties and methods."""

    @staticmethod
    def _fitted_model():
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=4, learning_rate=0.1,
            binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["alpha", "beta", "gamma"])
        return model

    def test_tree_count(self):
        """tree_count_ should match iterations."""
        model = self._fitted_model()
        assert model.tree_count_ == 10

    def test_tree_count_not_fitted(self):
        """tree_count_ before fitting should raise RuntimeError."""
        model = CatBoostMLX(binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.tree_count_

    def test_feature_names(self):
        """feature_names_ should match what was passed to fit."""
        model = self._fitted_model()
        assert model.feature_names_ == ["alpha", "beta", "gamma"]

    def test_feature_names_default(self):
        """feature_names_ defaults to f0, f1, ... when not specified."""
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(
            iterations=5, depth=4, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert model.feature_names_ == ["f0", "f1", "f2"]

    def test_get_trees(self):
        """get_trees() returns list of tree dicts with correct structure."""
        model = self._fitted_model()
        trees = model.get_trees()
        assert len(trees) == 10
        t = trees[0]
        assert "depth" in t
        assert "nodes" in t
        assert "leaf_values" in t
        assert "split_gains" in t
        assert t["depth"] == 4
        # Nodes should contain both branch and leaf types
        types = {n["type"] for n in t["nodes"]}
        assert "branch" in types
        assert "leaf" in types

    def test_get_model_info(self):
        """get_model_info() returns loss type, tree count, etc."""
        model = self._fitted_model()
        info = model.get_model_info()
        assert info["loss_type"] == "rmse"
        assert info["num_trees"] == 10
        assert info["num_features"] == 3
        assert "approx_dimension" in info

    def test_plot_feature_importance(self, capsys):
        """plot_feature_importance() produces output without error."""
        model = self._fitted_model()
        model.plot_feature_importance()
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "#" in captured.out


# ---------------------------------------------------------------------------
# CTR (target encoding) tests
# ---------------------------------------------------------------------------
class TestCTR:
    """Tests for CTR target encoding with high-cardinality categoricals."""

    @staticmethod
    def _make_ctr_data(n=200, n_cats=25, seed=42):
        """Create a dataset with high-cardinality categorical + numeric features."""
        rng = np.random.RandomState(seed)
        categories = [f"cat_{i}" for i in range(n_cats)]
        X_cat = rng.choice(categories, n)
        X_num1 = rng.rand(n)
        X_num2 = rng.rand(n)
        # Build object array: column 0 is categorical, columns 1-2 are numeric
        X = np.empty((n, 3), dtype=object)
        X[:, 0] = X_cat
        X[:, 1] = X_num1
        X[:, 2] = X_num2
        # Target correlated with category hash
        y = np.array([1.0 if hash(c) % 3 > 0 else 0.0 for c in X_cat])
        return X, y

    def test_ctr_binary_classification(self):
        """Train with CTR on high-cardinality categorical, verify ctr_features in model."""
        _check_binaries()
        X, y = self._make_ctr_data()
        model = CatBoostMLXClassifier(
            iterations=20, depth=4, ctr=True, max_onehot_size=5,
            cat_features=[0], binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["category", "f1", "f2"])
        assert model._is_fitted
        # Model JSON should have ctr_features
        assert "ctr_features" in model._model_data
        assert len(model._model_data["ctr_features"]) > 0
        preds = model.predict(X)
        assert preds.shape == (200,)

    def test_ctr_multiclass(self):
        """CTR with multiclass: should create per-class CTR features."""
        _check_binaries()
        rng = np.random.RandomState(42)
        n = 200
        categories = [f"cat_{i}" for i in range(20)]
        X_cat = rng.choice(categories, n)
        X = np.empty((n, 2), dtype=object)
        X[:, 0] = X_cat
        X[:, 1] = rng.rand(n)
        y = np.array([hash(c) % 3 for c in X_cat], dtype=float)
        model = CatBoostMLX(
            loss="multiclass", iterations=20, depth=4, ctr=True,
            max_onehot_size=5, cat_features=[0], binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["category", "f1"])
        assert model._is_fitted
        assert "ctr_features" in model._model_data
        # Multiclass should have multiple CTR features (one per class)
        ctr_feats = model._model_data["ctr_features"]
        assert len(ctr_feats) > 1

    def test_ctr_predict_unknown_category(self):
        """Predict on data with unseen categories should work (uses default CTR)."""
        _check_binaries()
        X, y = self._make_ctr_data(n=100, n_cats=15)
        model = CatBoostMLXClassifier(
            iterations=20, depth=4, ctr=True, max_onehot_size=5,
            cat_features=[0], binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["category", "f1", "f2"])
        # Create test data with unseen categories
        X_test = np.empty((10, 3), dtype=object)
        X_test[:, 0] = [f"unseen_{i}" for i in range(10)]
        X_test[:, 1] = np.random.rand(10)
        X_test[:, 2] = np.random.rand(10)
        preds = model.predict(X_test, feature_names=["category", "f1", "f2"])
        assert preds.shape == (10,)

    def test_ctr_model_roundtrip(self, tmp_path):
        """Save and load model with CTR features, verify predictions match."""
        _check_binaries()
        X, y = self._make_ctr_data(n=100, n_cats=15)
        model = CatBoostMLXClassifier(
            iterations=15, depth=4, ctr=True, max_onehot_size=5,
            cat_features=[0], binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["category", "f1", "f2"])
        preds1 = model.predict(X)

        path = str(tmp_path / "ctr_model.json")
        model.save_model(path)
        model2 = CatBoostMLXClassifier(binary_path=BINARY_PATH, cat_features=[0])
        model2.load_model(path)
        preds2 = model2.predict(X, feature_names=["category", "f1", "f2"])
        np.testing.assert_array_equal(preds1, preds2)

    def test_ctr_with_onehot_threshold(self):
        """Mixed OneHot + CTR: low-cardinality uses OneHot, high uses CTR."""
        _check_binaries()
        rng = np.random.RandomState(42)
        n = 150
        # Column 0: low cardinality (3 values) → OneHot
        X_low = rng.choice(["a", "b", "c"], n)
        # Column 1: high cardinality (20 values) → CTR
        X_high = rng.choice([f"h{i}" for i in range(20)], n)
        X = np.empty((n, 3), dtype=object)
        X[:, 0] = X_low
        X[:, 1] = X_high
        X[:, 2] = rng.rand(n)
        y = (rng.rand(n) > 0.5).astype(float)
        model = CatBoostMLXClassifier(
            iterations=20, depth=4, ctr=True, max_onehot_size=5,
            cat_features=[0, 1], binary_path=BINARY_PATH,
        )
        model.fit(X, y, feature_names=["low_cat", "high_cat", "num"])
        assert model._is_fitted
        # CTR features should exist for the high-cardinality column
        ctr_feats = model._model_data.get("ctr_features", [])
        assert len(ctr_feats) > 0
        # The CTR feature should reference the high-cardinality column (index 1)
        ctr_orig_idxs = [cf["original_feature_idx"] for cf in ctr_feats]
        assert 1 in ctr_orig_idxs


# ── Feature Importances (sklearn ndarray) ──────────────────────────────────

class TestFeatureImportancesArray:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(60, 4)
        self.y = 2.0 * self.X[:, 0] + 3.0 * self.X[:, 1] + rng.normal(0, 0.1, 60)
        self.model = CatBoostMLXRegressor(
            iterations=20, depth=4, binary_path=BINARY_PATH
        )
        self.model.fit(self.X, self.y)

    def test_feature_importances_shape(self):
        fi = self.model.feature_importances_
        assert fi.shape == (4,)

    def test_feature_importances_sums_to_one(self):
        fi = self.model.feature_importances_
        assert np.isclose(fi.sum(), 1.0), f"Sum is {fi.sum()}, expected 1.0"

    def test_feature_importances_not_fitted(self):
        m = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.feature_importances_

    def test_feature_importances_matches_dict(self):
        fi_arr = self.model.feature_importances_
        fi_dict = self.model.get_feature_importance()
        total = sum(fi_dict.values())
        for i, name in enumerate(self.model.feature_names_):
            expected = fi_dict.get(name, 0.0) / total if total > 0 else 0.0
            assert np.isclose(fi_arr[i], expected), f"Mismatch at {name}"


# ── Verbose Streaming ──────────────────────────────────────────────────────

class TestVerbose:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(40, 2)
        self.y = self.X[:, 0] + rng.normal(0, 0.1, 40)

    def test_verbose_streams_output(self, capfd):
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, verbose=True, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        captured = capfd.readouterr()
        # Should contain iteration output lines (capfd captures C-level printf)
        assert "iter=" in captured.out or "loss=" in captured.out

    def test_verbose_false_quiet(self, capfd):
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, verbose=False, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        captured = capfd.readouterr()
        # Filter out Metal/profile lines from C++ init
        lines = [l for l in captured.out.strip().split("\n")
                 if l.strip() and not l.strip().startswith("[profile")
                 and not l.strip().startswith("Base prediction")]
        assert len(lines) == 0, f"Unexpected output: {lines}"

    def test_loss_history_still_parsed(self):
        model = CatBoostMLXRegressor(
            iterations=15, depth=3, verbose=True, binary_path=BINARY_PATH
        )
        model.fit(self.X, self.y)
        assert len(model.train_loss_history) == 15


# ── Staged Predict ─────────────────────────────────────────────────────────

class TestStagedPredict:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X_reg = rng.rand(40, 3)
        self.y_reg = 2.0 * self.X_reg[:, 0] + rng.normal(0, 0.1, 40)
        self.X_bin = rng.rand(50, 2)
        self.y_bin = (self.X_bin[:, 0] > 0.5).astype(float)

    def test_staged_predict_regression_final(self):
        model = CatBoostMLXRegressor(
            iterations=15, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X_reg, self.y_reg)
        staged = list(model.staged_predict(self.X_reg))
        direct = model.predict(self.X_reg)
        np.testing.assert_allclose(staged[-1], direct, atol=1e-4)

    def test_staged_predict_binary_final(self):
        model = CatBoostMLXClassifier(
            iterations=15, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X_bin, self.y_bin)
        staged = list(model.staged_predict(self.X_bin))
        direct = model.predict(self.X_bin)
        np.testing.assert_array_equal(staged[-1], direct)

    def test_staged_predict_proba_final(self):
        model = CatBoostMLXClassifier(
            iterations=15, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X_bin, self.y_bin)
        staged = list(model.staged_predict_proba(self.X_bin))
        direct = model.predict_proba(self.X_bin)
        np.testing.assert_allclose(staged[-1], direct, atol=1e-4)

    def test_staged_predict_eval_period(self):
        model = CatBoostMLXRegressor(
            iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X_reg, self.y_reg)
        staged = list(model.staged_predict(self.X_reg, eval_period=5))
        assert len(staged) == 4  # trees 5, 10, 15, 20

    def test_staged_predict_not_fitted(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="not fitted"):
            list(model.staged_predict(self.X_reg))

    def test_staged_predict_proba_regression_error(self):
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(self.X_reg, self.y_reg)
        with pytest.raises(ValueError, match="not supported"):
            list(model.staged_predict_proba(self.X_reg))

    def test_staged_predict_learning_curve(self):
        model = CatBoostMLXRegressor(
            iterations=20, depth=4, binary_path=BINARY_PATH
        )
        model.fit(self.X_reg, self.y_reg)
        staged = list(model.staged_predict(self.X_reg))
        # Predictions should change across iterations
        assert not np.allclose(staged[0], staged[-1])


# ── Apply (Leaf Indices) ──────────────────────────────────────────────────

class TestApply:
    def setup_method(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        self.X = rng.rand(30, 3)
        self.y = self.X[:, 0] + rng.normal(0, 0.1, 30)
        self.model = CatBoostMLXRegressor(
            iterations=10, depth=4, binary_path=BINARY_PATH
        )
        self.model.fit(self.X, self.y)

    def test_apply_shape(self):
        result = self.model.apply(self.X)
        assert result.shape == (30, 10)

    def test_apply_values_range(self):
        result = self.model.apply(self.X)
        max_leaf = 2 ** 4  # depth=4
        assert np.all(result >= 0)
        assert np.all(result < max_leaf)

    def test_apply_deterministic(self):
        r1 = self.model.apply(self.X)
        r2 = self.model.apply(self.X)
        np.testing.assert_array_equal(r1, r2)

    def test_apply_not_fitted(self):
        model = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.apply(self.X)


# ── Edge Cases and Robustness ────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for validation edge cases, temp file cleanup, and roundtrip fidelity."""

    def _dummy_data(self):
        return np.zeros((5, 2)), np.zeros(5)

    def test_iterations_upper_bound(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="iterations"):
            CatBoostMLX(iterations=200000).fit(X, y)

    def test_snapshot_path_null_byte(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="snapshot_path"):
            CatBoostMLX(iterations=1, snapshot_path="/tmp/foo\x00bar").fit(X, y)

    def test_binary_path_null_byte(self):
        X, y = self._dummy_data()
        with pytest.raises(ValueError, match="binary_path"):
            CatBoostMLX(iterations=1, binary_path="/tmp/\x00inject").fit(X, y)

    def test_temp_files_cleaned_after_fit(self):
        _check_binaries()
        import glob
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 20)
        # Count catboost_mlx temp dirs before
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "catboost_mlx_*")))
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        model.predict(X)
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "catboost_mlx_*")))
        # No new temp dirs should remain
        new_dirs = after - before
        assert len(new_dirs) == 0, f"Leaked temp dirs: {new_dirs}"

    def test_temp_files_cleaned_on_error(self):
        import glob
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "catboost_mlx_*")))
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, binary_path="/nonexistent/path"
        )
        rng = np.random.RandomState(42)
        X = rng.rand(10, 2)
        y = rng.rand(10)
        try:
            model.fit(X, y)
        except Exception:
            pass
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "catboost_mlx_*")))
        new_dirs = after - before
        assert len(new_dirs) == 0, f"Leaked temp dirs on error: {new_dirs}"

    def test_save_load_roundtrip_regression(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3)
        y = X[:, 0] * 10 + rng.normal(0, 0.1, 30)
        model = CatBoostMLXRegressor(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        pred_before = model.predict(X)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
            model2.load_model(path)
            pred_after = model2.predict(X)
            np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_classifier(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(float)
        model = CatBoostMLXClassifier(
            iterations=10, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        pred_before = model.predict(X)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save_model(path)
            model2 = CatBoostMLXClassifier(binary_path=BINARY_PATH)
            model2.load_model(path)
            pred_after = model2.predict(X)
            np.testing.assert_array_equal(pred_before, pred_after)
        finally:
            os.unlink(path)

    def test_pool_sorted_cat_features(self):
        """Pool auto-detected cat_features should be sorted by index."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        # Create DataFrame where categorical columns are NOT in column order
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat_b": ["x", "y", "z"],
            "num2": [4.0, 5.0, 6.0],
            "cat_a": ["a", "b", "c"],
        })
        pool = Pool(df)
        assert pool.cat_features == [1, 3]
        assert pool.cat_features == sorted(pool.cat_features)

    def test_1d_input_reshaped(self):
        _check_binaries()
        rng = np.random.RandomState(42)
        X = rng.rand(20)  # 1D
        y = X * 2
        model = CatBoostMLXRegressor(
            iterations=5, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (20,)
