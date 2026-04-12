"""
test_nanobind_parity.py -- Parity and edge-case tests for the nanobind in-process path.

Verifies that the nanobind path (_fit_nanobind) produces results equivalent
to the subprocess path (csv_train/csv_predict), and tests edge cases specific
to the nanobind integration.

Run with: pytest tests/test_nanobind_parity.py -v
"""

import os
import unittest.mock

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor
import catboost_mlx.core as _core_module


def _has_nanobind():
    """Check if nanobind extension is available."""
    try:
        from catboost_mlx import _core  # noqa: F401
        return True
    except ImportError:
        return False


def _has_binaries():
    """Check if subprocess binaries exist."""
    return (os.path.isfile(os.path.join(BINARY_PATH, "csv_train"))
            and os.path.isfile(os.path.join(BINARY_PATH, "csv_predict")))


requires_nanobind = pytest.mark.skipif(
    not _has_nanobind(), reason="nanobind extension not compiled")
requires_binaries = pytest.mark.skipif(
    not _has_binaries(), reason="csv_train/csv_predict binaries not found")
requires_both = pytest.mark.skipif(
    not (_has_nanobind() and _has_binaries()),
    reason="Need both nanobind extension and subprocess binaries for parity tests")


def _train_nanobind(model_class, X, y, **kwargs):
    """Train using the nanobind path (default when available)."""
    model = model_class(binary_path=BINARY_PATH, **kwargs)
    model.fit(X, y)
    return model


def _train_subprocess(model_class, X, y, **kwargs):
    """Train using the subprocess path by temporarily disabling nanobind."""
    model = model_class(binary_path=BINARY_PATH, **kwargs)
    with unittest.mock.patch.object(_core_module, '_HAS_NANOBIND', False):
        model.fit(X, y)
    return model


# =============================================================================
# Parity tests: nanobind vs subprocess produce equivalent results
# =============================================================================

@requires_both
class TestParity:
    """Compare nanobind and subprocess paths for equivalent output."""

    def test_rmse_predictions_match(self):
        """RMSE regression predictions should be close between both paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, 60)

        m_nb = _train_nanobind(CatBoostMLXRegressor, X, y,
                               iterations=20, depth=3, learning_rate=0.1, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXRegressor, X, y,
                                 iterations=20, depth=3, learning_rate=0.1, random_seed=42)

        pred_nb = m_nb.predict(X)
        pred_sp = m_sp.predict(X)
        np.testing.assert_allclose(pred_nb, pred_sp, atol=1e-4,
                                   err_msg="RMSE predictions diverge between paths")

    def test_logloss_predictions_match(self):
        """Binary classification predictions should match between paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 3)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)

        m_nb = _train_nanobind(CatBoostMLXClassifier, X, y,
                               iterations=20, depth=3, learning_rate=0.1, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXClassifier, X, y,
                                 iterations=20, depth=3, learning_rate=0.1, random_seed=42)

        pred_nb = m_nb.predict(X)
        pred_sp = m_sp.predict(X)
        np.testing.assert_array_equal(pred_nb, pred_sp,
                                      err_msg="Logloss class predictions differ")

    def test_multiclass_predictions_match(self):
        """Multiclass predictions should match between paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(90, 3)
        y = (X[:, 0] * 3).astype(int).clip(0, 2).astype(float)

        m_nb = _train_nanobind(CatBoostMLXClassifier, X, y,
                               iterations=15, depth=3, learning_rate=0.1, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXClassifier, X, y,
                                 iterations=15, depth=3, learning_rate=0.1, random_seed=42)

        pred_nb = m_nb.predict(X)
        pred_sp = m_sp.predict(X)
        np.testing.assert_array_equal(pred_nb, pred_sp,
                                      err_msg="Multiclass predictions differ")

    def test_loss_history_match(self):
        """Loss history should be close between paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 50)

        m_nb = _train_nanobind(CatBoostMLXRegressor, X, y,
                               iterations=10, depth=3, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXRegressor, X, y,
                                 iterations=10, depth=3, random_seed=42)

        assert len(m_nb.train_loss_history) == 10
        assert len(m_sp.train_loss_history) == 10
        np.testing.assert_allclose(
            m_nb.train_loss_history, m_sp.train_loss_history, atol=1e-4,
            err_msg="Train loss history diverges")

    def test_feature_importance_match(self):
        """Feature importance should be close between paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 4)
        y = X[:, 0] * 3 + X[:, 1] + rng.normal(0, 0.1, 60)

        m_nb = _train_nanobind(CatBoostMLXRegressor, X, y,
                               iterations=20, depth=3, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXRegressor, X, y,
                                 iterations=20, depth=3, random_seed=42)

        imp_nb = m_nb.feature_importances_
        imp_sp = m_sp.feature_importances_
        # Both should identify f0 as most important
        assert np.argmax(imp_nb) == np.argmax(imp_sp), \
            "Most important feature differs between paths"

    def test_trees_built_match(self):
        """Number of trees should match between paths."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 2)
        y = rng.rand(50)

        m_nb = _train_nanobind(CatBoostMLXRegressor, X, y,
                               iterations=15, depth=3, random_seed=42)
        m_sp = _train_subprocess(CatBoostMLXRegressor, X, y,
                                 iterations=15, depth=3, random_seed=42)

        assert m_nb.tree_count_ == m_sp.tree_count_ == 15


# =============================================================================
# Nanobind-specific edge cases
# =============================================================================

@requires_nanobind
class TestNanobindEdgeCases:
    """Edge cases specific to the nanobind in-process path."""

    def test_single_feature(self):
        """Training with a single feature should work."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 1)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)
        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=10, depth=2)
        pred = model.predict(X)
        assert pred.shape == (30,)

    def test_many_features(self):
        """Training with many features should work."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 50)
        y = X[:, 0] + rng.normal(0, 0.1, 50)
        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=5, depth=3)
        pred = model.predict(X)
        assert pred.shape == (50,)

    def test_constant_target(self):
        """Training with constant target should not crash."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 2)
        y = np.ones(30) * 5.0
        model = _train_nanobind(CatBoostMLXRegressor, X, y, iterations=5)
        pred = model.predict(X)
        # All predictions should be close to 5.0
        np.testing.assert_allclose(pred, 5.0, atol=0.5)

    def test_integer_targets(self):
        """Integer target array should be handled (converted to float)."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = rng.randint(0, 3, 40)  # int64
        model = _train_nanobind(CatBoostMLXClassifier, X, y, iterations=10)
        pred = model.predict(X)
        assert set(pred).issubset({0, 1, 2})

    def test_float64_input(self):
        """float64 arrays should be safely downcast to float32."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3).astype(np.float64)
        y = rng.rand(40).astype(np.float64)
        model = _train_nanobind(CatBoostMLXRegressor, X, y, iterations=10)
        pred = model.predict(X)
        assert pred.shape == (40,)

    def test_fortran_order_input(self):
        """Fortran-ordered arrays should be converted to C-contiguous."""
        rng = np.random.RandomState(42)
        X = np.asfortranarray(rng.rand(40, 3))
        y = rng.rand(40)
        assert X.flags['F_CONTIGUOUS']
        model = _train_nanobind(CatBoostMLXRegressor, X, y, iterations=10)
        pred = model.predict(X)
        assert pred.shape == (40,)


# =============================================================================
# Nanobind categorical encoding
# =============================================================================

@requires_nanobind
class TestNanobindCategorical:
    """Test categorical feature encoding through the nanobind path."""

    def test_string_categorical(self):
        """String categorical features should be encoded to integers."""
        rng = np.random.RandomState(42)
        n = 40
        cats = np.array([f"cat_{i % 5}" for i in range(n)])
        f1 = rng.rand(n)
        X = np.column_stack([cats, f1])
        y = (np.array([i % 5 for i in range(n)]) > 2).astype(float)

        model = CatBoostMLXClassifier(
            iterations=10, depth=3, cat_features=[0], binary_path=BINARY_PATH
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (n,)

    def test_mixed_categorical_numeric(self):
        """Mix of categorical and numeric features."""
        rng = np.random.RandomState(42)
        n = 50
        cats = np.array([f"type_{i % 3}" for i in range(n)])
        f1 = rng.rand(n)
        f2 = rng.rand(n)
        X = np.column_stack([f1, cats, f2])
        y = rng.rand(n)

        model = CatBoostMLXRegressor(
            iterations=10, depth=3, cat_features=[1], binary_path=BINARY_PATH
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (n,)

    def test_high_cardinality_categorical(self):
        """Many unique categories should work."""
        rng = np.random.RandomState(42)
        n = 60
        cats = np.array([f"id_{i}" for i in range(n)])  # all unique
        f1 = rng.rand(n)
        X = np.column_stack([cats, f1])
        y = rng.rand(n)

        model = CatBoostMLXRegressor(
            iterations=5, depth=3, cat_features=[0], binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert model.tree_count_ == 5


# =============================================================================
# Nanobind eval_set and early stopping
# =============================================================================

@requires_nanobind
class TestNanobindEvalSet:
    """Test eval_set and early stopping through the nanobind path."""

    def test_eval_set_basic(self):
        """eval_set should be accepted and produce eval loss history."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 60)
        X_val = rng.rand(20, 3)
        y_val = X_val[:, 0] + rng.normal(0, 0.1, 20)

        model = CatBoostMLXRegressor(
            iterations=15, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        assert len(model.eval_loss_history) == 15
        assert len(model.train_loss_history) == 15

    def test_eval_set_early_stopping(self):
        """Early stopping with eval_set should stop before max iterations."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 80)
        # Use noise-only validation to trigger early stopping
        X_val = rng.rand(20, 3)
        y_val = rng.rand(20) * 100  # unrelated targets

        model = CatBoostMLXRegressor(
            iterations=200, depth=3, early_stopping_rounds=5,
            binary_path=BINARY_PATH
        )
        model.fit(X, y, eval_set=(X_val, y_val))
        assert model.tree_count_ < 200, "Early stopping should have triggered"

    def test_eval_fraction(self):
        """eval_fraction should split data and produce eval loss."""
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 100)

        model = CatBoostMLXRegressor(
            iterations=15, depth=3, eval_fraction=0.2,
            binary_path=BINARY_PATH
        )
        model.fit(X, y)
        assert len(model.train_loss_history) == 15
        assert len(model.eval_loss_history) == 15


# =============================================================================
# Nanobind loss history
# =============================================================================

@requires_nanobind
class TestNanobindLossHistory:
    """Test loss history tracking through the nanobind path."""

    def test_loss_history_length(self):
        """Loss history should have one entry per iteration."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = rng.rand(40)
        model = _train_nanobind(CatBoostMLXRegressor, X, y, iterations=25)
        assert len(model.train_loss_history) == 25

    def test_loss_history_decreasing(self):
        """Train loss should generally decrease over iterations."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 3)
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 80)
        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=30, depth=3)
        history = model.train_loss_history
        # First loss should be higher than last loss
        assert history[0] > history[-1], \
            f"Loss should decrease: first={history[0]:.4f} last={history[-1]:.4f}"

    def test_loss_history_no_eval_set(self):
        """Without eval_set, eval_loss_history should be empty."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = rng.rand(40)
        model = _train_nanobind(CatBoostMLXRegressor, X, y, iterations=10)
        assert len(model.train_loss_history) == 10
        assert len(model.eval_loss_history) == 0

    def test_loss_history_verbose_false(self):
        """Loss history should be populated even when verbose=False."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 2)
        y = rng.rand(40)
        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=10, verbose=False)
        assert len(model.train_loss_history) == 10


# =============================================================================
# Nanobind weights
# =============================================================================

@requires_nanobind
class TestNanobindWeights:
    """Test sample weights through the nanobind path."""

    def test_uniform_weights(self):
        """Uniform weights should produce same result as no weights."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 50)

        m_no_w = _train_nanobind(CatBoostMLXRegressor, X, y,
                                 iterations=10, depth=3, random_seed=42)
        m_w = CatBoostMLXRegressor(
            iterations=10, depth=3, random_seed=42, binary_path=BINARY_PATH
        )
        m_w.fit(X, y, sample_weight=np.ones(50))

        pred_no_w = m_no_w.predict(X)
        pred_w = m_w.predict(X)
        np.testing.assert_allclose(pred_no_w, pred_w, atol=1e-4)

    def test_nonuniform_weights(self):
        """Non-uniform weights should change predictions."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 2)
        y = rng.rand(50)

        w_heavy_first = np.ones(50)
        w_heavy_first[:10] = 10.0  # heavy weight on first 10

        w_heavy_last = np.ones(50)
        w_heavy_last[-10:] = 10.0  # heavy weight on last 10

        m1 = CatBoostMLXRegressor(
            iterations=20, depth=3, random_seed=42, binary_path=BINARY_PATH
        )
        m1.fit(X, y, sample_weight=w_heavy_first)

        m2 = CatBoostMLXRegressor(
            iterations=20, depth=3, random_seed=42, binary_path=BINARY_PATH
        )
        m2.fit(X, y, sample_weight=w_heavy_last)

        # Different weights should produce different predictions
        pred1 = m1.predict(X)
        pred2 = m2.predict(X)
        assert not np.allclose(pred1, pred2, atol=1e-6), \
            "Different weights should produce different predictions"


# =============================================================================
# Nanobind validation errors
# =============================================================================

@requires_nanobind
class TestNanobindValidation:
    """Test that validation errors are raised correctly in the nanobind path."""

    def test_nan_mode_forbidden(self):
        """nan_mode=forbidden with NaN data should raise RuntimeError."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        X[0, 0] = np.nan
        y = rng.rand(20)
        model = CatBoostMLXRegressor(
            iterations=5, nan_mode="forbidden", binary_path=BINARY_PATH
        )
        with pytest.raises(RuntimeError, match="NaN"):
            model.fit(X, y)

    def test_empty_dataset(self):
        """Empty dataset should raise ValueError."""
        X = np.zeros((0, 3))
        y = np.zeros(0)
        model = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="empty"):
            model.fit(X, y)

    def test_eval_set_mutual_exclusivity(self):
        """eval_set + eval_fraction > 0 should raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 2)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(
            iterations=10, eval_fraction=0.2, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            model.fit(X, y, eval_set=(X[:5], y[:5]))

    def test_eval_set_feature_mismatch(self):
        """eval_set with wrong feature count should raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3)
        y = rng.rand(20)
        model = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="features"):
            model.fit(X, y, eval_set=(rng.rand(10, 5), rng.rand(10)))


# =============================================================================
# Nanobind save/load roundtrip
# =============================================================================

@requires_nanobind
class TestNanobindSaveLoad:
    """Test that nanobind-trained models save and load correctly."""

    def test_save_load_roundtrip(self, tmp_path):
        """Model trained via nanobind should survive save/load."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = X[:, 0] + rng.normal(0, 0.1, 50)

        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=15, depth=3)
        pred_before = model.predict(X)

        path = str(tmp_path / "model.json")
        model.save_model(path)

        loaded = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        loaded.load_model(path)
        pred_after = loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)

    def test_save_load_preserves_metadata(self, tmp_path):
        """Saved model should preserve tree count and feature names."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3)
        y = rng.rand(50)

        model = _train_nanobind(CatBoostMLXRegressor, X, y,
                                iterations=10, depth=3)
        path = str(tmp_path / "model.json")
        model.save_model(path)

        loaded = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        loaded.load_model(path)

        assert loaded.tree_count_ == 10
        assert len(loaded.feature_names_in_) == 3
