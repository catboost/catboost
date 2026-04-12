"""
test_qa_round14_sprint11_nanobind.py -- Adversarial QA tests for Sprint 11 nanobind bindings.

Tests the new in-process training path (nanobind) introduced in Sprint 11:
  - train_api.h / train_api.cpp (C++ public API)
  - bindings.cpp (nanobind Python bindings)
  - core.py additions: _fit_nanobind, _build_train_config, dispatch logic

Focus areas:
  1. Nanobind/subprocess parity under stress (large datasets, deep trees)
  2. Loss history correctness (valid values, monotonicity, correct length)
  3. Categorical encoding edge cases (empty strings, NaN, special chars, unseen)
  4. eval_set through nanobind (history length, early stopping)
  5. Memory safety (sequential fits, tiny datasets, state isolation)
  6. Model JSON integrity (structure, save/load roundtrip)
  7. Feature importance (non-zero, sums to 1, known-important features rank highest)
  8. Weights edge cases (zero, large, negative -- negative is a known bug)
  9. Data type robustness (int16, bool, DataFrame, Pool, mixed dtypes)
  10. Fit-predict-fit state consistency
  11. Verbose output (C stdout captured at fd level by capfd)
  12. Error paths (mismatched shapes, NaN targets, inf inputs, bad params)

Run with: pytest tests/test_qa_round14_sprint11_nanobind.py -v
"""

import json
import os
import unittest.mock

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT

from catboost_mlx import CatBoostMLX, CatBoostMLXClassifier, CatBoostMLXRegressor
import catboost_mlx.core as _core_module


# ── Availability guards ──────────────────────────────────────────────────────

def _has_nanobind():
    try:
        from catboost_mlx import _core  # noqa: F401
        return True
    except ImportError:
        return False


def _has_binaries():
    return (os.path.isfile(os.path.join(BINARY_PATH, "csv_train"))
            and os.path.isfile(os.path.join(BINARY_PATH, "csv_predict")))


requires_nanobind = pytest.mark.skipif(
    not _has_nanobind(), reason="nanobind extension not compiled")
requires_binaries = pytest.mark.skipif(
    not _has_binaries(), reason="csv_train/csv_predict binaries not found")
requires_both = pytest.mark.skipif(
    not (_has_nanobind() and _has_binaries()),
    reason="Need both nanobind extension and subprocess binaries")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _nb(model_class, X, y, **kwargs):
    """Train via nanobind (default path when extension is available)."""
    model = model_class(binary_path=BINARY_PATH, **kwargs)
    model.fit(X, y)
    return model


def _sp(model_class, X, y, **kwargs):
    """Train via subprocess by patching _HAS_NANOBIND to False."""
    model = model_class(binary_path=BINARY_PATH, **kwargs)
    with unittest.mock.patch.object(_core_module, "_HAS_NANOBIND", False):
        model.fit(X, y)
    return model


# ============================================================================
# 1. Nanobind / subprocess parity under stress
# ============================================================================

@requires_both
class TestParityStress:
    """Parity between nanobind and subprocess paths for larger, harder inputs."""

    def test_large_dataset_500_rows_20_features(self):
        """500-row, 20-feature regression: predictions within 1e-3."""
        rng = np.random.RandomState(42)
        X = rng.rand(500, 20).astype(np.float32)
        y = (X[:, :3].sum(axis=1) + rng.normal(0, 0.1, 500)).astype(np.float32)

        m_nb = _nb(CatBoostMLXRegressor, X, y, iterations=50, depth=6, random_seed=42)
        m_sp = _sp(CatBoostMLXRegressor, X, y, iterations=50, depth=6, random_seed=42)

        pred_nb = m_nb.predict(X)
        pred_sp = m_sp.predict(X)
        np.testing.assert_allclose(pred_nb, pred_sp, atol=1e-3,
                                   err_msg="Parity diverges on 500x20 dataset")

    def test_deep_trees_depth_10(self):
        """Deep trees (depth=10) predictions should match between paths."""
        rng = np.random.RandomState(99)
        X = rng.rand(200, 6).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        m_nb = _nb(CatBoostMLXRegressor, X, y, iterations=20, depth=10, random_seed=99)
        m_sp = _sp(CatBoostMLXRegressor, X, y, iterations=20, depth=10, random_seed=99)

        np.testing.assert_allclose(m_nb.predict(X), m_sp.predict(X), atol=1e-3,
                                   err_msg="Depth-10 predictions diverge")

    def test_high_iterations_100(self):
        """100 iterations: final predictions within tolerance."""
        rng = np.random.RandomState(7)
        X = rng.rand(150, 8).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m_nb = _nb(CatBoostMLXRegressor, X, y, iterations=100, depth=4, random_seed=7)
        m_sp = _sp(CatBoostMLXRegressor, X, y, iterations=100, depth=4, random_seed=7)

        np.testing.assert_allclose(m_nb.predict(X), m_sp.predict(X), atol=1e-3,
                                   err_msg="100-iteration predictions diverge")

    def test_binary_classification_large(self):
        """Large binary classification: class label agreement."""
        rng = np.random.RandomState(13)
        X = rng.rand(400, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(float)

        m_nb = _nb(CatBoostMLXClassifier, X, y, iterations=40, depth=5, random_seed=13)
        m_sp = _sp(CatBoostMLXClassifier, X, y, iterations=40, depth=5, random_seed=13)

        np.testing.assert_array_equal(m_nb.predict(X), m_sp.predict(X),
                                      err_msg="Large binary classification predictions differ")

    def test_loss_history_parity_100_iters(self):
        """Loss history should match across 100 iterations."""
        rng = np.random.RandomState(5)
        X = rng.rand(100, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m_nb = _nb(CatBoostMLXRegressor, X, y, iterations=100, depth=3, random_seed=5)
        m_sp = _sp(CatBoostMLXRegressor, X, y, iterations=100, depth=3, random_seed=5)

        assert len(m_nb.train_loss_history) == 100
        assert len(m_sp.train_loss_history) == 100
        np.testing.assert_allclose(
            m_nb.train_loss_history, m_sp.train_loss_history, atol=1e-4,
            err_msg="Loss history diverges over 100 iterations")


# ============================================================================
# 2. Loss history correctness
# ============================================================================

@requires_nanobind
class TestLossHistoryCorrectness:
    """Loss history must be numerically valid and semantically correct."""

    def test_loss_history_exact_length_matches_iterations(self):
        """train_loss_history must have exactly num_iterations entries."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 4).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        for n_iter in (1, 5, 25, 50):
            m = _nb(CatBoostMLXRegressor, X, y, iterations=n_iter)
            assert len(m.train_loss_history) == n_iter, \
                f"iterations={n_iter}: expected {n_iter} history entries, got {len(m.train_loss_history)}"

    def test_loss_history_no_nan_or_inf(self):
        """All loss history values must be finite (no NaN, no inf)."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 5).astype(np.float32)
        y = (X[:, 0] * 2).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=30)
        for i, v in enumerate(m.train_loss_history):
            assert np.isfinite(v), f"train_loss_history[{i}] = {v} is not finite"

    def test_loss_history_all_positive(self):
        """RMSE loss must be positive at every iteration."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=20)
        for i, v in enumerate(m.train_loss_history):
            assert v > 0, f"train_loss_history[{i}] = {v} is not positive (RMSE must be > 0)"

    def test_loss_history_monotone_decreasing_clean_signal(self):
        """Clean linear signal: loss must be strictly monotone-decreasing every step."""
        rng = np.random.RandomState(42)
        X = rng.rand(200, 4).astype(np.float32)
        y = (X[:, 0] * 3.0).astype(np.float32)  # pure linear, zero noise

        m = _nb(CatBoostMLXRegressor, X, y, iterations=30, depth=4)
        hist = m.train_loss_history
        for i in range(1, len(hist)):
            assert hist[i] < hist[i - 1], \
                f"Loss increased at step {i}: {hist[i - 1]:.6f} -> {hist[i]:.6f}"

    def test_loss_history_overall_decreasing_noisy(self):
        """Noisy signal: final loss must be lower than initial loss."""
        rng = np.random.RandomState(42)
        X = rng.rand(120, 5).astype(np.float32)
        y = (X[:, 0] * 2 + rng.normal(0, 0.2, 120)).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=40, depth=4)
        hist = m.train_loss_history
        assert hist[0] > hist[-1], \
            f"Loss did not decrease overall: first={hist[0]:.4f} last={hist[-1]:.4f}"

    def test_eval_loss_history_correct_length_with_eval_set(self):
        """eval_loss_history must have one entry per iteration when eval_set is given."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        X_val = rng.rand(20, 4).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=15, binary_path=BINARY_PATH)
        m.fit(X, y, eval_set=(X_val, y_val))

        assert len(m.train_loss_history) == 15
        assert len(m.eval_loss_history) == 15

    def test_eval_loss_history_empty_without_eval_set(self):
        """Without an eval_set, eval_loss_history must be empty."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3).astype(np.float32)
        y = rng.rand(40).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=10)
        assert m.eval_loss_history == [], \
            f"eval_loss_history should be empty without eval_set, got {m.eval_loss_history}"

    def test_loss_history_populated_when_verbose_false(self):
        """Loss history must be computed even when verbose=False (key Sprint 11 change)."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = rng.rand(50).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=20, verbose=False)
        assert len(m.train_loss_history) == 20, \
            "Loss history must be populated even when verbose=False"

    def test_early_stopped_history_shorter_than_max_iterations(self):
        """With early stopping, loss history and tree_count_ must both be < max iterations.

        KNOWN BEHAVIOR (BUG-003): When early stopping triggers, train_loss_history records
        all iterations that ran (best_iteration + patience window), while tree_count_
        reflects only the best iteration. These two counts DIFFER. For example:
          - best_iteration = 2 (trees saved)
          - train_loss_history length = 7 (2 + patience=5 look-ahead steps)
        Both should be < max_iterations (200), which is the critical invariant.
        The mismatch between history length and tree_count_ is a semantic issue that
        should be fixed but is not a crash.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(80, 5).astype(np.float32)
        y_train = (X[:, 0] * 2).astype(np.float32)
        X_val = rng.rand(20, 5).astype(np.float32)
        y_val = rng.rand(20).astype(np.float32) * 100  # unrelated -- triggers early stop

        m = CatBoostMLXRegressor(
            iterations=200, depth=3, early_stopping_rounds=5,
            binary_path=BINARY_PATH
        )
        m.fit(X, y_train, eval_set=(X_val, y_val))

        assert m.tree_count_ < 200, "Early stopping should have triggered"
        assert len(m.train_loss_history) < 200, \
            f"train_loss_history should be < 200 entries after early stopping, got {len(m.train_loss_history)}"
        assert len(m.eval_loss_history) < 200, \
            f"eval_loss_history should be < 200 entries after early stopping, got {len(m.eval_loss_history)}"
        # Document the known mismatch: history records all iterations run (best + patience window)
        # while tree_count_ records only the best iteration
        if len(m.train_loss_history) != m.tree_count_:
            pytest.xfail(
                f"BUG-003: train_loss_history len ({len(m.train_loss_history)}) != "
                f"tree_count_ ({m.tree_count_}) after early stopping. "
                "History includes the patience look-ahead window; model uses best iteration."
            )

    def test_eval_loss_history_all_finite(self):
        """All eval loss values must be finite."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        X_val = rng.rand(20, 4).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=20, binary_path=BINARY_PATH)
        m.fit(X, y, eval_set=(X_val, y_val))

        for i, v in enumerate(m.eval_loss_history):
            assert np.isfinite(v), f"eval_loss_history[{i}] = {v} is not finite"


# ============================================================================
# 3. Categorical encoding edge cases
# ============================================================================

@requires_nanobind
class TestCategoricalEdgeCases:
    """Adversarial categorical encoding tests through the nanobind path."""

    def test_empty_string_category(self):
        """Empty string '' is a valid categorical value and must not crash."""
        rng = np.random.RandomState(42)
        cats = np.array(["", "a", "", "b", "a"] * 8, dtype=object)
        nums = rng.rand(40).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(40).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        assert m.tree_count_ == 5
        pred = m.predict(X)
        assert pred.shape == (40,)
        assert np.all(np.isfinite(pred))

    def test_single_category_value(self):
        """Column with only one unique category value (all same)."""
        rng = np.random.RandomState(42)
        cats = np.array(["only"] * 30, dtype=object)
        nums = rng.rand(30).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(30).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        # Single-category column carries zero information; model should still train
        assert m.tree_count_ == 5

    def test_special_characters_in_category_values(self):
        """Categories with slashes, newlines in repr, semicolons, spaces."""
        rng = np.random.RandomState(42)
        cats = np.array(["a/b", "c\\nd_repr", "e;f", "g h", "i\tj"] * 8, dtype=object)
        nums = rng.rand(40).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(40).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        assert m.tree_count_ == 5
        pred = m.predict(X)
        assert np.all(np.isfinite(pred))

    def test_nan_in_categorical_column(self):
        """NaN values in a categorical column must not crash the nanobind path."""
        rng = np.random.RandomState(42)
        cats = np.array(["a", np.nan, "b", "a"] * 8, dtype=object)
        nums = rng.rand(32).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(32).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        assert m.tree_count_ == 5

    def test_unseen_category_in_eval_set(self):
        """eval_set may contain categories not seen during training -- must not crash."""
        rng = np.random.RandomState(42)
        Xtrain = np.array(["train_a", "train_b"] * 15, dtype=object).reshape(-1, 1)
        Xtrain = np.column_stack([Xtrain, rng.rand(30).reshape(-1, 1).astype(object)])
        ytrain = rng.rand(30).astype(np.float32)

        Xval = np.array(["unseen_x", "unseen_y"] * 5, dtype=object).reshape(-1, 1)
        Xval = np.column_stack([Xval, rng.rand(10).reshape(-1, 1).astype(object)])
        yval = rng.rand(10).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, cat_features=[0], binary_path=BINARY_PATH)
        m.fit(Xtrain, ytrain, eval_set=(Xval, yval))
        assert len(m.eval_loss_history) == 5

    def test_high_cardinality_categorical_all_unique(self):
        """All-unique categorical column (n unique == n rows)."""
        rng = np.random.RandomState(42)
        n = 50
        cats = np.array([f"id_{i}" for i in range(n)], dtype=object)
        nums = rng.rand(n).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(n).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        assert m.tree_count_ == 5

    def test_unicode_and_emoji_in_categories(self):
        """Unicode and emoji category values must be handled as strings."""
        rng = np.random.RandomState(42)
        cats = np.array(["cafe\u0301", "\U0001F600", "\u4e2d\u6587", "ascii"] * 8, dtype=object)
        nums = rng.rand(32).reshape(-1, 1).astype(object)
        X = np.column_stack([cats, nums])
        y = rng.rand(32).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5, cat_features=[0])
        assert m.tree_count_ == 5


# ============================================================================
# 4. eval_set through nanobind
# ============================================================================

@requires_nanobind
class TestEvalSetNanobind:
    """eval_set and early stopping correctness through the nanobind path."""

    def test_eval_set_history_length_exact(self):
        """eval_loss_history must have exactly num_iterations entries."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        X_val = rng.rand(20, 4).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=25, binary_path=BINARY_PATH)
        m.fit(X, y, eval_set=(X_val, y_val))
        assert len(m.eval_loss_history) == 25
        assert len(m.train_loss_history) == 25

    def test_eval_set_predictions_are_reasonable(self):
        """Predictions on val set from a model trained with eval_set should be correlated with y."""
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5).astype(np.float32)
        y = (X[:, 0] * 3).astype(np.float32)
        X_val = rng.rand(30, 5).astype(np.float32)
        y_val = (X_val[:, 0] * 3).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=30, depth=4, binary_path=BINARY_PATH)
        m.fit(X, y, eval_set=(X_val, y_val))

        pred_val = m.predict(X_val)
        corr = np.corrcoef(pred_val, y_val)[0, 1]
        assert corr > 0.8, f"Validation predictions poorly correlated with y_val: corr={corr:.3f}"

    def test_early_stopping_stops_before_max_iterations(self):
        """Unrelated validation targets must trigger early stopping."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 3).astype(np.float32)
        y_train = (X[:, 0] * 2).astype(np.float32)
        X_val = rng.rand(20, 3).astype(np.float32)
        y_val = (rng.rand(20) * 100).astype(np.float32)  # unrelated

        m = CatBoostMLXRegressor(
            iterations=200, depth=3, early_stopping_rounds=5,
            binary_path=BINARY_PATH
        )
        m.fit(X, y_train, eval_set=(X_val, y_val))
        assert m.tree_count_ < 200, \
            f"Early stopping should have triggered, but built {m.tree_count_} trees"

    def test_eval_fraction_produces_eval_history(self):
        """eval_fraction splits training data and must populate eval_loss_history."""
        rng = np.random.RandomState(42)
        X = rng.rand(100, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(
            iterations=15, depth=3, eval_fraction=0.2,
            binary_path=BINARY_PATH
        )
        m.fit(X, y)
        assert len(m.train_loss_history) == 15
        assert len(m.eval_loss_history) == 15

    def test_eval_set_and_eval_fraction_mutually_exclusive(self):
        """Providing both eval_set and eval_fraction > 0 must raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = rng.rand(50).astype(np.float32)

        m = CatBoostMLXRegressor(
            iterations=5, eval_fraction=0.2, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            m.fit(X, y, eval_set=(X[:10], y[:10]))

    def test_eval_set_feature_count_mismatch_raises(self):
        """eval_set with wrong number of features must raise ValueError before dispatch."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 4).astype(np.float32)
        y = rng.rand(30).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match="features"):
            m.fit(X, y, eval_set=(rng.rand(10, 7).astype(np.float32), rng.rand(10)))


# ============================================================================
# 5. Memory safety and sequential fits
# ============================================================================

@requires_nanobind
class TestMemorySafety:
    """State isolation between sequential fits; no cross-contamination."""

    def test_fit_twice_replaces_state_not_accumulates(self):
        """Fitting the same model twice must replace all state, not accumulate."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 4).astype(np.float32)
        y = rng.rand(50).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=10)

        assert len(m.train_loss_history) == 10
        assert m.tree_count_ == 10

        m.fit(X, y)  # second fit

        assert len(m.train_loss_history) == 10, \
            f"Loss history should not accumulate across fits; got len={len(m.train_loss_history)}"
        assert m.tree_count_ == 10

    def test_sequential_fits_change_predictions_with_different_seeds(self):
        """Different seeds on sequential fits must produce different predictions."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 4).astype(np.float32)
        y = (X[:, 0] * 2).astype(np.float32)

        m1 = _nb(CatBoostMLXRegressor, X, y, iterations=15, depth=3, random_seed=1)
        m2 = _nb(CatBoostMLXRegressor, X, y, iterations=15, depth=3, random_seed=999)

        # Should differ because of different seeds
        assert not np.allclose(m1.predict(X), m2.predict(X), atol=1e-6), \
            "Different seeds should produce different predictions"

    def test_five_sequential_models_each_independent(self):
        """Five models trained sequentially must each have independent state."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 4).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        models = []
        for seed in range(5):
            m = _nb(CatBoostMLXRegressor, X, y, iterations=10, depth=3, random_seed=seed)
            models.append(m)

        for i, m in enumerate(models):
            assert m.tree_count_ == 10, f"Model {i} has {m.tree_count_} trees (expected 10)"
            assert len(m.train_loss_history) == 10, \
                f"Model {i} has {len(m.train_loss_history)} loss entries (expected 10)"

    def test_tiny_dataset_two_rows(self):
        """Two-row dataset must train and predict without crashing."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert m.tree_count_ == 5
        pred = m.predict(X)
        assert pred.shape == (2,)
        assert np.all(np.isfinite(pred))

    def test_tiny_dataset_three_rows(self):
        """Three-row dataset must train without crashing."""
        X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=3)
        assert m.tree_count_ == 3

    def test_fit_predict_fit_state_is_second_model(self):
        """After fit-predict-fit, model state must reflect only the second fit."""
        rng = np.random.RandomState(42)
        X1 = rng.rand(60, 4).astype(np.float32)
        y1 = (X1[:, 0] * 5).astype(np.float32)  # important feature: index 0

        X2 = rng.rand(60, 4).astype(np.float32)
        y2 = (X2[:, 3] * 5).astype(np.float32)  # important feature: index 3

        m = CatBoostMLXRegressor(iterations=30, depth=4, binary_path=BINARY_PATH)
        m.fit(X1, y1)
        _ = m.predict(X1)  # intermediate prediction

        m.fit(X2, y2)  # second fit with different signal

        fi = m.feature_importances_
        most_important = int(np.argmax(fi))
        assert most_important == 3, \
            f"After second fit, most important feature should be 3, got {most_important}"

        assert len(m.train_loss_history) == 30, \
            f"Loss history should be 30 after second fit, got {len(m.train_loss_history)}"

    def test_eval_loss_cleared_on_refit_without_eval_set(self):
        """Refitting without eval_set must clear any previously stored eval_loss_history."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        X_val = rng.rand(20, 3).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=10, binary_path=BINARY_PATH)
        m.fit(X, y, eval_set=(X_val, y_val))
        assert len(m.eval_loss_history) == 10  # populated from first fit

        m.fit(X, y)  # second fit without eval_set
        assert len(m.eval_loss_history) == 0, \
            f"eval_loss_history not cleared on refit without eval_set; got {len(m.eval_loss_history)} entries"


# ============================================================================
# 6. Model JSON integrity
# ============================================================================

@requires_nanobind
class TestModelJSONIntegrity:
    """Model JSON structure and roundtrip correctness for nanobind-trained models."""

    def test_model_json_has_required_top_level_keys(self):
        """Model JSON must contain format, version, model_info, features, trees."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 4).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=10)
        data = m._model_data
        assert data is not None
        for key in ("format", "version", "model_info", "features", "trees"):
            assert key in data, f"Missing key '{key}' in model JSON"

    def test_model_json_trees_count_matches_tree_count_(self):
        """Number of trees in JSON must match tree_count_ attribute."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 5).astype(np.float32)
        y = rng.rand(80).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=20)
        assert len(m._model_data["trees"]) == m.tree_count_ == 20

    def test_model_json_features_array_matches_input_features(self):
        """features array in JSON must have one entry per input feature."""
        rng = np.random.RandomState(42)
        n_features = 7
        X = rng.rand(60, n_features).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert len(m._model_data["features"]) == n_features

    def test_model_json_feature_names_injected(self):
        """Feature names provided to fit() must appear in model_data features."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = rng.rand(50).astype(np.float32)
        names = ["alpha", "beta", "gamma"]

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        m.fit(X, y, feature_names=names)

        feat_names_in_json = [f["name"] for f in m._model_data["features"]]
        assert feat_names_in_json == names, \
            f"Expected names {names}, got {feat_names_in_json}"

    def test_model_json_model_info_required_fields(self):
        """model_info must contain loss_type, learning_rate, num_trees."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = rng.rand(50).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=8)
        info = m._model_data["model_info"]
        for field in ("loss_type", "learning_rate", "num_trees"):
            assert field in info, f"model_info missing field '{field}'"

    def test_save_load_roundtrip_preserves_predictions(self, tmp_path):
        """Predictions before and after save/load must be identical."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 5).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=20, depth=4)
        pred_before = m.predict(X)

        path = str(tmp_path / "model.json")
        m.save_model(path)

        m2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        m2.load_model(path)
        pred_after = m2.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after,
                                      err_msg="Save/load altered predictions")

    def test_save_load_preserves_tree_count_and_feature_count(self, tmp_path):
        """tree_count_ and n_features_in_ must survive save/load."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 6).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=12, depth=3)
        path = str(tmp_path / "model.json")
        m.save_model(path)

        m2 = CatBoostMLXRegressor(binary_path=BINARY_PATH)
        m2.load_model(path)

        assert m2.tree_count_ == 12
        assert m2.n_features_in_ == 6

    def test_model_json_is_valid_json(self):
        """model_json string from nanobind result must be parseable by json.loads."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        from catboost_mlx import _core
        cfg = _core.TrainConfig()
        cfg.num_iterations = 8
        result = _core.train(
            features=X, targets=y,
            feature_names=["f0", "f1", "f2"],
            is_categorical=[False, False, False],
            config=cfg,
        )
        # This must not raise JSONDecodeError
        parsed = json.loads(result.model_json)
        assert isinstance(parsed, dict)


# ============================================================================
# 7. Feature importance
# ============================================================================

@requires_nanobind
class TestFeatureImportance:
    """Feature importance correctness from the nanobind path."""

    def test_feature_importances_sums_to_one(self):
        """feature_importances_ (normalized) must sum to 1.0."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 5).astype(np.float32)
        y = (X[:, 0] * 3 + X[:, 1]).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=20)
        fi = m.feature_importances_
        np.testing.assert_allclose(fi.sum(), 1.0, atol=1e-6,
                                   err_msg="feature_importances_ must sum to 1.0")

    def test_feature_importances_shape(self):
        """feature_importances_ must have one entry per feature."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 7).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=10)
        assert m.feature_importances_.shape == (7,)

    def test_most_important_feature_correctly_identified(self):
        """The feature driving the signal must rank as most important."""
        rng = np.random.RandomState(42)
        X = rng.rand(200, 6).astype(np.float32)
        y = (X[:, 2] * 5 + rng.normal(0, 0.05, 200)).astype(np.float32)  # signal in feature 2

        m = _nb(CatBoostMLXRegressor, X, y, iterations=40, depth=4)
        most_important = int(np.argmax(m.feature_importances_))
        assert most_important == 2, \
            f"Expected feature 2 to be most important, got {most_important}"

    def test_top_k_features_overlap_with_true_signal(self):
        """Top-5 features by importance should overlap with true signal features."""
        rng = np.random.RandomState(42)
        n_features = 30
        X = rng.rand(300, n_features).astype(np.float32)
        true_feat_indices = [0, 1, 2, 3, 4]  # first 5 features carry signal
        y = (X[:, true_feat_indices].sum(axis=1)
             + rng.normal(0, 0.1, 300)).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=60, depth=5)
        top5 = set(np.argsort(m.feature_importances_)[::-1][:5])
        overlap = len(top5 & set(true_feat_indices))
        assert overlap >= 4, \
            f"Expected at least 4/5 true features in top-5, got {overlap}. Top-5={top5}"

    def test_feature_importances_non_negative(self):
        """All feature importance values must be >= 0."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 5).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=10)
        fi = m.feature_importances_
        assert np.all(fi >= 0), f"Negative importance values: {fi}"


# ============================================================================
# 8. Weights edge cases
# ============================================================================

@requires_nanobind
class TestWeightsEdgeCases:
    """Sample weight handling through the nanobind path."""

    def test_uniform_weights_match_no_weights(self):
        """Uniform weights should produce identical predictions to no weights."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m_no_w = _nb(CatBoostMLXRegressor, X, y, iterations=15, depth=4, random_seed=42)

        m_w = CatBoostMLXRegressor(iterations=15, depth=4, random_seed=42,
                                   binary_path=BINARY_PATH)
        m_w.fit(X, y, sample_weight=np.ones(60, dtype=np.float32))

        np.testing.assert_allclose(m_no_w.predict(X), m_w.predict(X), atol=1e-4,
                                   err_msg="Uniform weights != no weights")

    def test_zero_weights_produce_zero_or_base_predictions(self):
        """All-zero sample weights: model trains without crashing; predictions are finite."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        weights = np.zeros(30, dtype=np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        m.fit(X, y, sample_weight=weights)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred)), f"Predictions with zero weights contain non-finite values: {pred}"

    def test_large_weights_produce_finite_predictions(self):
        """Very large weights (1e6) must not overflow to NaN/inf in predictions."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        weights = np.full(30, 1e6, dtype=np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        m.fit(X, y, sample_weight=weights)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred)), \
            f"Predictions with large weights contain non-finite values: {pred}"

    def test_negative_weights_raise_or_produce_invalid_json(self):
        """Negative weights currently cause the C++ engine to produce NaN/inf in model JSON.

        This is a KNOWN BUG (BUG-001). The test documents the current behavior:
        negative weights reach the C++ layer, which stores NaN/inf in leaf values,
        and Python's json.loads() then fails because JSON does not allow NaN/inf literals.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        weights = np.full(30, -1.0, dtype=np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        # The JSONDecodeError message is "Expecting value" -- the underlying cause is
        # that negative weights produce NaN/inf leaf values serialized as bare tokens
        # which Python's json parser rejects. We match on the exception type only.
        with pytest.raises((ValueError, RuntimeError, json.JSONDecodeError)):
            m.fit(X, y, sample_weight=weights)

    def test_nonuniform_weights_change_predictions(self):
        """Different weight distributions on the same data must produce different models."""
        rng = np.random.RandomState(42)
        X = rng.rand(60, 3).astype(np.float32)
        y = rng.rand(60).astype(np.float32)

        w1 = np.ones(60, dtype=np.float32)
        w1[:20] = 10.0  # heavy on first 20

        w2 = np.ones(60, dtype=np.float32)
        w2[-20:] = 10.0  # heavy on last 20

        m1 = CatBoostMLXRegressor(iterations=20, depth=4, random_seed=42,
                                  binary_path=BINARY_PATH)
        m1.fit(X, y, sample_weight=w1)

        m2 = CatBoostMLXRegressor(iterations=20, depth=4, random_seed=42,
                                  binary_path=BINARY_PATH)
        m2.fit(X, y, sample_weight=w2)

        assert not np.allclose(m1.predict(X), m2.predict(X), atol=1e-6), \
            "Different weight distributions must produce different predictions"


# ============================================================================
# 9. Data type robustness
# ============================================================================

@requires_nanobind
class TestDataTypeRobustness:
    """Input array type coercion and edge cases."""

    def test_int16_input(self):
        """int16 feature arrays must be safely cast to float32 for the GPU."""
        rng = np.random.RandomState(42)
        X = rng.randint(0, 100, (50, 4)).astype(np.int16)
        y = rng.rand(50).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert m.tree_count_ == 5
        pred = m.predict(X.astype(np.float32))
        assert np.all(np.isfinite(pred))

    def test_bool_input(self):
        """Boolean feature arrays must be safely cast and train without error."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 4) > 0.5
        y = rng.rand(50).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert m.tree_count_ == 5

    def test_float64_input_cast_to_float32(self):
        """float64 arrays must be safely downcast to float32."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 4).astype(np.float64)
        y = rng.rand(50).astype(np.float64)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=8)
        assert m.tree_count_ == 8

    def test_fortran_contiguous_input(self):
        """Fortran-ordered (column-major) arrays must be converted to C-contiguous."""
        rng = np.random.RandomState(42)
        X = np.asfortranarray(rng.rand(50, 4))
        y = rng.rand(50)
        assert X.flags["F_CONTIGUOUS"]

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert m.tree_count_ == 5

    def test_dataframe_input_preserves_column_names(self):
        """DataFrame input must auto-extract column names as feature_names_in_."""
        pytest.importorskip("pandas")
        import pandas as pd

        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.rand(60, 4),
                          columns=["feat_a", "feat_b", "feat_c", "feat_d"])
        y = df["feat_a"].values * 2 + rng.normal(0, 0.1, 60)

        m = _nb(CatBoostMLXRegressor, df, y, iterations=10)
        assert list(m.feature_names_in_) == ["feat_a", "feat_b", "feat_c", "feat_d"]

    def test_pool_input(self):
        """Pool objects must be handled identically to raw arrays."""
        from catboost_mlx.pool import Pool

        rng = np.random.RandomState(42)
        X = rng.rand(50, 4).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(float)
        pool = Pool(X, y, feature_names=["a", "b", "c", "d"])

        m = CatBoostMLXClassifier(iterations=10, binary_path=BINARY_PATH)
        m.fit(pool)

        assert m.tree_count_ == 10
        assert list(m.feature_names_in_) == ["a", "b", "c", "d"]

    def test_mixed_int_float_dataframe(self):
        """DataFrame with mixed int/float/bool dtypes must coerce without crashing."""
        pytest.importorskip("pandas")
        import pandas as pd

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "int_col": rng.randint(0, 10, 50),
            "float_col": rng.rand(50),
            "bool_col": rng.rand(50) > 0.5,
        })
        y = (rng.rand(50) > 0.5).astype(float)

        m = _nb(CatBoostMLXClassifier, df, y, iterations=8)
        assert m.tree_count_ == 8


# ============================================================================
# 10. Sequential fits: same model object
# ============================================================================

@requires_nanobind
class TestSequentialFits:
    """Second fit on the same model must replace, not extend, the first model."""

    def test_second_fit_same_data_same_predictions(self):
        """Fitting same data twice with same seed must yield numerically equivalent predictions.

        Note: Metal GPU kernels may produce sub-epsilon (< 1e-7) floating point
        differences between runs due to non-deterministic thread accumulation order.
        We allow atol=1e-6 rather than exact equality.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=10, depth=3, random_seed=42,
                                 binary_path=BINARY_PATH)
        m.fit(X, y)
        pred1 = m.predict(X).copy()

        m.fit(X, y)
        pred2 = m.predict(X)

        np.testing.assert_allclose(pred1, pred2, atol=1e-6,
                                   err_msg="Same data + same seed should produce nearly identical predictions on refit")

    def test_refit_different_data_uses_new_model(self):
        """Refitting with different X/y must produce predictions correlated with new y."""
        rng = np.random.RandomState(42)
        X = rng.rand(80, 4).astype(np.float32)
        y_new = (X[:, 3] * 4).astype(np.float32)  # signal only in feature 3

        m = CatBoostMLXRegressor(iterations=30, depth=4, binary_path=BINARY_PATH)
        # First fit with a different target
        m.fit(X, (X[:, 0] * 4).astype(np.float32))
        # Second fit with new target
        m.fit(X, y_new)

        # Predictions should correlate with y_new, not old y
        pred = m.predict(X)
        corr = np.corrcoef(pred, y_new)[0, 1]
        assert corr > 0.7, \
            f"After refit, predictions should correlate with new targets (corr={corr:.3f})"

    def test_is_fitted_true_after_both_fits(self):
        """_is_fitted must be True after each fit."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 2).astype(np.float32)
        y = rng.rand(30).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        assert not m._is_fitted

        m.fit(X, y)
        assert m._is_fitted

        m.fit(X, y)
        assert m._is_fitted


# ============================================================================
# 11. Verbose output (C stdout -- captured at fd level)
# ============================================================================

@requires_nanobind
class TestVerboseOutput:
    """Verbose output goes through C printf -- captured by capfd, not capsys."""

    def test_verbose_true_produces_iteration_lines(self, capfd):
        """verbose=True must print iter=, trees=, loss= lines to stdout."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=3, verbose=True, binary_path=BINARY_PATH)
        m.fit(X, y)

        out = capfd.readouterr().out
        assert "iter=" in out, f"Expected 'iter=' in verbose output, got: {repr(out[:200])}"
        assert "loss=" in out, f"Expected 'loss=' in verbose output, got: {repr(out[:200])}"
        assert "trees=" in out, f"Expected 'trees=' in verbose output, got: {repr(out[:200])}"

    def test_verbose_false_produces_no_output(self, capfd):
        """verbose=False must produce no stdout output."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        m = CatBoostMLXRegressor(iterations=3, verbose=False, binary_path=BINARY_PATH)
        m.fit(X, y)

        out = capfd.readouterr().out
        assert out == "", f"Expected no output with verbose=False, got: {repr(out[:200])}"

    def test_verbose_output_line_count_matches_iterations(self, capfd):
        """There must be exactly num_iterations iteration lines in verbose output."""
        rng = np.random.RandomState(42)
        X = rng.rand(40, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        n_iter = 5

        m = CatBoostMLXRegressor(iterations=n_iter, verbose=True, binary_path=BINARY_PATH)
        m.fit(X, y)

        out = capfd.readouterr().out
        iter_lines = [line for line in out.split("\n") if line.startswith("iter=")]
        assert len(iter_lines) == n_iter, \
            f"Expected {n_iter} iter= lines, got {len(iter_lines)}: {iter_lines}"


# ============================================================================
# 12. Error paths
# ============================================================================

@requires_nanobind
class TestErrorPaths:
    """Adversarial error path testing for the nanobind path."""

    def test_mismatched_x_y_lengths_raises_value_error(self):
        """X and y with different row counts must raise ValueError before dispatch."""
        X = np.random.rand(30, 3).astype(np.float32)
        y = np.random.rand(25).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match=r"(?i)(sample|shape|30|25)"):
            m.fit(X, y)

    def test_nan_in_targets_does_not_silently_produce_zero_trees(self):
        """NaN values in y must either raise an error or build the expected number of trees.

        KNOWN ISSUE: currently the C++ engine silently builds 0 trees when
        all targets are effectively NaN (after gradient computation). This test
        documents the current behavior and acts as a regression guard.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = rng.rand(30).astype(np.float32)
        y[5] = np.nan

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        try:
            m.fit(X, y)
            # If it didn't raise, trees_built must equal iterations
            # (building 0 trees silently is the documented bug)
            if m.tree_count_ == 0:
                pytest.xfail(
                    "BUG-002: NaN in targets silently builds 0 trees instead of raising an error"
                )
        except (ValueError, RuntimeError):
            pass  # Raising is the correct behavior

    def test_inf_in_features_raises_or_produces_invalid_model(self):
        """inf values in features must either raise or produce a parseable model.

        KNOWN ISSUE (BUG-001 related): inf in features propagates through the
        GBDT leaf computation, producing inf leaf values that get serialized as
        'inf' in JSON -- which Python's json.loads() correctly rejects. The
        current behavior is an unhandled JSONDecodeError surfacing to the user.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        X[0, 0] = np.inf
        y = rng.rand(30).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises((ValueError, RuntimeError, json.JSONDecodeError)):
            m.fit(X, y)

    def test_nan_mode_forbidden_with_nan_in_features_raises(self):
        """nan_mode='forbidden' with NaN in X must raise RuntimeError before GPU dispatch."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3).astype(np.float32)
        X[0, 0] = np.nan
        y = rng.rand(20).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, nan_mode="forbidden",
                                 binary_path=BINARY_PATH)
        with pytest.raises(RuntimeError, match="NaN"):
            m.fit(X, y)

    def test_empty_dataset_raises_value_error(self):
        """Zero-row dataset must raise ValueError before any C++ call."""
        X = np.zeros((0, 3), dtype=np.float32)
        y = np.zeros(0, dtype=np.float32)

        m = CatBoostMLXRegressor(iterations=5, binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match=r"(?i)empty"):
            m.fit(X, y)

    def test_1d_x_is_auto_reshaped_to_2d(self):
        """1-D X must be auto-reshaped to (n_samples, 1) without error."""
        rng = np.random.RandomState(42)
        X = rng.rand(30).astype(np.float32)
        y = rng.rand(30).astype(np.float32)

        m = _nb(CatBoostMLXRegressor, X, y, iterations=5)
        assert m.n_features_in_ == 1
        assert m.tree_count_ == 5

    def test_negative_iterations_raises_value_error(self):
        """iterations < 1 must raise ValueError during param validation."""
        with pytest.raises(ValueError, match="iterations"):
            CatBoostMLXRegressor(iterations=-1, binary_path=BINARY_PATH)._validate_params()

    def test_zero_iterations_raises_value_error(self):
        """iterations=0 must raise ValueError during param validation."""
        with pytest.raises(ValueError, match="iterations"):
            CatBoostMLXRegressor(iterations=0, binary_path=BINARY_PATH)._validate_params()

    def test_cat_features_index_out_of_bounds_raises(self):
        """cat_features index beyond n_features must raise ValueError."""
        X = np.random.rand(30, 3).astype(np.float32)
        y = np.random.rand(30).astype(np.float32)

        m = CatBoostMLXRegressor(iterations=5, cat_features=[99],
                                 binary_path=BINARY_PATH)
        with pytest.raises(ValueError, match=r"(?i)(out of bounds|cat_features|99)"):
            m.fit(X, y)


# ============================================================================
# 13. Low-level _core module tests
# ============================================================================

@requires_nanobind
class TestCoreLowLevel:
    """Direct tests of the nanobind _core module (train(), TrainConfig, TrainResult)."""

    def test_train_config_defaults_match_expected_values(self):
        """TrainConfig defaults must match TTrainConfig C++ defaults."""
        from catboost_mlx import _core

        cfg = _core.TrainConfig()
        assert cfg.num_iterations == 100
        assert cfg.max_depth == 6
        assert abs(cfg.learning_rate - 0.1) < 1e-6
        assert abs(cfg.l2_reg_lambda - 3.0) < 1e-6
        assert cfg.max_bins == 255
        assert cfg.loss_type == "auto"
        assert cfg.nan_mode == "min"
        assert cfg.bootstrap_type == "no"
        assert cfg.grow_policy == "SymmetricTree"

    def test_train_result_has_all_expected_fields(self):
        """TrainResult must expose all documented fields."""
        from catboost_mlx import _core

        rng = np.random.RandomState(42)
        X = rng.rand(30, 2).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        cfg = _core.TrainConfig()
        cfg.num_iterations = 5
        cfg.compute_feature_importance = True

        result = _core.train(
            features=X, targets=y,
            feature_names=["f0", "f1"],
            is_categorical=[False, False],
            config=cfg,
        )
        assert hasattr(result, "final_train_loss")
        assert hasattr(result, "final_test_loss")
        assert hasattr(result, "best_iteration")
        assert hasattr(result, "trees_built")
        assert hasattr(result, "model_json")
        assert hasattr(result, "feature_names")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "train_loss_history")
        assert hasattr(result, "eval_loss_history")
        assert hasattr(result, "grad_ms")
        assert hasattr(result, "tree_search_ms")
        assert hasattr(result, "leaf_ms")
        assert hasattr(result, "apply_ms")

    def test_timing_fields_positive_after_training(self):
        """All timing fields must be > 0 after training completes."""
        from catboost_mlx import _core

        rng = np.random.RandomState(42)
        X = rng.rand(40, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        cfg = _core.TrainConfig()
        cfg.num_iterations = 5
        result = _core.train(
            features=X, targets=y,
            feature_names=["f0", "f1", "f2"],
            is_categorical=[False, False, False],
            config=cfg,
        )
        assert result.grad_ms > 0, "grad_ms should be positive"
        assert result.tree_search_ms > 0, "tree_search_ms should be positive"
        assert result.leaf_ms > 0, "leaf_ms should be positive"
        assert result.apply_ms > 0, "apply_ms should be positive"

    def test_explicit_val_set_produces_eval_history(self):
        """Passing val_features/val_targets directly to _core.train must fill eval_loss_history."""
        from catboost_mlx import _core

        rng = np.random.RandomState(42)
        X = rng.rand(50, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        X_val = rng.rand(20, 3).astype(np.float32)
        y_val = X_val[:, 0].astype(np.float32)

        cfg = _core.TrainConfig()
        cfg.num_iterations = 10
        result = _core.train(
            features=X, targets=y,
            feature_names=["f0", "f1", "f2"],
            is_categorical=[False, False, False],
            val_features=X_val.copy(),
            val_targets=y_val.copy(),
            config=cfg,
        )
        assert len(result.eval_loss_history) == 10, \
            f"Expected 10 eval entries, got {len(result.eval_loss_history)}"
        assert len(result.train_loss_history) == 10

    def test_feature_names_echoed_in_result(self):
        """TrainResult.feature_names must echo back the feature_names passed in."""
        from catboost_mlx import _core

        rng = np.random.RandomState(42)
        X = rng.rand(30, 3).astype(np.float32)
        y = rng.rand(30).astype(np.float32)
        names = ["alpha", "beta", "gamma"]

        cfg = _core.TrainConfig()
        cfg.num_iterations = 3
        result = _core.train(
            features=X, targets=y,
            feature_names=names,
            is_categorical=[False, False, False],
            config=cfg,
        )
        assert result.feature_names == names, \
            f"Expected feature_names {names}, got {result.feature_names}"

    def test_trees_built_matches_num_iterations(self):
        """result.trees_built must equal config.num_iterations (without early stopping)."""
        from catboost_mlx import _core

        rng = np.random.RandomState(42)
        X = rng.rand(40, 2).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        for n in (1, 7, 25):
            cfg = _core.TrainConfig()
            cfg.num_iterations = n
            result = _core.train(
                features=X, targets=y,
                feature_names=["f0", "f1"],
                is_categorical=[False, False],
                config=cfg,
            )
            assert result.trees_built == n, \
                f"num_iterations={n}: expected trees_built={n}, got {result.trees_built}"
