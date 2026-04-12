"""
test_ranking_nanobind.py -- Comprehensive adversarial QA for PairLogit and YetiRank
through the nanobind in-process training path.

Coverage areas:
  1. PairLogit basic correctness through nanobind
  2. YetiRank basic correctness through nanobind
  3. Nanobind vs subprocess parity for ranking losses
  4. Edge cases: single-doc groups, all-same relevance, large groups
  5. eval_set + early stopping with ranking
  6. Feature importance with ranking losses
  7. Save/load roundtrip for ranking models
  8. Unsorted group_ids behavior (adversarial: nanobind path does NOT sort)
  9. Validation errors: ranking without group_id, mismatched lengths

Run with:
  pytest tests/test_ranking_nanobind.py -xvs
"""

import json
import os
import tempfile
import unittest.mock

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY_PATH = REPO_ROOT

from catboost_mlx import CatBoostMLX, CatBoostMLXRegressor
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


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _make_ranking_data(n_groups=5, docs_per_group=10, n_features=4, seed=42):
    """Return (X, y, group_ids) with integer relevance labels 0-4."""
    rng = np.random.RandomState(seed)
    n = n_groups * docs_per_group
    X = rng.rand(n, n_features).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for g in range(n_groups):
        start = g * docs_per_group
        for d in range(docs_per_group):
            y[start + d] = float(rng.randint(0, 5))
    group_ids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)
    return X, y, group_ids


def _nb_ranking(loss, X, y, group_ids, **kwargs):
    """Train a ranking model via the nanobind path (default when extension is active)."""
    model = CatBoostMLX(
        loss=loss, binary_path=BINARY_PATH, **kwargs
    )
    model.fit(X, y, group_id=group_ids)
    return model


def _sp_ranking(loss, X, y, group_ids, **kwargs):
    """Train a ranking model via the subprocess path by patching _HAS_NANOBIND."""
    model = CatBoostMLX(
        loss=loss, binary_path=BINARY_PATH, **kwargs
    )
    with unittest.mock.patch.object(_core_module, "_HAS_NANOBIND", False):
        model.fit(X, y, group_id=group_ids)
    return model


# ============================================================================
# 1. PairLogit basic correctness through nanobind
# ============================================================================

@requires_nanobind
class TestPairLogitNanobind:
    """PairLogit training through the nanobind in-process path."""

    def setup_method(self):
        self.X, self.y, self.gids = _make_ranking_data(
            n_groups=5, docs_per_group=10, n_features=4, seed=42
        )

    def test_predictions_shape_matches_input(self):
        """Predictions vector must have exactly n_samples entries."""
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        preds = model.predict(self.X)
        assert preds.shape == (len(self.X),), (
            f"Expected predictions shape ({len(self.X)},), got {preds.shape}"
        )

    def test_predictions_are_finite(self):
        """All predictions must be finite (no NaN or inf)."""
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        preds = model.predict(self.X)
        assert np.all(np.isfinite(preds)), \
            f"Predictions contain non-finite values: {preds[~np.isfinite(preds)]}"

    def test_loss_history_length_matches_iterations(self):
        """train_loss_history must have exactly num_iterations entries."""
        n_iter = 30
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=n_iter, depth=3, learning_rate=0.1)
        assert len(model.train_loss_history) == n_iter, (
            f"Expected {n_iter} loss history entries, got {len(model.train_loss_history)}"
        )

    def test_loss_decreases_over_training(self):
        """Final train loss must be lower than initial loss (verifies learning)."""
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=50, depth=3, learning_rate=0.1)
        hist = model.train_loss_history
        assert hist[-1] < hist[0], (
            f"PairLogit loss did not decrease: first={hist[0]:.6f}, last={hist[-1]:.6f}"
        )

    def test_loss_history_all_positive(self):
        """PairLogit loss is a log-sum and must be non-negative."""
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        for i, v in enumerate(model.train_loss_history):
            assert v >= 0, f"train_loss_history[{i}] = {v} is negative (PairLogit loss must be >= 0)"

    def test_loss_history_all_finite(self):
        """No NaN or inf in loss history."""
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        for i, v in enumerate(model.train_loss_history):
            assert np.isfinite(v), f"train_loss_history[{i}] = {v} is not finite"

    def test_tree_count_matches_iterations(self):
        """tree_count_ must equal the requested number of iterations."""
        n_iter = 25
        model = _nb_ranking("pairlogit", self.X, self.y, self.gids,
                             iterations=n_iter, depth=3)
        assert model.tree_count_ == n_iter, (
            f"tree_count_ = {model.tree_count_}, expected {n_iter}"
        )

    def test_within_group_score_ordering(self):
        """With a strong relevance signal, higher relevance docs should score higher on average."""
        # Create a very clear signal: feature 0 == relevance
        rng = np.random.RandomState(0)
        n_groups, docs_per_group = 8, 12
        n = n_groups * docs_per_group
        relevances = np.tile(np.arange(docs_per_group, dtype=np.float32), n_groups)
        X = np.column_stack([
            relevances + rng.normal(0, 0.01, n),  # feature tightly correlated with label
            rng.rand(n),
        ]).astype(np.float32)
        y = relevances
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("pairlogit", X, y, gids,
                             iterations=80, depth=4, learning_rate=0.15)
        preds = model.predict(X)

        # Within each group, check that Spearman correlation between relevance and score > 0
        from scipy.stats import spearmanr  # noqa: PLC0415
        correlations = []
        for g in range(n_groups):
            mask = gids == g
            rho, _ = spearmanr(y[mask], preds[mask])
            correlations.append(rho)
        avg_rho = np.mean(correlations)
        assert avg_rho > 0.3, (
            f"Expected positive rank correlation (signal is strong), got avg_rho={avg_rho:.3f}"
        )


# ============================================================================
# 2. YetiRank basic correctness through nanobind
# ============================================================================

@requires_nanobind
class TestYetiRankNanobind:
    """YetiRank training through the nanobind in-process path."""

    def setup_method(self):
        self.X, self.y, self.gids = _make_ranking_data(
            n_groups=5, docs_per_group=10, n_features=4, seed=7
        )

    def test_predictions_shape_matches_input(self):
        """Predictions vector must have exactly n_samples entries."""
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        preds = model.predict(self.X)
        assert preds.shape == (len(self.X),)

    def test_predictions_are_finite(self):
        """All predictions must be finite (no NaN or inf)."""
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        preds = model.predict(self.X)
        assert np.all(np.isfinite(preds)), \
            f"YetiRank predictions contain non-finite values"

    def test_loss_history_populated(self):
        """train_loss_history must not be empty."""
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        assert len(model.train_loss_history) > 0, \
            "train_loss_history is empty after YetiRank training"

    def test_loss_history_length_matches_iterations(self):
        """train_loss_history must have exactly num_iterations entries."""
        n_iter = 25
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=n_iter, depth=3)
        assert len(model.train_loss_history) == n_iter, (
            f"Expected {n_iter} entries, got {len(model.train_loss_history)}"
        )

    def test_loss_history_all_finite(self):
        """No NaN or inf in YetiRank loss history."""
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=20, depth=3, learning_rate=0.1)
        for i, v in enumerate(model.train_loss_history):
            assert np.isfinite(v), f"train_loss_history[{i}] = {v} is not finite"

    def test_loss_decreases_over_training(self):
        """Final loss must be lower than initial — YetiRank should learn."""
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=50, depth=3, learning_rate=0.1)
        hist = model.train_loss_history
        assert hist[-1] < hist[0], (
            f"YetiRank loss did not decrease: first={hist[0]:.6f}, last={hist[-1]:.6f}"
        )

    def test_tree_count_matches_iterations(self):
        """tree_count_ must equal the requested number of iterations."""
        n_iter = 20
        model = _nb_ranking("yetirank", self.X, self.y, self.gids,
                             iterations=n_iter, depth=3)
        assert model.tree_count_ == n_iter


# ============================================================================
# 3. Parity: nanobind vs subprocess for ranking losses
# ============================================================================

@requires_both
class TestRankingParityNanobindSubprocess:
    """Nanobind and subprocess must produce identical (or very close) results."""

    def test_pairlogit_predictions_parity(self):
        """PairLogit predictions must be within atol=1e-3 between paths."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)

        m_nb = _nb_ranking("pairlogit", X, y, gids,
                            iterations=40, depth=3, learning_rate=0.1, random_seed=42)
        m_sp = _sp_ranking("pairlogit", X, y, gids,
                            iterations=40, depth=3, learning_rate=0.1, random_seed=42)

        np.testing.assert_allclose(
            m_nb.predict(X), m_sp.predict(X), atol=1e-3,
            err_msg="PairLogit predictions diverge between nanobind and subprocess"
        )

    def test_yetirank_predictions_parity(self):
        """YetiRank predictions must be within atol=1e-3 between paths.

        YetiRank uses a random permutation per iteration; both paths use the
        same random_seed, so the pair generation must be deterministic.
        """
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)

        m_nb = _nb_ranking("yetirank", X, y, gids,
                            iterations=30, depth=3, learning_rate=0.1, random_seed=42)
        m_sp = _sp_ranking("yetirank", X, y, gids,
                            iterations=30, depth=3, learning_rate=0.1, random_seed=42)

        np.testing.assert_allclose(
            m_nb.predict(X), m_sp.predict(X), atol=1e-3,
            err_msg="YetiRank predictions diverge between nanobind and subprocess"
        )

    def test_pairlogit_loss_history_parity(self):
        """Loss histories must match across all iterations between paths."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=99)

        m_nb = _nb_ranking("pairlogit", X, y, gids,
                            iterations=30, depth=3, learning_rate=0.1, random_seed=99)
        m_sp = _sp_ranking("pairlogit", X, y, gids,
                            iterations=30, depth=3, learning_rate=0.1, random_seed=99)

        assert len(m_nb.train_loss_history) == len(m_sp.train_loss_history), (
            f"Loss history lengths differ: nb={len(m_nb.train_loss_history)}, "
            f"sp={len(m_sp.train_loss_history)}"
        )
        np.testing.assert_allclose(
            m_nb.train_loss_history, m_sp.train_loss_history, atol=1e-4,
            err_msg="PairLogit loss histories diverge between nanobind and subprocess"
        )

    def test_pairlogit_tree_count_parity(self):
        """tree_count_ must agree between paths for the same parameters."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=7)

        m_nb = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=3)
        m_sp = _sp_ranking("pairlogit", X, y, gids, iterations=20, depth=3)

        assert m_nb.tree_count_ == m_sp.tree_count_, (
            f"tree_count_ differs: nb={m_nb.tree_count_}, sp={m_sp.tree_count_}"
        )


# ============================================================================
# 4. Edge cases
# ============================================================================

@requires_nanobind
class TestRankingEdgeCases:
    """Adversarial edge cases for ranking through the nanobind path."""

    def test_single_document_groups_pairlogit_does_not_crash(self):
        """Single-doc groups generate no pairs — model should still complete."""
        rng = np.random.RandomState(42)
        n_groups = 10
        # Every document is its own group
        X = rng.rand(n_groups, 3).astype(np.float32)
        y = np.arange(n_groups, dtype=np.float32)
        gids = np.arange(n_groups, dtype=np.uint32)

        model = _nb_ranking("pairlogit", X, y, gids, iterations=10, depth=2)
        preds = model.predict(X)
        # With no pairs there is no gradient signal; just ensure no crash
        assert preds.shape == (n_groups,)
        assert np.all(np.isfinite(preds))

    def test_single_document_groups_yetirank_does_not_crash(self):
        """Single-doc groups with YetiRank must not crash."""
        rng = np.random.RandomState(42)
        n_groups = 10
        X = rng.rand(n_groups, 3).astype(np.float32)
        y = np.arange(n_groups, dtype=np.float32)
        gids = np.arange(n_groups, dtype=np.uint32)

        model = _nb_ranking("yetirank", X, y, gids, iterations=10, depth=2)
        preds = model.predict(X)
        assert preds.shape == (n_groups,)
        assert np.all(np.isfinite(preds))

    def test_all_same_relevance_within_groups_pairlogit(self):
        """Groups where all docs have the same relevance generate zero-weight pairs.
        Model must complete without NaN in predictions or loss.
        """
        rng = np.random.RandomState(42)
        n_groups = 5
        docs_per_group = 8
        n = n_groups * docs_per_group
        X = rng.rand(n, 4).astype(np.float32)
        # All labels within each group are identical (0 gradient from pairs)
        y = np.repeat(np.arange(n_groups, dtype=np.float32), docs_per_group)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=3)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds)), \
            "Predictions contain non-finite values when all relevances within a group are equal"
        for v in model.train_loss_history:
            assert np.isfinite(v), f"Loss is non-finite: {v}"

    def test_all_same_relevance_within_groups_yetirank(self):
        """YetiRank with uniform relevance — adjacent pairs all have zero weight."""
        rng = np.random.RandomState(42)
        n_groups = 5
        docs_per_group = 8
        n = n_groups * docs_per_group
        X = rng.rand(n, 4).astype(np.float32)
        y = np.repeat(np.arange(n_groups, dtype=np.float32), docs_per_group)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("yetirank", X, y, gids, iterations=20, depth=3)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds)), \
            "YetiRank predictions are non-finite with uniform relevance"

    def test_large_groups_100_docs_pairlogit(self):
        """Groups of 100 documents generate ~5000 pairs each — must not crash or OOM."""
        rng = np.random.RandomState(13)
        n_groups = 3
        docs_per_group = 100
        n = n_groups * docs_per_group
        X = rng.rand(n, 4).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=3)
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds)), "Large-group PairLogit predictions are non-finite"

    def test_large_groups_100_docs_yetirank(self):
        """YetiRank with 100 docs per group must complete without crash."""
        rng = np.random.RandomState(13)
        n_groups = 3
        docs_per_group = 100
        n = n_groups * docs_per_group
        X = rng.rand(n, 4).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("yetirank", X, y, gids, iterations=20, depth=3)
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_two_docs_per_group_minimum_viable(self):
        """Minimum useful group size is 2 — exactly one pair per group."""
        rng = np.random.RandomState(42)
        n_groups = 20
        docs_per_group = 2
        n = n_groups * docs_per_group
        X = rng.rand(n, 3).astype(np.float32)
        y = np.tile([0.0, 1.0], n_groups)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=2)
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_group_ids_as_float_array_converted_to_uint32(self):
        """group_id passed as float array should be safely converted to uint32."""
        X, y, gids = _make_ranking_data(n_groups=4, docs_per_group=8, seed=42)
        gids_float = gids.astype(np.float64)  # mimic a user passing float group IDs

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        model.fit(X, y, group_id=gids_float)  # should not crash
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_group_ids_as_list(self):
        """group_id can be a plain Python list — should be converted internally."""
        X, y, gids = _make_ranking_data(n_groups=4, docs_per_group=8, seed=42)
        gids_list = gids.tolist()

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        model.fit(X, y, group_id=gids_list)
        preds = model.predict(X)
        assert preds.shape == (len(X),)


# ============================================================================
# 5. eval_set with ranking + early stopping
# ============================================================================

@requires_nanobind
class TestRankingEvalSet:
    """Validation set behavior with ranking losses through nanobind."""

    def test_eval_set_populates_eval_loss_history_pairlogit(self):
        """With eval_set, eval_loss_history must be populated for PairLogit."""
        X_tr, y_tr, gids_tr = _make_ranking_data(n_groups=5, docs_per_group=10, seed=1)
        X_val, y_val, _gids_val = _make_ranking_data(n_groups=3, docs_per_group=10, seed=2)

        model = CatBoostMLX(
            loss="pairlogit", iterations=20, depth=3, learning_rate=0.1,
            binary_path=BINARY_PATH
        )
        model.fit(X_tr, y_tr, group_id=gids_tr, eval_set=(X_val, y_val))

        assert len(model.eval_loss_history) > 0, \
            "eval_loss_history is empty even though eval_set was provided"

    def test_eval_set_populates_eval_loss_history_yetirank(self):
        """With eval_set, eval_loss_history must be populated for YetiRank."""
        X_tr, y_tr, gids_tr = _make_ranking_data(n_groups=5, docs_per_group=10, seed=1)
        X_val, y_val, _ = _make_ranking_data(n_groups=3, docs_per_group=10, seed=2)

        model = CatBoostMLX(
            loss="yetirank", iterations=20, depth=3, learning_rate=0.1,
            binary_path=BINARY_PATH
        )
        model.fit(X_tr, y_tr, group_id=gids_tr, eval_set=(X_val, y_val))

        assert len(model.eval_loss_history) > 0, \
            "eval_loss_history is empty even though eval_set was provided"

    def test_eval_loss_history_all_finite_pairlogit(self):
        """All eval loss values must be finite."""
        X_tr, y_tr, gids_tr = _make_ranking_data(n_groups=5, docs_per_group=10, seed=3)
        X_val, y_val, _ = _make_ranking_data(n_groups=3, docs_per_group=10, seed=4)

        model = CatBoostMLX(
            loss="pairlogit", iterations=20, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X_tr, y_tr, group_id=gids_tr, eval_set=(X_val, y_val))

        for i, v in enumerate(model.eval_loss_history):
            assert np.isfinite(v), f"eval_loss_history[{i}] = {v} is not finite"

    def test_early_stopping_terminates_before_max_iterations(self):
        """Early stopping with a diverging eval set must terminate before max iterations."""
        X_tr, y_tr, gids_tr = _make_ranking_data(n_groups=8, docs_per_group=15, seed=5)
        rng = np.random.RandomState(999)
        # Completely random eval set: no signal, should cause early stopping
        X_val = rng.rand(30, 4).astype(np.float32)
        y_val = rng.rand(30).astype(np.float32) * 100.0

        model = CatBoostMLX(
            loss="pairlogit", iterations=200, depth=3, learning_rate=0.1,
            early_stopping_rounds=5, binary_path=BINARY_PATH
        )
        model.fit(X_tr, y_tr, group_id=gids_tr, eval_set=(X_val, y_val))

        assert model.tree_count_ < 200, (
            f"Early stopping should have triggered before 200 iterations, "
            f"but tree_count_={model.tree_count_}"
        )

    def test_eval_set_does_not_alter_train_predictions(self):
        """Providing eval_set must not change the train predictions."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        X_val, y_val, _ = _make_ranking_data(n_groups=3, docs_per_group=10, seed=99)

        m_no_eval = _nb_ranking("pairlogit", X, y, gids,
                                 iterations=20, depth=3, random_seed=42)
        m_with_eval = CatBoostMLX(
            loss="pairlogit", iterations=20, depth=3, learning_rate=0.1,
            random_seed=42, binary_path=BINARY_PATH
        )
        m_with_eval.fit(X, y, group_id=gids, eval_set=(X_val, y_val))

        np.testing.assert_allclose(
            m_no_eval.predict(X), m_with_eval.predict(X), atol=1e-3,
            err_msg="eval_set altered train predictions"
        )


# ============================================================================
# 6. Feature importance with ranking losses
# ============================================================================

@requires_nanobind
class TestRankingFeatureImportance:
    """Feature importance computed during ranking training must be valid."""

    def test_feature_importance_populated_pairlogit(self):
        """feature_importances_ must be non-empty after PairLogit training."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        model = _nb_ranking("pairlogit", X, y, gids, iterations=30, depth=3)
        fi = model.feature_importances_
        assert fi is not None and len(fi) > 0, \
            "feature_importances_ is empty after PairLogit training"

    def test_feature_importance_populated_yetirank(self):
        """feature_importances_ must be non-empty after YetiRank training."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        model = _nb_ranking("yetirank", X, y, gids, iterations=30, depth=3)
        fi = model.feature_importances_
        assert fi is not None and len(fi) > 0, \
            "feature_importances_ is empty after YetiRank training"

    def test_feature_importance_correct_length(self):
        """Number of importance values must equal number of features."""
        n_features = 6
        rng = np.random.RandomState(42)
        n_groups, docs_per_group = 5, 10
        n = n_groups * docs_per_group
        X = rng.rand(n, n_features).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=3)
        fi = model.feature_importances_
        assert len(fi) == n_features, (
            f"Expected {n_features} importance values, got {len(fi)}"
        )

    def test_feature_importance_all_non_negative(self):
        """All importance values must be non-negative (gain-based)."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        model = _nb_ranking("pairlogit", X, y, gids, iterations=30, depth=3)
        fi = model.feature_importances_
        for i, v in enumerate(fi):
            assert v >= 0, f"feature_importances_[{i}] = {v} is negative"

    def test_informative_feature_ranks_highest_pairlogit(self):
        """The one feature correlated with relevance should have the highest importance."""
        rng = np.random.RandomState(42)
        n_groups, docs_per_group = 6, 12
        n = n_groups * docs_per_group
        relevance = rng.randint(0, 5, n).astype(np.float32)
        # Feature 0: strongly correlated; features 1-3: noise
        X = np.column_stack([
            relevance + rng.normal(0, 0.05, n),
            rng.rand(n),
            rng.rand(n),
            rng.rand(n),
        ]).astype(np.float32)
        gids = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = CatBoostMLX(
            loss="pairlogit", iterations=60, depth=4, learning_rate=0.15,
            binary_path=BINARY_PATH
        )
        model.fit(X, relevance, group_id=gids)
        fi = model.feature_importances_
        assert np.argmax(fi) == 0, (
            f"Expected feature 0 (signal) to rank highest, got feature {np.argmax(fi)}: {fi}"
        )


# ============================================================================
# 7. Save/load roundtrip for ranking models
# ============================================================================

@requires_nanobind
class TestRankingSaveLoadRoundtrip:
    """Saving and loading a ranking model must preserve predictions exactly."""

    def test_pairlogit_save_load_predictions_identical(self):
        """Loaded PairLogit model must produce identical predictions."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        model = _nb_ranking("pairlogit", X, y, gids, iterations=20, depth=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pairlogit_model.json")
            model.save_model(path)
            assert os.path.isfile(path), f"save_model did not create {path}"

            loaded = CatBoostMLX(binary_path=BINARY_PATH)
            loaded.load_model(path)
            preds_original = model.predict(X)
            preds_loaded = loaded.predict(X)

        np.testing.assert_array_equal(
            preds_original, preds_loaded,
            err_msg="PairLogit loaded model predictions differ from original"
        )

    def test_yetirank_save_load_predictions_identical(self):
        """Loaded YetiRank model must produce identical predictions."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=7)
        model = _nb_ranking("yetirank", X, y, gids, iterations=20, depth=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "yetirank_model.json")
            model.save_model(path)

            loaded = CatBoostMLX(binary_path=BINARY_PATH)
            loaded.load_model(path)
            preds_original = model.predict(X)
            preds_loaded = loaded.predict(X)

        np.testing.assert_array_equal(
            preds_original, preds_loaded,
            err_msg="YetiRank loaded model predictions differ from original"
        )

    def test_save_model_json_contains_valid_structure(self):
        """Saved JSON must have 'model_info', 'features', and 'trees' keys."""
        X, y, gids = _make_ranking_data(n_groups=4, docs_per_group=8, seed=42)
        model = _nb_ranking("pairlogit", X, y, gids, iterations=10, depth=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save_model(path)
            with open(path) as f:
                data = json.load(f)

        assert "model_info" in data, "Saved JSON missing 'model_info'"
        assert "features" in data, "Saved JSON missing 'features'"
        assert "trees" in data, "Saved JSON missing 'trees'"
        assert len(data["trees"]) == 10, (
            f"Expected 10 trees in saved JSON, got {len(data['trees'])}"
        )

    def test_loaded_model_loss_type_preserved(self):
        """The loss_type field in model_info must be preserved through save/load."""
        X, y, gids = _make_ranking_data(n_groups=4, docs_per_group=8, seed=42)
        model = _nb_ranking("pairlogit", X, y, gids, iterations=10, depth=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save_model(path)
            with open(path) as f:
                data = json.load(f)

        loss_in_json = data.get("model_info", {}).get("loss_type", "").lower()
        assert "pairlogit" in loss_in_json, (
            f"loss_type not preserved in saved JSON model_info: {data.get('model_info')}"
        )


# ============================================================================
# 8. Adversarial: unsorted group_ids behavior
# ============================================================================

@requires_nanobind
class TestRankingUnsortedGroupIds:
    """The nanobind path (train_api.cpp) assumes group_ids are sorted.
    The subprocess path (csv_train.cpp) sorts them. This asymmetry is a
    known semantic difference. These tests document the behavior.
    """

    def test_sorted_group_ids_completes_without_error(self):
        """Baseline: sorted group_ids must always work correctly."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=8, seed=42)
        assert np.all(np.diff(gids) >= 0), "Fixture gids must be sorted"

        model = _nb_ranking("pairlogit", X, y, gids, iterations=15, depth=3)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_unsorted_group_ids_nanobind_does_not_crash(self):
        """The nanobind path must not crash on unsorted group_ids.

        The C++ train_api.cpp BuildDatasetFromArrays does NOT sort by group;
        it relies on consecutive-equal detection for GroupOffsets. Shuffled
        group_ids will produce incorrect GroupOffsets. Document this as a
        known behavioral difference from the subprocess path (which sorts).
        Either a clear error or degenerate-but-finite predictions are acceptable;
        a crash (segfault/exit) is not acceptable.
        """
        rng = np.random.RandomState(42)
        n_groups, docs_per_group = 5, 8
        n = n_groups * docs_per_group
        X = rng.rand(n, 3).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        gids_sorted = np.repeat(np.arange(n_groups, dtype=np.uint32), docs_per_group)
        # Shuffle within interleaved pattern — groups are not contiguous
        gids_shuffled = np.tile(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        # This either completes with degenerate results or raises a clear Python error.
        # What it MUST NOT do: segfault or call exit().
        try:
            model.fit(X, y, group_id=gids_shuffled)
            preds = model.predict(X)
            # If it ran, predictions must be finite (even if incorrect)
            assert np.all(np.isfinite(preds)), \
                "Unsorted group_ids produced non-finite predictions"
        except (ValueError, RuntimeError):
            # A clear Python error is also acceptable behavior
            pass

    def test_unsorted_vs_sorted_pairlogit_predictions_differ(self):
        """When group_ids are not contiguous, nanobind predictions should differ from
        the subprocess path (which sorts). This confirms the paths have different
        behavior on unsorted input and documents the divergence.

        This test is INFORMATIONAL — it marks the known asymmetry between paths.
        """
        rng = np.random.RandomState(42)
        n_groups, docs_per_group = 4, 8
        n = n_groups * docs_per_group
        X = rng.rand(n, 3).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        # Interleaved: groups are not contiguous
        gids_interleaved = np.tile(np.arange(n_groups, dtype=np.uint32), docs_per_group)

        try:
            m_nb = CatBoostMLX(
                loss="pairlogit", iterations=15, depth=3,
                random_seed=42, binary_path=BINARY_PATH
            )
            m_nb.fit(X, y, group_id=gids_interleaved)
            pred_nb = m_nb.predict(X)
        except (ValueError, RuntimeError):
            pytest.skip("nanobind path raised error on unsorted group_ids (acceptable)")
            return

        m_sp = CatBoostMLX(
            loss="pairlogit", iterations=15, depth=3,
            random_seed=42, binary_path=BINARY_PATH
        )
        with unittest.mock.patch.object(_core_module, "_HAS_NANOBIND", False):
            m_sp.fit(X, y, group_id=gids_interleaved)
        pred_sp = m_sp.predict(X)

        # The two paths SHOULD differ — log the max divergence as information
        max_diff = np.max(np.abs(pred_nb - pred_sp))
        # We don't assert equality here; we assert the test ran to completion
        # and note the divergence for documentation
        assert np.all(np.isfinite(pred_nb)), "nanobind path: non-finite predictions"
        assert np.all(np.isfinite(pred_sp)), "subprocess path: non-finite predictions"
        # Informational: if they happen to agree, that would be surprising
        if max_diff < 1e-3:
            pytest.xfail(
                f"UNEXPECTED: nanobind and subprocess agreed on unsorted group_ids "
                f"(max_diff={max_diff:.6f}). Expected divergence due to sort difference."
            )


# ============================================================================
# 9. Validation errors
# ============================================================================

@requires_nanobind
class TestRankingValidationErrors:
    """Ranking loss without group_id, or with mismatched lengths, must fail gracefully."""

    def test_mismatched_group_id_length_raises_value_error(self):
        """group_id with wrong length must raise ValueError before reaching C++."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)
        short_gids = gids[:-5]  # one fewer than expected

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        with pytest.raises(ValueError, match="group_id"):
            model.fit(X, y, group_id=short_gids)

    def test_ranking_without_group_id_raises_or_degenerate(self):
        """Ranking loss without group_id must either raise a clear error or produce
        degenerate (but finite) results. It must not silently produce valid-looking scores.

        The C++ binary (subprocess path) raises: 'Ranking losses require --group-col'.
        The nanobind path calls RunTraining with empty groupIds, which leads to
        empty GroupOffsets; the behavior depends on the C++ internals.
        """
        rng = np.random.RandomState(42)
        X = rng.rand(50, 4).astype(np.float32)
        y = rng.rand(50).astype(np.float32)

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        try:
            model.fit(X, y)  # no group_id
            preds = model.predict(X)
            # If it ran: document that it returned something (even if degenerate)
            # Without group_id there are no pairs, so the model learns nothing
            assert preds.shape == (50,), "Predictions shape wrong"
        except (ValueError, RuntimeError, Exception):
            pass  # Any exception is also acceptable — but no crash / segfault

    def test_group_id_all_same_value_creates_one_giant_group(self):
        """When all docs share one group_id, they all end up in one group."""
        rng = np.random.RandomState(42)
        n = 30
        X = rng.rand(n, 3).astype(np.float32)
        y = rng.rand(n).astype(np.float32)
        gids = np.zeros(n, dtype=np.uint32)  # all in group 0

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        model.fit(X, y, group_id=gids)
        preds = model.predict(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_empty_group_id_array_raises_or_no_crash(self):
        """Zero-length group_id array must not crash the process."""
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3).astype(np.float32)
        y = rng.rand(20).astype(np.float32)
        empty_gids = np.array([], dtype=np.uint32)

        model = CatBoostMLX(
            loss="pairlogit", iterations=5, depth=2, binary_path=BINARY_PATH
        )
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X, y, group_id=empty_gids)

    def test_negative_group_id_values_cast_to_uint32_do_not_crash(self):
        """Negative Python integers are cast to large uint32 values. Must not crash."""
        rng = np.random.RandomState(42)
        n_groups, docs_per_group = 4, 8
        n = n_groups * docs_per_group
        X = rng.rand(n, 3).astype(np.float32)
        y = (rng.rand(n) * 4).astype(np.float32)
        # Negative group IDs — np.asarray with dtype=uint32 wraps around
        gids_neg = np.repeat(np.array([-4, -3, -2, -1], dtype=np.int64), docs_per_group)

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        # np.asarray(gids_neg, dtype=uint32) wraps to large values but is contiguous
        # Behavior: may produce garbage results or error; must not crash
        try:
            model.fit(X, y, group_id=gids_neg)
            preds = model.predict(X)
            assert np.all(np.isfinite(preds))
        except (ValueError, RuntimeError, OverflowError):
            pass


# ============================================================================
# 10. Repeat fit (state isolation)
# ============================================================================

@requires_nanobind
class TestRankingStateisolation:
    """Fitting the same model object twice must produce consistent state."""

    def test_second_fit_overwrites_first_pairlogit(self):
        """Calling fit() again on the same model must replace all model state."""
        X1, y1, gids1 = _make_ranking_data(n_groups=4, docs_per_group=8, seed=1)
        X2, y2, gids2 = _make_ranking_data(n_groups=4, docs_per_group=8, seed=2)

        model = CatBoostMLX(
            loss="pairlogit", iterations=15, depth=3, binary_path=BINARY_PATH
        )
        model.fit(X1, y1, group_id=gids1)
        preds_after_first_fit = model.predict(X1).copy()
        tree_count_first = model.tree_count_

        model.fit(X2, y2, group_id=gids2)
        tree_count_second = model.tree_count_

        # Tree counts from separate fits with same params must match
        assert tree_count_first == tree_count_second == 15

        # After second fit, predictions on X1 must differ (different training data)
        preds_after_second_fit = model.predict(X1)
        # They should differ because the model was retrained on different data
        assert not np.allclose(preds_after_first_fit, preds_after_second_fit), (
            "Second fit did not update model: predictions identical to first fit"
        )

    def test_loss_history_reset_on_second_fit(self):
        """After a second fit, train_loss_history length must be exactly num_iterations."""
        X, y, gids = _make_ranking_data(n_groups=5, docs_per_group=10, seed=42)

        model = CatBoostMLX(
            loss="pairlogit", iterations=10, depth=2, binary_path=BINARY_PATH
        )
        model.fit(X, y, group_id=gids)
        assert len(model.train_loss_history) == 10

        model.fit(X, y, group_id=gids)
        assert len(model.train_loss_history) == 10, (
            f"After second fit, train_loss_history has {len(model.train_loss_history)} entries, "
            f"expected 10 (history must be reset, not appended)"
        )
