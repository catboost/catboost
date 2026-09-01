"""
Integration tests for Pool.set_label() -- edge cases and complex scenarios.

Tests cover: multiclass, ranking (YetiRank/PairLogit), eval_set interaction,
quantized pools, baseline, group_id, cat_features, polars/pandas input types,
prediction equivalence vs Pool reconstruction, and GPU (if available).
"""

import numpy as np
import pytest
from catboost import (
    Pool, CatBoostClassifier, CatBoostRegressor, CatBoostRanker, CatBoostError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_available():
    """Check if CatBoost GPU training is available."""
    try:
        p = Pool([[1, 2], [3, 4]], label=[0, 1])
        m = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        m.fit(p)
        return True
    except Exception:
        return False


GPU = _gpu_available()
SKIP_GPU = pytest.mark.skipif(not GPU, reason="GPU not available")

RNG_SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def X100(rng):
    """100 x 10 float32 feature matrix."""
    return rng.standard_normal((100, 10)).astype(np.float32)


@pytest.fixture
def X500(rng):
    """500 x 10 float32 feature matrix for ranking tests."""
    return rng.standard_normal((500, 10)).astype(np.float32)


# ===========================================================================
# Multiclass
# ===========================================================================

class TestSetLabelMulticlass:
    """Multiclass-specific edge cases."""

    def test_binary_to_multiclass(self, X100, rng):
        """set_label can change number of classes: binary -> 4-class."""
        y_bin = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(data=X100, label=y_bin)

        clf_bin = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf_bin.fit(pool)
        assert len(clf_bin.classes_) == 2

        y_multi = rng.integers(0, 4, 100).astype(np.float32)
        pool.set_label(y_multi)

        clf_multi = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf_multi.fit(pool)
        assert len(clf_multi.classes_) == 4

    def test_multiclass_to_binary(self, X100, rng):
        """set_label can reduce number of classes: 5-class -> binary."""
        y_multi = rng.integers(0, 5, 100).astype(np.float32)
        pool = Pool(data=X100, label=y_multi)

        clf = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool)
        assert len(clf.classes_) == 5

        y_bin = rng.integers(0, 2, 100).astype(np.float32)
        pool.set_label(y_bin)

        clf2 = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf2.fit(pool)
        assert len(clf2.classes_) == 2

    def test_multiclass_prediction_equivalence(self, X100, rng):
        """set_label multiclass predictions must match Pool reconstruction."""
        y = rng.integers(0, 3, 100).astype(np.float32)
        params = dict(iterations=10, verbose=0, random_seed=RNG_SEED)

        # Old way: rebuild Pool
        pool_old = Pool(data=X100, label=y)
        clf_old = CatBoostClassifier(**params)
        clf_old.fit(pool_old)
        preds_old = clf_old.predict_proba(pool_old)

        # New way: set_label
        pool_new = Pool(data=X100)
        pool_new.set_label(y)
        clf_new = CatBoostClassifier(**params)
        clf_new.fit(pool_new)
        preds_new = clf_new.predict_proba(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)


# ===========================================================================
# Learning to Rank
# ===========================================================================

class TestSetLabelRanking:
    """Ranking-specific tests (YetiRank, PairLogit, QueryRMSE)."""

    @pytest.fixture
    def ranking_data(self, X500, rng):
        """500 samples in 20 groups of 25."""
        group_id = np.repeat(np.arange(20), 25)
        y = rng.standard_normal(500).astype(np.float32)
        return X500, group_id, y

    def test_yetirank_with_set_label(self, ranking_data, rng):
        """YetiRank training works after set_label."""
        X, group_id, y = ranking_data
        pool = Pool(data=X, group_id=group_id)
        pool.set_label(y)

        ranker = CatBoostRanker(
            loss_function="YetiRank", iterations=5, verbose=0, random_seed=RNG_SEED,
        )
        ranker.fit(pool)
        preds = ranker.predict(pool)
        assert preds.shape == (500,)
        assert np.isfinite(preds).all()
        assert preds.std() > 0, "Model should produce non-constant predictions"

    def test_queryrmse_with_set_label(self, ranking_data, rng):
        """QueryRMSE training works after set_label."""
        X, group_id, y = ranking_data
        pool = Pool(data=X, group_id=group_id)
        pool.set_label(y)

        ranker = CatBoostRanker(
            loss_function="QueryRMSE", iterations=5, verbose=0, random_seed=RNG_SEED,
        )
        ranker.fit(pool)
        preds = ranker.predict(pool)
        assert preds.shape == (500,)
        assert np.isfinite(preds).all()
        assert preds.std() > 0, "Model should produce non-constant predictions"

    def test_ranking_prediction_equivalence(self, ranking_data, rng):
        """Ranking predictions must match Pool reconstruction."""
        X, group_id, y = ranking_data
        params = dict(
            loss_function="YetiRank", iterations=10, verbose=0, random_seed=RNG_SEED,
        )

        pool_old = Pool(data=X, label=y, group_id=group_id)
        ranker_old = CatBoostRanker(**params)
        ranker_old.fit(pool_old)
        preds_old = ranker_old.predict(pool_old)

        pool_new = Pool(data=X, group_id=group_id)
        pool_new.set_label(y)
        ranker_new = CatBoostRanker(**params)
        ranker_new.fit(pool_new)
        preds_new = ranker_new.predict(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)

    def test_ranking_set_label_changes_target(self, ranking_data, rng):
        """Changing ranking target via set_label produces different model."""
        X, group_id, y = ranking_data
        pool = Pool(data=X, group_id=group_id, label=y)
        params = dict(
            loss_function="YetiRank", iterations=10, verbose=0, random_seed=RNG_SEED,
        )

        ranker1 = CatBoostRanker(**params)
        ranker1.fit(pool)
        preds1 = ranker1.predict(pool)

        y2 = rng.standard_normal(500).astype(np.float32)
        pool.set_label(y2)
        ranker2 = CatBoostRanker(**params)
        ranker2.fit(pool)
        preds2 = ranker2.predict(pool)

        assert not np.allclose(preds1, preds2), "Different labels should give different ranking"

    def test_pairlogit_with_set_label(self, X500, rng):
        """PairLogit with explicit pairs works after set_label."""
        group_id = np.repeat(np.arange(20), 25)
        y = rng.standard_normal(500).astype(np.float32)
        # Generate pairs within each group
        pairs = []
        for g in range(20):
            base = g * 25
            for i in range(5):
                winner = base + rng.integers(0, 25)
                loser = base + rng.integers(0, 25)
                if winner != loser:
                    pairs.append((winner, loser))
        pairs = np.array(pairs, dtype=np.int32)

        pool = Pool(data=X500, group_id=group_id, pairs=pairs)
        pool.set_label(y)

        ranker = CatBoostRanker(
            loss_function="PairLogit", iterations=5, verbose=0, random_seed=RNG_SEED,
        )
        ranker.fit(pool)
        preds = ranker.predict(pool)
        assert preds.shape == (500,)
        assert np.isfinite(preds).all()

    def test_queryrmse_prediction_equivalence(self, ranking_data, rng):
        """QueryRMSE: set_label predictions must match Pool reconstruction."""
        X, group_id, y = ranking_data
        params = dict(
            loss_function="QueryRMSE", iterations=10, verbose=0, random_seed=RNG_SEED,
        )

        pool_old = Pool(data=X, label=y, group_id=group_id)
        ranker_old = CatBoostRanker(**params)
        ranker_old.fit(pool_old)
        preds_old = ranker_old.predict(pool_old)

        pool_new = Pool(data=X, group_id=group_id)
        pool_new.set_label(y)
        ranker_new = CatBoostRanker(**params)
        ranker_new.fit(pool_new)
        preds_new = ranker_new.predict(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)


# ===========================================================================
# eval_set Interaction
# ===========================================================================

class TestSetLabelEvalSet:
    """eval_set behavior with set_label."""

    def test_set_label_on_eval_set_between_runs(self, X100, rng):
        """Changing eval_set labels between fit() calls works."""
        X_train = X100
        X_eval = rng.standard_normal((50, 10)).astype(np.float32)

        y_train = rng.integers(0, 2, 100).astype(np.float32)
        y_eval1 = rng.integers(0, 2, 50).astype(np.float32)
        y_eval2 = 1 - y_eval1  # Flipped labels

        train_pool = Pool(X_train, label=y_train)
        eval_pool = Pool(X_eval, label=y_eval1)

        clf1 = CatBoostClassifier(iterations=10, verbose=0, random_seed=RNG_SEED)
        clf1.fit(train_pool, eval_set=eval_pool)
        metrics1 = clf1.get_best_score()

        eval_pool.set_label(y_eval2)
        clf2 = CatBoostClassifier(iterations=10, verbose=0, random_seed=RNG_SEED)
        clf2.fit(train_pool, eval_set=eval_pool)
        metrics2 = clf2.get_best_score()

        # Same model trained, but eval metrics should differ (different eval labels)
        val_key = "validation"
        assert val_key in metrics1, "Expected 'validation' key in metrics1"
        assert val_key in metrics2, "Expected 'validation' key in metrics2"
        for metric_name in metrics1[val_key]:
            assert metrics1[val_key][metric_name] != metrics2[val_key][metric_name], \
                f"Eval metric {metric_name} should differ with flipped eval labels"

    def test_eval_set_with_pool_reuse(self, X100, rng):
        """Same Pool used for both train and eval after label mutation."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y1)

        clf = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool, eval_set=pool)  # Same pool as train and eval
        preds1 = clf.predict_proba(pool)

        # Guarantee y2 != y1 elementwise so the "different labels -> different models"
        # assertion is unconditional (bitflip avoids rng collision that would silently skip).
        y2 = (1 - y1).astype(np.float32)

        pool.set_label(y2)

        clf2 = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf2.fit(pool, eval_set=pool)
        preds2 = clf2.predict_proba(pool)

        assert not np.array_equal(preds1, preds2), \
            "Flipped labels must produce different predictions"


# ===========================================================================
# Quantized Pools
# ===========================================================================

class TestSetLabelQuantized:
    """set_label on quantized pools -- labels are NOT quantized, so it should work."""

    def test_set_label_on_quantized_pool_works(self, X100, rng):
        """set_label works on quantized pools (quantization only affects features)."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        pool.quantize()
        assert pool.is_quantized()

        new_y = 1 - y  # Flip labels
        pool.set_label(new_y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, new_y)

    def test_quantized_pool_training_after_set_label(self, X100, rng):
        """Training on quantized pool with mutated labels produces valid model."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        pool.quantize()

        new_y = rng.integers(0, 2, 100).astype(np.float32)
        pool.set_label(new_y)

        clf = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool)
        preds = clf.predict_proba(pool)
        assert preds.shape == (100, 2)
        assert np.isfinite(preds).all()
        assert preds[:, 0].std() > 0, "Model should produce non-constant predictions"

    def test_quantized_prediction_equivalence(self, X100, rng):
        """Quantized pool: set_label predictions must match fresh quantized Pool."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        params = dict(iterations=10, verbose=0, random_seed=RNG_SEED)

        # Old way: build and quantize from scratch
        pool_old = Pool(X100, label=y)
        pool_old.quantize()
        clf_old = CatBoostClassifier(**params)
        clf_old.fit(pool_old)
        preds_old = clf_old.predict_proba(pool_old)

        # New way: quantize without label, then set_label
        pool_new = Pool(X100)
        pool_new.quantize()
        pool_new.set_label(y)
        clf_new = CatBoostClassifier(**params)
        clf_new.fit(pool_new)
        preds_new = clf_new.predict_proba(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)


# ===========================================================================
# Baseline
# ===========================================================================

class TestSetLabelBaseline:
    """Interaction between set_label and baseline."""

    def test_set_label_preserves_baseline(self, X100, rng):
        """Baseline values must survive set_label."""
        y = rng.standard_normal(100).astype(np.float32)
        baseline = rng.standard_normal(100).astype(np.float32)
        pool = Pool(X100, label=y, baseline=baseline.reshape(-1, 1))

        pool.set_label(np.zeros(100, dtype=np.float32))
        # Baseline is stored as-is (not recomputed), so compare bit-exact -- any cast drift
        # here would be a real bug that looser tolerance would mask.
        result_baseline = np.array(pool.get_baseline()).ravel()
        np.testing.assert_array_equal(result_baseline, baseline)

    def test_training_with_baseline_and_set_label(self, X100, rng):
        """Training with baseline + set_label produces valid model."""
        baseline = rng.standard_normal(100).astype(np.float32).reshape(-1, 1)
        pool = Pool(X100, baseline=baseline)

        y = rng.standard_normal(100).astype(np.float32)
        pool.set_label(y)

        reg = CatBoostRegressor(iterations=5, verbose=0, random_seed=RNG_SEED)
        reg.fit(pool)
        preds = reg.predict(pool)
        assert preds.shape == (100,)
        assert np.isfinite(preds).all()


# ===========================================================================
# Cat features
# ===========================================================================

class TestSetLabelCatFeatures:
    """Interaction between set_label and categorical features."""

    def test_cat_features_preserved_after_set_label(self):
        """Cat feature indices and hash maps unaffected by set_label."""
        rng = np.random.default_rng(RNG_SEED)
        n = 200
        cat_col = rng.choice(["a", "b", "c", "d"], n)
        num_cols = rng.standard_normal((n, 5)).astype(np.float32)

        data = np.column_stack([cat_col.reshape(-1, 1), num_cols.astype(str)])
        y = rng.integers(0, 2, n).astype(np.float32)
        pool = Pool(data, label=y, cat_features=[0])

        assert pool.get_cat_feature_indices() == [0]

        pool.set_label(1 - y)
        assert pool.get_cat_feature_indices() == [0]

        # Train should still work with cat features
        clf = CatBoostClassifier(iterations=3, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool)
        preds = clf.predict(pool)
        assert len(preds) == n

    def test_cat_features_training_equivalence(self):
        """Predictions with cat features must match Pool reconstruction."""
        rng = np.random.default_rng(RNG_SEED)
        n = 200
        cat_col = rng.choice(["x", "y", "z"], n)
        num_cols = rng.standard_normal((n, 5)).astype(np.float32)
        data = np.column_stack([cat_col.reshape(-1, 1), num_cols.astype(str)])
        y = rng.integers(0, 2, n).astype(np.float32)

        params = dict(iterations=10, verbose=0, random_seed=RNG_SEED)

        pool_old = Pool(data, label=y, cat_features=[0])
        clf_old = CatBoostClassifier(**params)
        clf_old.fit(pool_old)
        preds_old = clf_old.predict_proba(pool_old)

        pool_new = Pool(data, cat_features=[0])
        pool_new.set_label(y)
        clf_new = CatBoostClassifier(**params)
        clf_new.fit(pool_new)
        preds_new = clf_new.predict_proba(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)


# ===========================================================================
# Input types (pandas, polars)
# ===========================================================================

class TestSetLabelInputTypes:
    """Various input types for set_label."""

    def test_pandas_series(self, X100):
        pd = pytest.importorskip("pandas")
        pool = Pool(X100)
        y = pd.Series(np.ones(100, dtype=np.float32))
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, np.ones(100, dtype=np.float32))

    def test_pandas_dataframe_single_column(self, X100):
        pd = pytest.importorskip("pandas")
        pool = Pool(X100)
        y = pd.DataFrame({"target": np.ones(100, dtype=np.float32)})
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, np.ones(100, dtype=np.float32))

    def test_polars_series(self, X100):
        pl = pytest.importorskip("polars")
        pool = Pool(X100)
        y = pl.Series("target", np.ones(100, dtype=np.float32))
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, np.ones(100, dtype=np.float32))

    def test_python_list_of_ints(self, X100):
        pool = Pool(X100)
        y = [0, 1] * 50
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        expected = np.array([0, 1] * 50, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_float64_array(self, X100):
        pool = Pool(X100)
        y = np.ones(100, dtype=np.float64) * 3.14
        pool.set_label(y)
        # Storage is float32 regardless of input dtype; compare bit-exact against
        # the explicit float32 cast so we'd notice any unexpected extra rounding.
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, np.full(100, 3.14, dtype=np.float32))

    def test_bool_array(self, X100):
        pool = Pool(X100)
        y = np.array([True, False] * 50)
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        expected = np.array([1.0, 0.0] * 50, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# Error handling
# ===========================================================================

class TestSetLabelErrors:
    """Error cases that must be caught."""

    def test_string_labels_rejected(self, X100):
        # The exact error path depends on where conversion fails (py_to_tvector or C++ target
        # type validation). The contract is just "string labels must not silently succeed".
        pool = Pool(X100)
        with pytest.raises((CatBoostError, TypeError, ValueError)):
            pool.set_label(["cat", "dog"] * 50)

    def test_2d_multi_column_rejected(self, X100):
        pool = Pool(X100)
        with pytest.raises(CatBoostError, match=r"(dimension|1-D|1 target|length)"):
            pool.set_label(np.ones((100, 3), dtype=np.float32))

    def test_nan_labels_accepted_for_training(self, X100, rng):
        """NaN labels should be accepted by set_label (training may reject them)."""
        pool = Pool(X100)
        y = np.ones(100, dtype=np.float32)
        y[0] = np.nan
        # set_label itself should not reject NaN -- that's the trainer's job
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        assert np.isnan(result[0])

    def test_inf_labels_accepted(self, X100):
        """Inf labels should be accepted by set_label."""
        pool = Pool(X100)
        y = np.ones(100, dtype=np.float32)
        y[0] = np.inf
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        assert np.isinf(result[0])

    def test_empty_label_rejected(self, X100):
        pool = Pool(X100)
        with pytest.raises(CatBoostError):
            pool.set_label(np.array([], dtype=np.float32))


# ===========================================================================
# Prediction equivalence (the core guarantee)
# ===========================================================================

class TestPredictionEquivalence:
    """The critical invariant: set_label predictions == Pool reconstruction predictions."""

    def _compare(self, X, y, model_class, model_params, extra_pool_kwargs=None):
        """Train with old way vs set_label, assert predictions identical."""
        pool_kwargs = extra_pool_kwargs or {}

        pool_old = Pool(data=X, label=y, **pool_kwargs)
        model_old = model_class(**model_params)
        model_old.fit(pool_old)
        preds_old = model_old.predict(pool_old)

        pool_new = Pool(data=X, **pool_kwargs)
        pool_new.set_label(y)
        model_new = model_class(**model_params)
        model_new.fit(pool_new)
        preds_new = model_new.predict(pool_new)

        np.testing.assert_array_equal(preds_old, preds_new)

    def test_regression_equivalence(self, X100, rng):
        y = rng.standard_normal(100).astype(np.float32)
        self._compare(X100, y, CatBoostRegressor, dict(
            iterations=20, verbose=0, random_seed=RNG_SEED,
        ))

    def test_regression_equivalence_float64(self, X100, rng):
        """Guards against the constructor / set_label dtype path diverging.

        Both paths are expected to downcast to float32 internally, so predictions
        must match bit-exactly. If someone later preserves float64 in one path
        but not the other, this test fails.
        """
        y = rng.standard_normal(100)  # float64 -- no .astype here
        self._compare(X100, y, CatBoostRegressor, dict(
            iterations=20, verbose=0, random_seed=RNG_SEED,
        ))

    def test_binary_classification_equivalence(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        self._compare(X100, y, CatBoostClassifier, dict(
            iterations=20, verbose=0, random_seed=RNG_SEED,
        ))

    def test_multiclass_equivalence(self, X100, rng):
        y = rng.integers(0, 5, 100).astype(np.float32)
        self._compare(X100, y, CatBoostClassifier, dict(
            iterations=20, verbose=0, random_seed=RNG_SEED,
        ))

    def test_ranking_equivalence(self, X500, rng):
        group_id = np.repeat(np.arange(20), 25)
        y = rng.standard_normal(500).astype(np.float32)
        self._compare(X500, y, CatBoostRanker, dict(
            loss_function="YetiRank", iterations=10, verbose=0, random_seed=RNG_SEED,
        ), extra_pool_kwargs=dict(group_id=group_id))

    def test_weighted_equivalence(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        w = rng.random(100).astype(np.float32) + 0.1
        self._compare(X100, y, CatBoostClassifier, dict(
            iterations=20, verbose=0, random_seed=RNG_SEED,
        ), extra_pool_kwargs=dict(weight=w))


# ===========================================================================
# GPU (if available)
# ===========================================================================

class TestSetLabelGPU:
    """GPU-specific tests."""

    @SKIP_GPU
    def test_gpu_training_after_set_label(self, X100, rng):
        """set_label + GPU training produces valid predictions."""
        pool = Pool(X100)
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool.set_label(y)

        clf = CatBoostClassifier(
            iterations=5, task_type="GPU", devices="0", verbose=0, random_seed=RNG_SEED,
        )
        clf.fit(pool)
        preds = clf.predict_proba(pool)
        assert preds.shape == (100, 2)
        assert np.isfinite(preds).all()

    @SKIP_GPU
    def test_gpu_vs_cpu_set_label_consistency(self, X100, rng):
        """GPU and CPU should produce correlated predictions on the same Pool+labels.

        GPU and CPU implementations use different algorithms, so exact equality is not
        expected -- but predictions must be positively correlated; near-zero or negative
        correlation signals a real label/feature plumbing bug, not algorithmic variance.
        """
        pool = Pool(X100)
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool.set_label(y)

        params = dict(iterations=10, verbose=0, random_seed=RNG_SEED)

        clf_cpu = CatBoostClassifier(task_type="CPU", **params)
        clf_cpu.fit(pool)
        preds_cpu = clf_cpu.predict_proba(pool)[:, 1]

        pool.set_label(y)  # Re-set to ensure clean state for second fit
        clf_gpu = CatBoostClassifier(task_type="GPU", devices="0", **params)
        clf_gpu.fit(pool)
        preds_gpu = clf_gpu.predict_proba(pool)[:, 1]

        assert np.isfinite(preds_gpu).all()
        assert preds_gpu.shape == preds_cpu.shape

        # Use a tolerant but non-trivial correlation threshold -- rules out broken plumbing
        # without being fragile to algorithmic/rng differences between backends.
        corr = np.corrcoef(preds_cpu, preds_gpu)[0, 1]
        assert corr > 0.5, (
            "GPU and CPU predictions should correlate on same labels; got corr = {:.3f}".format(corr)
        )

    @SKIP_GPU
    def test_gpu_prediction_equivalence(self, X100, rng):
        """GPU: set_label predictions must match Pool reconstruction."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        params = dict(
            iterations=10, task_type="GPU", devices="0", verbose=0, random_seed=RNG_SEED,
        )

        pool_old = Pool(X100, label=y)
        clf_old = CatBoostClassifier(**params)
        clf_old.fit(pool_old)
        preds_old = clf_old.predict_proba(pool_old)

        pool_new = Pool(X100)
        pool_new.set_label(y)
        clf_new = CatBoostClassifier(**params)
        clf_new.fit(pool_new)
        preds_new = clf_new.predict_proba(pool_new)

        # GPU histogram builder uses atomic ops, so two identically-seeded runs are not
        # bit-exact. A tight rtol still catches set_label plumbing bugs without being flaky.
        np.testing.assert_allclose(preds_old, preds_new, rtol=1e-4, atol=1e-6)


# ===========================================================================
# Stress / sequential mutation
# ===========================================================================

class TestSetLabelStress:
    """Stress tests for repeated mutations."""

    def test_rapid_label_cycling(self, X100, rng):
        """Rapidly cycling labels 20 times should not corrupt Pool."""
        pool = Pool(X100)
        for i in range(20):
            y = rng.standard_normal(100).astype(np.float32) * (i + 1)
            pool.set_label(y)
            result = np.array(pool.get_label(), dtype=np.float32)
            np.testing.assert_array_almost_equal(result, y, decimal=5)

        # Final training must work
        reg = CatBoostRegressor(iterations=3, verbose=0, random_seed=RNG_SEED)
        reg.fit(pool)
        preds = reg.predict(pool)
        assert len(preds) == 100
        assert np.isfinite(preds).all()

    def test_alternating_cls_reg_targets(self, X100, rng):
        """Alternate classification and regression labels on same Pool."""
        pool = Pool(X100)

        for i in range(10):
            if i % 2 == 0:
                y = rng.integers(0, 2, 100).astype(np.float32)
                pool.set_label(y)
                clf = CatBoostClassifier(iterations=2, verbose=0, random_seed=RNG_SEED + i)
                clf.fit(pool)
                preds = clf.predict(pool)
                assert preds.shape == (100,)
                assert np.isfinite(preds.astype(float)).all()
            else:
                y = rng.standard_normal(100).astype(np.float32)
                pool.set_label(y)
                reg = CatBoostRegressor(iterations=2, verbose=0, random_seed=RNG_SEED + i)
                reg.fit(pool)
                preds = reg.predict(pool)
                assert preds.shape == (100,)
                assert np.isfinite(preds).all()

    def test_group_id_preserved_through_mutations(self, X500, rng):
        """group_id must survive multiple set_label calls."""
        group_id = np.repeat(np.arange(20), 25)
        pool = Pool(X500, group_id=group_id)

        for _ in range(5):
            y = rng.standard_normal(500).astype(np.float32)
            pool.set_label(y)

        # Ranking still works => group_id intact
        ranker = CatBoostRanker(
            loss_function="YetiRank", iterations=3, verbose=0, random_seed=RNG_SEED,
        )
        ranker.fit(pool)
        preds = ranker.predict(pool)
        assert preds.shape == (500,)
        assert np.isfinite(preds).all()


# ===========================================================================
# Additional coverage (found by code review)
# ===========================================================================

class TestSetLabelAdditional:
    """Tests for gaps identified by code review agents."""

    def test_non_contiguous_class_ids(self, X100, rng):
        """Sparse integer class labels (0, 5, 10) should work."""
        y = rng.choice([0, 5, 10], 100).astype(np.float32)
        pool = Pool(X100)
        pool.set_label(y)
        clf = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool)
        assert len(clf.classes_) == 3
        preds = clf.predict_proba(pool)
        assert preds.shape == (100, 3)

    def test_single_sample_pool(self):
        """set_label works on a pool with a single sample."""
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        pool = Pool(X)
        pool.set_label(np.array([1.0], dtype=np.float32))
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, [1.0])

    def test_set_label_with_2d_column_array(self, X100, rng):
        """(N, 1) shaped label array should be accepted and flattened."""
        y = rng.integers(0, 2, (100, 1)).astype(np.float32)
        pool = Pool(X100)
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        assert result.shape == (100,)

    def test_float64_precision_preserved(self, X100, rng):
        """float64 labels round-trip through float32 storage bit-exactly.

        This is the contract: underlying storage is float32, so get_label() must
        equal y.astype(float32) exactly. Looser tolerance (decimal=5) would hide
        an unexpected extra round-trip through double.
        """
        y = rng.standard_normal(100)  # float64
        pool = Pool(X100)
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, y.astype(np.float32))

    def test_set_weight_then_set_label(self, X100, rng):
        """set_weight + set_label in combination should not interfere."""
        pool = Pool(X100)
        w = rng.random(100).astype(np.float32) + 0.1
        pool.set_weight(w)
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool.set_label(y)

        result_w = np.array(pool.get_weight(), dtype=np.float32)
        result_y = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_almost_equal(result_w, w, decimal=5)
        np.testing.assert_array_equal(result_y, y)

        clf = CatBoostClassifier(iterations=5, verbose=0, random_seed=RNG_SEED)
        clf.fit(pool)
        preds = clf.predict_proba(pool)
        assert np.isfinite(preds).all()

    def test_integer_relevance_labels_ranking(self, X500, rng):
        """Ranking with integer relevance grades (0-4) after set_label."""
        group_id = np.repeat(np.arange(20), 25)
        y = rng.integers(0, 5, 500).astype(np.float32)
        pool = Pool(X500, group_id=group_id)
        pool.set_label(y)

        ranker = CatBoostRanker(
            loss_function="YetiRank", iterations=5, verbose=0, random_seed=RNG_SEED,
        )
        ranker.fit(pool)
        preds = ranker.predict(pool)
        assert preds.shape == (500,)
        assert np.isfinite(preds).all()


# ===========================================================================
# Additional coverage (from 5-agent review gaps H1-H9)
# ===========================================================================

class TestSetLabelCoverage:
    """Coverage added per review: dtype rejection, re-fit, pandas index, extreme values."""

    @pytest.mark.parametrize("bad", [
        np.array([b"a", b"b"] * 50),                      # bytes, dtype kind 'S'
        np.array([object(), object()] * 50, dtype=object),  # object, dtype kind 'O'
    ])
    def test_non_numeric_dtypes_rejected(self, X100, bad):
        """Dtypes S (bytes) and O (object) must not silently succeed."""
        pool = Pool(X100)
        with pytest.raises((CatBoostError, TypeError, ValueError)):
            pool.set_label(bad)

    def test_refit_same_pool_after_set_label(self, X100, rng):
        """fit -> set_label -> fit on same Pool must match fresh-Pool baseline."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        params = dict(iterations=8, verbose=0, random_seed=RNG_SEED)

        shared = Pool(X100, label=y1)
        CatBoostClassifier(**params).fit(shared)
        shared.set_label(y2)
        clf_refit = CatBoostClassifier(**params).fit(shared)
        preds_refit = clf_refit.predict_proba(shared)

        fresh = Pool(X100, label=y2)
        clf_fresh = CatBoostClassifier(**params).fit(fresh)
        preds_fresh = clf_fresh.predict_proba(fresh)

        # set_label must leave Pool in a state equivalent to a freshly-built one.
        np.testing.assert_allclose(preds_refit, preds_fresh, rtol=1e-6, atol=1e-8)

    def test_pandas_series_non_default_index(self, X100, rng):
        """pandas.Series with a shifted index must use values, not align by index."""
        import pandas as pd
        y = rng.integers(0, 2, 100).astype(np.float32)
        series = pd.Series(y, index=np.arange(1000, 1100))
        pool = Pool(X100)
        pool.set_label(series)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, y)

    @pytest.mark.parametrize("value", [1e30, -1e30, 1e-30])
    def test_extreme_finite_values(self, X100, value):
        """Extreme but finite labels must round-trip without overflow at set_label layer."""
        pool = Pool(X100)
        y = np.full(100, value, dtype=np.float32)
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, y)

    def test_negative_integer_labels(self, X100):
        """Negative integer labels (e.g. [-1, 0, 1]) must round-trip."""
        pool = Pool(X100)
        y = np.tile([-1, 0, 1, 2], 25).astype(np.int64)
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, y.astype(np.float32))


class TestSetLabelFitWarning:
    """The helper _set_pool_label_with_overwrite_warning must warn on silent overwrite."""

    def test_fit_with_prelabeled_pool_and_y_warns(self, X100, rng):
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100, label=y1)
        with pytest.warns(UserWarning, match=r"overrides the Pool's labels"):
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y2)

    def test_fit_with_unlabeled_pool_and_y_does_not_warn(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # promote UserWarning to error
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y)

    def test_second_fit_on_same_pool_warns(self, X100, rng):
        """Two consecutive fit(pool, y=...) calls: first no warn, second warns."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y1)

        with pytest.warns(UserWarning, match=r"overrides the Pool's labels"):
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y2)

    def test_warning_stacklevel_points_at_user_frame(self, X100, rng):
        """Warning must attribute to the user's code (this test file), not core.py."""
        import warnings
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100, label=y1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y2)
        matching = [w for w in caught if "overrides the Pool's labels" in str(w.message)]
        assert len(matching) == 1
        assert matching[0].filename.endswith("test_set_label.py"), (
            "Warning should attribute to user code, got filename {}".format(matching[0].filename)
        )


class TestScoreOverwriteWarning:
    """Cover the 3 score() call sites of _set_pool_label_with_overwrite_warning."""

    @pytest.mark.parametrize("model_cls", [CatBoostClassifier, CatBoostRegressor])
    def test_score_with_prelabeled_pool_warns(self, X100, rng, model_cls):
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100, label=y1)
        model = model_cls(iterations=3, verbose=0).fit(pool)
        with pytest.warns(UserWarning, match=r"score\(\) received a Pool.*overrides"):
            model.score(pool, y=y2)

    def test_ranker_score_with_prelabeled_pool_warns(self, X500, rng):
        group_id = np.repeat(np.arange(20), 25)
        y1 = rng.integers(0, 5, 500).astype(np.float32)
        y2 = rng.integers(0, 5, 500).astype(np.float32)
        pool = Pool(X500, label=y1, group_id=group_id)
        ranker = CatBoostRanker(
            loss_function="YetiRank", iterations=3, verbose=0, random_seed=RNG_SEED,
        ).fit(pool)
        with pytest.warns(UserWarning, match=r"score\(\) received a Pool.*overrides"):
            ranker.score(pool, y=y2)


class TestEvalSetFootgun:
    """fit(X=pool, eval_set=pool, y=y) silently mutated X's labels before our guard."""

    def test_same_pool_as_train_and_eval_with_y_raises(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        y_new = (1 - y).astype(np.float32)
        with pytest.raises(CatBoostError, match=r"same Pool object as both X and eval_set"):
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y_new, eval_set=pool)

    def test_same_pool_in_eval_set_list_with_y_raises(self, X100, rng):
        """eval_set can be a list; identity check must detect membership."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        y_new = (1 - y).astype(np.float32)
        with pytest.raises(CatBoostError, match=r"same Pool object as both X and eval_set"):
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y_new, eval_set=[pool])

    def test_same_pool_in_eval_set_tuple_with_y_raises(self, X100, rng):
        """eval_set can be a (X, y) tuple; identity check must match when X aliases."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        y_new = (1 - y).astype(np.float32)
        y_eval = (1 - y).astype(np.float32)
        with pytest.raises(CatBoostError, match=r"same Pool object as both X and eval_set"):
            CatBoostClassifier(iterations=3, verbose=0).fit(
                pool, y=y_new, eval_set=(pool, y_eval)
            )

    def test_same_pool_in_list_of_tuples_with_y_raises(self, X100, rng):
        """eval_set can be a list of (X, y) tuples; check each tuple's X slot."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        other = Pool(X100, label=y)
        y_new = (1 - y).astype(np.float32)
        with pytest.raises(CatBoostError, match=r"same Pool object as both X and eval_set"):
            CatBoostClassifier(iterations=3, verbose=0).fit(
                pool, y=y_new, eval_set=[(other, y), (pool, y)]
            )

    def test_different_pools_as_train_and_eval_does_not_raise(self, X100, rng):
        """Sanity: two distinct Pools must not trigger the guard."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool_train = Pool(X100, label=y)
        pool_eval = Pool(X100, label=y)
        CatBoostClassifier(iterations=3, verbose=0).fit(pool_train, eval_set=pool_eval)


class TestStrongExceptionGuarantee:
    """A failed set_label must leave the Pool in its prior state (C++ reorder guarantees this)."""

    def test_failed_set_label_preserves_state(self, X100, rng):
        y_orig = rng.integers(0, 2, 100).astype(np.float32)
        w_orig = np.linspace(0.1, 1.0, 100).astype(np.float32)
        pool = Pool(X100, label=y_orig, weight=w_orig)

        with pytest.raises((CatBoostError, TypeError, ValueError)):
            pool.set_label(np.zeros(50, dtype=np.float32))  # wrong length

        np.testing.assert_array_equal(np.array(pool.get_label(), dtype=np.float32), y_orig)
        np.testing.assert_array_almost_equal(
            np.array(pool.get_weight(), dtype=np.float32), w_orig, decimal=5
        )

        # Pool must still be usable: subsequent valid set_label + fit succeed.
        y_new = (1 - y_orig).astype(np.float32)
        pool.set_label(y_new)
        CatBoostClassifier(iterations=3, verbose=0).fit(pool)


class TestSetLabelQuantizeOrder:
    """set_label -> quantize (reverse of TestSetLabelQuantized's existing coverage)."""

    def test_set_label_then_quantize_preserves_label(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100)
        pool.set_label(y)
        pool.quantize()
        assert pool.is_quantized()
        result = np.array(pool.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, y)
        CatBoostClassifier(iterations=3, verbose=0).fit(pool)


class TestDtypePreservation:
    """get_label() should return the dtype the user passed in (matches constructor)."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_get_label_returns_input_dtype(self, X100, rng, dtype):
        y = rng.integers(0, 2, 100).astype(dtype)
        pool = Pool(X100)
        pool.set_label(y)
        result = np.array(pool.get_label())
        # Shadow target_type tracks input dtype; underlying storage is float32.
        assert result.dtype == np.dtype(dtype), (
            "get_label() should return input dtype {}, got {}".format(dtype, result.dtype)
        )


class TestSetLabelPolarsDataFrame:
    """Newly-added polars.DataFrame support (in addition to polars.Series)."""

    def test_single_column_polars_dataframe_accepted(self, X100, rng):
        pl = pytest.importorskip("polars")
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100)
        pool.set_label(pl.DataFrame({"target": y}))
        np.testing.assert_array_equal(np.array(pool.get_label(), dtype=np.float32), y)

    def test_multi_column_polars_dataframe_rejected(self, X100, rng):
        pl = pytest.importorskip("polars")
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100)
        with pytest.raises(CatBoostError):
            pool.set_label(pl.DataFrame({"a": y, "b": y}))


class TestNonFiniteValuesComprehensive:
    """Full NaN/Inf round-trip across all index positions and mixed cases."""

    def test_nan_inf_negative_inf_round_trip(self, X100):
        pool = Pool(X100)
        y = np.ones(100, dtype=np.float32)
        y[0], y[50], y[99] = np.nan, np.inf, -np.inf
        pool.set_label(y)
        result = np.array(pool.get_label(), dtype=np.float32)
        assert np.isnan(result[0])
        assert np.isposinf(result[50])
        assert np.isneginf(result[99])


# ===========================================================================
# Bug-hunt regression tests (from round-4 review)
# ===========================================================================

class TestStringLikeDtypesRejected:
    """Round-4 HIGH: dtype.kind in 'USV' (strings/bytes/void) must be rejected,
    not silently float-coerced through py_to_tvector's Python-level float() call."""

    def test_unicode_string_numeric_array_rejected(self, X100):
        pool = Pool(X100)
        # These parse successfully via Python float("1.5") -- the bug being guarded.
        y_str = np.array(["1.0"] * 100)  # dtype.kind == 'U'
        assert y_str.dtype.kind == 'U'
        with pytest.raises(CatBoostError, match=r"dtype kind"):
            pool.set_label(y_str)

    def test_bytes_string_array_rejected(self, X100):
        pool = Pool(X100)
        y_bytes = np.array([b"1.0"] * 100)  # dtype.kind == 'S'
        assert y_bytes.dtype.kind == 'S'
        with pytest.raises(CatBoostError, match=r"dtype kind"):
            pool.set_label(y_bytes)

    def test_datetime_array_rejected(self, X100):
        """datetime64 has dtype.kind == 'M'; not in our rejection set but
        py_to_tvector fails anyway -- test the full rejection path is clean."""
        pool = Pool(X100)
        y = np.array(["2024-01-01"] * 100, dtype="datetime64[D]")
        with pytest.raises((CatBoostError, TypeError, ValueError)):
            pool.set_label(y)


class TestFailedSetLabelPreservesStateRegression:
    """Round-4 adversarial scenarios: strong exception guarantee must survive
    unusual failure modes, not just wrong-length."""

    def test_failed_set_label_leaves_get_label_stable(self, X100, rng):
        """After a rejected set_label, get_label() must still return the prior label."""
        y_orig = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y_orig)

        # Reject via string-dtype path (exercises the 'USV' guard).
        with pytest.raises(CatBoostError):
            pool.set_label(np.array(["1.0"] * 100))

        np.testing.assert_array_equal(np.array(pool.get_label(), dtype=np.float32), y_orig)
        assert pool.has_label()

    def test_failed_set_label_then_set_weight_still_works(self, X100, rng):
        """A failed set_label must not poison subsequent set_weight on the same Pool."""
        y_orig = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y_orig)

        with pytest.raises(CatBoostError):
            pool.set_label(np.zeros(50, dtype=np.float32))  # wrong length

        # Wrong-length weight: cleanly raises.
        with pytest.raises(CatBoostError):
            pool.set_weight(np.ones(50, dtype=np.float32))

        # Correct-length weight: succeeds, Pool still usable end-to-end.
        pool.set_weight(np.ones(100, dtype=np.float32))
        CatBoostClassifier(iterations=3, verbose=0).fit(pool)


class TestEvalSetSingleTupleAndWarnOrder:
    """Round-4 coverage: single-element tuple eval_set alias, and raise-no-warn
    across all supported container types when length is wrong."""

    def test_single_element_tuple_alias_raises(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        y_new = (1 - y).astype(np.float32)
        with pytest.raises(CatBoostError, match=r"same Pool object as both X and eval_set"):
            CatBoostClassifier(iterations=3, verbose=0).fit(
                pool, y=y_new, eval_set=(pool,)
            )

    @pytest.mark.parametrize("make_y", [
        lambda: np.zeros(50, dtype=np.float32),
        lambda: [0.0] * 50,
    ], ids=["numpy", "list"])
    def test_wrong_length_y_raises_without_warning(self, X100, rng, make_y):
        """Warning helper's dry-run must fail-first, never warn-then-throw."""
        import warnings
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y1)
        bad_y = make_y()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # any UserWarning -> failure
            with pytest.raises(CatBoostError):
                CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=bad_y)


class TestTrueZeroRowPool:
    """Round-4 coverage: the (num_row==0, len(y)==0) early-return path."""

    def test_zero_row_pool_zero_label_succeeds(self):
        empty_X = np.empty((0, 5), dtype=np.float32)
        pool = Pool(empty_X)
        assert pool.num_row() == 0
        # set_label returns self on the zero-row path without calling into C++.
        # We only verify it does not raise; get_label() semantics on a zero-row
        # Pool are determined by the constructor path and not our concern.
        assert pool.set_label(np.array([], dtype=np.float32)) is pool

    def test_zero_row_pool_nonempty_label_raises(self):
        empty_X = np.empty((0, 5), dtype=np.float32)
        pool = Pool(empty_X)
        with pytest.raises(CatBoostError, match=r"Length of label=5"):
            pool.set_label(np.zeros(5, dtype=np.float32))


class TestErrorMessageQuality:
    """Round-4 coverage: error messages must pin shape/dtype details for docs/grep."""

    def test_2d_rejection_mentions_shape_and_label(self, X100):
        pool = Pool(X100)
        with pytest.raises(CatBoostError) as excinfo:
            pool.set_label(np.ones((100, 3), dtype=np.float32))
        msg = str(excinfo.value)
        assert "(100, 3)" in msg, "Expected offending shape in message, got: {}".format(msg)
        assert "label" in msg.lower(), "Expected 'label' keyword in message, got: {}".format(msg)


class TestSetLabelBasic:
    """Basic set_label functionality folded in from the former unit-test file."""

    def test_set_label_replaces_label(self, X100):
        pool = Pool(X100, label=np.zeros(100, dtype=np.float32))
        pool.set_label(np.ones(100, dtype=np.float32))
        np.testing.assert_array_equal(
            np.array(pool.get_label(), dtype=np.float32), np.ones(100, dtype=np.float32)
        )

    def test_set_label_returns_self(self, X100):
        pool = Pool(X100)
        assert pool.set_label(np.zeros(100, dtype=np.float32)) is pool

    def test_set_label_on_unlabeled_pool(self, X100, rng):
        pool = Pool(X100)
        assert not pool.has_label()
        y = rng.standard_normal(100).astype(np.float32)
        pool.set_label(y)
        assert pool.has_label()
        np.testing.assert_array_almost_equal(
            np.array(pool.get_label(), dtype=np.float32), y, decimal=5
        )

    @pytest.mark.parametrize("delta", [-1, +1])
    def test_off_by_one_length_raises(self, X100, delta):
        pool = Pool(X100)
        with pytest.raises(CatBoostError, match=r"[Ll]ength"):
            pool.set_label(np.zeros(100 + delta, dtype=np.float32))

    def test_accepts_python_list(self, X100):
        pool = Pool(X100)
        pool.set_label([1.0] * 100)
        np.testing.assert_array_equal(
            np.array(pool.get_label(), dtype=np.float32), np.ones(100, dtype=np.float32)
        )


class TestSetLabelPoolReuse:
    """End-to-end Pool reuse scenarios (the motivating use case)."""

    def test_reuse_across_cls_then_reg(self, X100, rng):
        pool = Pool(X100)
        pool.set_label(rng.integers(0, 2, 100).astype(np.float32))
        clf = CatBoostClassifier(iterations=3, verbose=0).fit(pool)
        assert clf.predict_proba(pool).shape == (100, 2)

        pool.set_label(rng.standard_normal(100).astype(np.float32))
        reg = CatBoostRegressor(iterations=3, verbose=0).fit(pool)
        assert reg.predict(pool).shape == (100,)

    def test_reuse_across_weight_schemas(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        params = dict(iterations=20, verbose=0, random_seed=RNG_SEED)

        pool.set_weight(np.ones(100, dtype=np.float32))
        p1 = CatBoostClassifier(**params).fit(pool).predict_proba(pool)

        pool.set_weight(np.linspace(0.01, 10.0, 100).astype(np.float32))
        p2 = CatBoostClassifier(**params).fit(pool).predict_proba(pool)

        assert np.max(np.abs(p1 - p2)) > 1e-3, (
            "Different weights should produce different predictions; "
            "max|Delta_p| = {:g}".format(np.max(np.abs(p1 - p2)))
        )

    def test_predict_only_pool_shared_across_models(self, X100, rng):
        predict_pool = Pool(X100)  # no label
        y_cls = rng.integers(0, 2, 100).astype(np.float32)
        clf = CatBoostClassifier(iterations=3, verbose=0).fit(Pool(X100, label=y_cls))
        y_reg = rng.standard_normal(100).astype(np.float32)
        reg = CatBoostRegressor(iterations=3, verbose=0).fit(Pool(X100, label=y_reg))
        assert clf.predict_proba(predict_pool).shape == (100, 2)
        assert reg.predict(predict_pool).shape == (100,)

    def test_set_label_preserves_features(self, X100):
        pool = Pool(X100, label=np.zeros(100, dtype=np.float32))
        features_before = np.array(pool.get_features())
        pool.set_label(np.ones(100, dtype=np.float32))
        np.testing.assert_array_equal(features_before, np.array(pool.get_features()))

    def test_set_label_preserves_weight(self, X100, rng):
        w = rng.random(100).astype(np.float32)
        pool = Pool(X100, label=np.zeros(100, dtype=np.float32), weight=w)
        pool.set_label(np.ones(100, dtype=np.float32))
        np.testing.assert_array_almost_equal(
            np.array(pool.get_weight(), dtype=np.float32), w, decimal=5
        )

    def test_multiple_sequential_set_label(self, X100):
        pool = Pool(X100, label=np.zeros(100, dtype=np.float32))
        for i in range(5):
            new_y = np.full(100, float(i), dtype=np.float32)
            pool.set_label(new_y)
            np.testing.assert_array_equal(
                np.array(pool.get_label(), dtype=np.float32), new_y
            )

    def test_fit_with_pool_and_y_initializes_label(self, X100, rng):
        pool = Pool(X100)
        y = rng.integers(0, 2, 100).astype(np.float32)
        clf = CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y)
        assert pool.has_label()
        assert len(clf.predict(pool)) == 100


# ===========================================================================
# Follow-up regression tests (from Round-6+ multi-agent audits)
# ===========================================================================

class TestSetLabelOnPreLabeledPool:
    """After Round-6 widen of C++ validation: Pool built with non-Float numeric label
    (Integer/Boolean/Python list coerced to int64) must accept set_label."""

    @pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int8, np.uint8])
    def test_pool_built_with_int_then_set_label(self, X100, rng, dtype):
        y_int = rng.integers(0, 2, 100).astype(dtype)
        pool = Pool(X100, label=y_int)  # TargetType = Integer
        new_y = rng.standard_normal(100).astype(np.float32)
        pool.set_label(new_y)
        # Verify set_label actually replaced the labels (not just didn't raise).
        np.testing.assert_allclose(
            np.array(pool.get_label(), dtype=np.float32), new_y, rtol=1e-5
        )

    def test_pool_built_with_bool_then_set_label(self, X100, rng):
        y_bool = rng.integers(0, 2, 100).astype(bool)
        pool = Pool(X100, label=y_bool)  # TargetType = Boolean
        pool.set_label(rng.standard_normal(100).astype(np.float32))
        assert pool.has_label()

    def test_pool_built_with_python_int_list_then_set_label(self, X100, rng):
        y_list = [int(v) for v in rng.integers(0, 2, 100)]  # int64 via np.asarray
        pool = Pool(X100, label=y_list)  # TargetType = Integer
        pool.set_label([float(v) for v in rng.standard_normal(100)])
        assert pool.has_label()

    def test_pool_built_with_string_labels_still_rejects_set_label(self, X100, rng):
        """String target still requires Pool reconstruction -- we only widen to
        accept Boolean/Integer; String transition is not supported."""
        y_str = np.array(["cat", "dog"] * 50)
        pool = Pool(X100, label=y_str)  # TargetType = String
        with pytest.raises(CatBoostError, match=r"(string|String)"):
            pool.set_label(np.zeros(100, dtype=np.float32))

    def test_pool_slice_on_int_parent_then_set_label(self, X100, rng):
        """Sliced pool inherits parent TargetType; Round-6 fix lets set_label
        work on the slice just as on the parent."""
        y_int = rng.integers(0, 5, 100).astype(np.int64)
        pool = Pool(X100, label=y_int)
        sliced = pool.slice(list(range(50)))
        sliced.set_label(np.ones(50, dtype=np.float32))
        result = np.array(sliced.get_label(), dtype=np.float32)
        np.testing.assert_array_equal(result, np.ones(50, dtype=np.float32))


class TestUnverifiedDocstringClaims:
    """Back docstring/doc-page claims with actual tests (Round-3 audit caught
    several promises with zero coverage)."""

    def test_int64_above_2_24_loses_precision_via_float32_storage(self, X100):
        """Docstring promises: 'Integer values above 2**24 lose precision.'
        Verify by round-tripping int64 values that don't fit exactly in float32.

        At 2^24, float32's 23-bit mantissa forces every second integer to
        collapse into its neighbor -- 100 consecutive ints should yield about
        51 distinct float32 values. A tighter bound catches regressions if
        storage ever silently widens to float64 (which would yield 100).
        """
        pool = Pool(X100)
        base = 2**24
        y = np.array([base + i for i in range(100)], dtype=np.int64)
        pool.set_label(y)
        result_f32 = np.array(pool.get_label()).astype(np.float32)
        unique_count = len(np.unique(result_f32))
        assert unique_count <= 55, (
            "Expected precision loss at 2^24 boundary (~51 distinct values); "
            "got {} unique. If this is 100, float32 storage may have regressed "
            "to float64.".format(unique_count)
        )

    def test_rejects_shape_n_2(self, X100):
        """Close the off-by-one coverage gap -- we tested (N, 3) but not (N, 2)."""
        pool = Pool(X100)
        with pytest.raises(CatBoostError, match=r"(1-D|\(100, 2\)|dimension)"):
            pool.set_label(np.ones((100, 2), dtype=np.float32))

    def test_rejects_void_dtype(self, X100):
        """dtype kind 'V' (structured/void) rejection has code guard but no test."""
        pool = Pool(X100)
        y_struct = np.zeros(100, dtype=[('a', 'i4'), ('b', 'f4')])
        assert y_struct.dtype.kind == 'V'
        with pytest.raises(CatBoostError, match=r"dtype kind"):
            pool.set_label(y_struct)

    @pytest.mark.parametrize("dtype", [np.bool_, np.uint8])
    def test_get_label_returns_input_dtype_extended(self, X100, rng, dtype):
        """Extend TestDtypePreservation parametrization: bool and uint8 were missing."""
        y = rng.integers(0, 2, 100).astype(dtype)
        pool = Pool(X100)
        pool.set_label(y)
        result = np.array(pool.get_label())
        assert result.dtype == np.dtype(dtype)


class TestEvalSetDtypeMismatchWarning:
    """Round-6: detect common set_label desync when train dtype != eval dtype."""

    def test_dtype_mismatch_warns(self, X100, rng):
        # Both pools start with float32 regression targets, so the model fits.
        y_reg_f32 = rng.standard_normal(100).astype(np.float32)
        train_pool = Pool(X100, label=y_reg_f32)
        eval_pool = Pool(X100, label=y_reg_f32)
        # Simulate user updating only the train Pool dtype via set_label
        # (e.g. re-casting to float64 for higher-precision regression targets)
        # while forgetting to sync the eval Pool.
        train_pool.set_label(rng.standard_normal(100).astype(np.float64))
        with pytest.warns(UserWarning, match=r"target dtype.*does not match"):
            CatBoostRegressor(iterations=3, verbose=0).fit(
                train_pool, eval_set=eval_pool
            )

    def test_matching_dtypes_no_warn(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        train_pool = Pool(X100, label=y)
        eval_pool = Pool(X100, label=y)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            CatBoostClassifier(iterations=3, verbose=0).fit(
                train_pool, eval_set=eval_pool
            )


class TestSetLabelTrainingModes:
    """Round-5/6 training-mode code paths not previously exercised."""

    def test_eval_fraction_uses_post_set_label(self, X100, rng):
        """fit(pool, y=new_y, eval_fraction=0.2): set_label fires before split;
        both train and eval halves see the new labels."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100, label=y1)
        with pytest.warns(UserWarning, match=r"overrides the Pool's labels"):
            CatBoostClassifier(
                iterations=3, verbose=0, eval_fraction=0.2
            ).fit(pool, y=y2)

    def test_eval_fraction_with_eval_set_still_raises(self, X100, rng):
        """Pre-existing mutual-exclusion check must survive our changes."""
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        other = Pool(X100, label=y)
        with pytest.raises(CatBoostError, match=r"[Bb]oth eval_fraction and eval_set"):
            CatBoostClassifier(
                iterations=3, verbose=0, eval_fraction=0.2
            ).fit(pool, y=(1 - y).astype(np.float32), eval_set=other)

    def test_cv_reads_current_labels_after_set_label(self, X100, rng):
        """catboost.cv() is a separate entry point; verify set_label is honored."""
        from catboost import cv
        pool = Pool(X100)
        pool.set_label(rng.integers(0, 2, 100).astype(np.float32))
        # Will raise if set_label labels were not picked up (cv requires labels).
        result = cv(
            pool,
            params={'iterations': 3, 'loss_function': 'Logloss', 'verbose': False},
            fold_count=3,
        )
        # cv returns a DataFrame with iteration-indexed metrics.
        assert len(result) == 3

    def test_eval_metrics_uses_current_labels(self, X100, rng):
        """Post-fit eval_metrics should read the Pool's current (post-set_label) labels."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        train_pool = Pool(X100, label=y1)
        model = CatBoostClassifier(iterations=3, verbose=0).fit(train_pool)
        eval_pool = Pool(X100, label=y1)
        metrics_before = np.asarray(
            model.eval_metrics(eval_pool, metrics=['Logloss'])['Logloss']
        )
        eval_pool.set_label(y2)
        metrics_after = np.asarray(
            model.eval_metrics(eval_pool, metrics=['Logloss'])['Logloss']
        )
        # eval_metrics returns per-iteration arrays -- compare elementwise.
        assert not np.allclose(metrics_before, metrics_after)


class TestSetLabelObjectiveLoopPattern:
    """Round-6: Optuna/hyperopt-style pattern -- Pool(X) built once, set_label per trial.
    This is the blessed idiom our PR most directly enables."""

    def test_many_label_updates_with_fit(self, X100, rng):
        pool = Pool(X100)
        for trial in range(10):
            y = rng.integers(0, 2, 100).astype(np.float32)
            pool.set_label(y)
            model = CatBoostClassifier(
                iterations=3, verbose=0, random_seed=trial
            ).fit(pool)
            preds = model.predict(pool)
            assert len(preds) == 100
            assert np.isfinite(preds).all()


class TestSetLabelSaveLoadRoundTrip:
    """Round-3 coverage gap: save/load+set_label roundtrip.

    Also exercises the _read_pool target_type detection fix: without the fix,
    loading a quantized pool set the Python target_type shadow to `str`, making
    get_label() return string-cast numbers.
    """

    def test_quantized_save_load_preserves_numeric_target_type(self, X100, rng, tmp_path):
        y = rng.integers(0, 5, 100).astype(np.int64)
        pool = Pool(X100, label=y)
        pool.quantize()
        save_path = str(tmp_path / "pool.cbquant")
        pool.save(save_path)

        loaded = Pool("quantized://" + save_path)
        # After the _read_pool fix, target_type is detected as a numeric python type
        # (int/float/bool), not str. Without the fix this assertion fails.
        assert loaded.target_type is not str, (
            "Loaded quantized pool target_type should be numeric (detected from "
            "RawTargetData.GetTargetType()), not the historical str hardcode"
        )

    def test_save_load_then_set_label_and_fit(self, X100, rng, tmp_path):
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        pool.quantize()
        save_path = str(tmp_path / "pool.cbquant")
        pool.save(save_path)

        loaded = Pool("quantized://" + save_path)
        # Our PR: set_label on a loaded quantized pool must work.
        loaded.set_label(rng.integers(0, 2, 100).astype(np.float32))
        CatBoostClassifier(iterations=3, verbose=0).fit(loaded)


class TestSetLabelZeroRowHasLabelSymmetry:
    """Round-3: document the has_label() semantics on zero-row Pool + set_label([]).
    The zero-row path returns self without calling _set_label, so has_label()
    reflects the pre-construction state."""

    def test_zero_row_unlabeled_pool_set_empty_label_stays_unlabeled(self):
        empty_X = np.empty((0, 5), dtype=np.float32)
        pool = Pool(empty_X)
        assert not pool.has_label()
        pool.set_label(np.array([], dtype=np.float32))
        # Zero-row early return leaves construction-time state intact.
        assert not pool.has_label()


class TestSetLabelFitYEndToEnd:
    """P0 end-to-end equivalence: fit(pool, y=new_y) must produce a model
    bit-for-bit identical to fit(Pool(X, label=new_y)). Tests the whole
    behavior change, not just the warning side-effect."""

    def test_fit_pool_y_matches_fresh_pool_training(self, X100, rng):
        y = rng.integers(0, 2, 100).astype(np.float32)
        params = dict(iterations=10, verbose=0, random_seed=RNG_SEED)

        # Path A: fit(pool, y=y) -- goes through set_label internally.
        pool_a = Pool(X100)  # unlabeled, so no overwrite warning
        clf_a = CatBoostClassifier(**params).fit(pool_a, y=y)
        preds_a = clf_a.predict_proba(pool_a)

        # Path B: fit(Pool(X, label=y)) -- direct construction.
        pool_b = Pool(X100, label=y)
        clf_b = CatBoostClassifier(**params).fit(pool_b)
        preds_b = clf_b.predict_proba(pool_b)

        # The whole point of the behavior change: these must match exactly.
        np.testing.assert_array_equal(preds_a, preds_b)


class TestSetLabelModelSaveLoadAfterFit:
    """P1: save_model/load_model round-trip after fit on a set_label'd Pool
    must yield bit-exact predictions (labels don't enter the model artifact,
    but verify no metadata leak / stale state breaks serialization)."""

    def test_save_load_after_set_label_fit(self, X100, rng, tmp_path):
        pool = Pool(X100)
        pool.set_label(rng.integers(0, 2, 100).astype(np.float32))
        model = CatBoostClassifier(
            iterations=10, verbose=0, random_seed=RNG_SEED
        ).fit(pool)
        preds_before = model.predict_proba(pool)

        save_path = str(tmp_path / "model.cbm")
        model.save_model(save_path)

        loaded = CatBoostClassifier()
        loaded.load_model(save_path)
        preds_after = loaded.predict_proba(pool)

        np.testing.assert_array_equal(preds_before, preds_after)


class TestSetLabelFromFilePool:
    """P1: Pool built from disk (TSV + column_description) -- TSV loader
    stores labels as String target type, so set_label raises with a clear
    message (not a cryptic C++ error). Documents and pins the limitation:
    to change labels on a file-loaded pool, reconstruct with a new label."""

    def test_tsv_file_pool_then_set_label_raises_clear_error(self, X100, rng, tmp_path):
        # Write a minimal TSV: label + 10 numeric features.
        y = rng.integers(0, 2, 100).astype(np.int64)
        rows = ["{}\t{}".format(int(y[i]), "\t".join(str(v) for v in X100[i])) for i in range(100)]
        tsv_path = tmp_path / "data.tsv"
        tsv_path.write_text("\n".join(rows))

        cd_path = tmp_path / "data.cd"
        cd_lines = ["0\tLabel"] + ["{}\tNum".format(i + 1) for i in range(X100.shape[1])]
        cd_path.write_text("\n".join(cd_lines))

        file_pool = Pool(data=str(tsv_path), column_description=str(cd_path))
        # TSV loader populates String target type. Our widened validation
        # rejects only String -> set_label on a TSV pool raises clearly.
        with pytest.raises(CatBoostError, match=r"string|String"):
            file_pool.set_label(rng.standard_normal(100).astype(np.float32))
        # Pool is still usable: training on the original string labels works.
        CatBoostClassifier(iterations=3, verbose=0).fit(file_pool)


class TestSetLabelIntrospectionFunctions:
    """P2 regression guard: document the 'post-fit set_label affects introspection'
    pitfall with tests. If future refactoring makes these APIs snapshot training
    labels instead of reading live from Pool, our doc page's pitfall note
    becomes inaccurate and needs updating."""

    def test_eval_metrics_reflects_current_pool_labels(self, X100, rng):
        """eval_metrics reads labels from Pool at call time, not at fit time."""
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y1)
        model = CatBoostClassifier(iterations=5, verbose=0).fit(pool)

        m_original = np.asarray(model.eval_metrics(pool, metrics=['Logloss'])['Logloss'])
        pool.set_label((1 - y1).astype(np.float32))
        m_flipped = np.asarray(model.eval_metrics(pool, metrics=['Logloss'])['Logloss'])
        # Labels fully inverted -> metrics differ substantially.
        assert not np.allclose(m_original, m_flipped)

    def test_calc_feature_statistics_reflects_current_pool_labels(self, X100, rng):
        """calc_feature_statistics['mean_target'] is the average label over bins
        defined by model borders. Changing labels must change these means.

        Uses enough iterations to guarantee the model creates borders on feature 0,
        otherwise mean_target would be empty for that feature.
        """
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y1)
        model = CatBoostClassifier(
            iterations=50, verbose=0, random_seed=RNG_SEED,
        ).fit(pool)

        stats1 = model.calc_feature_statistics(pool, feature=0, plot=False)
        mt1 = np.asarray(stats1['mean_target'])
        if mt1.size == 0:
            pytest.skip("Model did not create splits on feature 0 -- try another seed")

        pool.set_label((1 - y1).astype(np.float32))
        stats2 = model.calc_feature_statistics(pool, feature=0, plot=False)
        mt2 = np.asarray(stats2['mean_target'])

        # Both arrays share the same bin count (borders are model property, not
        # label-dependent). Flipped labels -> per-bin means must differ.
        assert not np.allclose(mt1, mt2)


class TestSetLabelTextFeaturesPreservation:
    """P2: set_label on Pool with text_features must leave text feature
    metadata untouched."""

    def test_text_features_preserved(self, rng):
        n = 50
        num = rng.standard_normal((n, 3)).astype(np.float32)
        text = np.array(
            [["red", "green"] if i % 2 == 0 else ["blue", "yellow"] for i in range(n)],
            dtype=object,
        )
        X = np.column_stack([text, num])
        y = rng.integers(0, 2, n).astype(np.float32)
        pool = Pool(
            data=X, label=y,
            text_features=[0, 1],
        )
        text_idx_before = pool.get_text_feature_indices()
        pool.set_label(rng.integers(0, 2, n).astype(np.float32))
        assert pool.get_text_feature_indices() == text_idx_before


class TestSetLabelInitModel:
    """P2: init_model + set_label that breaks label semantics -> clean error,
    not silent miscompute."""

    def test_init_model_regression_then_set_cls_labels_raises_cleanly(self, X100, rng):
        pool = Pool(X100, label=rng.standard_normal(100).astype(np.float32))
        base = CatBoostRegressor(iterations=5, verbose=0).fit(pool)
        # Now the user re-purposes the pool as a classifier and tries
        # incremental training -- CatBoost must raise clearly.
        pool.set_label(rng.integers(0, 2, 100).astype(np.float32))
        with pytest.raises(CatBoostError):
            CatBoostClassifier(iterations=5, verbose=0).fit(pool, init_model=base)


class TestSetLabelWarningSilenceable:
    """P3: UserWarning must play nicely with standard Python warning filters."""

    def test_overwrite_warning_silenceable(self, X100, rng):
        import warnings as _w
        y1 = rng.integers(0, 2, 100).astype(np.float32)
        y2 = (1 - y1).astype(np.float32)
        pool = Pool(X100, label=y1)
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=UserWarning)
            # Would normally emit the overwrite warning; filter suppresses it.
            CatBoostClassifier(iterations=3, verbose=0).fit(pool, y=y2)


class TestSetLabelBackwardCompat:
    """P3: users who never call set_label must see zero change in behavior.
    Trains a classic Pool+fit path and asserts no UserWarning is emitted."""

    def test_classic_fit_no_new_warnings(self, X100, rng):
        import warnings as _w
        y = rng.integers(0, 2, 100).astype(np.float32)
        pool = Pool(X100, label=y)
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)  # any UserWarning -> failure
            CatBoostClassifier(iterations=3, verbose=0).fit(pool)

    def test_classic_fit_with_eval_no_new_warnings(self, X100, rng):
        import warnings as _w
        y = rng.integers(0, 2, 100).astype(np.float32)
        train = Pool(X100, label=y)
        eval_p = Pool(X100, label=y)
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            CatBoostClassifier(iterations=3, verbose=0).fit(train, eval_set=eval_p)
