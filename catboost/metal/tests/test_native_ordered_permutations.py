"""Native Ordered host permutations and persistent selection-state acceptance."""

import os

import numpy as np
import pytest
from catboost import CatBoostError, CatBoostRegressor, Pool

from test_native_ordered_api import StopAfter, dataset, options


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal CatBoost package",
)


def estimator(permutations=4, **extra):
    return CatBoostRegressor(**options(**extra)).set_params(permutation_count=permutations)


@pytest.mark.parametrize("permutations", [1, 2, 3, 4, 7])
def test_native_ordered_permutation_count_and_seeded_models(permutations):
    x, target, weights = dataset()
    first = estimator(permutations, random_strength=0.8).fit(x, target, sample_weight=weights)
    second = estimator(permutations, random_strength=0.8).fit(x, target, sample_weight=weights)
    assert first.get_all_params()["permutation_count"] == permutations
    assert int(first.get_metadata()["metal_permutations"]) == permutations
    np.testing.assert_array_equal(first.get_leaf_values(), second.get_leaf_values())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    np.testing.assert_allclose(first.predict(x, task_type="GPU"), first.predict(x), rtol=3e-6, atol=3e-6)
    history = first.get_evals_result()["learn"]["RMSE"]
    assert history[-1] < history[0]


@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson", "MVS"])
def test_native_ordered_multiple_permutation_snapshot_resume_is_exact(tmp_path, bootstrap):
    x, target, weights = dataset()
    extra = dict(bootstrap_type=bootstrap, iterations=10, random_strength=0.9,
                 leaf_estimation_iterations=3, leaf_estimation_backtracking="AnyImprovement")
    if bootstrap == "Bayesian":
        extra["bagging_temperature"] = 0.7
    elif bootstrap != "No":
        extra["subsample"] = 0.8
    saved = dict(extra, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_file="ordered-p4.snapshot", snapshot_interval=0)
    pool = Pool(x, target, weight=weights)
    partial = estimator(**saved).fit(pool, eval_set=pool, use_best_model=False, callbacks=[StopAfter(4)])
    assert partial.tree_count_ == 4
    resumed = estimator(**saved).fit(pool, eval_set=pool, use_best_model=False)
    direct = estimator(**extra).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.get_leaf_weights(), direct.get_leaf_weights())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    assert resumed.get_evals_result() == direct.get_evals_result()
    completed = estimator(**saved).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(completed.predict(x), direct.predict(x))
    assert completed.tree_count_ == 10


@pytest.mark.parametrize("mode", ["depth_zero", "repeated_split"])
def test_native_ordered_snapshot_search_attempt_edge_cases(tmp_path, mode):
    x, target, weights = dataset()
    if mode == "repeated_split":
        x = (x[:, :1] > 0).astype(np.float32)
        target = (2 * x[:, 0] - 1).astype(np.float32)
    extra = dict(iterations=7, depth=0 if mode == "depth_zero" else 4)
    saved = dict(extra, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_file="ordered-edge.snapshot", snapshot_interval=0)
    estimator(3, **saved).fit(x, target, sample_weight=weights, callbacks=[StopAfter(3)])
    resumed = estimator(3, **saved).fit(x, target, sample_weight=weights)
    direct = estimator(3, **extra).fit(x, target, sample_weight=weights)
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    expected_leaves = 1 if mode == "depth_zero" else 2
    np.testing.assert_array_equal(direct.get_tree_leaf_counts(), np.full(7, expected_leaves))


def test_native_ordered_has_time_uses_identity_with_requested_multiple_permutations():
    x, target, weights = dataset()
    ordered = estimator(4, has_time=True).fit(x, target, sample_weight=weights)
    identity = estimator(1, has_time=True).fit(x, target, sample_weight=weights)
    assert ordered.get_params()["permutation_count"] == 4
    assert ordered.get_all_params()["permutation_count"] == 1
    assert int(ordered.get_metadata()["metal_permutations"]) == 1
    np.testing.assert_array_equal(ordered.get_leaf_values(), identity.get_leaf_values())
    np.testing.assert_array_equal(ordered.predict(x), identity.predict(x))


def test_native_ordered_default_uses_four_permutations():
    x, target, weights = dataset()
    default = CatBoostRegressor(**options()).fit(x, target, sample_weight=weights)
    explicit = estimator(4).fit(x, target, sample_weight=weights)
    assert default.get_all_params()["permutation_count"] == 4
    assert int(default.get_metadata()["metal_permutations"]) == 4
    np.testing.assert_array_equal(default.predict(x), explicit.predict(x))


def test_native_ordered_initial_model_snapshot_uses_new_tree_rng_count(tmp_path):
    x, target, weights = dataset()
    first = estimator(iterations=3).fit(x, target, sample_weight=weights)
    saved = dict(iterations=8, random_strength=0.7, allow_writing_files=True,
                 train_dir=str(tmp_path), save_snapshot=True, snapshot_interval=0,
                 snapshot_file="ordered-init.snapshot")
    partial = estimator(**saved).fit(x, target, sample_weight=weights, init_model=first,
                                     callbacks=[StopAfter(3)])
    assert partial.tree_count_ == 6
    resumed = estimator(**saved).fit(x, target, sample_weight=weights, init_model=first)
    direct = estimator(iterations=8, random_strength=0.7).fit(x, target, sample_weight=weights, init_model=first)
    assert resumed.tree_count_ == direct.tree_count_ == 11
    np.testing.assert_array_equal(resumed.get_leaf_values(), direct.get_leaf_values())
    np.testing.assert_array_equal(resumed.predict(x), direct.predict(x))
    assert resumed.get_evals_result() == direct.get_evals_result()


def test_native_ordered_snapshot_rejects_changed_permutation_count(tmp_path):
    x, target, weights = dataset()
    saved = dict(iterations=7, allow_writing_files=True, train_dir=str(tmp_path),
                 save_snapshot=True, snapshot_interval=0, snapshot_file="ordered-p4.snapshot")
    estimator(4, **saved).fit(x, target, sample_weight=weights, callbacks=[StopAfter(3)])
    with pytest.raises(CatBoostError, match="snapshot|Snapshot"):
        estimator(3, **saved).fit(x, target, sample_weight=weights)


def test_native_ordered_large_pool_uses_block_permutations():
    x, target, weights = dataset(50003)
    config = dict(iterations=2, depth=2, border_count=8)
    first = estimator(**config).set_params(fold_permutation_block=65).fit(x, target, sample_weight=weights)
    second = estimator(**config).set_params(fold_permutation_block=65).fit(x, target, sample_weight=weights)
    assert int(first.get_metadata()["metal_permutations"]) == 4
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert np.isfinite(first.predict(x)).all()
