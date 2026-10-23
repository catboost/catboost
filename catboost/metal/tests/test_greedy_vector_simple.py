"""Greedy vector Simple exports sampled weak leaves and shares them across histories."""

import ctypes as ct
import math
import platform

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_bootstrap import next_word, seed_for_item, uniforms
from test_greedy_vector_training import OBJECTIVES, no_cpu_fit, problem, route


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires actual Apple Silicon Metal",
)


def simple_problem(objective="MultiClass", policy="Depthwise", **options):
    return problem(objective, policy, leaf_estimation_method="Simple",
                   leaf_estimation_iterations=1, **options)


def sampled_weights(args, iteration):
    rows = args["bins"].shape[1]
    kind = args.get("bootstrap_type", "No")
    seed = args["random_seed"]
    absolute = args.get("iteration_offset", 0) + iteration
    u = uniforms(rows, seed=seed, iteration=absolute)
    if kind == "No":
        return np.ones(rows, np.float32)
    if kind == "Bernoulli":
        return np.float32(u < np.float32(args["subsample"]))
    if kind == "Bayesian":
        return np.float32((-np.log(u.astype(float) + 1e-20)) ** args.get("bagging_temperature", 1))
    result = np.zeros(rows, np.float32)
    rate = np.float32(-math.log(1 - np.float32(args["subsample"])))
    for row in range(rows):
        state = seed_for_item(row, seed=seed, iteration=absolute)
        log_probability = np.float32(0)
        while True:
            state, word = next_word(state)
            u = min(np.float32(word) * np.float32(2**-32), np.nextafter(np.float32(1), np.float32(0)))
            log_probability = np.float32(log_probability + np.float32(math.log(max(float(u), 2**-32))))
            if log_probability <= -rate:
                break
            result[row] += 1
    return result


def weak_leaves(args, active, tree, bins, iteration):
    x = active.astype(float)
    y = args["targets"]
    w = args["sample_weight"]
    if args["objective"] == "RMSEWithUncertainty":
        error = y.astype(float) - x[0]
        gradient = np.array([error, error**2 * np.exp(np.minimum(-2*x[1], 70)) - 1]) * w
    else:
        if args["objective"] == "MultiClass":
            full = np.vstack([x, np.zeros(x.shape[1])])
            exponential = np.exp(full - full.max(axis=0))
            probability = exponential / exponential.sum(axis=0)
        else:
            exponential = np.exp(-np.abs(x))
            probability = np.clip(np.where(x >= 0, 1/(1+exponential), exponential/(1+exponential)), 1e-7, 1-1e-7)
        gradient = ((np.arange(len(x))[:, None] == y) - probability[:len(x)]) * w
    draws = sampled_weights(args, iteration)
    gradient = np.float32(np.float32(gradient) * draws).astype(float)
    mass = np.float32(w * draws).astype(float)
    ids = route(tree, bins)
    leaves = len(tree.leaf_weights)
    weights = np.bincount(ids, weights=mass, minlength=leaves)
    values = np.zeros((leaves, args["classes"]), np.float32)
    l2 = args["l2_leaf_reg"] or np.float32(1e-20)
    for leaf in range(leaves):
        if weights[leaf] > 1e-20:
            values[leaf, :len(x)] = gradient[:, ids == leaf].sum(axis=1) / (weights[leaf] + l2)
        if args["objective"] == "MultiClass":
            values[leaf, :len(x)] = values[leaf, :len(x)].astype(float) + values[leaf, :len(x)].sum(dtype=float)
    values *= np.float32(args["learning_rate"])
    return values, weights, ids


@pytest.mark.parametrize("objective,policy", zip(OBJECTIVES, ("Depthwise", "Lossguide", "Region")))
@pytest.mark.parametrize("sampler", ["No", "Bayesian", "Bernoulli", "Poisson"])
def test_simple_final_sampled_leaf_statistics_match_cuda_equations(objective, policy, sampler):
    args, banks, _ = simple_problem(objective, policy, iterations=2, bootstrap_type=sampler)
    with _multiclass.Session(**args) as session:
        for iteration in range(2):
            active, before = session.optimization_predictions(), session.predictions()
            tree = session.step()
            values, weights, ids = weak_leaves(args, active, tree, banks[0], iteration)
            assert len(tree.leaf_weights) > 1, "Exercise statistics after the final split"
            np.testing.assert_allclose(tree.leaf_values, values, rtol=7e-5, atol=4e-6)
            np.testing.assert_allclose(tree.leaf_weights, weights, rtol=4e-6, atol=1e-5)
            np.testing.assert_allclose(session.predictions(), before + values[ids], rtol=7e-5, atol=4e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_simple_copies_the_searched_model_into_every_permutation(objective):
    args, banks, initial = simple_problem(objective, "Lossguide", count=4, iterations=3,
        bootstrap_type="Bernoulli", score_function="Cosine", random_strength=.27)
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial_predictions=initial)
        for iteration, search in enumerate((0, 3, 1)):
            before = session.permutation_state
            with _multiclass.Session(**(args | dict(bins=banks[search], iterations=1,
                    iteration_offset=iteration, initial_predictions=before["predictions"][search],
                    initial_optimization_predictions=before["optimization_predictions"][search]))) as single:
                expected = single.step()
            session.select_permutation(search)
            tree = session.step()
            for name in ("nodes", "leaf_values", "leaf_weights"):
                np.testing.assert_array_equal(getattr(tree, name), getattr(expected, name))
            after = session.permutation_state
            for index, bank in enumerate(banks):
                ids = route(tree, bank)
                np.testing.assert_allclose(after["predictions"][index],
                    np.float32(before["predictions"][index] + expected.leaf_values[ids]), rtol=3e-6, atol=1e-7)
            np.testing.assert_array_equal(session.predictions(), after["predictions"][-1])


@pytest.mark.parametrize("sampler", ["Bernoulli", "Poisson"])
def test_zero_sampled_mass_root_stays_zero_despite_original_occupancy(sampler):
    args, _, _ = simple_problem("MultiClassOneVsAll", depth=0, iterations=1, l2_leaf_reg=0,
        bootstrap_type=sampler, subsample=.25)
    args.update(bins=np.zeros((1, 4), np.uint8), targets=np.array([0, 1, 2, 0], np.uint32),
        sample_weight=np.ones(4, np.float32), initial_predictions=np.zeros((4, 3), np.float32),
        candidate_features=[], candidate_bins=[], candidate_types=None)
    for seed in range(4096):
        args["random_seed"] = seed
        if not sampled_weights(args, 0).any():
            break
    else:
        raise AssertionError("No all-zero bounded bootstrap fixture")
    with _multiclass.Session(**args) as session:
        tree = session.step()
        np.testing.assert_array_equal(tree.leaf_values, 0)
        np.testing.assert_array_equal(tree.leaf_weights, 0)
        np.testing.assert_array_equal(session.predictions(), args["initial_predictions"])


@pytest.mark.parametrize("masses", [[np.float32(1e-20)],
    [np.nextafter(np.float32(1e-20), np.float32(1))], [np.float32(2e-20)],
    [np.float32(1e-20), np.float32(1e-28)]])
def test_simple_mass_guard_and_zero_l2_normalization(masses):
    args, _, _ = simple_problem(depth=0, iterations=1, classes=2, learning_rate=1, l2_leaf_reg=0)
    rows = len(masses)
    args.update(bins=np.zeros((1, rows), np.uint8), targets=np.zeros(rows, np.uint32),
        sample_weight=np.array(masses), initial_predictions=np.zeros((rows, 2), np.float32),
        candidate_features=[], candidate_bins=[], candidate_types=None)
    with _multiclass.Session(**args) as session:
        active = session.optimization_predictions()
        tree = session.step()
        expected, weights, _ = weak_leaves(args, active, tree, args["bins"], 0)
        np.testing.assert_allclose(tree.leaf_values, expected, rtol=3e-6, atol=0)
        np.testing.assert_array_equal(tree.leaf_weights, np.float32(weights))


def test_simple_regularized_denominator_retains_double_exponent_range():
    args, _, _ = simple_problem("MultiClassOneVsAll", depth=0, iterations=1, classes=2,
        learning_rate=1, l2_leaf_reg=np.finfo(np.float32).max, bootstrap_type="Bayesian",
        bagging_temperature=20, random_seed=5)
    args.update(bins=np.zeros((1, 1), np.uint8), targets=np.zeros(1, np.uint32),
        sample_weight=np.array([1e29], np.float32), initial_predictions=np.zeros((1, 2), np.float32),
        candidate_features=[], candidate_bins=[], candidate_types=None)
    with _multiclass.Session(**args) as session:
        active = session.optimization_predictions()
        tree = session.step()
        expected, weights, _ = weak_leaves(args, active, tree, args["bins"], 0)
        assert weights[0] + float(args["l2_leaf_reg"]) > np.finfo(np.float32).max
        assert expected[0, 0] > .01
        np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-5, atol=1e-7)
        np.testing.assert_allclose(tree.leaf_weights, weights, rtol=5e-5)


def test_later_history_failure_restores_every_simple_cursor_and_retry():
    args, _, _ = simple_problem("RMSEWithUncertainty", depth=0, iterations=1, learning_rate=1, l2_leaf_reg=0)
    initial = np.zeros((3, 4, 2), np.float32)
    initial[:2, :, 0] = 20000
    initial[2, :, 1] = -34
    args.update(bins=np.zeros((1, 4), np.uint8), targets=np.full(4, 20000, np.float32),
        sample_weight=np.ones(4, np.float32), initial_predictions=initial[0],
        candidate_features=[], candidate_bins=[], candidate_types=None)
    with _multiclass.Session(**args) as session:
        session.configure_permutations(np.repeat(args["bins"][None], 3, axis=0), initial_predictions=initial)
        session.select_permutation(0)
        before = session.permutation_state
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Nonfinite multiclass objective"):
                session.step()
            assert session.completed_iterations == 0
            for name in before:
                np.testing.assert_array_equal(session.permutation_state[name], before[name])


@pytest.mark.parametrize("options", [dict(leaf_estimation_iterations=2),
    dict(objective="MultiRMSE"), dict(objective="MultiLogloss"), dict(objective="MultiCrossEntropy")])
def test_simple_rejects_non_cuda_combinations_before_native_load(monkeypatch, options):
    args, _, _ = simple_problem()
    args.update(options)
    monkeypatch.setattr(_multiclass, "build_library", lambda: pytest.fail("Invalid Simple parameters reached native load"))
    with pytest.raises(ValueError):
        _multiclass.Session(**args)


def test_greedy_feature_weight_abi_validates_transactionally_and_changes_search():
    args, _, _ = simple_problem("MultiClass", "Lossguide", iterations=1, depth=1, max_leaves=2)
    with _multiclass.Session(**args) as session:
        session.configure_greedy_feature_weights([0, 0, 1000, 0])
        invalid = np.array([1000, 0, np.nan, 0], np.float32)
        error = ct.create_string_buffer(2048)
        code = session._lib.cbm_multiclass_session_set_greedy_feature_weights(
            session._handle, 4, _multiclass._f32(invalid), error, len(error))
        assert code and b"finite" in error.value
        tree = session.step()
        assert tree.nodes[0, 0] == 2, "Invalid weights must not partially replace the configured search weights"
        weights = np.ones(4, np.float32)
        code = session._lib.cbm_multiclass_session_set_greedy_feature_weights(
            session._handle, 4, _multiclass._f32(weights), error, len(error))
        assert code and b"precede training" in error.value
