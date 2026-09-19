"""Sampled symmetric vector leaves and shared weak models across histories."""
import platform

import numpy as np
import pytest

from catboost_metal import _multiclass
from test_greedy_vector_simple import sampled_weights
from test_greedy_vector_training import no_cpu_fit, problem as greedy_problem
from test_multiclass_permutations import _leaf_ids


pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires actual Apple Silicon Metal")
OBJECTIVES = ("MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")


def problem(objective="MultiClass", count=1, **extra):
    args, banks, initial = greedy_problem(objective, "SymmetricTree", count=count, iterations=2, depth=2,
        leaf_estimation_method="Simple", leaf_estimation_iterations=1, **extra)
    args.pop("max_leaves")
    if objective in ("MultiRMSE", "MultiLogloss", "MultiCrossEntropy"):
        x = banks[0].astype(float)
        targets = np.column_stack((x[0] / 4 - .8, x[1] / 5 - .2, (x[2] > 3) + .2 * x[3]))
        if objective == "MultiLogloss":
            targets = targets > .5
        elif objective == "MultiCrossEntropy":
            targets = .1 + .8 / (1 + np.exp(-targets))
        args["targets"] = targets.astype(np.float32)
    return args, banks, initial


def expected_weak_model(args, active, tree, bins, iteration):
    x, y, weights = active.astype(float), args["targets"], args["sample_weight"]
    objective = args["objective"]
    if objective == "MultiRMSE":
        gradient = (y.T.astype(float) - x) * weights
    elif objective == "RMSEWithUncertainty":
        error = y.astype(float) - x[0]
        gradient = np.array([error, error**2 * np.exp(np.minimum(-2*x[1], 70)) - 1]) * weights
    else:
        if objective == "MultiClass":
            full = np.vstack((x, np.zeros(x.shape[1])))
            exponential = np.exp(full - full.max(axis=0))
            probability = exponential / exponential.sum(axis=0)
        else:
            exponential = np.exp(-np.abs(x))
            probability = np.where(x >= 0, 1/(1+exponential), exponential/(1+exponential))
            if objective == "MultiClassOneVsAll":
                probability = np.clip(probability, 1e-7, 1-1e-7)
        target = (np.arange(len(x))[:, None] == y) if objective in ("MultiClass", "MultiClassOneVsAll") else y.T
        gradient = (target - probability[:len(x)]) * weights
    draws = sampled_weights(args, iteration)
    gradient = np.float32(np.float32(gradient) * draws).astype(float)
    mass = np.float32(weights * draws).astype(float)
    ids = _leaf_ids(tree, bins)
    leaf_weights = np.bincount(ids, weights=mass, minlength=1 << tree.depth)
    values = np.zeros((len(leaf_weights), args["classes"]), np.float32)
    ridge = args["l2_leaf_reg"] or np.float32(1e-20)
    for leaf, weight in enumerate(leaf_weights):
        if weight > 1e-20:
            values[leaf, :len(x)] = gradient[:, ids == leaf].sum(axis=1) / (weight + ridge)
        if objective == "MultiClass":
            values[leaf, :len(x)] = values[leaf, :len(x)].astype(float) + values[leaf, :len(x)].sum(dtype=float)
    values *= np.float32(args["learning_rate"])
    return values, leaf_weights, ids


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("sampler", ["No", "Bayesian", "Bernoulli", "Poisson"])
def test_symmetric_simple_matches_sampled_gradient_equations(objective, sampler):
    args, banks, _ = problem(objective, bootstrap_type=sampler)
    with _multiclass.Session(**args) as session:
        for iteration in range(2):
            active, before = session.optimization_predictions(), session.predictions()
            tree = session.step()
            values, weights, ids = expected_weak_model(args, active, tree, banks[0], iteration)
            assert tree.depth >= 1
            np.testing.assert_allclose(tree.leaf_values, values, rtol=7e-5, atol=5e-6)
            np.testing.assert_allclose(tree.leaf_weights, weights, rtol=4e-6, atol=1e-5)
            np.testing.assert_allclose(session.predictions(), before + values[ids], rtol=7e-5, atol=5e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_history_receives_the_selected_sampled_model(objective):
    args, banks, initial = problem(objective, count=4, bootstrap_type="Bernoulli", random_strength=.25)
    with _multiclass.Session(**args) as session:
        session.configure_permutations(banks, initial_predictions=initial)
        for iteration, search in enumerate((3, 1)):
            before = session.permutation_state
            single_args = args | dict(bins=banks[search], iterations=1, iteration_offset=iteration,
                initial_predictions=before["predictions"][search],
                initial_optimization_predictions=before["optimization_predictions"][search])
            with _multiclass.Session(**single_args) as single:
                expected = single.step()
            session.select_permutation(search)
            tree = session.step()
            for name in ("split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
                np.testing.assert_array_equal(getattr(tree, name), getattr(expected, name))
            after = session.permutation_state
            for index, bank in enumerate(banks):
                ids = _leaf_ids(tree, bank)
                np.testing.assert_allclose(after["predictions"][index],
                    np.float32(before["predictions"][index] + expected.leaf_values[ids]), rtol=4e-6, atol=1e-7)


@pytest.mark.parametrize("sampler", ["Bernoulli", "Poisson"])
def test_empty_sampled_root_has_zero_simple_mass_and_values(sampler):
    args, _, _ = problem("MultiClassOneVsAll", bootstrap_type=sampler)
    args.update(depth=0, iterations=1, l2_leaf_reg=0, subsample=.25,
        bins=np.zeros((1, 4), np.uint8), targets=np.array([0, 1, 2, 0], np.uint32),
        sample_weight=np.ones(4, np.float32), initial_predictions=np.zeros((4, 3), np.float32),
        candidate_features=[], candidate_bins=[], candidate_types=None)
    for seed in range(4096):
        args["random_seed"] = seed
        if not sampled_weights(args, 0).any():
            break
    else:
        pytest.fail("No bounded all-zero sampled fixture")
    with _multiclass.Session(**args) as session:
        tree = session.step()
        np.testing.assert_array_equal(tree.leaf_values, 0)
        np.testing.assert_array_equal(tree.leaf_weights, 0)
        np.testing.assert_array_equal(session.predictions(), args["initial_predictions"])


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_simple_ignores_iterative_backtracking(backtracking):
    args, _, _ = problem("MultiClass", bootstrap_type="Bayesian")
    with _multiclass.Session(**args) as direct, _multiclass.Session(**(args | dict(leaf_estimation_backtracking=backtracking))) as trial:
        for _ in range(2):
            first, second = direct.step(), trial.step()
            np.testing.assert_array_equal(second.leaf_values, first.leaf_values)
            np.testing.assert_array_equal(second.leaf_weights, first.leaf_weights)
            np.testing.assert_array_equal(trial.predictions(), direct.predictions())


def test_failed_later_history_restores_simple_cursors():
    args, _, _ = problem("RMSEWithUncertainty")
    initial = np.zeros((3, 4, 2), np.float32)
    initial[:2, :, 0] = 20000
    initial[2, :, 1] = -34
    args.update(depth=0, iterations=1, learning_rate=1, l2_leaf_reg=0,
        bins=np.zeros((1, 4), np.uint8), targets=np.full(4, 20000, np.float32),
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
