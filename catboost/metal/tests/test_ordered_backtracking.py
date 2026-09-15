"""CUDA's coupled Ordered leaf walker, independently checked on Metal.

One trial step applies to all fold tasks and full-model estimation together;
line-searching each prefix separately is a different algorithm. No CPU fits.
"""

import numpy as np
import pytest

from catboost_metal import _ordered
import cuda_ordered_reference as reference
from test_ordered_training import apple_silicon, prohibit_cpu_training, dataset, options, compare


HARD_CASES = [("Poisson", None, "Newton"), ("Poisson", None, "Gradient"), ("Lq", 3, "Gradient")]


def hard_problem(objective, parameter, method, **overrides):
    bins, targets, features, borders, weights = dataset(seed=1947)
    targets = (np.exp(.8 * targets + 1.3).astype(np.float32)
               if objective == "Poisson" else targets * 3)
    config = options(objective, iterations=2, depth=2, bias=0, learning_rate=.12,
                     l2_leaf_reg=.2, objective_param=parameter, leaf_estimation_method=method,
                     leaf_estimation_iterations=4, leaf_estimation_backtracking="AnyImprovement",
                     sample_weight=weights)
    config.update(overrides)
    return bins, targets, features, borders, config


@pytest.mark.parametrize("objective,param,method", HARD_CASES)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_common_task_step_matches_cuda_walker(objective, param, method, normalize, permutation_count, backtracking):
    bins, targets, features, borders, config = hard_problem(objective, param, method,
        permutation_count=permutation_count, fold_size_loss_normalization=normalize,
        leaf_estimation_backtracking=backtracking)
    expected = reference.train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)
    assert any(not trial["accepted"] for trace in expected["backtracking_traces"] for trial in trace)
    for trace in expected["backtracking_traces"]:
        for trial in trace:
            threshold = trial["score_before"]
            if backtracking == "Armijo":
                threshold += 1e-5 * trial["step"] * trial["direction_dot"]
            assert trial["accepted"] == (trial["score_after"] >= threshold)


@pytest.mark.parametrize("objective,param,method,normalize", [
    ("Poisson", None, "Newton", True), ("Lq", 3, "Gradient", False),
])
def test_global_acceptance_can_worsen_one_prefix_and_differs_from_separate_walkers(
        monkeypatch, objective, param, method, normalize):
    bins, targets, features, borders, config = hard_problem(objective, param, method,
        iterations=1, fold_size_loss_normalization=normalize)
    coupled = reference.train_reference(bins, targets, features, borders, **config)
    assert any(trial["accepted"] and np.any(trial["task_scores_after"] < trial["task_scores_before"] - 1e-6)
               for trial in coupled["backtracking_traces"][0])
    original = reference.estimate_tasks_reference

    def independently(tasks, leaves, **kwargs):
        results = [original([task], leaves, **kwargs) for task in tasks]
        return (np.concatenate([result[0] for result in results]),
                np.concatenate([result[1] for result in results]),
                [trial for result in results for trial in result[2]])

    with monkeypatch.context() as scoped:
        scoped.setattr(reference, "estimate_tasks_reference", independently)
        separate = reference.train_reference(bins, targets, features, borders, **config)
    assert np.max(np.abs(coupled["task_leaf_values"] - separate["task_leaf_values"])) > 1e-4
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        session.step()
        state = session.state()
    np.testing.assert_allclose(state["cursors"], coupled["cursors"], rtol=8e-6, atol=3e-6)
    assert np.max(np.abs(state["cursors"] - separate["cursors"])) > 1e-4


SINGLE_STEP_CASES = [
    ("RMSE", None, "Newton"), ("Logloss", None, "Newton"), ("CrossEntropy", None, "Gradient"),
    ("Poisson", None, "Newton"), ("Huber", .7, "Newton"), ("Expectile", .7, "Gradient"),
    ("Lq", 2.5, "Newton"), ("Tweedie", 1.3, "Newton"),
    ("LogLinQuantile", .7, "Gradient"), ("Quantile", .7, "Gradient"),
    ("MAE", None, "Gradient"), ("MAPE", None, "Gradient"),
]


@pytest.mark.parametrize("objective,param,method", SINGLE_STEP_CASES)
@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_one_leaf_step_bypasses_backtracking_exactly(objective, param, method, backtracking):
    bins, targets, features, borders, weights = dataset(objective if objective in ("Logloss", "CrossEntropy") else "RMSE")
    if objective in ("Poisson", "Tweedie", "LogLinQuantile"):
        targets = np.exp(.5 * targets).astype(np.float32)
    config = options(objective, iterations=3, permutation_count=4, sample_weight=weights,
                     objective_param=param, leaf_estimation_method=method, leaf_estimation_iterations=1)
    plain = _ordered.train(bins, targets, features, borders, **config)
    checked = _ordered.train(bins, targets, features, borders,
                             **dict(config, leaf_estimation_backtracking=backtracking))
    np.testing.assert_array_equal(plain.predictions, checked.predictions)
    np.testing.assert_array_equal(plain.leaf_values, checked.leaf_values)


@pytest.mark.parametrize("objective,param,method", SINGLE_STEP_CASES)
@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_all_scalar_negative_objectives_drive_common_step_acceptance(objective, param, method, backtracking):
    bins, targets, features, borders, weights = dataset(objective if objective in ("Logloss", "CrossEntropy") else "RMSE")
    if objective in ("Poisson", "Tweedie", "LogLinQuantile"):
        targets = np.exp(.5 * targets).astype(np.float32)
    config = options(objective, iterations=2, depth=2, sample_weight=weights,
                     objective_param=param, leaf_estimation_method=method, leaf_estimation_iterations=4,
                     leaf_estimation_backtracking=backtracking, fold_size_loss_normalization=True)
    expected = reference.train_reference(bins, targets, features, borders, **config)
    actual = _ordered.train(bins, targets, features, borders, **config)
    compare(actual, expected)


@pytest.mark.parametrize("objective", ["Quantile", "MAE", "MAPE"])
@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_exact_leaf_estimation_ignores_backtracking(objective, backtracking):
    bins, targets, features, borders, weights = dataset(rows=33)
    config = options(objective, iterations=2, permutation_count=4, sample_weight=weights,
                     leaf_estimation_method="Exact", leaf_estimation_iterations=4)
    plain = _ordered.train(bins, targets, features, borders, **config)
    checked = _ordered.train(bins, targets, features, borders,
                             **dict(config, leaf_estimation_backtracking=backtracking))
    np.testing.assert_array_equal(plain.predictions, checked.predictions)
    np.testing.assert_array_equal(plain.leaf_values, checked.leaf_values)


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_common_step_sampling_snapshot_resume_is_exact(backtracking, permutation_count):
    bins, targets, features, borders, config = hard_problem("Poisson", None, "Newton",
        iterations=5, permutation_count=permutation_count, leaf_estimation_backtracking=backtracking,
        bootstrap_type="MVS", subsample=.6, random_strength=.5)
    whole = _ordered.train(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **dict(config, iterations=2)) as session:
        session.step()
        session.step()
        state = session.state()
    resumed = _ordered.train(bins, targets, features, borders,
                             **dict(config, iterations=3, initial_state=state))
    np.testing.assert_array_equal(resumed.predictions, whole.predictions)
    np.testing.assert_array_equal(resumed.leaf_values, whole.leaf_values[2:])
    assert resumed.stats["bootstrap_state"] == whole.stats["bootstrap_state"]


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_rejected_trials_extend_budget_until_first_success(backtracking):
    bins = np.zeros((1, 33), np.uint8)
    targets = np.full(33, 100, np.float32)
    empty = np.empty(0, np.uint32)
    config = options("Lq", iterations=1, depth=0, bias=0, objective_param=3,
                     leaf_estimation_method="Gradient", leaf_estimation_iterations=2,
                     leaf_estimation_backtracking=backtracking, l2_leaf_reg=.2)
    expected = reference.train_reference(bins, targets, empty, empty, **config)
    trace = expected["backtracking_traces"][0]
    assert len(trace) > config["leaf_estimation_iterations"]
    assert len(trace) <= 100 and trace[-1]["accepted"]
    assert not any(trial["accepted"] for trial in trace[:-1])
    actual = _ordered.train(bins, targets, empty, empty, **config)
    compare(actual, expected)


@pytest.mark.parametrize("backtracking", ["AnyImprovement", "Armijo"])
def test_nonfinite_poisson_trials_shrink_safely_and_zero_mass_prefix_stays_zero(backtracking):
    bins = np.zeros((1, 33), np.uint8)
    targets, weights = np.full(33, 10000, np.float32), np.ones(33, np.float32)
    targets[:2], weights[:2] = 1e30, 0
    empty = np.empty(0, np.uint32)
    config = options("Poisson", iterations=1, depth=0, bias=0, l2_leaf_reg=.2,
                     sample_weight=weights, leaf_estimation_method="Newton", leaf_estimation_iterations=2,
                     leaf_estimation_backtracking=backtracking, fold_size_loss_normalization=True)
    expected = reference.train_reference(bins, targets, empty, empty, **config)
    trace = expected["backtracking_traces"][0]
    assert not np.isfinite(trace[0]["score_after"]) and not trace[0]["accepted"]
    assert len(trace) > 2 and trace[-1]["accepted"] and np.isfinite(trace[-1]["score_after"])
    with _ordered.Session(bins, targets, empty, empty, **config) as session:
        session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_array_equal(state["cursors"][:5], 0)
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)


def test_invalid_backtracking_rule_is_rejected():
    bins, targets, features, borders, _ = dataset()
    with pytest.raises((ValueError, RuntimeError)):
        _ordered.Session(bins, targets, features, borders,
                          **options(leaf_estimation_backtracking="Unknown"))
