"""Actual Metal vector-tree training. References never fit CPU models."""
import numpy as np
import pytest

from catboost_metal import _multioutput, _multioutput_math


OBJECTIVES = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")


def data(objective, rows=513):
    rng = np.random.default_rng(152)
    bins = rng.integers(0, 16, (4, rows), dtype=np.uint8)
    continuous = np.column_stack([(bins[0].astype(float) - 7) / 5,
                                 (bins[1].astype(float) - 8) / 6,
                                 (bins[2].astype(float) - 7) / 5]).astype(np.float32)
    if objective == "MultiRMSE":
        targets = continuous
    elif objective == "RMSEWithUncertainty":
        targets = continuous[:, 0] + rng.normal(size=rows).astype(np.float32) * (.1 + bins[1] / 20)
    elif objective == "MultiLogloss":
        targets = (continuous > 0).astype(np.float32)
    else:
        targets = (1 / (1 + np.exp(-continuous))).astype(np.float32)
    cf = np.repeat(np.arange(4, dtype=np.uint32), 15)
    cb = np.tile(np.arange(15, dtype=np.uint32), 4)
    return bins, targets, cf, cb


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("backtracking", ["No", "AnyImprovement", "Armijo"])
def test_shared_vector_trees_learn_and_cursor_matches_tree_traversal(objective, backtracking):
    bins, targets, cf, cb = data(objective)
    result = _multioutput.train(bins, targets, cf, cb, objective=objective, iterations=12,
        depth=3, learning_rate=.2, leaf_estimation_iterations=3,
        leaf_estimation_backtracking=backtracking)
    assert result.loss[-1] < result.loss[0] * (.95 if objective == "MultiCrossEntropy" else .85)
    expected = np.zeros_like(result.predictions)
    for tree, depth in enumerate(result.depths):
        leaf = np.zeros(bins.shape[1], np.uint32)
        for level in range(depth):
            leaf |= (bins[result.split_features[tree, level]] > result.split_bins[tree, level]).astype(np.uint32) << level
        expected += result.leaf_values[tree, leaf]
    # Training applies learning_rate * unscaled leaf + cursor in one GPU FMA;
    # exported leaves are separately rounded after learning-rate scaling.
    np.testing.assert_allclose(result.predictions, expected, rtol=5e-7, atol=5e-7)
    assert result.leaf_values.shape[-1] == (2 if objective == "RMSEWithUncertainty" else 3)
    assert "Apple" in result.stats["device"]


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_one_tree_original_weight_leaf_direction_matches_math(objective, method):
    bins, targets, _, _ = data(objective, 257)
    weights = np.linspace(.1, 3, bins.shape[1], dtype=np.float32)
    bins = bins[:1]
    result = _multioutput.train(bins, targets, [0], [7], objective=objective, iterations=1,
        depth=1, learning_rate=.2, l2_leaf_reg=2.5, sample_weight=weights,
        leaf_estimation_method=method, bootstrap_type="Bayesian", random_seed=71)
    assert result.depths[0] == 1
    dimensions = result.predictions.shape[1]
    reference = _multioutput_math.objective_and_leaf_directions(
        np.zeros((dimensions, bins.shape[1]), np.float32), targets.T,
        objective=objective, leaf_ids=(bins[0] > 7).astype(np.uint32), leaves=2,
        sample_weight=weights, l2_leaf_reg=2.5, leaf_estimation_method=method)
    np.testing.assert_allclose(result.leaf_values[0], reference["directions"] * np.float32(.2), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_permutation_continuation_is_exact(objective):
    bins, targets, cf, cb = data(objective, 129)
    banks = np.stack([bins, bins[:, ::-1], np.roll(bins, 1, axis=1), bins])
    opts = dict(objective=objective, depth=2, learning_rate=.1, leaf_estimation_iterations=2,
                bootstrap_type="Bayesian", random_seed=99, random_strength=.4)
    with _multioutput.Session(bins, targets, cf, cb, iterations=6, **opts) as full:
        full.configure_permutations(banks)
        for i in range(6):
            full.select_permutation(i % 4); full.step()
        expected = full.result()
    with _multioutput.Session(bins, targets, cf, cb, iterations=2, **opts) as prefix:
        prefix.configure_permutations(banks)
        for i in range(2):
            prefix.select_permutation(i % 4); prefix.step()
        state = prefix.permutation_state
    with _multioutput.Session(bins, targets, cf, cb, iterations=4, iteration_offset=2,
        initial_predictions=state["predictions"][-1],
        initial_optimization_predictions=state["optimization_predictions"][-1], **opts) as resumed:
        resumed.configure_permutations(banks, initial_predictions=state["predictions"],
            optimization_predictions=state["optimization_predictions"])
        for i in range(2, 6):
            resumed.select_permutation(i % 4); resumed.step()
        actual = resumed.result()
    np.testing.assert_array_equal(actual.leaf_values, expected.leaf_values[2:])
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    np.testing.assert_array_equal(actual.loss, expected.loss[2:])


@pytest.mark.parametrize("objective,targets", [
    ("MultiRMSE", np.ones(3)), ("MultiRMSE", np.ones((3, 1))),
    ("MultiLogloss", np.full((3, 2), .2)), ("MultiCrossEntropy", np.full((3, 2), 1.2)),
    ("RMSEWithUncertainty", np.ones((3, 2))),
])
def test_bad_vector_targets_rejected_before_gpu(monkeypatch, objective, targets):
    from catboost_metal import _multiclass
    monkeypatch.setattr(_multiclass, "build_library", lambda: pytest.fail("Must reject before GPU load"))
    with pytest.raises(ValueError):
        _multioutput.train(np.zeros((1, 3), np.uint8), targets, [], [], objective=objective)
