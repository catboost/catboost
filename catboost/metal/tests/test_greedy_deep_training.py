"""Lossguide path depth is independent of symmetric-tree depth limits."""
import platform

import numpy as np
import pytest

from catboost_metal import _greedy
from catboost_metal._greedy_inference import predict_bins, tree_depth
from catboost_metal._greedy_training import run_training


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="Actual Apple Silicon Metal GPU required")
@pytest.mark.parametrize("objective", ["RMSE", "Logloss"])
@pytest.mark.parametrize("depth,capacity,expected", [(100, 31, 30), (100, 101, 100),
    (2**32 - 1, 33, 32), (2**32 - 1, 1, 0)])
def test_deep_lossguide_uses_leaf_bound_and_gpu_predictions(tmp_path, objective, depth, capacity, expected):
    bins = np.zeros((1, 32), np.uint8)
    targets = np.ones(32, np.float32)
    weights = np.linspace(.25, 2., 32, dtype=np.float32)
    candidates = np.zeros(1, np.uint32)
    options = dict(grow_policy="Lossguide", iterations=2, depth=depth, max_leaves=capacity,
        learning_rate=.2, l2_leaf_reg=3., bias=0., score_function="NewtonL2", objective=objective,
        sample_weight=weights, eval_bins=bins, eval_targets=targets, eval_weight=weights,
        use_best_model=False, save_snapshot=True, snapshot_file=tmp_path / "deep.npz")
    result = run_training(bins, targets, candidates, candidates, **options)
    prediction = 0.
    for tree in result.trees:
        assert len(tree.leaf_values) == expected + 1
        assert tree_depth(tree) == expected
        # CUDA Lossguide retains a defined zero-gain winner. A constant feature
        # therefore creates an increasingly deep left path with empty right leaves.
        assert len(tree.nodes) == 2 * expected + 1
        probability = 1 / (1 + np.exp(-prediction))
        gradient = (1 - prediction if objective == "RMSE" else 1 - probability) * weights.sum(dtype=float)
        denominator = weights.sum(dtype=float) * (1 if objective == "RMSE" else probability * (1 - probability))
        increment = .2 * gradient / (denominator + 3)
        assert tree.leaf_values[0] == pytest.approx(increment, rel=2e-6)
        np.testing.assert_array_equal(tree.leaf_values[1:], 0.)
        prediction += increment
    np.testing.assert_allclose(result.predictions, prediction, rtol=2e-6)
    np.testing.assert_array_equal(predict_bins(bins, result.trees), result.predictions)
    np.testing.assert_array_equal(result.eval_predictions, result.predictions)
    extended = run_training(bins, targets, candidates, candidates, **dict(options, iterations=3))
    full = _greedy.train(bins, targets, candidates, candidates,
        **{key: value for key, value in dict(options, iterations=3).items()
           if key not in {"eval_bins", "eval_targets", "eval_weight", "use_best_model", "save_snapshot", "snapshot_file"}})
    assert extended.resumed_iterations == 2
    np.testing.assert_array_equal(extended.predictions, full.predictions)
    np.testing.assert_array_equal(extended.loss, full.loss)


def test_deep_lossguide_default_leaf_capacity_is_cuda_31():
    with _greedy.TrainingSession(np.zeros((1, 8), np.uint8), np.ones(8), [0], [0],
                               iterations=1, depth=100, grow_policy="Lossguide") as session:
        tree = session.step()
    assert len(tree.leaf_values) == 31 and tree_depth(tree) == 30
