"""CUDA equation oracles only: no CPU model fitting."""
import numpy as np
import pytest

from catboost_metal import _multioutput_math as math


@pytest.mark.parametrize("objective,dimensions", [("MultiRMSE", 2), ("MultiRMSE", 7),
                                                  ("MultiRMSE", 64), ("RMSEWithUncertainty", 2),
                                                  ("MultiLogloss", 7), ("MultiCrossEntropy", 7)])
@pytest.mark.parametrize("rows", [1, 257, 4099])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_gpu_objective_and_partition_leaves(objective, dimensions, rows, method):
    rng = np.random.default_rng(rows + dimensions)
    predictions = rng.normal(0, .5, (dimensions, rows)).astype(np.float32)
    targets = rng.normal(size=rows if objective == "RMSEWithUncertainty" else (dimensions, rows)).astype(np.float32)
    if objective == "MultiLogloss":
        targets = (targets > 0).astype(np.float32)
    elif objective == "MultiCrossEntropy":
        targets = (1 / (1 + np.exp(-targets))).astype(np.float32)
    weights = rng.uniform(.01, 3, rows).astype(np.float32)
    if rows > 1:
        weights[::7] = 0
    ids = rng.integers(0, 7, rows, dtype=np.uint32)
    result = math.objective_and_leaf_directions(predictions, targets, objective=objective,
        leaf_ids=ids, leaves=9, sample_weight=weights, l2_leaf_reg=2.5,
        leaf_estimation_method=method)
    w = weights.astype(float)
    if objective == "MultiRMSE":
        error = targets.astype(float) - predictions
        gradients = error * w
        hessian = np.broadcast_to(w, gradients.shape)
        loss = (error * error).sum(axis=0) * w
    elif objective == "RMSEWithUncertainty":
        error = targets.astype(float) - predictions[0]
        ev = np.exp(np.minimum(-2 * predictions[1].astype(float), 70))
        norm = error * error * ev
        gradients = np.array([w * error, w * (norm - 1)])
        hessian = np.array([w, 2 * w * norm])
        loss = w * (.9189385332046 + predictions[1] + .5 * norm)
    else:
        probability = 1 / (1 + np.exp(-predictions.astype(float)))
        gradients = (targets - probability) * w
        hessian = probability * (1 - probability) * w
        loss = (np.maximum(predictions, 0) - targets * predictions
                + np.log1p(np.exp(-abs(predictions.astype(float))))).mean(axis=0) * w
    np.testing.assert_allclose(result["gradients"], gradients, rtol=2e-5, atol=4e-6)
    np.testing.assert_allclose(result["hessian_diagonal"], hessian, rtol=2e-5, atol=4e-6)
    np.testing.assert_allclose(result["weighted_losses"], loss, rtol=2e-5, atol=4e-6)
    expected = np.zeros((9, dimensions))
    for leaf in range(9):
        selected = ids == leaf
        if w[selected].sum() > 0:
            denominator = w[selected].sum() if method == "Gradient" else hessian[:, selected].sum(axis=1)
            expected[leaf] = gradients[:, selected].sum(axis=1) / (denominator + 2.5)
    np.testing.assert_allclose(result["directions"], expected, rtol=3e-5, atol=5e-6)
    assert result["stats"]["kernel_dispatches"] == 3
    assert "Apple" in result["stats"]["device"]


def test_uncertainty_mean_is_natural_gradient_and_exponent_is_clamped():
    predictions = np.array([[.5, .5, .5], [-50, -35, -10]], np.float32)
    targets = np.array([.500001, .500001, .500001], np.float32)
    result = math.objective_and_leaf_directions(predictions, targets, objective="RMSEWithUncertainty")
    np.testing.assert_array_equal(result["gradients"][0], targets - predictions[0])
    assert result["hessian_diagonal"][0].tolist() == [1, 1, 1]
    assert result["gradients"][1, 0] == result["gradients"][1, 1]
    assert np.isfinite(result["directions"]).all()


def test_multilabel_does_not_use_onevsall_probability_clipping():
    predictions = np.array([[-30., -50.], [30., 50.]], np.float32)
    targets = np.array([[0., 0.], [1., 1.]], np.float32)
    result = math.objective_and_leaf_directions(predictions, targets, objective="MultiLogloss")
    assert 0 < -result["gradients"][0, 1] < 1e-20
    assert result["gradients"][1].tolist() == [0., 0.]
    assert result["hessian_diagonal"][1].tolist() == [0., 0.]


def test_uncertainty_singular_newton_leaf_reports_failure():
    with pytest.raises(RuntimeError, match="leaf solve failed"):
        math.objective_and_leaf_directions(np.zeros((2, 5), np.float32), np.zeros(5),
            objective="RMSEWithUncertainty", l2_leaf_reg=0)
    result = math.objective_and_leaf_directions(np.zeros((2, 5), np.float32), np.zeros(5),
        objective="RMSEWithUncertainty", l2_leaf_reg=0, leaf_estimation_method="Gradient")
    np.testing.assert_array_equal(result["directions"], [[0, -1]])


def test_masked_overflow_and_large_partition():
    rows = 65539
    predictions = np.zeros((2, rows), np.float32)
    predictions[:, 0] = 1e30
    targets = np.ones((2, rows), np.float32)
    weights = np.ones(rows, np.float32)
    weights[0] = 0
    result = math.objective_and_leaf_directions(predictions, targets, sample_weight=weights, l2_leaf_reg=0)
    np.testing.assert_array_equal(result["directions"], [[1, 1]])
    assert result["weighted_losses"][0] == 0


@pytest.mark.parametrize("kwargs", [
    {"objective": "MultiRMSEWithMissingValues"}, {"leaf_ids": [-1, 0, 0]},
    {"leaf_ids": [0., 0., 0.]}, {"leaf_ids": [0, 0, 2], "leaves": 2},
    {"sample_weight": [0, 0, 0]}, {"sample_weight": [1, -1, 1]},
    {"l2_leaf_reg": float("nan")}, {"leaf_estimation_method": "Exact"},
])
def test_invalid_inputs_before_native_load(monkeypatch, kwargs):
    monkeypatch.setattr(math, "build_library", lambda: pytest.fail("Must reject before native load"))
    with pytest.raises(ValueError):
        math.objective_and_leaf_directions(np.zeros((2, 3)), np.ones((2, 3)), **kwargs)


def test_uncertainty_target_shape_rejected_before_pointer(monkeypatch):
    monkeypatch.setattr(math, "build_library", lambda: pytest.fail("Must reject before native load"))
    with pytest.raises(ValueError, match="targets"):
        math.objective_and_leaf_directions(np.zeros((2, 3)), np.ones((2, 3)), objective="RMSEWithUncertainty")
