"""Finite CUDA double direction dots must survive float32 product overflow."""

import platform

import numpy as np
import pytest

from catboost_metal import _multiclass


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal required",
)


@pytest.mark.parametrize("method", ("Newton", "Gradient"))
def test_large_finite_ova_noise_preserves_direction_dot(method, monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        pytest.fail("Vector Langevin regression must not fit CPU CatBoost")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)
    events = []
    gradient_noise = np.array([1e25, -2e25], np.float64)

    def noise(event, count):
        events.append((event, count))
        assert count == 2
        assert event in (1, 2)
        return gradient_noise.copy() if event == 1 else np.zeros(count, np.float64)

    # Balanced labels at zero logits give clean gradient zero and per-class
    # Hessian 4 * .25. CUDA adds lambda before noise, solves in double, then
    # rounds each direction to float. Its full noisy G dot direction is finite
    # in double but much larger than FLT_MAX. One leaf iteration still computes
    # that dot before bypassing backtracking and publishing its finite update.
    denominator = (1.0 if method == "Newton" else 4.0) + 1.0
    direction = (gradient_noise / denominator).astype(np.float32)
    source_dot = np.dot(gradient_noise, direction.astype(np.float64))
    assert np.isfinite(source_dot) and source_dot > np.finfo(np.float32).max
    expected = (direction * np.float32(.25)).reshape(1, 2)

    with _multiclass.Session(
        bins=np.zeros((1, 4), np.uint8),
        targets=np.array([0, 1, 0, 1], np.uint32),
        sample_weight=np.ones(4, np.float32),
        initial_predictions=np.zeros((4, 2), np.float32),
        classes=2,
        objective="MultiClassOneVsAll",
        candidate_features=np.array([], np.uint32),
        candidate_bins=np.array([], np.uint32),
        iterations=1,
        depth=0,
        learning_rate=.25,
        l2_leaf_reg=1,
        random_strength=0,
        bootstrap_type="No",
        leaf_estimation_method=method,
        leaf_estimation_iterations=1,
        leaf_estimation_backtracking="Armijo",
    ) as session:
        session.configure_langevin(2, noise)
        tree = session.step()
        prediction = session.predictions()
        assert session.completed_iterations == 1

    assert tree.depth == 0 and np.isfinite(tree.loss)
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=2e-6, atol=0)
    np.testing.assert_allclose(prediction, np.repeat(expected, 4, axis=0), rtol=2e-6, atol=0)
    np.testing.assert_array_equal(tree.leaf_weights, [4])
    assert events == [(1, 2), (2, 2)]
