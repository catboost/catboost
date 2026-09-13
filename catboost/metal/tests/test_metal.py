"""Checks of the restricted CUDA-to-Metal port, without CPU model training.

Run with PYTHONPATH=catboost/metal/python. The arithmetic oracle translates
upstream CUDA formulas independently; installed CatBoost is used only for its
quantizer, model loader, and inference. GPU tests skip on non-Apple hosts.
"""

import platform

import numpy as np
import pytest
from catboost import CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor
from catboost_metal import _native
from catboost_metal.regressor import quantize_features
from cuda_reference import train_reference


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("These port checks must not invoke CatBoost CPU training")

    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)


@pytest.fixture(scope="session")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal GPU checks require macOS on Apple Silicon")
    info = _native.device_info()
    assert info["backend"] == "Metal"
    assert info["name"]
    return info


def _check_native(bins, targets, candidate_features, candidate_bins, **options):
    actual = _native.train(bins, targets, candidate_features, candidate_bins, **options)
    expected = train_reference(bins, targets, candidate_features, candidate_bins, **options)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, tree_depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :tree_depth],
                                      expected["split_features"][tree, :tree_depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :tree_depth],
                                      expected["split_bins"][tree, :tree_depth])
    np.testing.assert_array_equal(actual.leaf_weights, expected["leaf_weights"])
    for name in ("leaf_values", "predictions", "rmse"):
        np.testing.assert_allclose(getattr(actual, name), expected[name],
                                   rtol=3e-5, atol=3e-5, err_msg=name)
        assert np.isfinite(getattr(actual, name)).all()
    assert actual.stats["device"]
    assert actual.stats["kernel_dispatches"] > 0
    assert np.isfinite(actual.stats["gpu_seconds"])
    assert actual.stats["gpu_seconds"] >= 0
    return actual


def _options(score_function, **overrides):
    result = dict(iterations=1, depth=3, learning_rate=0.25, l2_leaf_reg=2.0,
                  bias=0.0, score_function=score_function)
    result.update(overrides)
    return result


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_odd_rows_multiple_trees_match_cuda_reference(metal_device, score_function):
    rng = np.random.default_rng(7103)
    bins = rng.integers(0, 8, size=(5, 1031), dtype=np.uint8)
    targets = (7 * (bins[3] > 4).astype(np.float32)
               - 3 * (bins[0] > 2).astype(np.float32)
               + 2 * (bins[4] > 6).astype(np.float32)
               + rng.normal(0, 0.05, 1031)).astype(np.float32)
    features = np.repeat(np.arange(5, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 5)
    result = _check_native(bins, targets, features, borders,
                           **_options(score_function, iterations=5))
    assert result.rmse[-1] < result.rmse[0] * 0.5


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_bin_equality_ties_and_repeated_winner(metal_device, score_function):
    # Both columns describe exactly the same split. Candidate order intentionally
    # differs from feature order: CUDA's within-kernel tie break is candidate index.
    bins = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
    targets = np.array([-2, -2, 2, 2], dtype=np.float32)
    features = np.array([1, 0], dtype=np.uint32)
    borders = np.array([0, 0], dtype=np.uint32)
    result = _check_native(bins, targets, features, borders,
                           **_options(score_function, l2_leaf_reg=0, learning_rate=1))
    assert result.depths.tolist() == [1]
    assert result.split_features[0, 0] == 1
    # Equality belongs to the left child, and the first split is leaf bit zero.
    np.testing.assert_array_equal(result.leaf_values[0, :2], [-2, 2])
    np.testing.assert_array_equal(result.predictions, targets)


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_high_byte_bins_include_128_through_255(metal_device, score_function):
    bins = np.tile(np.arange(256, dtype=np.uint8), 2)[None, :]
    bins = np.concatenate((bins, np.array([[255]], np.uint8)), axis=1)
    targets = np.where(bins[0] > 192, 8, -3).astype(np.float32)
    features = np.zeros(255, dtype=np.uint32)
    borders = np.arange(255, dtype=np.uint32)
    result = _check_native(bins, targets, features, borders,
                           **_options(score_function, depth=1, l2_leaf_reg=0,
                                      learning_rate=1))
    assert result.split_bins[0, 0] == 192
    np.testing.assert_array_equal(result.predictions, targets)


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_zero_l2_empty_leaves_stay_finite(metal_device, score_function):
    bins = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2]], dtype=np.uint8)
    targets = np.array([-3, -3, -3, 0, 0, 0, 3, 3, 3], dtype=np.float32)
    features = np.array([0, 0], dtype=np.uint32)
    borders = np.array([0, 1], dtype=np.uint32)
    result = _check_native(bins, targets, features, borders,
                           **_options(score_function, l2_leaf_reg=0, learning_rate=1))
    assert result.depths.tolist() == [2]
    assert np.count_nonzero(result.leaf_weights[0, :4] == 0) == 1
    assert np.all(result.leaf_values[result.leaf_weights == 0] == 0)
    np.testing.assert_array_equal(result.predictions, targets)


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_zero_residual_does_not_add_gain_filter(metal_device, score_function):
    bins = np.array([[0, 1, 0, 1, 0]], dtype=np.uint8)
    targets = np.full(5, 1.25, dtype=np.float32)
    result = _check_native(bins, targets, np.array([0], np.uint32),
                           np.array([0], np.uint32),
                           **_options(score_function, iterations=2, bias=1.25))
    # Every score is zero: choose candidate zero, then stop at its repetition.
    np.testing.assert_array_equal(result.depths, [1, 1])
    np.testing.assert_array_equal(result.predictions, targets)


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_native_no_candidates_emits_depth_zero(metal_device, score_function):
    bins = np.zeros((2, 7), dtype=np.uint8)
    targets = np.arange(7, dtype=np.float32)
    empty = np.array([], dtype=np.uint32)
    result = _check_native(bins, targets, empty, empty,
                           **_options(score_function, iterations=2, learning_rate=1,
                                      l2_leaf_reg=0))
    np.testing.assert_array_equal(result.depths, [0, 0])
    np.testing.assert_array_equal(result.predictions, np.full(7, 3))


@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_wrapper_matches_reference_and_standard_model_roundtrip(
        metal_device, score_function, tmp_path):
    rng = np.random.default_rng(803)
    features = rng.normal(size=(259, 4)).astype(np.float32)
    targets = (5 * (features[:, 2] > 0.25) - 2 * (features[:, 0] > -0.4)
               + 0.15 * features[:, 1]).astype(np.float32)
    model = CatBoostMetalRegressor(iterations=4, depth=3, border_count=15,
                                  learning_rate=0.3, l2_leaf_reg=2,
                                  score_function=score_function).fit(features, targets)
    _, bins, candidate_features, candidate_bins = quantize_features(features, 15)
    reference = train_reference(
        bins, targets, candidate_features, candidate_bins, iterations=4, depth=3,
        learning_rate=0.3, l2_leaf_reg=2, bias=model.bias_, score_function=score_function)
    np.testing.assert_array_equal(model.tree_depths_, reference["depths"])
    np.testing.assert_allclose(model.training_predictions_, reference["predictions"],
                               rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(model.predict(features), model.training_predictions_,
                               rtol=3e-6, atol=3e-6)
    measured_rmse = np.sqrt(np.mean((targets.astype(np.float64)
                                    - model.training_predictions_) ** 2))
    np.testing.assert_allclose(model.loss_history_[-1], measured_rmse, rtol=3e-6)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.training_stats_["device"] == metal_device["name"]
    assert model.training_stats_["kernel_dispatches"] > 0
    assert model.tree_count_ == 4

    probes = rng.normal(size=(73, 4)).astype(np.float32)
    # Include values equal to an exported border and their immediate neighbours.
    border = model.borders_[2][0]
    probes[:3, 2] = [np.nextafter(border, np.float32(-np.inf)), border,
                     np.nextafter(border, np.float32(np.inf))]
    predictions = model.predict(probes)
    for format in ("cbm", "json"):
        path = tmp_path / f"metal.{format}"
        model.save_model(path, format=format)
        restored = CatBoostRegressor().load_model(str(path), format=format)
        np.testing.assert_array_equal(restored.predict(probes), predictions)
    np.testing.assert_array_equal(model.to_catboost().predict(probes), predictions)
    assert model.predict(np.empty((0, 4))).shape == (0,)


def test_wrapper_all_constant_features(metal_device, tmp_path):
    features = np.full((17, 3), 4.0, dtype=np.float32)
    targets = np.arange(17, dtype=np.float32)
    model = CatBoostMetalRegressor(iterations=3, depth=4).fit(features, targets)
    np.testing.assert_array_equal(model.tree_depths_, [0, 0, 0])
    np.testing.assert_array_equal(model.predict(features), np.full(17, 8.0))
    path = tmp_path / "constant.cbm"
    model.save_model(path)
    restored = CatBoostRegressor().load_model(str(path))
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))


@pytest.mark.parametrize("kwargs", [
    {"iterations": 0}, {"iterations": True}, {"depth": 17}, {"depth": -1},
    {"depth": 1.5}, {"border_count": 256}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"learning_rate": 1e100},
    {"l2_leaf_reg": -1}, {"score_function": "Unknown"},
    {"boosting_type": "Unknown"}, {"task_type": "GPU"},
    {"loss_function": "Logloss"}, {"bootstrap_type": "Uniform"},
    {"random_strength": -1}, {"one_hot_max_size": 256},
    {"leaf_estimation_iterations": 0},
])
def test_invalid_configuration_never_needs_device(monkeypatch, kwargs):
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    with pytest.raises((ValueError, TypeError)):
        CatBoostMetalRegressor(**kwargs)


@pytest.mark.parametrize("features,targets", [
    ([[1], [float("nan")]], [1, 2]), ([[1], [float("inf")]], [1, 2]),
    ([[1e100], [1]], [1, 2]), ([["cat"], ["dog"]], [1, 2]),
    ([[1 + 2j], [2 + 1j]], [1, 2]), ([[1], [2]], [1, float("inf")]),
    ([[1], [2]], [[1], [2]]), ([[1], [2]], [1]),
    ([], []), (np.empty((0, 2)), []), (np.empty((2, 0)), [1, 2]),
    ([[1, 2], [3]], [1, 2]),
])
def test_invalid_training_data_never_needs_device(monkeypatch, features, targets):
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    with pytest.raises(ValueError):
        CatBoostMetalRegressor().fit(features, targets)


def test_unfitted_model_never_needs_device(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    model = CatBoostMetalRegressor()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1]])
    with pytest.raises(RuntimeError, match="fit"):
        model.save_model(tmp_path / "unfitted.cbm")
    with pytest.raises(RuntimeError, match="fit"):
        model.to_catboost()
