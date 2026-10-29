"""Weighted/binary CUDA-port checks; no upstream CPU training is permitted."""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor, _native
from catboost_metal.regressor import quantize_features
from cuda_extended_reference import initial_bias, sigmoid, train_reference, weighted_loss


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Weighted/objective tests must not invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", fail)
    monkeypatch.setattr(CatBoostClassifier, "fit", fail)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal training checks require Apple Silicon")
    return _native.device_info()


def _kwargs(objective="RMSE", **overrides):
    kwargs = dict(iterations=1, depth=3, learning_rate=0.25, l2_leaf_reg=2,
                  bias=0, score_function="Cosine", objective=objective,
                  leaf_estimation_iterations=1, leaf_estimation_backtracking="No")
    kwargs.update(overrides)
    return kwargs


def _loss(result):
    return result.loss if hasattr(result, "loss") else result.rmse


def _compare(bins, targets, features, borders, **kwargs):
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :depth],
                                      expected["split_features"][tree, :depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :depth],
                                      expected["split_bins"][tree, :depth])
    for key in ("leaf_weights", "leaf_values", "predictions"):
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=5e-5,
                                   atol=5e-5, err_msg=key)
        assert np.isfinite(getattr(actual, key)).all()
    np.testing.assert_allclose(_loss(actual), expected["loss"], rtol=5e-5, atol=5e-6)
    assert np.isfinite(_loss(actual)).all()
    assert actual.stats["kernel_dispatches"] > 0
    return actual, expected


def _two_leaf_data():
    return (np.array([[0, 0, 1, 1]], np.uint8), np.array([0], np.uint32),
            np.array([0], np.uint32))


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_scalar_oracle_iterated_rmse_has_hand_derived_geometric_update(steps):
    bins, features, borders = _two_leaf_data()
    labels = np.array([0, 4, 10, 1000], np.float32)
    weights = np.array([1, 3, 2, 0], np.float32)
    result = train_reference(bins, labels, features, borders,
                             **_kwargs(learning_rate=0.5, sample_weight=weights,
                                       leaf_estimation_iterations=steps))
    # Recomputed residuals with Hessian damping lambda=2, without ridge-gradient
    # subtraction: unshrunk update -> unregularized weighted leaf mean.
    expected = [1.5 * (1 - (1 / 3) ** steps), 5 * (1 - 0.5 ** steps)]
    np.testing.assert_allclose(result["leaf_values"][0, :2], expected, rtol=1e-13)
    np.testing.assert_array_equal(result["leaf_weights"][0, :2], [4, 2])


@pytest.mark.parametrize("objective,labels,expected", [
    ("Logloss", [0, 1], [-0.2, 3 / 7]),
    ("CrossEntropy", [0.2, 0.8], [-0.12, 9 / 35]),
])
def test_scalar_oracle_binary_newton_step_has_hand_derived_values(objective, labels, expected):
    result = train_reference(np.array([[0, 1]], np.uint8), np.array(labels, np.float32),
                             np.array([0], np.uint32), np.array([0], np.uint32),
                             **_kwargs(objective, learning_rate=0.5,
                                       sample_weight=np.array([2, 6], np.float32)))
    np.testing.assert_allclose(result["leaf_values"][0, :2], expected, rtol=1e-6)
    np.testing.assert_allclose(result["loss"][0], np.log(2), rtol=1e-14)


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
@pytest.mark.parametrize("steps", [1, 4])
def test_weighted_objectives_and_newton_iterations_match_cuda_oracle(
        metal_device, objective, score_function, steps):
    rng = np.random.default_rng(4728)
    bins = rng.integers(0, 8, size=(4, 259), dtype=np.uint8)
    signal = (2 * (bins[1] > 4).astype(float) - (bins[3] > 2)
              + 0.3 * rng.normal(size=259))
    labels = signal if objective == "RMSE" else sigmoid(signal)
    if objective == "Logloss":
        labels = (rng.random(259) < labels).astype(float)
    labels = np.asarray(labels, np.float32)
    weights = (2.0 ** rng.integers(-3, 4, 259)).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(4, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 4)
    result, _ = _compare(bins, labels, features, borders,
                         **_kwargs(objective, iterations=3, depth=2,
                                   score_function=score_function, sample_weight=weights,
                                   leaf_estimation_iterations=steps))
    assert _loss(result)[-1] < _loss(result)[0]


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
def test_weight_and_regularization_scaling_preserves_training(metal_device, objective):
    bins, features, borders = _two_leaf_data()
    targets_by_objective = {"RMSE": [0, 3, 2, 1], "Logloss": [0, 0, 1, 1],
                            "CrossEntropy": [0, 0.25, 0.75, 1]}
    labels = np.array(targets_by_objective[objective], np.float32)
    weights = np.array([0.25, 2, 4, 0.5], np.float32)
    options = _kwargs(objective, iterations=2, leaf_estimation_iterations=3)
    one, _ = _compare(bins, labels, features, borders, **options, sample_weight=weights)
    scaled = _native.train(bins, labels, features, borders,
                           **{**options, "l2_leaf_reg": 16}, sample_weight=weights * 8)
    np.testing.assert_allclose(scaled.predictions, one.predictions, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(_loss(scaled), _loss(one), rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
def test_zero_weight_leaf_remains_zero_with_no_l2(metal_device, objective):
    labels = np.array([1, 1, 0, 0], np.float32)
    bins, features, borders = _two_leaf_data()
    weights = np.array([1, 3, 0, 0], np.float32)
    result, _ = _compare(bins, labels, features, borders,
                         **_kwargs(objective, l2_leaf_reg=0, sample_weight=weights,
                                   leaf_estimation_iterations=3))
    assert np.all(result.leaf_values[result.leaf_weights == 0] == 0)


def test_zero_weight_outlier_cannot_overflow_weighted_metric(metal_device):
    bins, features, borders = _two_leaf_data()
    labels = np.array([0, 4, 10, 1e20], np.float32)
    _compare(bins, labels, features, borders,
             **_kwargs(sample_weight=np.array([1, 3, 2, 0], np.float32)))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("order", [[0, 1, 2], [0, 2, 1], [1, 0, 2]])
def test_cancelling_large_terms_preserve_small_leaf_update(metal_device, weighted, order):
    bins = np.zeros((1, 3), np.uint8)
    empty = np.array([], np.uint32)
    if weighted:
        labels = np.array([1, 1, -1], np.float32)
        weights = np.array([2 ** 24, 1, 2 ** 24], np.float32)
    else:
        labels = np.array([2 ** 24, 1, -(2 ** 24)], np.float32)
        weights = None
    labels = labels[order]
    if weights is not None:
        weights = weights[order]
    options = _kwargs(depth=0, l2_leaf_reg=0, learning_rate=1, sample_weight=weights)
    expected = train_reference(bins, labels, empty, empty, **options)
    result = _native.train(bins, labels, empty, empty, **options)
    np.testing.assert_allclose(result.predictions, expected["predictions"],
                               rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("raw_bias", [-100.0, 100.0])
def test_binary_extreme_raw_values_have_finite_loss_and_updates(metal_device, raw_bias):
    labels = np.array([0, 0, 1, 1], np.float32)
    bins, features, borders = _two_leaf_data()
    result, _ = _compare(bins, labels, features, borders,
                         **_kwargs("Logloss", bias=raw_bias, l2_leaf_reg=3,
                                   sample_weight=np.array([1, 2, 3, 4], np.float32)))
    assert np.isfinite(_loss(result)).all()


@pytest.mark.parametrize("raw_bias,target", [[-50.0, 0.0], [50.0, 1.0]])
def test_binary_confident_correct_predictions_preserve_small_positive_loss(
        metal_device, raw_bias, target):
    """Required stability improvement over CUDA's cancelling float loss formula."""
    empty = np.array([], np.uint32)
    result = _native.train(np.zeros((1, 3), np.uint8), np.full(3, target, np.float32),
                           empty, empty, **_kwargs("CrossEntropy", depth=0, bias=raw_bias))
    np.testing.assert_allclose(_loss(result), np.log1p(np.exp(-50.0)), rtol=1e-5, atol=0)


@pytest.mark.parametrize("weight", [0.5e-20, 2e-20, 1e-10])
def test_tiny_leaf_weights_preserve_cuda_zero_l2_normalization(metal_device, weight):
    empty = np.array([], np.uint32)
    options = _kwargs(depth=0, learning_rate=1, l2_leaf_reg=0,
                      sample_weight=np.array([weight], np.float32))
    result, expected = _compare(np.zeros((1, 1), np.uint8), np.ones(1, np.float32),
                                empty, empty, **options)
    np.testing.assert_allclose(result.predictions, expected["predictions"], rtol=1e-5, atol=1e-7)


def test_binary_derivatives_respect_cuda_float32_probability_saturation(metal_device):
    empty = np.array([], np.uint32)
    result, expected = _compare(np.zeros((1, 1), np.uint8), np.ones(1, np.float32),
                                empty, empty, **_kwargs("CrossEntropy", depth=0, bias=20,
                                                       l2_leaf_reg=1e-20, learning_rate=1))
    np.testing.assert_array_equal(expected["leaf_values"], 0)
    np.testing.assert_array_equal(result.predictions, [20])


def test_weighted_regressor_wrapper_and_cbm_roundtrip(metal_device, tmp_path):
    rng = np.random.default_rng(68)
    X = rng.normal(size=(137, 3)).astype(np.float32)
    y = (2 * X[:, 0] + 3 * (X[:, 2] > 0)).astype(np.float32)
    weights = np.linspace(0, 3, y.size).astype(np.float32)
    model = CatBoostMetalRegressor(iterations=3, depth=2, learning_rate=0.3,
                                  l2_leaf_reg=2, border_count=7,
                                  leaf_estimation_iterations=3).fit(X, y, sample_weight=weights)
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, y, features, borders,
                               **_kwargs(iterations=3, depth=2, learning_rate=0.3,
                                         bias=model.bias_, sample_weight=weights,
                                         leaf_estimation_iterations=3))
    np.testing.assert_allclose(model.bias_, initial_bias(y, weights), rtol=1e-6)
    np.testing.assert_allclose(model.predict(X), expected["predictions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(model.loss_history_[-1],
                               weighted_loss(y, model.predict(X), weights, "RMSE"), rtol=1e-5)
    path = tmp_path / "weighted.cbm"
    model.save_model(path)
    restored = CatBoostRegressor().load_model(str(path))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


@pytest.mark.parametrize("boost_from_average", [False, True])
@pytest.mark.parametrize("class_labels", [[-7, 13], [-0.125, 0.75], ["negative", "positive"]])
def test_classifier_labels_weights_probabilities_and_roundtrip(
        metal_device, tmp_path, boost_from_average, class_labels):
    from catboost_metal import CatBoostMetalClassifier
    rng = np.random.default_rng(138)
    X = rng.normal(size=(149, 3)).astype(np.float32)
    encoded = ((X[:, 1] + 0.4 * X[:, 0]) > 0.6).astype(np.float32)
    negative, positive = class_labels
    labels = np.where(encoded == 1, positive, negative)
    weights = (0.25 + rng.random(encoded.size) * 2).astype(np.float32)
    model = CatBoostMetalClassifier(iterations=3, depth=2, learning_rate=0.2,
                                   l2_leaf_reg=2, border_count=7,
                                   leaf_estimation_iterations=3,
                                   boost_from_average=boost_from_average).fit(
                                       X, labels, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, class_labels)
    bias = initial_bias(encoded, weights, objective="Logloss",
                        boost_from_average=boost_from_average)
    np.testing.assert_allclose(model.bias_, bias, atol=1e-6)
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, encoded, features, borders,
                               **_kwargs("Logloss", iterations=3, depth=2,
                                         learning_rate=0.2, bias=bias,
                                         leaf_estimation_iterations=3, sample_weight=weights))
    raw = model.predict(X, prediction_type="RawFormulaVal")
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(raw, expected["predictions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(probabilities[:, 1], sigmoid(raw), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_array_equal(model.predict(X), np.where(raw > 0, positive, negative))
    np.testing.assert_allclose(model.predict(X, prediction_type="RawFormulaVal", task_type="METAL"),
                               raw, rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(model.predict_proba(X, task_type="METAL"),
                               probabilities, rtol=5e-6, atol=5e-6)
    np.testing.assert_array_equal(model.predict(X, task_type="METAL"), model.predict(X))
    for format in ("cbm", "json"):
        path = tmp_path / f"classifier.{format}"
        model.save_model(path, format=format)
        restored = CatBoostClassifier().load_model(str(path), format=format)
        np.testing.assert_array_equal(restored.classes_, model.classes_)
        np.testing.assert_array_equal(restored.predict_proba(X), probabilities)
        np.testing.assert_array_equal(restored.predict(X).reshape(-1), model.predict(X).reshape(-1))


def test_classifier_class_weights_match_effective_sample_weights(metal_device):
    from catboost_metal import CatBoostMetalClassifier
    X = np.arange(30, dtype=np.float32).reshape(15, 2)
    labels = np.array([-7, 13, -7] * 5)
    sample_weight = np.linspace(0.5, 2, 15).astype(np.float32)
    params = dict(iterations=2, depth=2, border_count=7, learning_rate=0.2,
                  l2_leaf_reg=2, leaf_estimation_iterations=2)
    weighted = CatBoostMetalClassifier(**params, class_weights={-7: 0.5, 13: 3}).fit(
        X, labels, sample_weight=sample_weight)
    effective = sample_weight * np.where(labels == -7, 0.5, 3).astype(np.float32)
    plain = CatBoostMetalClassifier(**params).fit(X, labels, sample_weight=effective)
    np.testing.assert_allclose(weighted.predict_proba(X), plain.predict_proba(X),
                               rtol=1e-6, atol=1e-7)


def test_cross_entropy_wrapper_preserves_soft_labels(metal_device, tmp_path):
    from catboost_metal import CatBoostMetalClassifier
    X = np.arange(39, dtype=np.float32).reshape(13, 3)
    targets = np.linspace(0.1, 0.9, 13).astype(np.float32)
    weights = np.linspace(1, 3, 13).astype(np.float32)
    model = CatBoostMetalClassifier(loss_function="CrossEntropy", iterations=2,
                                   depth=2, learning_rate=0.2, l2_leaf_reg=2,
                                   border_count=7, leaf_estimation_iterations=3).fit(
                                       X, targets, sample_weight=weights)
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, targets, features, borders,
                               **_kwargs("CrossEntropy", iterations=2, depth=2,
                                         learning_rate=0.2, sample_weight=weights,
                                         leaf_estimation_iterations=3))
    np.testing.assert_allclose(model.predict(X, prediction_type="RawFormulaVal"),
                               expected["predictions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(model.loss_history_, expected["loss"], rtol=1e-5)
    path = tmp_path / "soft-labels.cbm"
    model.save_model(path)
    restored = CatBoostClassifier().load_model(str(path))
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize("weights", [[1], [-1, 2], [0, 0], [1, np.nan],
                                      [1, np.inf], [[1], [2]], [1, 1e100]])
def test_invalid_sample_weights_rejected_before_gpu(monkeypatch, weights):
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Unexpected GPU training"))
    if hasattr(_native, "Session"):
        monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Unexpected GPU session"))
    with pytest.raises(ValueError):
        CatBoostMetalRegressor().fit([[0], [1]], [0, 1], sample_weight=weights)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_leaf_iterations_rejected_before_gpu(steps):
    with pytest.raises(ValueError):
        CatBoostMetalRegressor(leaf_estimation_iterations=steps)


@pytest.mark.parametrize("replacement", [
    {"sample_weight": np.array([1], np.float32)},
    {"sample_weight": np.array([[1], [2]], np.float32)},
    {"sample_weight": np.array([1, -1], np.float32)},
    {"sample_weight": np.array([0, 0], np.float32)},
    {"sample_weight": np.array([1, np.nan], np.float32)},
    {"sample_weight": np.array([1, np.inf], np.float32)},
    {"sample_weight": np.array([1, 1e100])},
    {"objective": "MultiClass"},
    {"objective": "CrossEntropy", "targets": np.array([-0.1, 1], np.float32)},
    {"objective": "CrossEntropy", "targets": np.array([0, 1.1], np.float32)},
    {"leaf_estimation_iterations": 0},
    {"leaf_estimation_iterations": True},
    {"leaf_estimation_iterations": 1.5},
    {"leaf_estimation_backtracking": "Unknown"},
])
def test_native_weighted_invalid_inputs_rejected_before_gpu(monkeypatch, replacement):
    monkeypatch.setattr(_native, "build_library", lambda: pytest.fail("Reached native runtime"))
    args = dict(bins=np.array([[0, 1]], np.uint8), targets=np.array([0, 1], np.float32),
                candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32),
                **_kwargs(depth=1))
    args.update(replacement)
    with pytest.raises(ValueError):
        _native.train(**args)
