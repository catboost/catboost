"""CUDA greedy score-calcer equations, evaluated on the actual Metal GPU."""
import ctypes as ct
import math

import numpy as np
import pytest

from catboost_metal import _multiclass, _multioutput_math


SCORES = {"SolarL2": 4, "LOOL2": 5, "SatL2": 6}


def gpu_calcer(gradients, weights, kind):
    gradients = np.asarray(gradients, np.float64)
    weights = np.asarray(weights, np.float32)
    if gradients.ndim == 1:
        gradients = gradients[:, None]
        weights = weights[:, None]
    pairs = np.empty((*gradients.shape, 2), np.float32)
    pairs[..., 0] = gradients
    pairs[..., 1] = gradients - pairs[..., 0].astype(np.float64)
    output = np.empty(len(gradients), np.float32)
    lib = _multioutput_math._load(_multioutput_math.build_library())
    function = lib.cbm_vector_score_math
    f32 = ct.POINTER(ct.c_float)
    function.argtypes = [ct.c_uint32] * 3 + [f32] * 3 + [ct.c_char_p, ct.c_size_t]
    function.restype = ct.c_int
    error = ct.create_string_buffer(1024)
    code = function(len(gradients), gradients.shape[1], SCORES[kind],
        pairs.ctypes.data_as(f32), weights.ctypes.data_as(f32), output.ctypes.data_as(f32), error, len(error))
    assert code == 0, error.value.decode()
    return output


def reference(gradients, weights, kind):
    """Literal score_calcers.cuh double leaf expressions, float Score field."""
    score = np.float32(0)
    for g, w in zip(gradients, weights):
        g, w = float(g), float(w)
        if kind == "SolarL2":
            term = -g * g * (1 + 2 * math.log(w + 1)) / w if w > float(np.float32(1e-20)) else 0
        elif kind == "LOOL2":
            adjust = np.float32(w / (w - 1)) if w > 1 else np.float32(0)
            adjust = np.float32(adjust * adjust)
            term = float(adjust) * (-g * g) / w if w > 0 else 0
        else:
            adjust = np.float32(w * (w - 2) / (w * w - 3 * w + 1)) if w > 2 else np.float32(0)
            term = float(adjust) * ((-g * g) / w) if w > 0 else 0
        score = np.float32(float(score) + term)
    return score


@pytest.mark.parametrize("kind", SCORES)
def test_extra_score_calcers_match_cuda_float_accumulator(kind):
    rng = np.random.default_rng(992)
    weights = np.exp(rng.uniform(-3, 10, (1031, 24))).astype(np.float32)
    # A low component tests the missing MultiClass gradient's wider sum.
    gradients = rng.normal(size=weights.shape) * weights * (1 + 2**-30)
    result = gpu_calcer(gradients, weights, kind)
    expected = np.array([reference(g, w, kind) for g, w in zip(gradients, weights)])
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=1e-4)


@pytest.mark.parametrize("kind,threshold", [("SolarL2", 1e-20), ("LOOL2", 1), ("SatL2", 2)])
def test_score_strict_weight_threshold(kind, threshold):
    at = np.float32(threshold)
    weights = np.array([0, np.nextafter(at, np.float32(0)), at, np.nextafter(at, np.float32(np.inf))], np.float32)
    gradients = np.full(len(weights), 1e-10 if kind == "SolarL2" else .5)
    result = gpu_calcer(gradients, weights, kind)
    expected = [reference([g], [w], kind) for g, w in zip(gradients, weights)]
    np.testing.assert_array_equal(result[:3], np.zeros(3))
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=1e-9)
    assert result[3] != 0


def test_sat_preserves_negative_adjustment_and_adjacent_pole_signs():
    pole = np.float32((3 + math.sqrt(5)) / 2)
    weights = np.array([2.1, 2.5, np.nextafter(pole, np.float32(0)), pole,
                        np.nextafter(pole, np.float32(np.inf)), 3., 1e20, 1e30], np.float32)
    gradients = np.ones(len(weights))
    result = gpu_calcer(gradients, weights, "SatL2")
    expected = [reference([g], [w], "SatL2") for g, w in zip(gradients, weights)]
    np.testing.assert_allclose(result, expected, rtol=5e-7, atol=0)
    below = weights.astype(np.float64) < (3 + math.sqrt(5)) / 2
    assert (result[below] > 0).all() and (result[~below] < 0).all()


@pytest.mark.parametrize("kind", SCORES)
def test_large_finite_leaf_score_avoids_intermediate_square_overflow(kind):
    weights = np.array([1e20, 1e25, 1e30], np.float32)
    gradients = np.array([1e19, 1e24, 1e29])
    result = gpu_calcer(gradients, weights, kind)
    assert np.isfinite(result).all()
    expected = [reference([g], [w], kind) for g, w in zip(gradients, weights)]
    np.testing.assert_allclose(result, expected, rtol=3e-6)


@pytest.mark.parametrize("kind,weight,gradient", [
    ("LOOL2", np.nextafter(np.float32(1), np.float32(2)), 1e-24),
    ("SatL2", np.nextafter(np.float32((3 + math.sqrt(5))/2), np.float32(3)), 1e-22),
])
def test_adjustment_restores_finite_score_after_tiny_gradient_square(kind, weight, gradient):
    expected = reference([gradient], [weight], kind)
    assert abs(expected) > np.finfo(np.float32).tiny
    result = gpu_calcer([gradient], [weight], kind)
    np.testing.assert_allclose(result, [expected], rtol=3e-6, atol=0)


def initial_gradients(objective, targets, weights):
    if objective == "MultiClass":
        active = weights[:, None] * ((targets[:, None] == np.arange(2)).astype(np.float32) - np.float32(1 / 3))
        return np.column_stack([active, -active.astype(np.float64).sum(axis=1)]), 2
    if objective == "MultiClassOneVsAll":
        return weights[:, None] * ((targets[:, None] == np.arange(3)).astype(np.float32) - np.float32(.5)), 3
    if objective == "RMSEWithUncertainty":
        return np.column_stack([weights * targets, weights * (targets * targets - 1)]), 2
    if objective == "MultiRMSE":
        return weights[:, None] * targets, 3
    return weights[:, None] * (targets - np.float32(.5)), 3


def root_scores(bins, gradients, weights, dimensions, kind):
    output = []
    parent = gradients[:, :dimensions].sum(axis=0, dtype=np.float64)
    total_weight = weights.sum(dtype=np.float64)
    for feature in bins:
        selected = feature == 0
        left_weight = np.float32(weights[selected].sum(dtype=np.float64))
        right_weight = np.float32(total_weight - float(left_weight))
        left = gradients[selected, :dimensions].sum(axis=0, dtype=np.float64).astype(np.float32)
        right = (parent - left.astype(np.float64)).astype(np.float32)
        terms, term_weights = [], []
        for l, r in zip(left, right):
            terms.extend((float(l), float(r))); term_weights.extend((left_weight, right_weight))
        if dimensions < gradients.shape[1]:
            l = left.sum(dtype=np.float64)
            terms.extend((-l, -(parent.sum() - l))); term_weights.extend((left_weight, right_weight))
        output.append(reference(terms, term_weights, kind))
    return np.asarray(output, np.float32)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll", "MultiRMSE",
                                       "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy"])
@pytest.mark.parametrize("kind", SCORES)
@pytest.mark.parametrize("penalty", [False, True])
def test_vector_tree_winner_matches_full_gradient_cuda_calcer(objective, kind, penalty):
    rng = np.random.default_rng(913)
    rows, features = 129, 19
    bins = rng.integers(0, 2, (features, rows), dtype=np.uint8)
    weights = rng.choice(np.array([.25, .5, 1, 2], np.float32), rows)
    if objective in ("MultiClass", "MultiClassOneVsAll"):
        targets = rng.integers(0, 3, rows, dtype=np.uint32)
    elif objective == "RMSEWithUncertainty":
        targets = rng.integers(-3, 4, rows).astype(np.float32)
    elif objective == "MultiRMSE":
        targets = rng.integers(-3, 4, (rows, 3)).astype(np.float32)
    else:
        targets = rng.integers(0, 2 if objective == "MultiLogloss" else 5, (rows, 3)).astype(np.float32)
        if objective == "MultiCrossEntropy":
            targets /= 4
    gradients, dimensions = initial_gradients(objective, targets, weights)
    scores = root_scores(bins, gradients, weights, dimensions, kind)
    counts = np.arange(1, features + 1, dtype=np.uint32) * 9 if penalty else np.zeros(features, np.uint32)
    feature_weights = np.ones(features, np.float32)
    if penalty:
        feature_weights = np.power((1 + counts.astype(np.float32) / np.float32(counts.max())).astype(np.float32).astype(float), -.7).astype(np.float32)
    expected = int(np.argmin(scores * feature_weights))
    with _multiclass.Session(bins, targets, np.arange(features), np.zeros(features, np.uint32),
        classes=2 if objective == "RMSEWithUncertainty" else 3, objective=objective,
        depth=1, iterations=1, score_function=kind, sample_weight=weights,
        l2_leaf_reg=3, random_strength=1e20) as session:
        session.configure_feature_penalties(counts, model_size_reg=.7)
        step = session.step()
    assert step.depth == 1
    assert step.split_features.tolist() == [expected]


@pytest.mark.parametrize("kind", SCORES)
def test_vector_extra_scores_ignore_l2_and_noise(kind):
    rng = np.random.default_rng(58)
    bins = rng.integers(0, 2, (8, 257), dtype=np.uint8)
    targets = rng.normal(size=(257, 3)).astype(np.float32)
    first = _multiclass.train(bins, targets, np.arange(8), np.zeros(8, np.uint32),
        classes=3, objective="MultiRMSE", score_function=kind, depth=1, iterations=1,
        l2_leaf_reg=0, random_strength=0)
    second = _multiclass.train(bins, targets, np.arange(8), np.zeros(8, np.uint32),
        classes=3, objective="MultiRMSE", score_function=kind, depth=1, iterations=1,
        l2_leaf_reg=5000, random_strength=1e20)
    np.testing.assert_array_equal(first.split_features, second.split_features)
    assert not np.array_equal(first.leaf_values, second.leaf_values)


def test_sat_positive_raw_score_stops_growth_but_marks_winning_ctr_used():
    # w=2.5 is inside Sat's negative-adjustment interval, so the raw score is
    # positive. A zero feature multiplier cannot turn it into a growing tree.
    with _multiclass.Session(np.zeros((1, 2), np.uint8), np.ones((2, 3), np.float32),
        [0], [0], classes=3, objective="MultiRMSE", score_function="SatL2", depth=1,
        iterations=1, sample_weight=np.full(2, 1.25, np.float32)) as session:
        session.configure_feature_penalties([10], model_size_reg=5000)
        step = session.step()
        assert session.feature_penalty_state["used_features"].tolist() == [1]
    assert step.depth == 0


@pytest.mark.parametrize("kind,weight", [("LOOL2", 2), ("SatL2", 3)])
def test_calcer_rounds_accumulator_after_every_leaf(kind, weight):
    gradients = np.ones((1, 201))
    gradients[0, 0] = 8192
    weights = np.full_like(gradients, weight, dtype=np.float32)
    result = gpu_calcer(gradients, weights, kind)
    expected = reference(gradients[0], weights[0], kind)
    np.testing.assert_array_equal(result, [expected])
    first = float(reference(gradients[0, :1], weights[0, :1], kind))
    small = float(reference([1], [weight], kind))
    assert expected != np.float32(first + 200 * small)


@pytest.mark.parametrize("kind", SCORES)
def test_missing_last_class_gradient_changes_winner(kind):
    for seed in range(100):
        rng = np.random.default_rng(seed)
        bins = rng.integers(0, 2, (9, 49), dtype=np.uint8)
        targets = rng.integers(0, 3, 49, dtype=np.uint32)
        weights = rng.choice(np.array([.5, 1, 2], np.float32), 49)
        gradients, dimensions = initial_gradients("MultiClass", targets, weights)
        full = root_scores(bins, gradients, weights, dimensions, kind)
        omitted = root_scores(bins, gradients[:, :2], weights, dimensions, kind)
        winner = int(np.argmin(full))
        if winner != int(np.argmin(omitted)):
            break
    else:
        pytest.fail("Fixture must distinguish the missing last-class contribution")
    result = _multiclass.train(bins, targets, np.arange(9), np.zeros(9, np.uint32),
        classes=3, objective="MultiClass", depth=1, iterations=1, score_function=kind, sample_weight=weights)
    assert result.split_features[0, 0] == winner


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll", "MultiRMSE",
                                       "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy"])
@pytest.mark.parametrize("kind", SCORES)
def test_extra_score_public_model_matches_standard_reader(objective, kind, monkeypatch):
    from catboost import CatBoost
    from catboost_metal import CatBoostMetalRegressor, CatBoostMetalClassifier
    monkeypatch.setattr(CatBoost, "fit", lambda *args, **kwargs: pytest.fail("CPU fitting is forbidden"))
    rng = np.random.default_rng(441)
    X = rng.normal(size=(113, 4)).astype(np.float32)
    if objective in ("MultiClass", "MultiClassOneVsAll"):
        y = np.argmax(X[:, :3], axis=1)
    elif objective == "RMSEWithUncertainty":
        y = X[:, 0]
    elif objective == "MultiRMSE":
        y = X[:, :3]
    elif objective == "MultiLogloss":
        y = (X[:, :3] > 0).astype(np.float32)
    else:
        y = 1 / (1 + np.exp(-X[:, :3]))
    cls = CatBoostMetalRegressor if objective in ("MultiRMSE", "RMSEWithUncertainty") else CatBoostMetalClassifier
    model = cls(loss_function=objective, score_function=kind, iterations=3, depth=2,
                leaf_estimation_iterations=1).fit(X, y)
    np.testing.assert_allclose(model.predict(X[:7], prediction_type="RawFormulaVal", task_type="GPU"),
        model.to_catboost().predict(X[:7], prediction_type="RawFormulaVal"), atol=1e-12)
    assert model.tree_count_ == 3
    assert model.training_stats_["kernel_dispatches"] > 0
