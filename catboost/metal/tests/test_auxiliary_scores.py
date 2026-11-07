"""CUDA Plain SolarL2/LOOL2 scalar scores verified on the Metal trainer.

score_calcers.cuh supplies the per-child formulas; DocParallel's
ComputeWeakTarget supplies first-order weighted derivatives. Tests compare
independent candidate arithmetic, including weight thresholds and tied
candidates across reduction groups, without invoking CPU CatBoost training.
"""

import math
import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor, _native
from cuda_auxiliary_score_reference import candidate_scores, score_children
from cuda_scalar_reference import objective_terms, train_reference
from test_newton_scores import _bootstrap_factors


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Solar/LOO arithmetic tests cannot invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Solar/LOO checks require Apple Silicon")
    return _native.device_info()


def options(score, **overrides):
    result = dict(iterations=1, depth=1, learning_rate=0.2, l2_leaf_reg=3, bias=0,
                  score_function=score, objective="RMSE", objective_param=None,
                  leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No")
    result.update(overrides)
    return result


def compare(bins, targets, features, borders, **kwargs):
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :depth], expected["split_features"][tree, :depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :depth], expected["split_bins"][tree, :depth])
    for key in ("leaf_weights", "leaf_values", "predictions"):
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=8e-5, atol=3e-5, err_msg=key)
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=8e-5, atol=3e-5)
    return actual


@pytest.mark.parametrize("score,total,weight,expected", [
    ("SolarL2", 2, 2, -2 * (1 + 2 * math.log(3))),
    ("SolarL2", 10, 0, 0),
    ("SolarL2", 1e-10, np.float32(1e-20), 0),
    ("LOOL2", 2, 2, -8),
    ("LOOL2", 10, 0, 0), ("LOOL2", 10, 0.75, 0), ("LOOL2", 10, 1, 0),
    ("LOOL2", 1, np.nextafter(np.float32(1), np.float32(2)), -70368752566272),
])
def test_score_matches_hand_derived_leaf_arithmetic(score, total, weight, expected):
    assert score_children([total], [weight], score) == float(np.float32(expected))


def test_loo_rounds_running_score_after_each_child_like_cuda():
    # Each weight-two child contributes -2*G². At -200,000,000 the
    # float32 spacing is16, so eight successive contributions of -2 vanish.
    sums, weights = [10000] + [1] * 8, [2] * 9
    actual = score_children(sums, weights, "LOOL2")
    assert actual == -200000000
    assert actual != float(np.float32(-200000000 - 8 * 2))


WINNERS = [
    ("SolarL2", [-2, -1, -2, -3, 1, 3, -1, 0], [.5, .25, .25, 1, 1, .25, 2, .5], 0, 1),
    ("LOOL2", [-4, 2, -2, 2, -4, -4, -1, -2], [.25, .5, 3, 3, 1, 1, 3, 2], 1, 0),
]


@pytest.mark.parametrize("score,targets,weights,winner,l2_winner", WINNERS)
def test_auxiliary_formula_changes_winner_from_l2(metal_device, score, targets, weights, winner, l2_winner):
    bins = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1]], np.uint8)
    targets, weights = np.asarray(targets, np.float32), np.asarray(weights, np.float32)
    features, borders = np.arange(2, dtype=np.uint32), np.zeros(2, np.uint32)
    scores = candidate_scores(bins, targets * weights, weights, features, borders, score)
    assert int(np.argmin(scores)) == winner
    actual = compare(bins, targets, features, borders, **options(score, sample_weight=weights))
    l2 = _native.train(bins, targets, features, borders, **options("L2", sample_weight=weights))
    assert actual.split_features[0, 0] == winner
    assert l2.split_features[0, 0] == l2_winner


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
def test_structure_uses_sample_weights_when_all_objective_hessians_are_zero(metal_device, score):
    bins = np.array([[0, 0, 0, 0], [0, 1, 0, 1]], np.uint8)
    targets, weights = np.array([-1, 1, -1, 1], np.float32), np.full(4, 2, np.float32)
    _, gradients, hessians = objective_terms(targets, np.zeros(4), "Huber", .4)
    assert np.all(hessians == 0)
    features, borders = np.arange(2, dtype=np.uint32), np.zeros(2, np.uint32)
    correct = candidate_scores(bins, weights * gradients, weights, features, borders, score)
    wrong = candidate_scores(bins, weights * gradients, weights * hessians, features, borders, score)
    assert np.argmin(correct) == 1
    assert np.argmin(wrong) == 0
    actual = compare(bins, targets, features, borders,
                     **options(score, objective="Huber", objective_param=.4, sample_weight=weights))
    assert actual.split_features[0, 0] == 1


@pytest.mark.parametrize("score,boundary,target", [("SolarL2", 1e-20, 1e10), ("LOOL2", 1, 1)])
@pytest.mark.parametrize("direction,expected", [(-1, 0), (0, 0), (1, 1)])
def test_float32_weight_threshold_is_strict_and_uses_weight_not_count(metal_device, score, boundary,
                                                                    target, direction, expected):
    weight = np.float32(boundary)
    if direction:
        weight = np.nextafter(weight, np.float32(0 if direction < 0 else np.inf))
    # Candidate zero has zero summed residual; candidate one separates signs.
    # With two observations in total, using a count-based cutoff gives the
    # wrong result for either score at/below its weight boundary.
    bins = np.array([[0, 0], [0, 1]], np.uint8)
    targets, weights = np.array([-target, target], np.float32), np.full(2, weight, np.float32)
    features, borders = np.arange(2, dtype=np.uint32), np.zeros(2, np.uint32)
    scores = candidate_scores(bins, targets * weights, weights, features, borders, score)
    assert np.argmin(scores) == expected
    actual = _native.train(bins, targets, features, borders, **options(score, sample_weight=weights))
    assert actual.split_features[0, 0] == expected


@pytest.mark.parametrize("score,weight,target", [
    ("SolarL2", np.nextafter(np.float32(1e-20), np.float32(np.inf)), 1),
    ("LOOL2", np.nextafter(np.float32(1), np.float32(2)), 1e-25),
    ("SolarL2", 1e30, 1), ("LOOL2", 1e30, 1),
])
def test_representable_score_survives_intermediate_square_underflow_or_overflow(metal_device, score,
                                                                              weight, target):
    # CUDA computes the leaf contribution in double. All resulting scores
    # here fit float32, while squaring a float32 gradient first loses them.
    bins = np.array([[0, 0], [0, 1]], np.uint8)
    targets, weights = np.array([-target, target], np.float32), np.full(2, weight, np.float32)
    features, borders = np.arange(2, dtype=np.uint32), np.zeros(2, np.uint32)
    scores = candidate_scores(bins, targets * weights, weights, features, borders, score)
    assert np.isfinite(scores).all() and scores[1] < scores[0]
    actual = _native.train(bins, targets, features, borders, **options(score, sample_weight=weights))
    assert actual.split_features[0, 0] == 1
    assert np.isfinite(actual.predictions).all()


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
@pytest.mark.parametrize("objective,param,bias", [("RMSE", None, 0), ("Logloss", None, -1.1),
                                                ("Poisson", None, math.log(4)), ("Huber", .4, 0)])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_weighted_multitree_auxiliary_scores_match_scalar_reference(metal_device, score, objective, param,
                                                                   bias, method):
    rng = np.random.default_rng(743)
    bins = rng.integers(0, 8, (4, 263), dtype=np.uint8)
    signal = .8 * (bins[0] > 3) - .5 * (bins[2] > 4) + rng.normal(0, .1, 263)
    targets = ((signal > .2).astype(float) if objective == "Logloss" else
               np.exp(signal) if objective == "Poisson" else signal).astype(np.float32)
    weights = rng.uniform(.25, 2, 263).astype(np.float32)
    weights[::17] = 0
    compare(bins, targets, np.repeat(np.arange(4, dtype=np.uint32), 7),
            np.tile(np.arange(7, dtype=np.uint32), 4),
            **options(score, iterations=3, depth=3, objective=objective, objective_param=param,
                      bias=bias, sample_weight=weights, leaf_estimation_method=method,
                      leaf_estimation_iterations=3))


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
def test_score_ignores_regularization_but_leaf_fitting_uses_it(metal_device, score):
    rng = np.random.default_rng(910)
    bins = rng.integers(0, 8, (4, 259), dtype=np.uint8)
    targets = (2 * (bins[2] > 3) + rng.normal(0, .3, 259)).astype(np.float32)
    features = np.repeat(np.arange(4, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 4)
    fits = [_native.train(bins, targets, features, borders, **options(score, depth=3, l2_leaf_reg=value))
            for value in (.001, 1000)]
    np.testing.assert_array_equal(fits[0].depths, fits[1].depths)
    np.testing.assert_array_equal(fits[0].split_features, fits[1].split_features)
    np.testing.assert_array_equal(fits[0].split_bins, fits[1].split_bins)
    assert np.max(np.abs(fits[0].leaf_values - fits[1].leaf_values)) > .1


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
@pytest.mark.parametrize("kind", ["No", "Bernoulli", "Bayesian", "MVS"])
def test_bootstrap_uses_original_structure_weights_and_ignores_score_noise(metal_device, score, kind):
    rng = np.random.default_rng(2917)
    bins = rng.integers(0, 8, (5, 1031), dtype=np.uint8)
    targets = (rng.random(1031) < .3).astype(np.float32)
    weights = rng.uniform(.25, 3, 1031).astype(np.float32)
    weights[::19] = 0
    features, borders = np.repeat(np.arange(5, dtype=np.uint32), 7), np.tile(np.arange(7, dtype=np.uint32), 5)
    _, unweighted, _ = objective_terms(targets, np.full(1031, np.float32(-1.1)), "Logloss")
    gradient = (weights * unweighted).astype(np.float32)
    factors = _bootstrap_factors(kind, gradient, seed=719)
    scores = candidate_scores(bins, (gradient * factors).astype(np.float32),
                              (weights * factors).astype(np.float32), features, borders, score)
    winner = int(np.argmin(scores))
    sampling = dict(bootstrap_type=kind, random_seed=719)
    if kind in ("Bernoulli", "MVS"):
        sampling["subsample"] = .6
    if kind == "MVS":
        sampling["mvs_reg"] = .75
    if kind == "Bayesian":
        sampling["bagging_temperature"] = 1
    kwargs = options(score, objective="Logloss", bias=-1.1, sample_weight=weights)
    actual = _native.train(bins, targets, features, borders, **kwargs, **sampling, random_strength=1000)
    quiet = _native.train(bins, targets, features, borders, **kwargs, **sampling, random_strength=0)
    assert actual.split_features[0, 0] == features[winner]
    assert actual.split_bins[0, 0] == borders[winner]
    np.testing.assert_array_equal(actual.split_features, quiet.split_features)
    np.testing.assert_array_equal(actual.split_bins, quiet.split_bins)
    np.testing.assert_array_equal(actual.predictions, quiet.predictions)
    expected = train_reference(bins, targets, features[winner:winner + 1], borders[winner:winner + 1], **kwargs)
    np.testing.assert_allclose(actual.leaf_weights, expected["leaf_weights"], rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(actual.leaf_values, expected["leaf_values"], rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
@pytest.mark.parametrize("zero_gradient", [False, True])
def test_global_candidate_ties_and_repeated_winner_stop(metal_device, score, zero_gradient):
    bins = np.full((513, 17), 128, np.uint8)
    bins[257:, 8:] = 255
    targets = np.array([-9.] * 8 + [8.] * 9, np.float32)
    if zero_gradient:
        targets[:] = 0
    features, borders = np.arange(513, dtype=np.uint32), np.full(513, 128, np.uint32)
    actual = _native.train(bins, targets, features, borders, **options(score, depth=4))
    assert actual.split_features[0, 0] == (0 if zero_gradient else 257)
    # After the perfect split, the constant candidate ties it and wins by
    # lower index. CUDA adds that unused winner even though it changes no
    # membership; only its next repeated win stops growth.
    assert actual.depths[0] == (1 if zero_gradient else 2)
    if not zero_gradient:
        assert actual.split_features[0, 1] == 0
    assert np.isfinite(actual.predictions).all()


@pytest.mark.parametrize("score", ["SolarL2", "LOOL2"])
@pytest.mark.parametrize("classification", [False, True])
def test_public_auxiliary_score_model_metadata_exports_and_gpu_inference(metal_device, tmp_path,
                                                                         score, classification):
    rng = np.random.default_rng(831)
    features = rng.normal(size=(259, 3)).astype(np.float32)
    targets = features[:, 0] - .3 * features[:, 1]
    if classification:
        targets = np.where(targets > .2, 13, -7)
    cls = CatBoostMetalClassifier if classification else CatBoostMetalRegressor
    loader = CatBoostClassifier if classification else CatBoostRegressor
    weights = rng.uniform(.2, 2, len(targets)).astype(np.float32)
    model = cls(iterations=3, depth=3, score_function=score, random_strength=100,
                border_count=7, leaf_estimation_iterations=3).fit(features, targets, sample_weight=weights)
    raw = model.predict(features, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(features, prediction_type="RawFormulaVal", task_type="METAL"),
                               raw, rtol=5e-6, atol=5e-6)
    for format in ("cbm", "json"):
        path = tmp_path / ("auxiliary." + format)
        model.save_model(path, format=format)
        restored = loader().load_model(str(path), format=format)
        assert restored.get_all_params()["score_function"] == score
        restored_raw = restored.predict(features, prediction_type="RawFormulaVal")
        if format == "cbm":
            np.testing.assert_array_equal(restored_raw, raw)
        else:
            # JSON decimal parsing can change the final double result by one
            # ULP; retain the established two-ULP serialization-only bound.
            np.testing.assert_array_max_ulp(restored_raw, raw, maxulp=2)
