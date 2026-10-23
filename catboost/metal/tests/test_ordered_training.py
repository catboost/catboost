"""Connected Metal Ordered training against independent CUDA-derived equations.

No CPU CatBoost fitting is used. The oracle lives in cuda_ordered_reference;
Metal traversal separately verifies that only the full-estimation task is
exported to the trained model. Tests inspect real packed fold cursor state.
"""

import copy
import math
import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import _ordered
from cuda_ordered_reference import train_reference


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Ordered tests must not invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", fail)
    monkeypatch.setattr(CatBoostClassifier, "fit", fail)


@pytest.fixture(scope="module", autouse=True)
def apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Connected Ordered training requires Apple Silicon Metal")


def dataset(objective="RMSE", rows=127, seed=8142):
    rng = np.random.default_rng(seed)
    bins = rng.integers(0, 8, (3, rows), dtype=np.uint8)
    signal = (.37 * bins[0].astype(np.float64) - .29 * bins[1]
              + .43 * (bins[2] > 3) + .16 * rng.normal(size=rows) - .5)
    if objective == "RMSE":
        targets = signal
    else:
        probability = 1 / (1 + np.exp(-signal))
        targets = ((rng.uniform(size=rows) < probability).astype(np.float64)
                   if objective == "Logloss" else probability)
    weights = (2.0 ** rng.integers(-2, 3, rows)).astype(np.float32)
    weights[13::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 3)
    return bins, np.asarray(targets, np.float32), features, borders, weights


def options(objective="RMSE", **overrides):
    result = dict(iterations=5, depth=3, learning_rate=.17, l2_leaf_reg=2.5,
                  bias=.125, objective=objective, score_function="Cosine",
                  leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                  permutation_count=1, random_seed=42)
    result.update(overrides)
    return result


def compare(actual, expected):
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :depth],
                                      expected["split_features"][tree, :depth], err_msg=f"tree {tree}")
        np.testing.assert_array_equal(actual.split_bins[tree, :depth],
                                      expected["split_bins"][tree, :depth], err_msg=f"tree {tree}")
    for key in ("leaf_values", "leaf_weights", "predictions"):
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=8e-6, atol=3e-6,
                                   err_msg=key)
        assert np.isfinite(getattr(actual, key)).all()
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=8e-6, atol=3e-6)
    assert actual.stats["kernel_dispatches"] > 0
    assert actual.stats["gpu_seconds"] >= 0


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("score_function", ["Cosine", "NewtonCosine"])
@pytest.mark.parametrize("leaf_estimation_method", ["Newton", "Gradient"])
@pytest.mark.parametrize("leaf_estimation_iterations", [1, 3])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_numeric_ordered_training_matches_cuda_equations(
        objective, score_function, leaf_estimation_method,
        leaf_estimation_iterations, permutation_count):
    bins, targets, features, borders, weights = dataset(objective)
    config = options(objective, score_function=score_function,
                     leaf_estimation_method=leaf_estimation_method,
                     leaf_estimation_iterations=leaf_estimation_iterations,
                     permutation_count=permutation_count, sample_weight=weights)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_array_equal(state["descriptors"], expected["folds"])
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)
    assert state["iteration_offset"] == config["iterations"]


SCALAR_CASES = [
    ("Poisson", None, "Newton"), ("Poisson", None, "Gradient"),
    ("Huber", .8, "Newton"), ("Huber", .8, "Gradient"),
    ("Expectile", .7, "Newton"), ("Expectile", .7, "Gradient"),
    ("Lq", 1.4, "Gradient"), ("Lq", 2.7, "Newton"), ("Lq", 2.7, "Gradient"),
    ("Tweedie", 1.3, "Newton"), ("Tweedie", 1.3, "Gradient"),
    ("LogLinQuantile", .65, "Gradient"), ("Quantile", .65, "Gradient"),
    ("MAE", None, "Gradient"), ("MAPE", None, "Gradient"),
]


@pytest.mark.parametrize("objective,objective_param,leaf_estimation_method", SCALAR_CASES)
@pytest.mark.parametrize("score_function", ["Cosine", "NewtonCosine"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_additional_scalar_objectives_match_cuda_ordered_equations(
        objective, objective_param, leaf_estimation_method, score_function, permutation_count):
    bins, targets, features, borders, weights = dataset()
    if objective in ("Poisson", "Tweedie", "LogLinQuantile"):
        targets = np.exp(.5 * targets).astype(np.float32)
    # Undamped superquadratic Gradient steps on small prefixes can explode;
    # use a modest step so this test measures the port's equations, not a
    # deliberately unstable optimization trajectory spanning 18 loss digits.
    rate = .02 if objective == "Lq" and objective_param > 2 and leaf_estimation_method == "Gradient" else .07
    config = options(objective, iterations=4, learning_rate=rate, bias=0,
                     objective_param=objective_param, leaf_estimation_method=leaf_estimation_method,
                     leaf_estimation_iterations=2, score_function=score_function,
                     permutation_count=permutation_count, sample_weight=weights)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)


@pytest.mark.parametrize("objective,objective_param,method", [
    ("Huber", 0, "Newton"), ("Expectile", 0, "Newton"), ("Expectile", 1, "Gradient"),
    ("Lq", 1, "Gradient"), ("Lq", 2, "Newton"),
    ("Quantile", 0, "Gradient"), ("Quantile", 1, "Gradient"),
    ("LogLinQuantile", 0, "Gradient"), ("LogLinQuantile", 1, "Gradient"),
])
def test_scalar_parameter_boundaries_and_normalized_leaf_steps(objective, objective_param, method):
    bins, targets, features, borders, weights = dataset(rows=67)
    if objective == "LogLinQuantile":
        targets = np.exp(.5 * targets).astype(np.float32)
    config = options(objective, iterations=3, depth=2, learning_rate=.07,
                     objective_param=objective_param, leaf_estimation_method=method,
                     leaf_estimation_iterations=2, sample_weight=weights,
                     fold_size_loss_normalization=True)
    compare(_ordered.train(bins, targets, features, borders, **config),
            train_reference(bins, targets, features, borders, **config))


@pytest.mark.parametrize("objective", ["Quantile", "LogLinQuantile"])
def test_quantile_default_alpha_is_one_half(objective):
    bins, targets, features, borders, _ = dataset()
    config = options(objective, iterations=2, leaf_estimation_method="Gradient")
    default = _ordered.train(bins, targets, features, borders, **config)
    explicit = _ordered.train(bins, targets, features, borders, **dict(config, objective_param=.5))
    np.testing.assert_array_equal(default.predictions, explicit.predictions)


def test_mae_reports_full_absolute_error_but_uses_half_gradient_for_training():
    bins, targets, features, borders, weights = dataset()
    config = options(iterations=4, leaf_estimation_method="Gradient", sample_weight=weights)
    mae = _ordered.train(bins, targets, features, borders, **dict(config, objective="MAE"))
    median = _ordered.train(bins, targets, features, borders,
                            **dict(config, objective="Quantile", objective_param=.5))
    np.testing.assert_array_equal(mae.leaf_values, median.leaf_values)
    np.testing.assert_array_equal(mae.predictions, median.predictions)
    np.testing.assert_allclose(mae.loss, 2 * median.loss, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("objective,param,method", [
    ("Poisson", None, "Newton"), ("Tweedie", 1.7, "Newton"),
    ("Lq", 2.7, "Newton"), ("Quantile", .3, "Gradient"),
])
def test_extended_objective_state_resume_is_exact(objective, param, method):
    bins, targets, features, borders, weights = dataset()
    if objective in ("Poisson", "Tweedie"):
        targets = np.exp(.5 * targets).astype(np.float32)
    config = options(objective, iterations=5, learning_rate=.07, permutation_count=4,
                     sample_weight=weights, objective_param=param, leaf_estimation_method=method)
    expected = _ordered.train(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **dict(config, iterations=2)) as session:
        session.step()
        session.step()
        state = session.state()
    actual = _ordered.train(bins, targets, features, borders,
                            **dict(config, iterations=3, initial_state=state))
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    np.testing.assert_array_equal(actual.leaf_values, expected.leaf_values[2:])


ORDERED_SAMPLERS = [
    pytest.param({"bootstrap_type": "Bayesian", "bagging_temperature": 1.5}, id="Bayesian"),
    pytest.param({"bootstrap_type": "Bernoulli", "subsample": .55}, id="Bernoulli"),
    pytest.param({"bootstrap_type": "Poisson", "subsample": .65}, id="Poisson"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .55}, id="MVS-auto"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .55, "mvs_reg": .7}, id="MVS-fixed"),
]


@pytest.mark.parametrize("sampling", ORDERED_SAMPLERS)
@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("observations", ["TestOnly", "LearnAndTest"])
def test_occurrence_sampling_and_score_noise_repeat_and_resume_exactly(sampling, permutation_count, observations):
    bins, targets, features, borders, weights = dataset()
    config = options(iterations=5, permutation_count=permutation_count, sample_weight=weights,
                     observations_to_bootstrap=observations, random_strength=.8, **sampling)
    whole = _ordered.train(bins, targets, features, borders, **config)
    repeated = _ordered.train(bins, targets, features, borders, **config)
    np.testing.assert_array_equal(whole.predictions, repeated.predictions)
    np.testing.assert_array_equal(whole.leaf_values, repeated.leaf_values)
    with _ordered.Session(bins, targets, features, borders, **dict(config, iterations=2)) as session:
        session.step()
        session.step()
        state = session.state()
    resumed = _ordered.train(bins, targets, features, borders,
                             **dict(config, iterations=3, initial_state=state))
    np.testing.assert_array_equal(resumed.predictions, whole.predictions)
    np.testing.assert_array_equal(resumed.leaf_values, whole.leaf_values[2:])
    np.testing.assert_array_equal(resumed.loss, whole.loss[2:])
    assert resumed.stats["bootstrap_state"]["iteration_offset"] == 5
    if sampling["bootstrap_type"] == "MVS" and "mvs_reg" not in sampling:
        assert state["mvs_lambda"] is not None and state["mvs_lambda"] >= 0
        assert resumed.stats["bootstrap_state"]["mvs_lambda"] == whole.stats["bootstrap_state"]["mvs_lambda"]


@pytest.mark.parametrize("sampling", ORDERED_SAMPLERS)
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_bootstrap_and_noise_do_not_enter_fixed_structure_leaf_estimation(sampling, permutation_count):
    bins, targets, _, _, weights = dataset()
    empty = np.empty(0, dtype=np.uint32)
    config = options(iterations=4, depth=0, sample_weight=weights, permutation_count=permutation_count)
    states, results = [], []
    for modifiers in ({}, dict(sampling, random_strength=3, observations_to_bootstrap="LearnAndTest")):
        with _ordered.Session(bins, targets, empty, empty, **dict(config, **modifiers)) as session:
            for _ in range(config["iterations"]):
                session.step()
            results.append(session.result())
            states.append(session.state())
    np.testing.assert_array_equal(results[0].leaf_values, results[1].leaf_values)
    np.testing.assert_array_equal(results[0].leaf_weights, results[1].leaf_weights)
    np.testing.assert_array_equal(states[0]["cursors"], states[1]["cursors"])


@pytest.mark.parametrize("sampling", ORDERED_SAMPLERS)
def test_sampling_prefixes_changes_the_structure_search(sampling):
    bins, targets, features, borders, weights = dataset(rows=257, seed=927)
    config = options(iterations=6, depth=3, sample_weight=weights, **sampling)
    tail_only = _ordered.train(bins, targets, features, borders,
                               **dict(config, observations_to_bootstrap="TestOnly"))
    both = _ordered.train(bins, targets, features, borders,
                          **dict(config, observations_to_bootstrap="LearnAndTest"))
    assert (not np.array_equal(tail_only.depths, both.depths)
            or not np.array_equal(tail_only.split_features, both.split_features)
            or not np.array_equal(tail_only.split_bins, both.split_bins))


@pytest.mark.parametrize("sampling", [
    {"bootstrap_type": "Bayesian", "bagging_temperature": 0},
    {"bootstrap_type": "Bernoulli", "subsample": 1},
    {"bootstrap_type": "MVS", "subsample": 1},
])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_sampling_identity_boundaries_match_no_bootstrap(sampling, permutation_count):
    bins, targets, features, borders, weights = dataset()
    config = options(sample_weight=weights, permutation_count=permutation_count)
    plain = _ordered.train(bins, targets, features, borders, **config)
    identity = _ordered.train(bins, targets, features, borders, **dict(config, **sampling))
    np.testing.assert_array_equal(plain.predictions, identity.predictions)
    np.testing.assert_array_equal(plain.leaf_values, identity.leaf_values)


@pytest.mark.parametrize("iteration_offset", [0, 7])
def test_score_noise_uses_quality_only_variance_and_fresh_feature_streams_per_depth(iteration_offset):
    from cuda_ordered_reference import _score_candidate
    from test_score_noise import _normal

    bins, targets, features, borders, weights = dataset(rows=127, seed=994)
    targets[:2] = [80, -120]  # Large prefix gradients must not enter noise variance.
    strength, seed = 5, 182
    config = options(iterations=1, depth=3, bias=0, random_seed=seed,
                     sample_weight=weights, random_strength=strength, iteration_offset=iteration_offset)
    actual = _ordered.train(bins, targets, features, borders, **config)
    repeated = _ordered.train(bins, targets, features, borders, **config)
    np.testing.assert_array_equal(actual.predictions, repeated.predictions)
    reference_options = {key: value for key, value in config.items() if key != "random_strength"}
    initial = train_reference(bins, targets, features, borders, **dict(reference_options, iterations=0))
    folds, maps = initial["folds"][:-1], initial["permutations"]
    gradients = (weights * targets).astype(np.float32)
    weak = np.where(np.abs(gradients) < np.float32(1e-15), 0,
                    gradients / (weights + np.float32(1e-15)))
    quality_rows = np.arange(int(folds[0, 0]), targets.size)
    variance = np.dot(weights[quality_rows].astype(np.float64), weak[quality_rows].astype(np.float64) ** 2) / len(quality_rows)
    scale = np.float32(strength * math.sqrt(variance) /
                       (1 + math.exp(iteration_offset * float(np.float32(config["learning_rate"])) - math.log(targets.size))))
    derivatives = [(gradients[:int(end)].astype(np.float64), weights[:int(end)].astype(np.float64))
                   for _, end, _, _ in folds]
    leaves, chosen, expected_features, expected_borders = np.zeros(targets.size, np.int64), set(), [], []
    streams = []
    for level in range(config["depth"]):
        noise = np.asarray([np.float32(_normal(feature, seed=seed, iteration=iteration_offset, stream=level + 1)) * scale
                            for feature in range(bins.shape[0])], np.float32)
        streams.append(noise)
        scores = [np.float32(_score_candidate(bins, int(feature), int(border), leaves,
                    derivatives, folds, maps, config["l2_leaf_reg"], False, 1 << level)) + noise[feature]
                  for feature, border in zip(features, borders)]
        winner = int(np.argmin(scores))
        if winner in chosen:
            break
        chosen.add(winner)
        feature, border = int(features[winner]), int(borders[winner])
        expected_features.append(feature)
        expected_borders.append(border)
        leaves |= (bins[feature] > border).astype(np.int64) << level
    assert len(streams) >= 2
    assert not np.array_equal(streams[0], streams[1])
    assert int(actual.depths[0]) == len(chosen)
    np.testing.assert_array_equal(actual.split_features[0, :len(chosen)], expected_features)
    np.testing.assert_array_equal(actual.split_bins[0, :len(chosen)], expected_borders)


@pytest.mark.parametrize("score_function", ["Cosine", "NewtonCosine"])
@pytest.mark.parametrize("mvs_reg", [None, .7])
@pytest.mark.parametrize("observations", ["TestOnly", "LearnAndTest"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_first_mvs_tree_samples_packed_structure_denominators_from_cuda_source(
        score_function, mvs_reg, observations, permutation_count):
    """Dynamic CUDA calls BootstrappedWeights(&target.Weights), not gradients.

    Sources: methods/oblivious_tree_structure_searcher.cpp:59-73 and
    gpu_data/bootstrap.h:59-64. Initial automatic lambda squares packed mean
    absolute W. The sorted-prefix threshold oracle is independent of Metal's
    bisection; RNG helpers reproduce the explicitly documented Metal row stream.
    """
    from catboost_metal._ordered_rng import OrderedSelectionRng
    from cuda_ordered_reference import _derivatives, _score_candidate
    from test_bootstrap import _mvs_threshold, uniforms

    bins, targets, features, borders, weights = dataset("CrossEntropy", rows=127, seed=4427)
    baseline = np.random.default_rng(917).normal(scale=.7, size=targets.size).astype(np.float32)
    config = options("CrossEntropy", iterations=1, depth=3, score_function=score_function,
        initial_predictions=baseline, sample_weight=weights, permutation_count=permutation_count,
        iteration_offset=7, random_seed=817, bootstrap_type="MVS", subsample=.55,
        mvs_reg=mvs_reg, observations_to_bootstrap=observations)
    actual = _ordered.train(bins, targets, features, borders, **config)
    oracle_options = {key: value for key, value in config.items()
                      if key not in ("bootstrap_type", "subsample", "mvs_reg", "observations_to_bootstrap")}
    initial = train_reference(bins, targets, features, borders, **dict(oracle_options, iterations=0))
    selected = OrderedSelectionRng(config["random_seed"], permutation_count,
                                  iteration_offset=config["iteration_offset"]).select()
    folds = initial["folds"][:-1][initial["folds"][:-1, 3] == selected]
    maps, derivatives, packed_denominators = initial["permutations"], [], []
    for _, end, _, permutation in folds:
        order = maps[permutation, :end]
        gradient, curvature = _derivatives(targets[order].astype(np.float64), baseline[order].astype(np.float64),
                                           weights[order].astype(np.float64), "CrossEntropy")
        denominator = curvature if score_function == "NewtonCosine" else weights[order]
        derivatives.append((gradient, denominator))
        packed_denominators.extend(denominator)
    packed = np.asarray(packed_denominators, np.float32).astype(np.float64)
    regularization = float(np.float32(np.abs(packed).mean() ** 2 if mvs_reg is None else mvs_reg))
    magnitude = np.sqrt(packed ** 2 + regularization)
    fraction = float(np.float32(config["subsample"]))
    thresholds = np.asarray([_mvs_threshold(magnitude[start:start + 8192], fraction)
                             for start in range(0, len(packed), 8192)])
    threshold = np.repeat(thresholds, 8192)[:len(packed)]
    probability = np.minimum(magnitude / threshold, 1)
    sampled = np.zeros(len(packed))
    keep = ((probability > np.finfo(np.float32).eps)
            & (uniforms(len(packed), seed=config["random_seed"], iteration=config["iteration_offset"]) < probability))
    sampled[keep] = 1 / probability[keep]
    position, sampled_derivatives = 0, []
    for (prefix, end, _, _), (gradient, denominator) in zip(folds, derivatives):
        factors = sampled[position:position + end].copy()
        if observations == "TestOnly":
            factors[:prefix] = 1
        sampled_derivatives.append((gradient * factors, denominator * factors))
        position += int(end)
    leaves, chosen, expected_features, expected_borders = np.zeros(targets.size, np.int64), set(), [], []
    for level in range(config["depth"]):
        scores = [_score_candidate(bins, int(feature), int(border), leaves, sampled_derivatives,
                  folds, maps, config["l2_leaf_reg"], False, 1 << level)
                  for feature, border in zip(features, borders)]
        winner = int(np.argmin(scores))
        if winner in chosen or scores[winner] >= np.finfo(np.float32).max:
            break
        chosen.add(winner)
        feature, border = int(features[winner]), int(borders[winner])
        expected_features.append(feature)
        expected_borders.append(border)
        leaves |= (bins[feature] > border).astype(np.int64) << level
    assert int(actual.depths[0]) == len(chosen)
    np.testing.assert_array_equal(actual.split_features[0, :len(chosen)], expected_features)
    np.testing.assert_array_equal(actual.split_bins[0, :len(chosen)], expected_borders)
    if mvs_reg is None:
        # Subsequent automatic lambda uses only the shrunk exported model.
        full_values = actual.leaf_values[0, :1 << int(actual.depths[0])]
        expected_next = np.float32(np.abs(full_values.astype(np.float64)).mean() ** 2)
        assert actual.stats["bootstrap_state"]["mvs_lambda"] == pytest.approx(expected_next, rel=2e-6, abs=1e-10)


EXACT_CASES = [("Quantile", alpha) for alpha in (0, .2, .5, 1)] + [("MAE", None), ("MAPE", None)]


@pytest.mark.parametrize("objective,alpha", EXACT_CASES)
@pytest.mark.parametrize("permutation_count", [1, 2, 4])
def test_exact_leaf_solver_uses_each_prefix_cursor_and_full_estimation_task(objective, alpha, permutation_count):
    bins, targets, features, borders, weights = dataset(rows=33, seed=5748)
    targets *= 3
    baseline = np.random.default_rng(445).normal(scale=2, size=targets.size).astype(np.float32)
    targets[0], baseline[0], weights[0] = -100, 20, 0
    targets[2], baseline[2], weights[2] = 100, -20, 0
    # Alpha=1 produces an algebraic score tie at level2 on this fixture;
    # endpoint leaf selection is checked at the two unambiguous levels.
    # The separate tie test below verifies the optional third split's values.
    depth = 2 if objective == "Quantile" and alpha == 1 else 3
    config = options(objective, iterations=4, depth=depth, learning_rate=.17, objective_param=alpha,
                     leaf_estimation_method="Exact", leaf_estimation_iterations=3,
                     initial_predictions=baseline, sample_weight=weights, permutation_count=permutation_count)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)


def test_exact_alpha_one_tied_split_still_estimates_each_task_correctly():
    """Different reduction precision may stop or add an equally scoring split.

    After (feature1,border4),(feature0,border1), candidates indices0,1,11
    all score -3.827738851882299 mathematically. Double reduction differs by
    4.4e-16 here; float32 selects index0 while the double oracle selects the
    already-used index11. Compare score optimality and actual task equations.
    """
    from cuda_ordered_reference import _derivatives, _score_candidate, exact_leaf_value
    bins, targets, features, borders, weights = dataset(rows=33, seed=5748)
    targets *= 3
    baseline = np.random.default_rng(445).normal(scale=2, size=targets.size).astype(np.float32)
    targets[0], baseline[0], weights[0] = -100, 20, 0
    targets[2], baseline[2], weights[2] = 100, -20, 0
    config = options("Quantile", iterations=1, depth=3, objective_param=1,
                     leaf_estimation_method="Exact", initial_predictions=baseline, sample_weight=weights)
    initial = train_reference(bins, targets, features, borders, **dict(config, iterations=0))
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        session.step()
        actual, state = session.result(), session.state()
    np.testing.assert_array_equal(actual.split_features[0, :2], [1, 0])
    np.testing.assert_array_equal(actual.split_bins[0, :2], [4, 1])
    folds, maps, derivative_pairs = initial["folds"][:-1], initial["permutations"], []
    for _, end, _, _ in folds:
        gradient, _ = _derivatives(targets[:end].astype(float), baseline[:end].astype(float),
                                   weights[:end].astype(float), "Quantile", 1)
        derivative_pairs.append((gradient, weights[:end]))
    leaves = (bins[1] > 4).astype(np.int64) | ((bins[0] > 1).astype(np.int64) << 1)
    scores = np.asarray([_score_candidate(bins, int(feature), int(border), leaves,
                         derivative_pairs, folds, maps, config["l2_leaf_reg"], False, 4)
                         for feature, border in zip(features, borders)])
    np.testing.assert_allclose(scores[[0, 1, 11]], scores.min(), rtol=0, atol=1e-14)
    depth = int(actual.depths[0])
    assert depth in (2, 3)
    if depth == 3:
        feature, border = int(actual.split_features[0, 2]), int(actual.split_bins[0, 2])
        chosen = np.flatnonzero((features == feature) & (borders == border))[0]
        assert scores[chosen] == pytest.approx(scores.min(), rel=3e-7)
        leaves |= (bins[feature] > border).astype(np.int64) << 2
    for task, (prefix, end, offset, _) in enumerate(state["descriptors"]):
        values = np.zeros(1 << depth)
        for leaf in range(1 << depth):
            selected = leaves[:prefix] == leaf
            values[leaf] = float(np.float32(config["learning_rate"])) * exact_leaf_value(
                targets[:prefix][selected], baseline[:prefix][selected], weights[:prefix][selected], "Quantile", 1)
        np.testing.assert_allclose(state["cursors"][offset:offset + end],
                                   baseline[:end] + values[leaves[:end]], rtol=8e-6, atol=3e-6)
        if task + 1 == len(state["descriptors"]):
            np.testing.assert_allclose(actual.leaf_values[0, :1 << depth], values, rtol=8e-6, atol=3e-6)


@pytest.mark.parametrize("objective,alpha", [("Quantile", 0), ("Quantile", 1), ("MAE", None), ("MAPE", None)])
def test_exact_constant_tree_ignores_ridge_step_count_and_zero_mass_prefixes(objective, alpha):
    bins, targets, _, _, weights = dataset(rows=33)
    weights[:2] = 0
    empty = np.empty(0, np.uint32)
    config = options(objective, iterations=3, depth=0, objective_param=alpha,
                     sample_weight=weights, leaf_estimation_method="Exact")
    with _ordered.Session(bins, targets, empty, empty, **dict(config, l2_leaf_reg=0, leaf_estimation_iterations=1)) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    changed = _ordered.train(bins, targets, empty, empty,
        **dict(config, l2_leaf_reg=1e5, leaf_estimation_iterations=5, fold_size_loss_normalization=True))
    np.testing.assert_array_equal(actual.predictions, changed.predictions)
    np.testing.assert_array_equal(actual.leaf_values, changed.leaf_values)
    np.testing.assert_array_equal(state["cursors"][:5], np.full(5, np.float32(config["bias"])))


def test_exact_mape_weights_use_original_target_with_nonzero_baseline():
    from cuda_ordered_reference import exact_leaf_value
    bins = np.zeros((1, 9), dtype=np.uint8)
    targets = np.asarray([1, 3, 9, 10, 12, 14, 16, 18, 20], np.float32)
    baseline = np.asarray([20, 20, -20, -20, -20, -20, -20, -20, -20], np.float32)
    weights = np.ones(9, np.float32)
    empty = np.empty(0, np.uint32)
    config = options("MAPE", iterations=1, depth=0, learning_rate=1, sample_weight=weights,
                     initial_predictions=baseline, leaf_estimation_method="Exact")
    actual = _ordered.train(bins, targets, empty, empty, **config)
    correct = exact_leaf_value(targets, baseline, weights, "MAPE")
    # Simulate the inspected old CUDA mistake only to prove fixture separation.
    residual_based = exact_leaf_value(targets - baseline, np.zeros(9), weights, "MAPE")
    assert correct != residual_based
    np.testing.assert_array_equal(actual.leaf_values[0], [correct])
    np.testing.assert_array_equal(actual.predictions, baseline + correct)


@pytest.mark.parametrize("objective,alpha", [("Quantile", .2), ("MAE", None), ("MAPE", None)])
def test_exact_sampling_and_all_fold_state_resume_match_uninterrupted_training(objective, alpha):
    bins, targets, features, borders, weights = dataset(rows=33)
    config = options(objective, iterations=5, permutation_count=4, sample_weight=weights,
                     objective_param=alpha, leaf_estimation_method="Exact",
                     bootstrap_type="MVS", subsample=.6, random_strength=.7)
    expected = _ordered.train(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **dict(config, iterations=2)) as session:
        session.step()
        session.step()
        state = session.state()
    actual = _ordered.train(bins, targets, features, borders,
                            **dict(config, iterations=3, initial_state=state))
    np.testing.assert_array_equal(actual.predictions, expected.predictions)
    np.testing.assert_array_equal(actual.leaf_values, expected.leaf_values[2:])


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("score_function", ["Cosine", "NewtonCosine"])
@pytest.mark.parametrize("leaf_estimation_method", ["Newton", "Gradient"])
def test_normalized_folds_preserve_cuda_score_and_leaf_ridge_scales(
        objective, score_function, leaf_estimation_method):
    bins, targets, features, borders, weights = dataset(objective)
    config = options(objective, iterations=3, score_function=score_function,
                     leaf_estimation_method=leaf_estimation_method,
                     leaf_estimation_iterations=3, sample_weight=weights,
                     fold_size_loss_normalization=True, permutation_count=4,
                     fold_len_multiplier=1.7)
    expected = train_reference(bins, targets, features, borders, **config)
    actual = _ordered.train(bins, targets, features, borders, **config)
    compare(actual, expected)


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_state_resume_is_exact_and_model_predictions_alone_are_insufficient(objective, permutation_count):
    bins, targets, features, borders, weights = dataset(objective)
    config = options(objective, iterations=7, permutation_count=permutation_count,
                     leaf_estimation_iterations=3, sample_weight=weights)
    whole = _ordered.train(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **dict(config, iterations=3)) as session:
        for _ in range(3):
            session.step()
        first, state = session.result(), session.state()
    with _ordered.Session(bins, targets, features, borders,
                          **dict(config, iterations=4, initial_state=state)) as session:
        for _ in range(4):
            session.step()
        second = session.result()
        assert session.state()["iteration_offset"] == 7
    for key in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(getattr(whole, key),
                                      np.concatenate([getattr(first, key), getattr(second, key)]))
    np.testing.assert_array_equal(whole.predictions, second.predictions)
    np.testing.assert_array_equal(whole.loss, np.concatenate([first.loss, second.loss[1:]]))
    model_only = _ordered.train(bins, targets, features, borders,
        **dict(config, iterations=4, initial_predictions=first.predictions, iteration_offset=3))
    assert not np.array_equal(second.predictions, model_only.predictions)


def test_exported_full_model_matches_independent_metal_inference():
    from catboost_metal._inference import predict_bins
    bins, targets, features, borders, weights = dataset()
    config = options(permutation_count=4, sample_weight=weights, leaf_estimation_iterations=3)
    trained = _ordered.train(bins, targets, features, borders, **config)
    prediction = predict_bins(bins, trained.depths, trained.split_features,
                              trained.split_bins, trained.leaf_values, bias=config["bias"])
    np.testing.assert_allclose(prediction, trained.predictions, rtol=3e-6, atol=1e-6)


def test_supplied_permutations_and_initial_predictions_preserve_original_row_mapping():
    bins, targets, features, borders, weights = dataset(rows=67)
    rng = np.random.default_rng(851)
    permutations = np.stack([rng.permutation(targets.size) for _ in range(4)]).astype(np.uint32)
    baseline = rng.normal(scale=.2, size=targets.size).astype(np.float32)
    config = options(iterations=4, depth=2, sample_weight=weights, initial_predictions=baseline,
                     permutation_count=4, permutations=permutations)
    compare(_ordered.train(bins, targets, features, borders, **config),
            train_reference(bins, targets, features, borders, **config))


def test_later_quality_targets_do_not_change_earliest_fold_cursor_with_fixed_structure():
    bins = np.zeros((1, 33), dtype=np.uint8)
    empty = np.empty(0, dtype=np.uint32)
    targets = np.arange(33, dtype=np.float32)
    config = options(iterations=4, depth=0, learning_rate=.25, bias=0)
    states = []
    for changed in (False, True):
        y = targets.copy()
        if changed:
            y[2:] += 100
        with _ordered.Session(bins, y, empty, empty, **config) as session:
            for _ in range(config["iterations"]):
                session.step()
            states.append(session.state())
    first = states[0]["descriptors"][0]
    _, end, offset, _ = map(int, first)
    np.testing.assert_array_equal(states[0]["cursors"][offset:offset + end],
                                  states[1]["cursors"][offset:offset + end])
    assert not np.array_equal(states[0]["cursors"], states[1]["cursors"])


def test_zero_weight_prefix_and_empty_leaves_do_not_generate_nonfinite_values():
    bins, targets, features, borders, weights = dataset()
    weights[:5] = 0
    config = options(sample_weight=weights, l2_leaf_reg=0, leaf_estimation_iterations=3,
                     fold_size_loss_normalization=True)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_array_equal(state["cursors"][:5], np.full(5, np.float32(config["bias"])))
    assert np.isfinite(state["cursors"]).all()


def test_repeat_training_is_deterministic_and_closed_sessions_reject_calls():
    bins, targets, features, borders, _ = dataset()
    config = options(iterations=2, permutation_count=4)
    first = _ordered.train(bins, targets, features, borders, **config)
    session = _ordered.Session(bins, targets, features, borders, **config)
    initial = session.result()
    assert initial.completed_iterations == 0
    assert len(initial.loss) == 1
    np.testing.assert_array_equal(initial.predictions, np.full(targets.size, np.float32(config["bias"])))
    step0, step1 = session.step(), session.step()
    assert step0.completed_iterations == 1 and not step0.finished
    assert step1.completed_iterations == 2 and step1.finished
    second = session.result()
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.leaf_values, second.leaf_values)
    with pytest.raises(RuntimeError, match="remaining|finished|complete"):
        session.step()
    session.close()
    session.close()
    for call in (session.step, session.result, session.state):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@pytest.mark.parametrize("mutation", ["targets", "bins", "weights", "ridge", "cursors", "descriptors", "fingerprint"])
def test_invalid_restored_state_is_rejected_and_next_training_still_works(mutation):
    bins, targets, features, borders, weights = dataset()
    config = options(iterations=2, sample_weight=weights)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        session.step()
        state = session.state()
    state, bins, targets, weights = copy.deepcopy(state), bins.copy(), targets.copy(), weights.copy()
    config["sample_weight"] = weights
    if mutation == "targets":
        targets[0] += .5
    elif mutation == "bins":
        bins[0, 0] = (bins[0, 0] + 1) % 8
    elif mutation == "weights":
        weights[0] += .5
    elif mutation == "ridge":
        config["l2_leaf_reg"] += 1
    elif mutation == "cursors":
        state["cursors"][0] = np.nan
    elif mutation == "descriptors":
        state["descriptors"][0, 0] += 1
    else:
        state["fingerprint"] = "invalid"
    with pytest.raises((ValueError, RuntimeError)):
        _ordered.Session(bins, targets, features, borders, **dict(config, initial_state=state))
    recovered = _ordered.train(bins, targets, features, borders, **config)
    assert np.isfinite(recovered.predictions).all()


@pytest.mark.parametrize("override", [
    {"score_function": "L2"}, {"score_function": "NewtonL2"},
    {"bootstrap_type": "Unknown"}, {"leaf_estimation_backtracking": "Unknown"},
    {"leaf_estimation_method": "Exact"}, {"objective": "MultiClass"},
    {"permutation_count": 0}, {"fold_len_multiplier": 1}, {"min_fold_size": 0}, {"random_strength": -1},
])
def test_unsupported_or_invalid_options_fail_explicitly(override):
    bins, targets, features, borders, _ = dataset()
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _ordered.Session(bins, targets, features, borders, **options(**override))


@pytest.mark.parametrize("override", [
    {"objective": "Huber"}, {"objective": "Huber", "objective_param": -.1},
    {"objective": "Expectile"}, {"objective": "Expectile", "objective_param": 1.1},
    {"objective": "Lq"}, {"objective": "Lq", "objective_param": .9},
    {"objective": "Lq", "objective_param": 1.5, "leaf_estimation_method": "Newton"},
    {"objective": "Tweedie"}, {"objective": "Tweedie", "objective_param": 1},
    {"objective": "Tweedie", "objective_param": 2},
    {"objective": "Quantile", "leaf_estimation_method": "Newton"},
    {"objective": "MAE", "leaf_estimation_method": "Newton"},
    {"objective": "MAPE", "leaf_estimation_method": "Newton"},
    {"objective": "LogLinQuantile", "leaf_estimation_method": "Newton"},
    {"objective": "Quantile", "objective_param": -.1, "leaf_estimation_method": "Gradient"},
])
def test_invalid_scalar_objective_settings_are_rejected(override):
    bins, targets, features, borders, _ = dataset()
    with pytest.raises((ValueError, RuntimeError)):
        _ordered.Session(bins, targets, features, borders, **options(**override))
