"""Connected supplied-pair PairLogit training; independent fixed-tree algebra."""

import platform

import numpy as np
import pytest

from catboost_metal import _native
from cuda_reference import _score_children
from cuda_scalar_reference import auxiliary_score_children
from test_bootstrap import uniforms
from test_pairwise_kernels import reference


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Actual Apple Silicon Metal GPU required",
)


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker

    def forbidden(*args, **kwargs):
        pytest.fail("PairLogit validation must not fit a CPU CatBoost model")

    for kind in (CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker):
        monkeypatch.setattr(kind, "fit", forbidden)


def problem(**changes):
    rng = np.random.default_rng(72481)
    rows = 261
    bins = rng.integers(0, 4, (3, rows), dtype=np.uint8)
    offsets = np.array([0, 61, 131, rows], np.uint32)
    winners, losers = [], []
    signal = 1.7 * (bins[0] > 1) - .6 * (bins[1] > 0) + rng.normal(0, .3, rows)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        a = rng.integers(begin, end, 403)
        b = begin + (a - begin + rng.integers(1, end - begin, 403)) % (end - begin)
        prefer_a = signal[a] > signal[b]
        winners.extend(np.where(prefer_a, a, b))
        losers.extend(np.where(prefer_a, b, a))
    weights = rng.lognormal(0, .7, len(winners)).astype(np.float32)
    weights[::19] = 0
    return dict(
        bins=bins, targets=np.zeros(rows, np.float32),
        candidate_features=np.repeat(np.arange(3, dtype=np.uint32), 3),
        candidate_bins=np.tile(np.arange(3, dtype=np.uint32), 3),
        objective="PairLogit", pair_winners=np.asarray(winners, np.uint32),
        pair_losers=np.asarray(losers, np.uint32), pair_weights=weights, group_offsets=offsets,
        iterations=3, depth=2, learning_rate=.2, l2_leaf_reg=2, bias=0,
        score_function="Cosine", leaf_estimation_method="Newton",
        leaf_estimation_iterations=4, leaf_estimation_backtracking="No",
        initial_predictions=rng.normal(0, .1, rows).astype(np.float32),
    ) | changes


def leaf_ids(step, bins):
    ids = np.zeros(bins.shape[1], np.uint32)
    for level in range(step.depth):
        feature, border = step.split_features[level], step.split_bins[level]
        right = bins[feature] == border if step.split_types[level] else bins[feature] > border
        ids |= right.astype(np.uint32) << level
    return ids


def training_terms(cursor, winners, losers, weights):
    """Independent float32 pair equations with a double loss/row accumulator.

    Near sigmoid saturation, rounding the probability only after a float64
    sigmoid changes a tiny Hessian materially. Float32 exp/add/div and edge
    products are required before the compensated row projection. The loss
    remains a separate double-precision softplus oracle.
    """
    terms = reference(cursor, winners, losers, weights)
    cursor, weights = np.asarray(cursor, np.float32), np.asarray(weights, np.float32)
    difference = cursor[winners] - cursor[losers]
    exponential = np.exp(-np.abs(difference))
    denominator = np.float32(1) + exponential
    probability = np.where(difference >= 0, np.float32(1) / denominator, exponential / denominator)
    probability = np.maximum(probability, np.float32(1e-40))
    edge_g = weights * (np.float32(1) - probability)
    edge_h = (weights * probability) * (np.float32(1) - probability)
    gradient, hessian = np.zeros(cursor.size), np.zeros(cursor.size)
    np.add.at(gradient, winners, edge_g); np.add.at(gradient, losers, -edge_g)
    np.add.at(hessian, winners, edge_h); np.add.at(hessian, losers, edge_h)
    terms["gradients"], terms["curvature"] = gradient, hessian
    return terms


def leaf_reference(args, cursor, ids, count):
    """CUDA's diagonal leaf walker, recomputing all supplied edges per trial."""
    weights = args["pair_weights"]
    if weights is None:
        weights = np.ones(len(args["pair_winners"]), np.float32)
    winners, losers = args["pair_winners"], args["pair_losers"]
    incident = reference(cursor, winners, losers, weights)["incident_weights"].astype(np.float32)
    masses = np.bincount(ids, weights=incident.astype(np.float64), minlength=count)
    l2 = float(np.float32(args["l2_leaf_reg"])) or float(np.float32(1e-20))
    point = np.zeros(count, np.float32)
    trace = []

    def project(values):
        raw = (cursor + values[ids]).astype(np.float32)
        terms = training_terms(raw, winners, losers, weights)
        gradient = terms["gradients"].astype(np.float32).astype(np.float64)
        hessian = terms["curvature"].astype(np.float32).astype(np.float64)
        return (-terms["objective"][0], np.bincount(ids, weights=gradient, minlength=count),
                np.bincount(ids, weights=hessian, minlength=count))

    def direction(g, h):
        diagonal = (masses if args["leaf_estimation_method"] == "Gradient" else h) + l2
        return np.divide(g, diagonal + 1e-20, out=np.zeros(count), where=masses >= 1e-20).astype(np.float32)

    def candidate(delta, step):
        result = (point.astype(np.float64) + step * delta.astype(np.float64)).astype(np.float32)
        result[masses < 1e-20] = 0
        return result

    iterations = args["leaf_estimation_iterations"]
    mode = args["leaf_estimation_backtracking"]
    value, g, h = project(point)
    if mode == "No" or iterations == 1:
        for _ in range(iterations):
            point = candidate(direction(g, h), 1)
            value, g, h = project(point)
            trace.append((1., True))
    else:
        attempts, updated = 0, False
        while attempts < iterations:
            delta = direction(g, h)
            dot = np.dot(g, delta.astype(np.float64))
            step = 1.
            while attempts < iterations or (not updated and attempts < 100):
                trial = candidate(delta, step)
                next_value, next_g, next_h = project(trial)
                threshold = value + (1e-5 * step * dot if mode == "Armijo" else 0)
                accept = np.isfinite(next_value) and next_value >= threshold
                attempts += 1
                trace.append((step, bool(accept)))
                if accept:
                    point, value, g, h = trial, next_value, next_g, next_h
                    updated = True
                    break
                step *= .5
    # CUDA NeedZeroAverage uses all solved leaves, including empty ones.
    point = (point - np.float32(point.mean(dtype=np.float64))).astype(np.float32)
    return (point * np.float32(args["learning_rate"])).astype(np.float32), masses, trace


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
def test_every_tree_matches_recomputed_edge_leaf_oracle(method, score, mode):
    args = problem(leaf_estimation_method=method, score_function=score, leaf_estimation_backtracking=mode)
    cursor = args["initial_predictions"].copy()
    with _native.Session(**args) as session:
        initial_terms = reference(cursor, args["pair_winners"], args["pair_losers"], args["pair_weights"])
        assert session.result().loss[0] == pytest.approx(initial_terms["objective"][0] / initial_terms["objective"][1], rel=3e-6)
        for _ in range(args["iterations"]):
            step = session.step()
            ids = leaf_ids(step, args["bins"])
            values, masses, _ = leaf_reference(args, cursor, ids, 1 << step.depth)
            np.testing.assert_allclose(step.leaf_values, values, rtol=4e-4, atol=2e-5)
            np.testing.assert_allclose(step.leaf_weights, masses, rtol=3e-6, atol=3e-4)
            np.testing.assert_allclose(step.leaf_values.mean(dtype=np.float64), 0, atol=2e-8)
            cursor = (cursor + step.leaf_values[ids]).astype(np.float32)
            np.testing.assert_array_equal(session.predictions(), cursor)
            terms = reference(cursor, args["pair_winners"], args["pair_losers"], args["pair_weights"])
            assert step.loss == pytest.approx(terms["objective"][0] / terms["objective"][1], rel=3e-6)
            assert step.stats["kernel_dispatches"] > 0


BOOTSTRAPS = [dict(bootstrap_type="No"), dict(bootstrap_type="Bayesian", bagging_temperature=1.4),
              dict(bootstrap_type="Bernoulli", subsample=.55), dict(bootstrap_type="Poisson", subsample=.55),
              dict(bootstrap_type="MVS", subsample=.55)]


def structure_reference(args, factors=None):
    terms = training_terms(args["initial_predictions"], args["pair_winners"], args["pair_losers"], args["pair_weights"])
    gradients = terms["gradients"].astype(np.float32)
    weights = terms["curvature" if args["score_function"].startswith("Newton") else "incident_weights"].astype(np.float32)
    if factors is not None:
        gradients, weights = gradients * factors, weights * factors
    gradients, weights = gradients.astype(np.float64), weights.astype(np.float64)
    ids, chosen = np.zeros(args["bins"].shape[1], np.uint32), []
    for level in range(args["depth"]):
        scores = []
        for feature, border in zip(args["candidate_features"], args["candidate_bins"]):
            trial = ids | (args["bins"][feature] > border).astype(np.uint32) << level
            sums = np.bincount(trial, weights=gradients, minlength=2 << level)
            masses = np.bincount(trial, weights=weights, minlength=2 << level)
            family = args["score_function"].removeprefix("Newton")
            scores.append(auxiliary_score_children(sums, masses, family) if family in ("SolarL2", "LOOL2")
                          else _score_children(sums, masses, args["l2_leaf_reg"], family))
        winner = int(np.argmin(scores))
        split = int(args["candidate_features"][winner]), int(args["candidate_bins"][winner])
        if split in chosen:
            break
        chosen.append(split)
        ids |= (args["bins"][split[0]] > split[1]).astype(np.uint32) << level
    return chosen


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"])
@pytest.mark.parametrize("bootstrap", ["No", "Bernoulli", "Bayesian"])
def test_structure_scores_use_completed_incident_derivatives_before_row_sampling(score, bootstrap):
    args = problem(iterations=1, score_function=score, bootstrap_type=bootstrap,
                   random_seed=79342, iteration_offset=9)
    factors = None
    if bootstrap != "No":
        draws = uniforms(args["bins"].shape[1], seed=args["random_seed"], iteration=args["iteration_offset"])
        if bootstrap == "Bernoulli":
            args["subsample"] = .55
            factors = (draws < np.float32(.55)).astype(np.float32)
        else:
            args["bagging_temperature"] = 1.4
            factors = np.power(-np.log(draws.astype(np.float64) + 1e-20), np.float32(1.4)).astype(np.float32)
    expected = structure_reference(args, factors)
    with _native.Session(**args) as session:
        step = session.step()
    assert list(zip(step.split_features, step.split_bins)) == expected


@pytest.mark.parametrize("sampling", BOOTSTRAPS)
def test_structure_bootstrap_leaves_use_original_incident_mass(sampling):
    args = problem(iterations=1, random_seed=89231, **sampling)
    with _native.Session(**args) as session:
        step = session.step()
    values, weights, _ = leaf_reference(args, args["initial_predictions"], leaf_ids(step, args["bins"]), 1 << step.depth)
    np.testing.assert_allclose(step.leaf_values, values, rtol=4e-4, atol=2e-5)
    np.testing.assert_allclose(step.leaf_weights, weights, rtol=3e-6, atol=3e-4)
    assert weights.sum() == pytest.approx(2 * args["pair_weights"].sum(dtype=np.float64), rel=2e-6)


@pytest.mark.parametrize("sampling", BOOTSTRAPS)
@pytest.mark.parametrize("mode", ["No", "Armijo"])
def test_exact_cursor_and_sampler_continuation(sampling, mode):
    args = problem(iterations=5, random_seed=8942, random_strength=.3,
                   leaf_estimation_backtracking=mode, **sampling)
    complete = _native.train(**args)
    first = _native.train(**(args | {"iterations": 2}))
    bootstrap = first.stats["bootstrap_state"]
    continuation = args | dict(iterations=3, initial_predictions=first.predictions,
                               iteration_offset=bootstrap["iteration_offset"],
                               initial_mvs_lambda=bootstrap["mvs_lambda"])
    resumed = _native.train(**continuation)
    for key in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(getattr(resumed, key), getattr(complete, key)[2:])
    np.testing.assert_array_equal(resumed.predictions, complete.predictions)
    np.testing.assert_array_equal(resumed.loss, complete.loss[2:])


def test_supplied_pairs_ignore_labels_and_default_to_unit_edge_weights():
    args = problem(pair_weights=None)
    first = _native.train(**args)
    changed = _native.train(**(args | {"targets": np.arange(args["bins"].shape[1], dtype=np.float32),
                                     "pair_weights": np.ones(len(args["pair_winners"]), np.float32)}))
    for key in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights", "predictions", "loss"):
        np.testing.assert_array_equal(getattr(first, key), getattr(changed, key))


def test_depth_zero_centers_the_only_leaf_to_zero():
    args = problem(depth=0)
    result = _native.train(**args)
    np.testing.assert_array_equal(result.depths, 0)
    np.testing.assert_array_equal(result.leaf_values, 0)
    np.testing.assert_array_equal(result.predictions, args["initial_predictions"])
    np.testing.assert_array_equal(result.loss, np.full_like(result.loss, result.loss[0]))


@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
@pytest.mark.parametrize("iterations", [1, 2])
def test_first_accepted_pairwise_step_can_exceed_backtracking_budget(mode, iterations):
    args = dict(
        bins=np.array([[0, 1, 0, 1], [0, 0, 1, 1]], np.uint8), targets=np.zeros(4, np.float32),
        candidate_features=np.array([0, 1], np.uint32), candidate_bins=np.array([0, 0], np.uint32),
        objective="PairLogit", pair_winners=np.array([3, 2, 2, 2, 2, 0, 2, 0, 1], np.uint32),
        pair_losers=np.array([0, 1, 3, 1, 1, 1, 3, 2, 3], np.uint32),
        pair_weights=np.array([.20243001, 2.5679834, 1.1656985, .2832399, 2.0732398,
                               2.0251286, 1.5428388, 1.8459527, 2.06207], np.float32),
        initial_predictions=np.array([-1.5496596, -14.044306, .9945111, 9.262004], np.float32),
        iterations=1, depth=2, learning_rate=1., l2_leaf_reg=.001, bias=0, score_function="L2",
        leaf_estimation_method="Newton", leaf_estimation_iterations=iterations,
        leaf_estimation_backtracking=mode,
    )
    with _native.Session(**args) as session:
        initial_loss = session.result().loss[0]
        step = session.step()
    assert step.depth == 2
    values, _, trace = leaf_reference(args, args["initial_predictions"], leaf_ids(step, args["bins"]), 4)
    np.testing.assert_allclose(step.leaf_values, values, rtol=4e-4, atol=2e-3)
    if iterations == 1:
        assert trace == [(1., True)]
        assert step.loss > initial_loss
    else:
        assert trace == [(1., False), (.5, False), (.25, False), (.125, False), (.0625, False),
                         (.03125, False), (.015625, False), (.0078125, True)]
        assert step.loss < initial_loss


@pytest.mark.parametrize("change", [
    dict(pair_winners=np.array([0, 1]), pair_losers=np.array([1]), pair_weights=None),
    dict(pair_winners=np.array([0]), pair_losers=np.array([0]), pair_weights=None),
    dict(pair_winners=np.array([0]), pair_losers=np.array([1000]), pair_weights=None),
    dict(pair_winners=np.array([0]), pair_losers=np.array([80]), pair_weights=None),
    dict(pair_weights=np.zeros(1209, np.float32)), dict(sample_weight=np.ones(261, np.float32)),
    dict(leaf_estimation_method="Exact"),
])
def test_invalid_supplied_pair_configuration_is_rejected(change):
    with pytest.raises((ValueError, RuntimeError), match="(?i)(pair|weight|Exact|endpoint|group)"):
        _native.train(**problem(**change))
