"""Combination GPU training checked against independent component equations.

No CPU CatBoost fit supplies an expected tree. The oracle differentiates each
component at the same raw point, sums its weighted derivatives, and projects
those sums onto the selected leaves. In particular, Gradient leaves use the
outer object weights, not the weighted sum used during split search.
"""

import ctypes as ct
import platform
import threading

import numpy as np
import pytest

from catboost_metal import _native
from cuda_scalar_reference import objective_terms, weighted_loss
from cuda_querywise_reference import query_terms, query_loss
from cuda_reference import _score_children
from test_pairwise_training import training_terms as pair_terms
from test_yeti_rank_kernels import reference as yeti_terms


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Combination requires actual Apple Silicon Metal",
)


class Component(ct.Structure):
    _fields_ = [("objective", ct.c_uint32), ("weight", ct.c_float), ("param", ct.c_float),
                ("border", ct.c_float), ("beta", ct.c_float), ("lambda_", ct.c_float),
                ("permutations", ct.c_uint32), ("reserved", ct.c_uint32),
                ("decay", ct.c_float), ("reserved1", ct.c_uint32 * 3)]


class Options(ct.Structure):
    _fields_ = [("component_count", ct.c_uint32), ("group_count", ct.c_uint32),
                ("pair_count", ct.c_uint32), ("reserved", ct.c_uint32)]


NAMES = ["RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber", "Expectile", "Lq",
         "Tweedie", "LogLinQuantile", "Quantile", "MAE", "MAPE", "QueryRMSE", "QuerySoftMax", "PairLogit"]


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor
    def forbidden(*args, **kwargs):
        pytest.fail("Combination private acceptance must not fit CPU CatBoost")
    for kind in (CatBoost, CatBoostClassifier, CatBoostRanker, CatBoostRegressor):
        monkeypatch.setattr(kind, "fit", forbidden)


def component(name, weight, param=0, **kwargs):
    result = Component()
    result.objective, result.weight, result.param = 17 if name == "YetiRank" else NAMES.index(name), weight, param
    result.beta, result.lambda_ = 1, .01
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


def problem():
    rng = np.random.default_rng(68091)
    bins = rng.integers(0, 4, (3, 271), dtype=np.uint8)
    target = (.4 + .2 * bins[0] + .03 * rng.normal(size=271)).astype(np.float32)
    weights = rng.uniform(.2, 2, 271).astype(np.float32)
    weights[::23] = 0
    return dict(bins=bins, targets=target, weights=weights,
                cursor=rng.normal(0, .15, 271).astype(np.float32),
                offsets=np.array([0, 1, 8, 61, 129, 271], np.uint32),
                winners=np.array([4, 14, 27, 55, 80, 94, 170, 200, 254], np.uint32),
                losers=np.array([2, 34, 43, 60, 121, 113, 240, 251, 261], np.uint32),
                pairs=np.array([.3, 2, .9, 0, 1.7, .8, 3, .5, 1], np.float32))


def session(data, components, *, method="Newton", backtracking="No", depth=1,
            iterations=2, leaf_iterations=3, score="Cosine", initial=None):
    """Use the additive C ABI while reusing existing session lifecycle readers."""
    result = _native.Session.__new__(_native.Session)
    result._handle, result._lock = ct.c_void_p(), threading.RLock()
    result._completed, result._permutation_count = 0, 1
    result._permutations_configured = result._feature_penalties_configured = False
    result._lib = _native._load(_native.build_library())
    result.objective = "Combination"
    rows = data["targets"].size
    features = np.repeat(np.arange(3, dtype=np.uint32), 3) if depth else np.array([], np.uint32)
    borders = np.tile(np.arange(3, dtype=np.uint32), 3) if depth else np.array([], np.uint32)
    result._params = _native.SessionParams(_native.TrainParams(rows, 3, features.size, 4,
        iterations, depth, {"L2": 0, "Cosine": 1, "NewtonL2": 2, "NewtonCosine": 3}[score], .2, data.get("l2", 2), 0),
        19, leaf_iterations, {"No": 0, "AnyImprovement": 1, "Armijo": 2}[backtracking], 0)
    objective = _native.ObjectiveOptions(19, int(method == "Gradient"), 0, 0)
    grouped = any(c.objective >= 12 for c in components)
    paired = any(c.objective == 14 for c in components)
    options = Options(len(components), data["offsets"].size - 1 if grouped else 0,
                      data["pairs"].size if paired else 0, 0)
    values = (Component * len(components))(*components)
    operation = result._lib.cbm_session_create_combination
    operation.argtypes = [ct.POINTER(_native.SessionParams), ct.POINTER(_native.ObjectiveOptions),
        ct.POINTER(Options), ct.POINTER(Component)] + [ct.c_void_p] * 11 + [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_size_t]
    operation.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    code = operation(ct.byref(result._params), ct.byref(objective), ct.byref(options), values,
        _native._u32(data["offsets"]) if grouped else None,
        _native._u32(data["winners"]) if paired else None,
        _native._u32(data["losers"]) if paired else None,
        _native._f32(data["pairs"]) if paired else None,
        _native._u8(data["bins"]), _native._f32(data["targets"]), _native._f32(data["weights"]),
        _native._f32(data["cursor"] if initial is None else initial),
        _native._u32(features), _native._u32(borders), None, ct.byref(result._handle), error, len(error))
    result._check(code, error)
    return result


def terms(data, components, raw, yeti_seeds=None):
    gradient, hessian, weak_weights = (np.zeros(raw.size, np.float32) for _ in range(3))
    loss = metric = 0
    ordered = sorted(components, key=lambda c: c.objective < 12)
    for c in ordered:
        name, scale = ("YetiRank" if c.objective == 17 else NAMES[c.objective]), float(c.weight)
        if name == "YetiRank":
            assert yeti_seeds is not None
            g, h = yeti_terms(data["targets"], data["weights"], raw, data["offsets"],
                permutations=c.permutations, decay=c.decay, seed=next(yeti_seeds))[1].T
            mass, value, final, scale = h, 0., 0., -scale
        elif name == "PairLogit":
            part = pair_terms(raw, data["winners"], data["losers"], data["pairs"])
            g, h, mass = part["gradients"], part["curvature"], part["incident_weights"]
            value, denominator = part["objective"]
            final = value / denominator
        elif c.objective >= 12:
            g, h, value, denominator = query_terms(data["targets"], raw, data["weights"],
                data["offsets"], name, c.beta, c.lambda_)
            mass = data["weights"]
            final = np.sqrt(value / denominator) if name == "QueryRMSE" else value / denominator
        else:
            target = (data["targets"] > c.border).astype(np.float32) if name == "Logloss" else data["targets"]
            v, g, h = objective_terms(target, raw, name, c.param)
            g, h = g * data["weights"], h * data["weights"]
            value = np.dot(v, data["weights"])
            mass = data["weights"]
            final = weighted_loss(target, raw, data["weights"], name, c.param)
        # CUDA materializes each component in float before coefficient multiply.
        gradient += np.float32(scale) * np.asarray(g, np.float32)
        hessian += np.float32(scale) * np.asarray(h, np.float32)
        weak_weights += np.float32(scale) * np.asarray(mass, np.float32)
        loss += scale * value
        metric += scale * final
    return gradient.astype(float), hessian.astype(float), weak_weights.astype(float), loss, metric


def leaf_oracle(data, components, cursor, ids, leaves, method, mode, iterations):
    masses = np.bincount(ids, weights=data["weights"], minlength=leaves)
    point = np.zeros(leaves, np.float32)
    def project(candidate):
        g, h, _, loss, _ = terms(data, components, (cursor + candidate[ids]).astype(np.float32))
        return -loss, np.bincount(ids, weights=g, minlength=leaves), np.bincount(ids, weights=h, minlength=leaves)
    value, g, h = project(point)
    attempts, updated = 0, False
    while attempts < iterations:
        diagonal = (masses if method == "Gradient" else h) + data.get("l2", 2)
        direction = np.divide(g, diagonal + 1e-20, out=np.zeros(leaves), where=diagonal > 0).astype(np.float32)
        dot = np.dot(g, direction)
        step = 1.
        while True:
            trial = (point.astype(float) + step * direction).astype(np.float32)
            trial[masses < 1e-20] = 0
            next_value, next_g, next_h = project(trial)
            attempts += 1
            accepted = mode == "No" or iterations == 1 or next_value >= value + (1e-5 * step * dot if mode == "Armijo" else 0)
            if accepted:
                point, value, g, h = trial, next_value, next_g, next_h
                updated = True
                break
            if attempts >= iterations and (updated or attempts >= 100):
                break
            step /= 2
    return point * np.float32(.2), masses


PROFILES = [
    ("RMSE", 0), ("Logloss", 0), ("CrossEntropy", 0), ("Poisson", 0),
    ("Huber", .8), ("Expectile", .3), ("Lq", 2.4), ("Tweedie", 1.3),
    ("LogLinQuantile", .35), ("Quantile", .7), ("MAE", .5), ("MAPE", 0),
    ("QueryRMSE", 0), ("QuerySoftMax", 0), ("PairLogit", 0),
]


@pytest.mark.parametrize("name,param", PROFILES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_component_algebra_leaves_and_metric(name, param, method):
    data = problem()
    data["targets"] = np.clip(data["targets"], 0, 1)
    components = [component(name, .7, param, border=.6), component("RMSE", 1.3)]
    with session(data, components, method=method) as model:
        cursor = data["cursor"].copy()
        initial = model.result().loss[0]
        assert initial == pytest.approx(terms(data, components, cursor)[4], rel=3e-6, abs=3e-6)
        for _ in range(2):
            step = model.step()
            ids = np.zeros(cursor.size, np.uint32)
            for level, (feature, border) in enumerate(zip(step.split_features, step.split_bins)):
                ids |= (data["bins"][feature] > border).astype(np.uint32) << level
            expected, masses = leaf_oracle(data, components, cursor, ids, 1 << step.depth, method, "No", 3)
            np.testing.assert_allclose(step.leaf_values, expected, rtol=2e-5, atol=3e-6)
            np.testing.assert_allclose(step.leaf_weights, masses, rtol=2e-6, atol=3e-6)
            cursor = (cursor + expected[ids]).astype(np.float32)
            assert step.loss == pytest.approx(terms(data, components, cursor)[4], rel=2e-5, abs=3e-6)
        np.testing.assert_allclose(model.result().predictions, cursor, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
@pytest.mark.parametrize("query", [False, True])
def test_backtracking_uses_sum_of_raw_objectives(method, mode, query):
    data = problem()
    components = [component("Huber", .3, .3), component("QueryRMSE" if query else "Poisson", 1.7)]
    with session(data, components, method=method, backtracking=mode, depth=1, iterations=1, leaf_iterations=5) as model:
        step = model.step()
        ids = np.zeros(data["targets"].size, np.uint32)
        for level, (feature, border) in enumerate(zip(step.split_features, step.split_bins)):
            ids |= (data["bins"][feature] > border).astype(np.uint32) << level
        expected, mass = leaf_oracle(data, components, data["cursor"], ids, 1 << step.depth, method, mode, 5)
        np.testing.assert_allclose(step.leaf_values, expected, rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(step.leaf_weights, mass, rtol=2e-6, atol=3e-6)


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine"])
def test_search_uses_weighted_component_weights(score):
    data = problem()
    components = [component("PairLogit", 11), component("RMSE", .25)]
    g, h, mass, _, _ = terms(data, components, data["cursor"])
    weights = h if score.startswith("Newton") else mass
    scores = []
    for feature in range(3):
        for border in range(3):
            right = data["bins"][feature] > border
            sums = [g[~right].sum(), g[right].sum()]
            masses = [weights[~right].sum(), weights[right].sum()]
            scores.append(_score_children(np.asarray(sums), np.asarray(masses), 2, score.removeprefix("Newton")))
    with session(data, components, score=score, iterations=1, leaf_iterations=1) as model:
        step = model.step()
        assert step.depth == 1
        assert (step.split_features[0], step.split_bins[0]) == divmod(int(np.argmin(scores)), 3)


@pytest.mark.parametrize("profile", ["scalar", "query", "pair"])
def test_staged_growth_and_resumed_cursors_match_full_steps(profile):
    data = problem()
    components = [component({"scalar": "Huber", "query": "QueryRMSE", "pair": "PairLogit"}[profile], .7, .8),
                  component("RMSE", 1.3)]
    with session(data, components, iterations=3, depth=2) as full:
        full.step()
        first = full.result().predictions.copy()
        full.step()
        full.step()
        expected = full.result()
    with session(data, components, iterations=2, depth=2, initial=first) as resumed:
        for _ in range(2):
            resumed.begin_tree()
            while not resumed.grow_tree()["finished"]:
                pass
            resumed.finish_tree()
        actual = resumed.result()
    for key in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key)[1:])
    np.testing.assert_array_equal(actual.predictions, expected.predictions)


@pytest.mark.parametrize("field,value", [("weight", 0), ("weight", -1), ("weight", np.inf),
    ("weight", np.nan), ("objective", 16), ("param", np.inf), ("reserved", 1)])
def test_invalid_component_is_rejected_before_training(field, value):
    components = [component("RMSE", .7), component("Huber", 1.3, .8)]
    setattr(components[0], field, value)
    with pytest.raises(RuntimeError, match="Combination|component"):
        with session(problem(), components):
            pass


@pytest.mark.parametrize("bad_target", [-.1, 0, np.finfo(np.float32).max])
def test_querysoftmax_component_validates_its_own_target(bad_target):
    data = problem()
    if bad_target < 0:
        # Keep the total target mass positive: aggregate validation alone is
        # insufficient to detect this invalid individual target.
        data["targets"][1] = bad_target
    else:
        data["targets"][:] = bad_target
    components = [component("QuerySoftMax", .7), component("RMSE", 1.3)]
    with pytest.raises(RuntimeError, match="Combination|QuerySoftMax|target"):
        with session(data, components):
            pass


SEED_CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint64))


def install_seed_callback(model):
    consumed = []
    def next_seed(_context, output):
        seed = 71093 + len(consumed) * 104729
        consumed.append(seed)
        output[0] = seed
        return 0
    callback = SEED_CALLBACK(next_seed)
    operation = model._lib.cbm_session_set_combination_yeti_seed_callback
    operation.argtypes = [ct.c_void_p, SEED_CALLBACK, ct.c_void_p, ct.c_char_p, ct.c_size_t]
    operation.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    model._check(operation(model._handle, callback, None, error, len(error)), error)
    return callback, consumed


def stochastic_leaf_oracle(data, components, ids, count, method, mode, iterations, consumed):
    """Full CUDA oracle evaluations, including rejected/final-unused trials."""
    seeds = iter(consumed)
    yetis = sum(c.objective == 17 for c in components)
    # Structure evaluates one weak target even with a depth-zero tree.
    for _ in range(yetis):
        next(seeds)
    point = np.zeros(count, np.float32)
    mass = np.bincount(ids, weights=data["weights"], minlength=count)
    evaluations = 0
    def project(candidate):
        nonlocal evaluations
        evaluations += 1
        g, h, _, loss, _ = terms(data, components,
            np.float32(data["cursor"] + candidate[ids]), seeds)
        return -loss, np.bincount(ids, weights=g, minlength=count), np.bincount(ids, weights=h, minlength=count)
    value, g, h = project(point)
    attempts, accepted = 0, False
    trace = []
    while attempts < iterations:
        diagonal = (mass if method == "Gradient" else h) + data.get("l2", 2)
        direction = np.divide(g, diagonal + 1e-20, out=np.zeros(count), where=diagonal > 0).astype(np.float32)
        dot = np.dot(g, direction)
        step = 1.
        if iterations == 1:
            point = np.float32(point + direction)
            point[mass < 1e-20] = 0
            break
        while attempts < iterations or (not accepted and attempts < 100):
            trial = np.float32(point.astype(float) + step * direction.astype(float))
            trial[mass < 1e-20] = 0
            next_value, next_g, next_h = project(trial)
            take = mode == "No" or next_value >= value + (1e-5 * step * dot if mode == "Armijo" else 0)
            trace.append(take)
            attempts += 1
            if take:
                point, value, g, h = trial, next_value, next_g, next_h
                accepted = True
                break
            step /= 2
    assert list(seeds) == [], "Runtime consumed unused speculative seed draws"
    assert len(consumed) == yetis * (1 + evaluations)
    return np.float32(point * np.float32(.2)), trace


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("iterations", [1, 3])
def test_yeti_coefficients_and_actual_oracle_seed_calls(method, mode, iterations):
    data = problem()
    for key in ("targets", "weights", "cursor"):
        data[key] = data[key][:67].copy()
    data["targets"] = np.clip(data["targets"], 0, 1)
    data["bins"] = data["bins"][:, :67].copy()
    data["offsets"] = np.array([0, 19, 40, 67], np.uint32)
    components = [component("YetiRank", .03, permutations=3, decay=.8),
                  component("RMSE", 2), component("YetiRank", .05, permutations=5, decay=.9)]
    with session(data, components, method=method, backtracking=mode,
                 iterations=1, leaf_iterations=iterations) as model:
        callback, consumed = install_seed_callback(model)
        step = model.step()
        ids = np.zeros(67, np.uint32)
        for level, (feature, border) in enumerate(zip(step.split_features, step.split_bins)):
            ids |= np.uint32(data["bins"][feature] > border) << level
        expected, _ = stochastic_leaf_oracle(data, components, ids, 1 << step.depth,
            method, mode, iterations, consumed)
        np.testing.assert_allclose(step.leaf_values, expected, rtol=8e-5, atol=3e-6)
        # Combination never applies top-level YetiRank's zero-average transform.
        assert abs(step.leaf_values.mean()) > .001
        assert callback is not None  # keep callback storage alive through step


def test_yeti_negative_regularized_curvature_yields_zero_newton_direction():
    data = problem()
    for key in ("targets", "weights", "cursor"):
        data[key] = data[key][:67].copy()
    data["targets"] = np.clip(data["targets"], 0, 1)
    data["bins"] = data["bins"][:, :67].copy()
    data["offsets"] = np.array([0, 19, 40, 67], np.uint32)
    components = [component("YetiRank", 10000, permutations=3, decay=.8), component("RMSE", .001)]
    with session(data, components, depth=0, iterations=1, leaf_iterations=1) as model:
        callback, consumed = install_seed_callback(model)
        step = model.step()
        expected, _ = stochastic_leaf_oracle(data, components, np.zeros(67, np.uint32), 1,
            "Newton", "No", 1, consumed)
        np.testing.assert_array_equal(expected, 0)
        np.testing.assert_array_equal(step.leaf_values, expected)
        assert callback is not None


@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
def test_rejected_stochastic_trials_consume_only_actual_oracle_draws(mode):
    data = problem()
    for key in ("targets", "weights", "cursor"):
        data[key] = data[key][:67].copy()
    data["targets"] = np.clip(data["targets"], 0, 1)
    data["cursor"][:] = -5
    data["bins"] = data["bins"][:, :67].copy()
    data["offsets"] = np.array([0, 19, 40, 67], np.uint32)
    data["l2"] = .001
    components = [component("Huber", 1, .2), component("RMSE", .001),
                  component("YetiRank", .000001, permutations=3, decay=.8)]
    with session(data, components, depth=0, iterations=1, leaf_iterations=3, backtracking=mode) as model:
        callback, consumed = install_seed_callback(model)
        step = model.step()
        expected, trace = stochastic_leaf_oracle(data, components, np.zeros(67, np.uint32), 1,
            "Newton", mode, 3, consumed)
        assert any(not accepted for accepted in trace), "Fixture must reject an overshooting trial"
        assert len(trace) > 3, "First acceptance is allowed to exceed the iteration budget"
        assert trace[-1]
        np.testing.assert_allclose(step.leaf_values, expected, rtol=8e-5, atol=3e-6)
        assert callback is not None
