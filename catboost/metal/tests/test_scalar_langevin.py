"""Native scalar Langevin walker equations and callback boundaries, no CPU fit."""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _native
from test_regularization import configure

pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Apple GPU required")


class Noise:
    noise_type = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_double))
    seed_type = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint64))

    def __init__(self, zero=False, negative_hessian=False):
        self.events, self.seeds = [], []
        self.zero, self.negative_hessian = zero, negative_hessian
        @self.noise_type
        def callback(context, event, count, output):
            np.ctypeslib.as_array(output, shape=(count,))[:] = self.draw(event, count)
            return 0
        @self.seed_type
        def seed(context, event, output):
            self.seeds.append(event)
            output[0] = 731 + len(self.seeds)
            return 0
        self.callback, self.seed = callback, seed

    def draw(self, event, count):
        self.events.append((event, count))
        if self.zero:
            return np.zeros(count)
        if event == 2 and self.negative_hessian:
            return np.full(count, -100.)
        scales = {1: .8, 2: -.15, 3: .22, 4: -.07}
        return scales[event] * (np.arange(count, dtype=float) + 1) * (1 + .13 * (len(self.events) - 1))

    def install(self, session, weak=False, temperature=2):
        function = session._lib.cbm_session_set_langevin
        function.argtypes = [ct.c_void_p, ct.c_float, ct.c_uint32, self.noise_type,
            self.seed_type, ct.c_void_p, ct.c_char_p, ct.c_size_t]
        error = ct.create_string_buffer(4096)
        assert function(session._handle, temperature, weak, self.callback, self.seed, None, error, len(error)) == 0, error.value.decode()


def arguments(iterations=4, mode="No", method="Newton"):
    targets = np.array([.25, 2, 1, 4, 3, .5, 2], np.float32)
    weights = np.array([.5, 1, 2, 3, 1, .25, 4], np.float32)
    return dict(bins=np.zeros((1, len(targets)), np.uint8), targets=targets,
        candidate_features=np.array([], np.uint32), candidate_bins=np.array([], np.uint32),
        iterations=1, depth=0, learning_rate=1, l2_leaf_reg=.4, bias=0, score_function="Cosine",
        objective="RMSE", sample_weight=weights, leaf_estimation_iterations=iterations,
        leaf_estimation_backtracking=mode, leaf_estimation_method=method)


def open_session(args):
    # Private native coverage bypasses the older standalone Python Simple
    # guard; the additive native setter is the interface under test here.
    simple = args["leaf_estimation_method"] == "Simple"
    session = _native.Session(**{**args, "leaf_estimation_method": "Newton" if simple else args["leaf_estimation_method"]})
    if simple:
        options = _native.ObjectiveOptions(0, 3, 0, 0)
        error = ct.create_string_buffer(4096)
        assert session._lib.cbm_session_set_objective(session._handle, ct.byref(options), error, len(error)) == 0, error.value.decode()
    return session


def reference(targets, weights, baselines, noise, iterations, mode, normalize, ridge):
    count = len(baselines)
    point = np.zeros(count, np.float32)
    mass = weights.sum(dtype=float)
    l2 = float(np.float32(.4))
    def evaluate(candidate):
        value, gradient = [], []
        for task in range(count):
            raw = np.float32(baselines[task] + candidate[task])
            residual = np.float32(targets - raw)
            gradient.append(np.float32(weights * residual).sum(dtype=float))
            value.append(-np.float32(weights * np.float32(residual * residual)).sum(dtype=float))
        value, gradient = np.array(value), np.array(gradient)
        diagonal = np.full(count, mass)
        if normalize:
            value /= mass; gradient /= mass; diagonal /= mass
        if ridge:
            value -= .5 * l2 * candidate.astype(float) ** 2
            gradient -= l2 * candidate.astype(float)
        return value.sum(), gradient, diagonal + l2
    value, gradient, diagonal = evaluate(point)
    gradient += noise.draw(1, count)
    diagonal += noise.draw(2, count)
    step, updated, fresh = 1., False, True
    for attempt in range(100):
        if attempt >= iterations and updated:
            break
        if fresh:
            direction = np.zeros(count, np.float32)
            valid = diagonal > 0
            direction[valid] = (gradient[valid] / (diagonal[valid] + 1e-20)).astype(np.float32)
            dot = gradient @ direction.astype(float)
        candidate = np.float32(point.astype(float) + step * direction.astype(float))
        if iterations == 1:
            return candidate
        candidate_value, candidate_gradient, candidate_diagonal = evaluate(candidate)
        candidate_gradient += noise.draw(3, count)
        threshold = value + (1e-5 * step * dot if mode == "Armijo" else 0)
        if mode == "No" or (np.isfinite(candidate_value) and candidate_value >= threshold):
            candidate_gradient += noise.draw(4, count)
            point, value, gradient, diagonal = candidate, candidate_value, candidate_gradient, candidate_diagonal
            fresh, updated, step = True, True, 1.
        else:
            step *= .5; fresh = False
    return point


@pytest.mark.parametrize("iterations", [1, 4])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("histories,normalize,ridge", [(1, False, False), (3, True, True)])
def test_scalar_global_langevin_walker(iterations, mode, histories, normalize, ridge):
    args = arguments(iterations, mode)
    baselines = np.repeat(np.linspace(-.2, .3, histories, dtype=np.float32)[:, None], len(args["targets"]), axis=1)
    expected_noise, actual_noise = Noise(), Noise()
    expected = reference(args["targets"], args["sample_weight"], baselines, expected_noise,
        iterations, mode, normalize, ridge)
    with open_session(args) as session:
        session.configure_permutations(np.zeros((histories, 1, len(args["targets"])), np.uint8), baselines)
        configure(session, normalize, ridge)
        actual_noise.install(session)
        session.step()
        actual = session.permutation_state["predictions"]
    np.testing.assert_allclose(actual, np.float32(baselines + expected[:, None]), rtol=8e-6, atol=4e-6)
    assert actual_noise.events == expected_noise.events
    assert actual_noise.seeds == [0]
    if iterations == 1:
        assert actual_noise.events == [(1, histories), (2, histories)]
    if iterations > 1 and mode == "No":
        assert len(actual_noise.events) == 2 + 2 * iterations


def test_negative_initial_langevin_diagonal_gives_zero_direction():
    args = arguments(1)
    noise = Noise(negative_hessian=True)
    with open_session(args) as session:
        noise.install(session)
        session.step()
        actual = session.result()
    np.testing.assert_array_equal(actual.leaf_values, 0)
    assert noise.events == [(1, 1), (2, 1)]


@pytest.mark.parametrize("method", ["Newton", "Exact", "Simple"])
def test_zero_temperature_preserves_leaf_callbacks_and_skip_boundaries(method):
    args = arguments(4, method=method)
    if method == "Exact":
        args.update(objective="Quantile", objective_param=.5, leaf_estimation_iterations=1)
    if method == "Simple":
        args["leaf_estimation_iterations"] = 1
    noise = Noise(zero=True)
    with open_session(args) as session:
        noise.install(session, weak=True, temperature=0)
        session.step()
        actual = session.result()
    with open_session(args) as session:
        session.step()
        expected = session.result()
    np.testing.assert_allclose(actual.leaf_values, expected.leaf_values, rtol=3e-6, atol=2e-6)
    assert noise.seeds == [0]
    assert len(noise.events) == (10 if method == "Newton" else 0)


def test_feature_parallel_simple_keeps_initial_leaf_noise_events():
    args = arguments(1, method="Simple")
    noise = Noise()
    with open_session(args) as session:
        noise.install(session, weak=False)
        session.step()
    assert noise.events == [(1, 1), (2, 1)]


def test_empty_leaf_remains_in_langevin_vector_then_regularizes_to_zero():
    args = arguments(1)
    args.update(depth=1, candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32))
    noise = Noise()
    with open_session(args) as session:
        noise.install(session)
        tree = session.step()
    assert noise.events == [(1, 2), (2, 2)]
    assert tree.leaf_weights[1] == 0
    assert tree.leaf_values[1] == 0
