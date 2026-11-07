"""Greedy ridge and Langevin use sequential CUDA leaf-oracle contracts."""
import ctypes as ct
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from catboost_metal import _greedy
from test_greedy_training import route
from test_scalar_langevin import Noise, arguments, reference
from test_yeti_rank_training import oracle_leaves, problem as yeti_problem


pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                               reason="Apple Silicon Metal required")


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CatBoost fitting is forbidden")

    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def ridge(session, enabled):
    call = session._lib.cbm_greedy_session_set_add_ridge
    call.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_char_p, ct.c_size_t]
    call.restype = ct.c_int
    error = ct.create_string_buffer(4096)
    if call(session._handle, enabled, error, len(error)):
        raise RuntimeError(error.value.decode())


class GreedyNoise(Noise):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeline, self.oracle_seeds = [], []

        @self.noise_type
        def callback(context, event, count, output):
            self.timeline.append(event)
            np.ctypeslib.as_array(output, shape=(count,))[:] = self.draw(event, count)
            return 0

        @self.seed_type
        def seed(context, event, output):
            self.timeline.append(event)
            self.seeds.append(event)
            output[0] = 91037 + len(self.timeline)
            if event == 6:
                self.oracle_seeds.append(output[0])
            return 0

        self.callback, self.seed = callback, seed

    def install(self, session, temperature=2):
        function = session._lib.cbm_greedy_session_set_langevin
        function.argtypes = [ct.c_void_p, ct.c_float, self.noise_type,
            self.seed_type, ct.c_void_p, ct.c_char_p, ct.c_size_t]
        function.restype = ct.c_int
        error = ct.create_string_buffer(4096)
        if function(session._handle, temperature, self.callback, self.seed, None, error, len(error)):
            raise RuntimeError(error.value.decode())


@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_greedy_ridge_matches_explicit_regularized_equations(mode, method):
    args = arguments(4, mode, method)
    baseline = np.zeros((1, len(args["targets"])), np.float32)
    expected = reference(args["targets"], args["sample_weight"], baseline, Noise(zero=True),
                         4, mode, False, True)
    with _greedy.TrainingSession(**args) as session:
        with pytest.raises(RuntimeError, match="ridge flag"):
            ridge(session, 2)
        ridge(session, True)
        tree = session.step()
        np.testing.assert_allclose(tree.leaf_values, expected, rtol=5e-6, atol=3e-6)
        with pytest.raises(RuntimeError, match="before the first tree"):
            ridge(session, False)
    # Ridge changes later directions; accepting the option as a no-op fails.
    with _greedy.TrainingSession(**args) as session:
        unregularized = session.step().leaf_values
    assert not np.allclose(unregularized, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("iterations", [1, 4])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("histories,ridge_enabled,policy", [
    (1, False, "Lossguide"), (3, True, "Depthwise"), (3, False, "Region")])
def test_greedy_langevin_walks_histories_in_source_order(iterations, mode, histories, ridge_enabled, policy):
    args = arguments(iterations, mode)
    args.update(grow_policy=policy, max_leaves=3)
    baselines = np.repeat(np.linspace(-.2, .3, histories, dtype=np.float32)[:, None], len(args["targets"]), axis=1)
    expected_noise, actual_noise = Noise(), GreedyNoise()
    # Each DocParallel task owns a walker. Noise calls advance one shared stream
    # in ascending history order, even when search chooses the final history.
    expected = np.array([reference(args["targets"], args["sample_weight"], baselines[p:p + 1],
        expected_noise, iterations, mode, False, ridge_enabled)[0] for p in range(histories)], np.float32)
    with _greedy.TrainingSession(**args) as session:
        session.configure_permutations(np.zeros((histories, 1, len(args["targets"])), np.uint8), baselines)
        session.select_permutation(histories - 1)
        ridge(session, ridge_enabled)
        actual_noise.install(session)
        session.step()
        actual = session.permutation_state["predictions"]
    np.testing.assert_allclose(actual, np.float32(baselines + expected[:, None]), rtol=8e-6, atol=4e-6)
    assert actual_noise.events == expected_noise.events
    assert actual_noise.seeds == []  # No candidates, no bootstrap, no weak noise.
    if iterations == 1:
        assert actual_noise.events == [(1, 1), (2, 1)] * histories


def test_negative_initial_greedy_langevin_diagonal_has_zero_direction():
    args = arguments(1)
    noise = GreedyNoise(negative_hessian=True)
    with _greedy.TrainingSession(**args) as session:
        noise.install(session)
        tree = session.step()
    np.testing.assert_array_equal(tree.leaf_values, 0)
    assert noise.events == [(1, 1), (2, 1)]


@pytest.mark.parametrize("method", ["Newton", "Exact", "Simple"])
@pytest.mark.parametrize("sampling", ["No", "Bayesian"])
def test_greedy_zero_temperature_and_skipped_estimators(method, sampling):
    args = arguments(4, method=method)
    args.update(bootstrap_type=sampling)
    if method == "Exact":
        args.update(objective="Quantile", objective_param=.5, leaf_estimation_iterations=1)
    elif method == "Simple":
        args["leaf_estimation_iterations"] = 1
    noise = GreedyNoise(zero=True)
    with _greedy.TrainingSession(**args) as session:
        ridge(session, True)
        noise.install(session, temperature=0)
        actual = session.step()
    with _greedy.TrainingSession(**args) as session:
        ridge(session, True)
        expected = session.step()
    np.testing.assert_allclose(actual.leaf_values, expected.leaf_values, rtol=4e-6, atol=3e-6)
    assert noise.seeds == ([0] if sampling != "No" else [])
    assert len(noise.events) == (10 if method == "Newton" else 0)


def raw_step(session):
    capacity = session._params.max_leaves
    nodes = np.zeros((2 * capacity - 1, 6), np.uint32)
    values, weights = np.zeros(capacity, np.float32), np.zeros(capacity, np.float32)
    info, error = _greedy.StepInfo(), ct.create_string_buffer(4096)
    code = session._lib.cbm_greedy_session_step(session._handle, ct.byref(info),
        nodes.ctypes.data_as(ct.POINTER(_greedy.Node)),
        values.ctypes.data_as(ct.POINTER(ct.c_float)), weights.ctypes.data_as(ct.POINTER(ct.c_float)), error, len(error))
    if code:
        raise RuntimeError(error.value.decode())
    return SimpleNamespace(nodes=nodes[:info.node_count], leaf_values=values[:info.leaf_count],
                           leaf_weights=weights[:info.leaf_count])


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("sampling", ["No", "Bayesian"])
def test_greedy_yeti_langevin_joint_oracle_seeds_follow_actual_calls(iterations, sampling):
    args = yeti_problem(iterations=1, depth=1, leaf_estimation_iterations=iterations,
                        bootstrap_type=sampling)
    args.update(objective="YetiRank", grow_policy="Lossguide", max_leaves=2)
    noise = GreedyNoise(zero=True)
    with _greedy.TrainingSession(**args) as session:
        noise.install(session, temperature=0)
        # Call the native operation directly: the legacy Python controller's
        # precomputed Yeti packet is deliberately absent in callback mode.
        tree = raw_step(session)
    ids = route(tree, args["bins"])
    expected, weights = oracle_leaves(args, args["initial_predictions"], ids,
        len(tree.leaf_values), noise.oracle_seeds[:iterations])
    np.testing.assert_allclose(tree.leaf_values, expected, rtol=2e-4, atol=3e-6)
    np.testing.assert_allclose(tree.leaf_weights, weights, rtol=5e-6, atol=2e-5)
    prefix = ([0] if sampling != "No" else []) + [5, 7, 6, 1, 2]
    assert noise.timeline == prefix + ([6, 3, 4] * iterations if iterations > 1 else [])


def test_greedy_langevin_rejects_invalid_configuration_before_mutation():
    args = arguments(1)
    noise = GreedyNoise()
    with _greedy.TrainingSession(**args) as session:
        for temperature in (-1., float("nan"), float("inf")):
            with pytest.raises(RuntimeError, match="temperature"):
                noise.install(session, temperature)
        noise.install(session)
        with pytest.raises(RuntimeError, match="once"):
            noise.install(session)
        session.step()
