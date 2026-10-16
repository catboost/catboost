"""Independent coupled Ordered prefix equations with injected Langevin noise."""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _ordered
from cuda_ordered_reference import numeric_folds, _task_descriptors
from test_ordered_training import prohibit_cpu_training
from test_scalar_langevin import Noise


pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                                reason="Apple GPU required")


def install(session, noise, ridge):
    # Isolate ctypes signatures from other private ABI tests in the same run.
    library = ct.CDLL(session._lib._name)
    setter = library.cbm_ordered_session_set_langevin
    setter.argtypes = [ct.c_void_p, ct.c_float, Noise.noise_type, Noise.seed_type,
                      ct.c_void_p, ct.c_char_p, ct.c_size_t]
    error = ct.create_string_buffer(4096)
    assert setter(session._handle, 0, noise.callback, noise.seed, None, error, len(error)) == 0, error.value.decode()
    regularize = library.cbm_ordered_set_add_ridge_to_target_function
    regularize.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_char_p, ct.c_size_t]
    assert regularize(session._handle, ridge, error, len(error)) == 0, error.value.decode()
    return library


def walk(tasks, noise, iterations, mode, normalize, ridge):
    """Source joint walker, independent NumPy sums over each original prefix."""
    mass = np.array([weight.sum(dtype=float) for _, _, weight in tasks])
    l2 = float(np.float32(.4))
    point = np.zeros(len(tasks), np.float32)

    def evaluate(candidate):
        values, gradients = [], []
        for index, (target, base, weight) in enumerate(tasks):
            residual = np.float32(target - np.float32(base + candidate[index]))
            values.append(-np.float32(weight * np.float32(residual * residual)).sum(dtype=float))
            gradients.append(np.float32(weight * residual).sum(dtype=float))
        values, gradients, diagonal = np.array(values), np.array(gradients), mass.copy()
        if normalize:
            values = np.divide(values, mass, out=np.zeros_like(values), where=mass > 0)
            gradients = np.divide(gradients, mass, out=np.zeros_like(gradients), where=mass > 0)
            diagonal = (mass > 0).astype(float)
        if ridge:
            values -= .5 * l2 * candidate.astype(float) ** 2
            gradients -= l2 * candidate.astype(float)
        return values.sum(), gradients, diagonal + l2

    value, gradient, diagonal = evaluate(point)
    gradient += noise.draw(1, len(tasks))
    diagonal += noise.draw(2, len(tasks))
    fresh, updated, step = True, False, 1.
    trace = []
    for attempt in range(100):
        if attempt >= iterations and updated:
            break
        if fresh:
            direction = np.divide(gradient, diagonal + 1e-20,
                                  out=np.zeros_like(gradient), where=diagonal > 0).astype(np.float32)
            dot = gradient @ direction.astype(float)
        candidate = np.float32(point.astype(float) + step * direction.astype(float))
        candidate[mass <= 1e-20] = 0
        if iterations == 1:
            return candidate, trace
        candidate_value, candidate_gradient, candidate_diagonal = evaluate(candidate)
        candidate_gradient += noise.draw(3, len(tasks))
        threshold = value + (1e-5 * step * dot if mode == "Armijo" else 0)
        accepted = mode == "No" or (np.isfinite(candidate_value) and candidate_value >= threshold)
        trace.append(accepted)
        if accepted:
            candidate_gradient += noise.draw(4, len(tasks))
            point, value, gradient, diagonal = candidate, candidate_value, candidate_gradient, candidate_diagonal
            fresh, updated, step = True, True, 1.
        else:
            step *= .5
            fresh = False
    return point, trace


def problem(permutations, iterations, mode, normalize, zero_prefix=False):
    target = np.array([.25, 2, 1, 4, 3, .5, 2, 1, 5, 2, .5, 3], np.float32)
    weight = np.array([.5, 1, 2, 3, 1, .25, 4, .5, 2, 3, 1, .25], np.float32)
    if zero_prefix:
        weight[:2] = 0
    base = np.linspace(-.2, .3, len(target), dtype=np.float32)
    orders = np.stack([np.roll(np.arange(len(target)), shift * 3)
                       for shift in range(permutations)]).astype(np.uint32)
    descriptors, _ = _task_descriptors(numeric_folds(len(target), min_fold_size=2), len(target), permutations)
    tasks = [(target[orders[p, :end]], base[orders[p, :end]], weight[orders[p, :end]])
             for end, _, _, p in descriptors]
    config = dict(iterations=1, depth=0, learning_rate=.17, l2_leaf_reg=.4, objective="RMSE",
                  sample_weight=weight, initial_predictions=base, leaf_estimation_iterations=iterations,
                  leaf_estimation_backtracking=mode, permutation_count=permutations, permutations=orders,
                  min_fold_size=2, fold_size_loss_normalization=normalize)
    return target, base, orders, descriptors, tasks, config


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("mode", ["No", "AnyImprovement", "Armijo"])
@pytest.mark.parametrize("normalize,ridge", [(False, False), (True, False), (True, True)])
def test_coupled_prefix_langevin_equations(permutations, iterations, mode, normalize, ridge):
    target, base, orders, descriptors, tasks, config = problem(permutations, iterations, mode, normalize)
    expected_noise, actual_noise = Noise(), Noise()
    values, trace = walk(tasks, expected_noise, iterations, mode, normalize, ridge)
    empty = np.empty(0, np.uint32)
    with _ordered.Session(np.zeros((1, len(target)), np.uint8), target, empty, empty, **config) as session:
        library = install(session, actual_noise, ridge)
        np.testing.assert_array_equal(session.state()["descriptors"], descriptors)
        session.step()
        state, result = session.state(), session.result()
    updates = np.float32(values * np.float32(config["learning_rate"]))
    expected = np.concatenate([np.float32(base[orders[p, :end]] + updates[index])
                               for index, (_, end, _, p) in enumerate(descriptors)])
    np.testing.assert_allclose(state["cursors"], expected, rtol=8e-6, atol=4e-6)
    np.testing.assert_allclose(result.leaf_values[0, 0], updates[-1], rtol=8e-6, atol=4e-6)
    assert actual_noise.events == expected_noise.events
    assert actual_noise.seeds == [0]
    if iterations == 1:
        assert actual_noise.events == [(1, len(tasks)), (2, len(tasks))]
    elif mode == "No":
        assert len(actual_noise.events) == 2 + 2 * iterations
    elif permutations == 4 and normalize:
        assert False in trace, "fixture must exercise rejected common steps"


@pytest.mark.parametrize("negative_hessian,zero_prefix", [(True, False), (False, True)])
def test_noisy_empty_or_nonpositive_diagonal_tasks_remain_zero(negative_hessian, zero_prefix):
    target, base, orders, descriptors, tasks, config = problem(1, 1, "No", True, zero_prefix)
    expected_noise, actual_noise = Noise(negative_hessian=negative_hessian), Noise(negative_hessian=negative_hessian)
    values, _ = walk(tasks, expected_noise, 1, "No", True, True)
    empty = np.empty(0, np.uint32)
    with _ordered.Session(np.zeros((1, len(target)), np.uint8), target, empty, empty, **config) as session:
        library = install(session, actual_noise, True)
        session.step()
        state = session.state()
    expected = np.concatenate([np.float32(base[orders[p, :end]] +
                                          np.float32(values[index] * np.float32(.17)))
                               for index, (_, end, _, p) in enumerate(descriptors)])
    np.testing.assert_allclose(state["cursors"], expected, rtol=8e-6, atol=4e-6)
    np.testing.assert_array_equal(values if negative_hessian else values[:1], 0)
    assert actual_noise.events == expected_noise.events
