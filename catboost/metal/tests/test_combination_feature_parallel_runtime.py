"""FeatureParallel Combination uses one leaf walker over every history.

Expected directions and losses come from independent component equations. The
oracle sums raw objectives across tasks before accepting a shared step, and its
stochastic stream is ordered by evaluation, history, then Yeti component.
"""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _native
from test_combination_runtime import (
    component, install_seed_callback, leaf_oracle, problem, session, stochastic_leaf_oracle, terms)
from test_pairwise_training import leaf_ids


pytestmark = pytest.mark.skipif(
    platform.system() != 'Darwin' or platform.machine() != 'arm64',
    reason='requires Apple Silicon Metal')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU CatBoost fitting is forbidden')

    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def small_problem():
    data = problem()
    for key in ('targets', 'weights', 'cursor'):
        data[key] = data[key][:67].copy()
    data['targets'] = np.clip(data['targets'], 0, 1)
    data['bins'] = data['bins'][:, :67].copy()
    data['offsets'] = np.array([0, 19, 40, 67], np.uint32)
    return data


def oracle(data, components, cursors, assignments, leaves, method, mode, iterations, *, weak_draws=0):
    """Optimize concatenated task coordinates using one scalar line search."""
    consumed = []

    def seeds():
        while True:
            value = 71093 + len(consumed) * 104729
            consumed.append(value)
            yield value

    stream = seeds()
    for _ in range(weak_draws):
        next(stream)
    masses = np.array([np.bincount(ids, weights=data['weights'], minlength=leaves)
                       for ids in assignments])
    point = np.zeros_like(masses, np.float32)
    evaluations = 0

    def evaluate(candidate):
        nonlocal evaluations
        evaluations += 1
        values, gradients, hessians = [], [], []
        for cursor, ids, shift in zip(cursors, assignments, candidate):
            g, h, _, loss, _ = terms(data, components, np.float32(cursor + shift[ids]), stream)
            values.append(-loss)
            gradients.append(np.bincount(ids, weights=g, minlength=leaves))
            hessians.append(np.bincount(ids, weights=h, minlength=leaves))
        return np.array(values), np.array(gradients), np.array(hessians)

    def direction(g, h):
        denominator = (masses if method == 'Gradient' else h) + data.get('l2', 2)
        return np.divide(g, denominator + 1e-20, out=np.zeros_like(g),
                         where=denominator > 0).astype(np.float32)

    def trial_point(step, delta):
        value = np.float32(point.astype(float) + step * delta.astype(float))
        value[masses < 1e-20] = 0
        return value

    values, gradient, hessian = evaluate(point)
    trace = []
    if mode == 'No' or iterations == 1:
        for _ in range(iterations):
            point = trial_point(1., direction(gradient, hessian))
            if iterations > 1:
                values, gradient, hessian = evaluate(point)
    else:
        attempts, updated, step, delta = 0, False, 1., None
        while attempts < iterations or (not updated and attempts < 100):
            if delta is None:
                delta = direction(gradient, hessian)
                dot = float(np.sum(gradient * delta))
            trial = trial_point(step, delta)
            next_values, next_gradient, next_hessian = evaluate(trial)
            threshold = values.sum() + (1e-5 * step * dot if mode == 'Armijo' else 0.)
            accepted = bool(np.isfinite(next_values.sum()) and next_values.sum() >= threshold)
            trace.append(dict(step=step, accepted=accepted, before=values.copy(), after=next_values.copy()))
            attempts += 1
            if accepted:
                point, values, gradient, hessian = trial, next_values, next_gradient, next_hessian
                updated, step, delta = True, 1., None
            else:
                step *= .5
    return np.float32(point * np.float32(.2)), masses, trace, consumed, evaluations


def final_cursors(cursors, assignments, values):
    return np.stack([np.float32(cursor + value[ids])
                     for cursor, ids, value in zip(cursors, assignments, values)])


def configure(model, banks, cursors):
    model.configure_permutations(banks, cursors)
    model.set_feature_activity(np.ones(banks.shape[1], np.uint8))
    model.select_permutation(len(banks) - 1)


def first_accepted(trace):
    return next(row['step'] for row in trace if row['accepted'])


@pytest.mark.parametrize('mode', ['AnyImprovement', 'Armijo'])
@pytest.mark.parametrize('histories', [2, 3])
def test_shared_step_uses_sum_of_all_history_objectives(mode, histories):
    data = problem()
    components = [component('Huber', .3, .3), component('Poisson', 1.7)]
    shifts = [-2., 3.] if histories == 2 else [-2., -2., 3.]
    cursors = np.stack([np.full_like(data['cursor'], shift) for shift in shifts])
    banks = np.stack([data['bins']] * histories)
    assignments = np.zeros(cursors.shape, np.uint32)
    expected, masses, trace, _, _ = oracle(
        data, components, cursors, assignments, 1, 'Newton', mode, 3)
    independent, independent_steps = [], []
    for cursor, ids in zip(cursors, assignments):
        values, _, individual, _, _ = oracle(
            data, components, cursor[None, :], ids[None, :], 1, 'Newton', mode, 3)
        reference, _ = leaf_oracle(data, components, cursor, ids, 1, 'Newton', mode, 3)
        np.testing.assert_array_equal(values[0], reference)
        independent.append(values[0])
        independent_steps.append(first_accepted(individual))
    assert independent_steps == ([.5, 1.] if histories == 2 else [.5, .5, 1.])
    assert first_accepted(trace) == (1. if histories == 2 else .5)
    assert np.max(np.abs(expected - independent)) > .1
    first_changes = trace[0]['after'] - trace[0]['before']
    assert first_changes[0] < -3900 and first_changes[-1] > 5500
    if histories == 2:
        # An individual task may worsen when the summed objective improves.
        assert trace[0]['accepted'] and first_changes.sum() > 1600
    else:
        # Duplicating that task changes the shared decision, not its direction.
        assert not trace[0]['accepted'] and first_changes.sum() < -2200
    with session(data, components, depth=0, iterations=1, leaf_iterations=3, backtracking=mode) as model:
        configure(model, banks, cursors)
        tree = model.step()
        assert tree.depth == 0
        np.testing.assert_allclose(tree.leaf_values, expected[-1], rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_weights, masses[-1], rtol=3e-6, atol=1e-5)
        # The last exported task alone cannot expose P2's difference: its first
        # task is the one changed by the global decision.
        np.testing.assert_allclose(model.permutation_state['predictions'],
                                   final_cursors(cursors, assignments, expected), rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('mode', ['No', 'Armijo'])
@pytest.mark.parametrize('iterations', [1, 3])
def test_stochastic_walker_callback_orders_evaluations_then_tasks(mode, iterations):
    data = small_problem()
    data['l2'] = 5
    components = [component('YetiRank', .3, permutations=3, decay=.8),
                  component('RMSE', 2), component('YetiRank', .5, permutations=5, decay=.9)]
    rng = np.random.default_rng(48711)
    banks = np.stack([data['bins'], *[rng.integers(0, 4, data['bins'].shape, dtype=np.uint8)
                                     for _ in range(2)]])
    cursors = np.stack([np.float32(data['cursor'] + shift) for shift in (0., -.7, .4)])
    with session(data, components, iterations=1, leaf_iterations=iterations, backtracking=mode) as model:
        configure(model, banks, cursors)
        callback, consumed = install_seed_callback(model)
        tree = model.step()
        assignments = np.stack([leaf_ids(tree, bank) for bank in banks])
        values, masses, _, expected_draws, evaluations = oracle(
            data, components, cursors, assignments, 1 << tree.depth, 'Newton', mode, iterations, weak_draws=2)
        assert consumed == expected_draws
        assert len(consumed) == 2 + evaluations * len(banks) * 2
        if mode == 'No' and iterations > 1:
            # Running complete per-history walks would consume the same number
            # of callbacks but assign them to different oracle evaluations.
            chunk = 2 * evaluations
            wrong = []
            for history, ids in enumerate(assignments):
                history_draws = expected_draws[:2] + expected_draws[2 + history * chunk:2 + (history + 1) * chunk]
                value, _ = stochastic_leaf_oracle(data | dict(cursor=cursors[history]), components,
                    ids, 1 << tree.depth, 'Newton', 'No', iterations, history_draws)
                wrong.append(value)
            assert not np.allclose(values[-1], wrong[-1], rtol=8e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_values, values[-1], rtol=8e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_weights, masses[-1], rtol=3e-6, atol=1e-5)
        np.testing.assert_allclose(model.permutation_state['predictions'],
                                   final_cursors(cursors, assignments, values), rtol=8e-5, atol=3e-6)
        assert abs(values[-1].mean()) > .001  # Combination does not center Yeti components as top-level YetiRank.
        assert callback is not None


@pytest.mark.parametrize('activity_first', [False, True])
@pytest.mark.parametrize('depth', [0, 1])
def test_dynamic_simple_maps_to_one_gradient_step_and_draws_each_task(activity_first, depth):
    data = small_problem()
    components = [component('YetiRank', .03, permutations=3, decay=.8),
                  component('RMSE', 2), component('YetiRank', .05, permutations=5, decay=.9)]
    rng = np.random.default_rng(472)
    banks = np.stack([data['bins'], *[rng.integers(0, 4, data['bins'].shape, dtype=np.uint8)
                                     for _ in range(3)]])
    cursors = np.stack([np.float32(data['cursor'] + .2 * history) for history in range(4)])
    with session(data, components, method='Gradient', depth=depth, iterations=1, leaf_iterations=1) as model:
        model.configure_permutations(banks, cursors)
        active = np.ones(banks.shape[1], np.uint8)
        if activity_first:
            model.set_feature_activity(active)
        options = _native.ObjectiveOptions(19, 3, 0, 0)  # Private Simple enum.
        error = ct.create_string_buffer(2048)
        model._check(model._lib.cbm_session_set_objective(
            model._handle, ct.byref(options), error, len(error)), error)
        if not activity_first:
            model.set_feature_activity(active)
        model.select_permutation(2)
        # Supply only the weak draws. Even a depth-zero tree must consume
        # these before replacing the weak packet with a leaf callback.
        weak = np.array([876123, 918723], np.uint64)
        setter = model._lib.cbm_session_set_combination_yeti_seeds
        setter.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint64), ct.c_char_p, ct.c_size_t]
        setter.restype = ct.c_int
        model._check(setter(model._handle, len(weak), weak.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                            error, len(error)), error)
        model.begin_tree()
        while not model.grow_tree()['finished']:
            pass
        callback, consumed = install_seed_callback(model)
        tree = model.finish_tree()
        assignments = np.stack([leaf_ids(tree, bank) for bank in banks])
        values, masses, _, expected_draws, evaluations = oracle(
            data, components, cursors, assignments, 1 << tree.depth, 'Gradient', 'No', 1)
        assert evaluations == 1
        assert consumed == expected_draws and len(consumed) == len(banks) * 2
        np.testing.assert_allclose(tree.leaf_values, values[-1], rtol=8e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_weights, masses[-1], rtol=3e-6, atol=1e-5)
        np.testing.assert_allclose(model.permutation_state['predictions'],
                                   final_cursors(cursors, assignments, values), rtol=8e-5, atol=3e-6)
        assert callback is not None
