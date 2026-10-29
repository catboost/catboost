"""Card7 additive leaf export: exact GPU cursor reconstruction and API lifecycle."""
import ctypes as ct
import platform

import numpy as np
import pytest

from catboost_metal import _greedy
from test_greedy_permutations import inputs, fixed
from test_greedy_simple_runtime import frozen_leaves
from test_greedy_training import POLICIES, route


pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64',
                               reason='Apple Silicon Metal required')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU CatBoost fitting is forbidden')

    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def capacity(session):
    p = session._params
    return min(p.max_leaves, p.depth + 1 if p.policy == 2 else 1 << min(p.depth, 16))


def export_call(session, *, count=None, stride=None, output=None, null=False, handle=None):
    count = session._permutation_count if count is None else count
    stride = capacity(session) if stride is None else stride
    output = np.full((max(count, 1), max(stride, 1)), 813.25, np.float32) if output is None else output
    call = session._lib.cbm_greedy_session_copy_last_permutation_leaves
    call.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_float), ct.c_char_p, ct.c_size_t]
    call.restype = ct.c_int
    error = ct.create_string_buffer(4096)
    code = call(session._handle if handle is None else handle, count, stride,
                None if null else output.ctypes.data_as(ct.POINTER(ct.c_float)), error, len(error))
    return code, error.value.decode(), output


def exported(session):
    code, error, output = export_call(session)
    assert code == 0, error
    return output


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('selected', [0, 2])
@pytest.mark.parametrize('case', [('RMSE', None, 'Newton'), ('Logloss', None, 'Gradient'),
                                  ('Quantile', .31, 'Exact')])
def test_all_history_leaves_reproduce_exact_cursor_updates(policy, selected, case):
    args, banks, cursors = inputs(policy, case, kind='Bernoulli', count=3)
    args.update(iterations=2, learning_rate=.19)
    with _greedy.Session(**args) as session:
        session.configure_permutations(banks, cursors)
        for iteration in range(2):
            before = session.permutation_state['predictions'].copy()
            session.select_permutation(selected if iteration == 0 else 2-selected)
            tree = session.step()
            dispatches = session._info().stats.kernel_dispatches
            values = exported(session)
            np.testing.assert_array_equal(exported(session), values)
            assert session._info().stats.kernel_dispatches == dispatches
            leaves = len(tree.leaf_values)
            np.testing.assert_array_equal(values[:, leaves:], 0)
            np.testing.assert_array_equal(values[-1, :leaves], tree.leaf_values)
            state = session.permutation_state['predictions']
            for p, bank in enumerate(banks):
                expected, _, ids = fixed(tree, bank, args, before[p])
                np.testing.assert_allclose(values[p, :leaves], expected, rtol=3e-4, atol=3e-6)
                np.testing.assert_array_equal(state[p], np.float32(before[p] + values[p, ids]))
            # Different cursors/banks must produce different per-history leaves.
            assert not np.array_equal(values[0, :leaves], values[1, :leaves])


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('selected', [0, 2])
def test_simple_retains_same_scaled_sampled_leaves_for_every_history(policy, selected):
    args, banks, cursors = inputs(policy, ('Logloss', None, 'Simple'), 'Bayesian', count=3)
    args.update(iterations=2, leaf_estimation_iterations=1, learning_rate=.23,
                iteration_offset=0, bagging_temperature=1., subsample=.71)
    with _greedy.Session(**args) as session:
        session.configure_permutations(banks, cursors)
        for iteration in range(2):
            before = session.permutation_state['predictions'].copy()
            session.select_permutation(selected)
            tree = session.step()
            values = exported(session)
            leaves = len(tree.leaf_values)
            expected, _ = frozen_leaves(args, before[selected], route(tree, banks[selected]), leaves, iteration)
            np.testing.assert_allclose(values[selected, :leaves], expected, rtol=5e-5, atol=3e-6)
            np.testing.assert_array_equal(values[:, :leaves], np.repeat(tree.leaf_values[None], 3, axis=0))
            for p in range(3):
                np.testing.assert_array_equal(session.permutation_state['predictions'][p],
                    np.float32(before[p] + values[p, route(tree, banks[p])]))
            np.testing.assert_array_equal(values[:, leaves:], 0)


@pytest.mark.parametrize('configured', [False, True])
def test_single_history_export_matches_public_tree(configured):
    args, banks, cursors = inputs(count=1)
    args['iterations'] = 1
    with _greedy.Session(**args) as session:
        if configured:
            session.configure_permutations(banks, cursors)
        tree = session.step()
        values = exported(session)
        np.testing.assert_array_equal(values[0, :len(tree.leaf_values)], tree.leaf_values)
        np.testing.assert_array_equal(values[0, len(tree.leaf_values):], 0)


def test_padding_is_cleared_when_last_tree_has_fewer_leaves():
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    args = dict(bins=bins, targets=np.array([1, 1, 2, 2], np.float32),
                candidate_features=np.array([0], np.uint32), candidate_bins=np.array([0], np.uint32),
                objective='RMSE', grow_policy='Depthwise', iterations=2, depth=3, max_leaves=8,
                learning_rate=1., l2_leaf_reg=0., score_function='L2')
    with _greedy.Session(**args) as session:
        session.configure_permutations(np.repeat(bins[None], 3, axis=0))
        first = session.step()
        assert len(first.leaf_values) == 2
        assert np.count_nonzero(exported(session)[:, 1]) == 3
        second = session.step()
        assert len(second.leaf_values) == 1
        values = exported(session)
        np.testing.assert_array_equal(values[:, 1:], 0)
        np.testing.assert_array_equal(values[:, 0], second.leaf_values[0])


@pytest.mark.parametrize('bad', ['before', 'count_low', 'count_high', 'stride_low', 'stride_high', 'null', 'closed'])
def test_invalid_export_is_atomic_and_does_not_damage_previous_tree(bad):
    args, banks, cursors = inputs(count=3)
    args['iterations'] = 2
    with _greedy.Session(**args) as session:
        session.configure_permutations(banks, cursors)
        valid = None
        if bad != 'before':
            session.step()
            valid = exported(session)
        count, stride = 3, capacity(session)
        kwargs = {}
        if bad == 'count_low': count -= 1
        if bad == 'count_high': count += 1
        if bad == 'stride_low': stride -= 1
        if bad == 'stride_high': stride += 1
        if bad == 'null': kwargs['null'] = True
        if bad == 'closed':
            kwargs['handle'] = ct.c_void_p(session._handle.value)
            session.close()
        sentinel = np.full((4, capacity(session)+1), 813.25, np.float32)
        code, error, output = export_call(session, count=count, stride=stride, output=sentinel, **kwargs)
        assert code != 0 and error
        np.testing.assert_array_equal(output, 813.25)
        if valid is not None and bad != 'closed':
            np.testing.assert_array_equal(exported(session), valid)
            session.step()  # A rejected read must not poison training.


def test_rejected_step_keeps_the_previous_export_usable():
    args, banks, cursors = inputs(count=3)
    with _greedy.Session(**args) as session:
        session.configure_permutations(banks, cursors)
        session.step()
        values = exported(session)
        error = ct.create_string_buffer(4096)
        code = session._lib.cbm_greedy_session_step(session._handle, None, None, None, None, error, len(error))
        assert code and b'output buffers' in error.value
        np.testing.assert_array_equal(exported(session), values)
        assert session._info().completed_iterations == 1


def test_langevin_exports_each_history_after_its_sequential_walker():
    from test_greedy_langevin import GreedyNoise
    from test_scalar_langevin import Noise, arguments, reference

    args = arguments(3, 'No')
    args.update(grow_policy='Region', depth=3, max_leaves=4, learning_rate=.37)
    baselines = np.repeat(np.array([-.2, .1, .3], np.float32)[:, None], len(args['targets']), axis=1)
    expected_noise = Noise()
    expected = np.array([reference(args['targets'], args['sample_weight'], baselines[p:p+1],
        expected_noise, 3, 'No', False, False)[0] for p in range(3)], np.float32)
    expected = np.float32(expected * np.float32(args['learning_rate']))
    with _greedy.Session(**args) as session:
        session.configure_permutations(np.zeros((3, 1, len(args['targets'])), np.uint8), baselines)
        session.select_permutation(2)
        actual_noise = GreedyNoise()
        actual_noise.install(session)
        tree = session.step()
        values = exported(session)
        np.testing.assert_allclose(values[:, 0], expected, rtol=8e-6, atol=4e-6)
        np.testing.assert_array_equal(values[:, 1:], 0)
        np.testing.assert_array_equal(values[-1, :len(tree.leaf_values)], tree.leaf_values)
        np.testing.assert_array_equal(session.permutation_state['predictions'],
                                      np.float32(baselines + values[:, :1]))
        assert actual_noise.events == expected_noise.events


def test_later_history_failure_does_not_publish_a_partial_tree():
    from test_greedy_langevin import GreedyNoise
    from test_scalar_langevin import arguments

    args = arguments(1)
    args.update(iterations=2)
    with _greedy.Session(**args) as session:
        session.configure_permutations(np.zeros((3, 1, len(args['targets'])), np.uint8))
        noise = GreedyNoise(zero=True)
        calls = [0]

        @noise.noise_type
        def fail_later(context, event, count, output):
            calls[0] += 1
            if calls[0] == 9:  # First tree=6 calls; fail history1 of tree2.
                return 1
            np.ctypeslib.as_array(output, shape=(count,))[:] = 0
            return 0

        noise.callback = fail_later
        noise.install(session)
        session.step()
        saved = exported(session)
        with pytest.raises(RuntimeError, match='callback'):
            session.step()
        assert session._info().completed_iterations == 1
        code, error, values = export_call(session)
        assert code and 'successfully completed tree' in error
        np.testing.assert_array_equal(values, 813.25)
        assert saved.shape == (3, 1)


def test_prepared_yeti_tree_cannot_be_mistaken_for_completed_export():
    from test_greedy_yeti_runtime import inputs as yeti_inputs

    args = yeti_inputs(objective='YetiRank', iterations=2, leaf_estimation_iterations=1)
    with _greedy.Session(**args) as session:
        session.step()
        exported(session)
        error = ct.create_string_buffer(4096)
        seed = ct.c_uint64(17011)
        assert not session._lib.cbm_greedy_session_set_yeti_oracle_seeds(
            session._handle, 1, ct.byref(seed), error, len(error)), error.value.decode()
        attempts = ct.c_uint32()
        assert not session._lib.cbm_greedy_session_prepare_yeti_tree(
            session._handle, ct.byref(attempts), error, len(error)), error.value.decode()
        code, error, values = export_call(session)
        assert code and 'prepared' in error
        np.testing.assert_array_equal(values, 813.25)
