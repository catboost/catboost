"""Greedy classic YetiRank against independent seeded CUDA target equations.

These tests exercise the counted private C ABI directly. Public controllers own
PFound and RNG persistence; the runtime consumes explicit weak/leaf packets and
reports the stochastic oracle's zero scalar loss.
"""
from contextlib import contextmanager
import ctypes as ct
import platform
from types import SimpleNamespace

import numpy as np
import pytest

from catboost_metal import _greedy
from catboost_metal._native import BootstrapOptions, ObjectiveOptions, ScoreNoiseOptions
from catboost_metal._yeti import YetiOptions
from cuda_reference import _score_children
from test_greedy_training import POLICIES, route
from test_yeti_rank_training import oracle_leaves, problem, terms


pytestmark = pytest.mark.skipif(
    platform.system() != 'Darwin' or platform.machine() != 'arm64',
    reason='requires Apple Silicon Metal')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU CatBoost fitting is forbidden')

    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(policy='Lossguide', **extra):
    return problem(**(dict(grow_policy=policy, max_leaves=6, depth=3,
                          random_seed=2**63 + 817) | extra))


def seed_packet(args, iteration, count=1):
    chunk = args['leaf_estimation_iterations'] + int(args['leaf_estimation_iterations'] > 1)
    return [0xdef0123400000011 + iteration * 97 + j for j in range(1 + count * chunk)]


def assert_same_tree(actual, expected):
    for key in ('nodes', 'leaf_values', 'leaf_weights'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    assert actual.loss == expected.loss == 0.


@pytest.fixture(scope='module')
def runtime():
    lib = _greedy._load(_greedy.build_library())
    u8, u32, u64, f32 = (ct.POINTER(kind) for kind in
                         (ct.c_uint8, ct.c_uint32, ct.c_uint64, ct.c_float))
    tail = [ct.c_char_p, ct.c_size_t]
    lib.cbm_greedy_session_create_yeti.argtypes = [
        ct.POINTER(_greedy.Params), ct.POINTER(ObjectiveOptions), ct.POINTER(YetiOptions),
        u32, ct.c_uint64, u8, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + tail
    lib.cbm_greedy_session_create_yeti.restype = ct.c_int
    for name in ('set_yeti_oracle_seeds', 'set_yeti_leaf_seeds'):
        function = getattr(lib, 'cbm_greedy_session_' + name)
        function.argtypes = [ct.c_void_p, ct.c_uint32, u64] + tail
        function.restype = ct.c_int
    lib.cbm_greedy_session_prepare_yeti_tree.argtypes = [ct.c_void_p, u32] + tail
    lib.cbm_greedy_session_prepare_yeti_tree.restype = ct.c_int

    def ptr(array, kind):
        return array.ctypes.data_as(kind)

    class Session:
        def __init__(self, args):
            self.handle = ct.c_void_p()
            self.count = 1
            self.error = ct.create_string_buffer(4096)
            self.bins = np.ascontiguousarray(args['bins'], np.uint8)
            self.rows = self.bins.shape[1]
            self.max_leaves = args['max_leaves']
            cf = np.ascontiguousarray(args['candidate_features'], np.uint32)
            cb = np.ascontiguousarray(args['candidate_bins'], np.uint32)
            types = np.ascontiguousarray(args.get('candidate_types', np.zeros(len(cf))), np.uint8)
            grid = int(self.bins.max()) + 1
            if len(cb):
                grid = max(grid, int((cb + np.where(types, 1, 2)).max()))
            p = _greedy.Params(
                self.rows, len(self.bins), len(cf), grid, args['iterations'], args['depth'],
                self.max_leaves, args.get('min_data_in_leaf', 1), POLICIES.index(args['grow_policy']),
                17, _greedy.SCORES.index(args['score_function']), 0, args['leaf_estimation_iterations'],
                0, 0, 0, args['learning_rate'], args['l2_leaf_reg'], 0., 0)
            objective = ObjectiveOptions(17, 0, 1., 0)
            offsets = np.ascontiguousarray(args['group_offsets'], np.uint32)
            yeti = YetiOptions(len(offsets) - 1, args['permutations'], args['decay'],
                               args.get('legacy_prefix_centering', False))
            y, w, cursor = [np.ascontiguousarray(args[key], np.float32) for key in
                            ('targets', 'sample_weight', 'initial_predictions')]
            self.check(lib.cbm_greedy_session_create_yeti(
                ct.byref(p), ct.byref(objective), ct.byref(yeti), ptr(offsets, u32), len(offsets),
                ptr(self.bins, u8), ptr(y, f32), ptr(w, f32), ptr(cursor, f32),
                ptr(cf, u32), ptr(cb, u32), ptr(types, u8), ct.byref(self.handle),
                self.error, len(self.error)))
            try:
                seed = args['random_seed']
                bootstrap = BootstrapOptions(
                    ('No', 'Bayesian', 'Bernoulli', 'Poisson').index(args.get('bootstrap_type', 'No')),
                    seed & 0xffffffff, seed >> 32, args.get('iteration_offset', 0),
                    args.get('bagging_temperature', 1.), args.get('subsample', 1.), 0., 0, 0., 0, 0, 0)
                noise = ScoreNoiseOptions(args.get('random_strength', 0.), 0, 0, 0)
                self.check(lib.cbm_greedy_session_set_bootstrap(
                    self.handle, ct.byref(bootstrap), self.error, len(self.error)))
                self.check(lib.cbm_greedy_session_set_score_noise(
                    self.handle, ct.byref(noise), self.error, len(self.error)))
            except Exception:
                lib.cbm_greedy_session_close(self.handle)
                raise

        def check(self, code):
            if code:
                raise RuntimeError(self.error.value.decode())

        def set_seeds(self, seeds, *, leaf_only=False):
            seeds = np.ascontiguousarray(seeds, np.uint64)
            name = 'set_yeti_leaf_seeds' if leaf_only else 'set_yeti_oracle_seeds'
            self.check(getattr(lib, 'cbm_greedy_session_' + name)(
                self.handle, len(seeds), ptr(seeds, u64), self.error, len(self.error)))

        def prepare(self):
            attempts = ct.c_uint32()
            self.check(lib.cbm_greedy_session_prepare_yeti_tree(
                self.handle, ct.byref(attempts), self.error, len(self.error)))
            return attempts.value

        def step(self, seeds=None, *, staged=False):
            if seeds is not None:
                self.set_seeds(seeds[:1] if staged else seeds)
            if staged:
                self.prepare()
                self.set_seeds(seeds[1:], leaf_only=True)
            nodes = np.empty((2 * self.max_leaves - 1, 6), np.uint32)
            values, weights = np.empty(self.max_leaves, np.float32), np.empty(self.max_leaves, np.float32)
            info = _greedy.StepInfo()
            self.check(lib.cbm_greedy_session_step(
                self.handle, ct.byref(info), nodes.ctypes.data_as(ct.POINTER(_greedy.Node)),
                ptr(values, f32), ptr(weights, f32), self.error, len(self.error)))
            return SimpleNamespace(nodes=nodes[:info.node_count].copy(),
                                   leaf_values=values[:info.leaf_count].copy(),
                                   leaf_weights=weights[:info.leaf_count].copy(), loss=info.loss,
                                   completed_iterations=info.completed_iterations, finished=bool(info.finished))

        def configure(self, banks, cursors=None):
            banks = np.ascontiguousarray(banks, np.uint8)
            self.count = len(banks)
            pointers = (u8 * self.count)(*(ptr(bank, u8) for bank in banks))
            initial = None
            if cursors is not None:
                cursors = np.ascontiguousarray(cursors, np.float32)
                initial = (f32 * self.count)(*(ptr(cursor, f32) for cursor in cursors))
            self.check(lib.cbm_greedy_session_set_permutations(
                self.handle, self.count, pointers, initial, None, None, self.error, len(self.error)))

        def select(self, index):
            self.check(lib.cbm_greedy_session_select_permutation(
                self.handle, index, self.error, len(self.error)))

        def cursors(self):
            cursors = np.empty((self.count, self.rows), np.float32)
            lambdas, valid = np.empty(self.count, np.float32), np.empty(self.count, np.uint8)
            self.check(lib.cbm_greedy_session_copy_permutation_state(
                self.handle, self.count, ptr(cursors, f32), ptr(lambdas, f32), ptr(valid, u8),
                self.error, len(self.error)))
            np.testing.assert_array_equal(lambdas, 0)
            np.testing.assert_array_equal(valid, 0)
            return cursors

    @contextmanager
    def create(args):
        session = Session(args)
        try:
            yield session
        finally:
            lib.cbm_greedy_session_close(session.handle)

    return create


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('leaf_iterations', [1, 3])
@pytest.mark.parametrize('legacy', [False, True])
def test_each_greedy_leaf_matches_explicit_yeti_oracle(runtime, policy, leaf_iterations, legacy):
    args = inputs(policy, leaf_estimation_iterations=leaf_iterations, legacy_prefix_centering=legacy)
    cursor = args['initial_predictions'].copy()
    with runtime(args) as session:
        for iteration in range(args['iterations']):
            seeds = seed_packet(args, iteration)
            tree = session.step(seeds)
            ids = route(tree, args['bins'])
            values, weights = oracle_leaves(args, cursor, ids, len(tree.leaf_values),
                                            seeds[1:1 + leaf_iterations])
            np.testing.assert_allclose(tree.leaf_values, values, rtol=2e-4, atol=2e-6)
            np.testing.assert_allclose(tree.leaf_weights, weights, rtol=4e-6, atol=1e-5)
            assert abs(tree.leaf_values.mean(dtype=float)) < 1e-7
            assert tree.loss == 0.
            assert tree.completed_iterations == iteration + 1
            assert tree.finished == (iteration == args['iterations'] - 1)
            cursor = np.float32(cursor + tree.leaf_values[ids])
            np.testing.assert_array_equal(session.cursors()[0], cursor)


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('score', ['L2', 'Cosine', 'NewtonL2', 'NewtonCosine'])
def test_root_scores_use_incident_mass_for_all_score_families(runtime, policy, score):
    args = inputs(policy, iterations=1, depth=1, max_leaves=2, score_function=score)
    rng = np.random.default_rng(98104)
    args['targets'] = rng.uniform(0, 1, len(args['targets'])).astype(np.float32)
    args['sample_weight'] = np.exp(rng.normal(0, 1.5, len(args['targets']))).astype(np.float32)
    args['sample_weight'][::17] = 0
    types = np.asarray(args['candidate_features'] == 2, np.uint8)
    args['candidate_types'] = types
    seeds = seed_packet(args, 0)
    gradient, mass = terms(args, args['initial_predictions'], seeds[0]).astype(float).T
    name = score.removeprefix('Newton')

    def winner_with(denominator):
        before = _score_children([gradient.sum()], [denominator.sum()], args['l2_leaf_reg'], name)
        gains = []
        for feature, border, kind in zip(args['candidate_features'], args['candidate_bins'], types):
            right = args['bins'][feature] == border if kind else args['bins'][feature] > border
            masses = [denominator[~right].sum(), denominator[right].sum()]
            gains.append(0. if min(masses) < 1e-20 else _score_children(
                [gradient[~right].sum(), gradient[right].sum()], masses, args['l2_leaf_reg'], name) - before)
        winner = int(np.argmin(gains))
        assert gains[winner] < 0
        return winner

    winner = winner_with(mass)
    # This fixture must fail if non-Newton scoring accidentally falls back to
    # the original observation weights as it does for scalar objectives.
    assert winner != winner_with(args['sample_weight'].astype(float))
    with runtime(args) as session:
        tree = session.step(seeds)
    assert tree.nodes[0, :3].tolist() == [int(args['candidate_features'][winner]),
                                        int(args['candidate_bins'][winner]), int(types[winner])]


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('leaf_iterations', [1, 3])
@pytest.mark.parametrize('bootstrap', ['No', 'Bayesian'])
def test_history_seed_chunks_follow_dataset_order_and_resume_exactly(runtime, policy, leaf_iterations, bootstrap):
    args = inputs(policy, iterations=4, leaf_estimation_iterations=leaf_iterations,
                  bootstrap_type=bootstrap, random_strength=.7)
    rng = np.random.default_rng(44173)
    banks = np.stack([args['bins'], *[rng.integers(0, 4, args['bins'].shape, dtype=np.uint8)
                                     for _ in range(3)]])
    cursors = np.tile(args['initial_predictions'], (4, 1))
    selection = [3, 1, 0, 2]  # Includes final-bank search, whose exported leaves must survive later banks.
    chunk = leaf_iterations + int(leaf_iterations > 1)
    trees = []
    with runtime(args) as session:
        session.configure(banks, cursors)
        for iteration, selected in enumerate(selection):
            seeds = seed_packet(args, iteration, 4)
            session.select(selected)
            # Check structure with a separate single-history search using the
            # identical weak seed and absolute bootstrap/noise iteration.
            search_args = args | dict(bins=banks[selected], initial_predictions=session.cursors()[selected],
                                      iterations=1, iteration_offset=iteration)
            with runtime(search_args) as search:
                expected = search.step([seeds[0], *seeds[1 + selected * chunk:1 + (selected + 1) * chunk]])
            tree = session.step(seeds, staged=bool(iteration % 2))
            np.testing.assert_array_equal(tree.nodes, expected.nodes)
            trees.append(tree)
            for history in range(4):
                ids = route(tree, banks[history])
                values, weights = oracle_leaves(args, cursors[history], ids, len(tree.leaf_values),
                    seeds[1 + history * chunk:1 + history * chunk + leaf_iterations])
                cursors[history] = np.float32(cursors[history] + values[ids])
                if history == 3:
                    np.testing.assert_allclose(tree.leaf_values, values, rtol=3e-4, atol=3e-6)
                    np.testing.assert_allclose(tree.leaf_weights, weights, rtol=4e-6, atol=1e-5)
            np.testing.assert_allclose(session.cursors(), cursors, rtol=5e-4, atol=5e-6)
            if iteration == 1:
                checkpoint = session.cursors()
        final = session.cursors()
    resumed_args = args | dict(iterations=2, iteration_offset=2, initial_predictions=checkpoint[-1])
    with runtime(resumed_args) as resumed:
        resumed.configure(banks, checkpoint)
        for iteration in (2, 3):
            resumed.select(selection[iteration])
            assert_same_tree(resumed.step(seed_packet(args, iteration, 4), staged=True), trees[iteration])
        np.testing.assert_array_equal(resumed.cursors(), final)


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('boundary', ['depth_zero', 'one_leaf', 'empty_candidates', 'normal'])
def test_two_phase_packet_matches_direct_step_at_topology_boundaries(runtime, policy, boundary):
    args = inputs(policy, iterations=1, random_strength=.7)
    if boundary == 'depth_zero':
        args['depth'] = 0
    elif boundary == 'one_leaf':
        args['max_leaves'] = 1
    elif boundary == 'empty_candidates':
        args.update(candidate_features=np.empty(0, np.uint32), candidate_bins=np.empty(0, np.uint32))
    seeds = seed_packet(args, 0)
    with runtime(args) as direct:
        expected = direct.step(seeds)
        prediction = direct.cursors()
    with runtime(args) as staged:
        staged.set_seeds(seeds[:1])
        attempts = staged.prepare()
        if boundary == 'empty_candidates':
            assert attempts == 0
        elif boundary in ('depth_zero', 'one_leaf'):
            # CUDA consumes its root scorer draw before capacity/termination.
            assert attempts == 1
        else:
            assert 1 <= attempts <= args['max_leaves']
        # An incomplete tree is not a resumable checkpoint. Cursor export is
        # guarded until the leaf packet completes the prepared topology.
        with pytest.raises(RuntimeError, match='prepared'):
            staged.cursors()
        staged.set_seeds(seeds[1:], leaf_only=True)
        assert_same_tree(staged.step(), expected)
        np.testing.assert_array_equal(staged.cursors(), prediction)


@pytest.mark.parametrize('case', ['missing_weak', 'short_full', 'leaves_before_prepare',
                                  'short_leaves', 'replace_weak_after_prepare'])
def test_malformed_seed_handoffs_reject_without_advancing_cursor(runtime, case):
    args = inputs(iterations=1)
    seeds = seed_packet(args, 0)
    with runtime(args) as session:
        if case == 'missing_weak':
            operation = session.prepare
        elif case == 'short_full':
            operation = lambda: session.set_seeds(seeds[:-1])
        elif case == 'leaves_before_prepare':
            operation = lambda: session.set_seeds(seeds[1:], leaf_only=True)
        else:
            session.set_seeds(seeds[:1])
            session.prepare()
            operation = (lambda: session.set_seeds(seeds[1:-1], leaf_only=True)) if case == 'short_leaves' else (
                lambda: session.set_seeds(seeds[:1]))
        with pytest.raises(RuntimeError):
            operation()
        if case in ('short_leaves', 'replace_weak_after_prepare'):
            with pytest.raises(RuntimeError, match='prepared'):
                session.cursors()
            session.set_seeds(seeds[1:], leaf_only=True)
            actual = session.step()
        else:
            np.testing.assert_array_equal(session.cursors()[0], args['initial_predictions'])
            actual = session.step(seeds)
    with runtime(args) as clean:
        assert_same_tree(actual, clean.step(seeds))


@pytest.mark.parametrize('policy', POLICIES)
def test_empty_history_leaf_coordinates_participate_in_zero_average(runtime, policy):
    args = inputs(policy, iterations=1, score_function='L2')
    # The estimation bank has a different partition, including empty leaves.
    # Retain two occupied leaves so the uncentered walk has a nonzero mean.
    with runtime(args) as search:
        structure = search.step(seed_packet(args, 0))
    search_ids = route(structure, args['bins'])
    leaves = np.unique(search_ids)
    assert len(leaves) > 2
    rows = [np.flatnonzero(search_ids == leaf)[0] for leaf in leaves[:2]]
    repeated_rows = np.resize(rows, args['bins'].shape[1])
    banks = np.stack([args['bins'], args['bins'][:, repeated_rows]])
    seeds = seed_packet(args, 0, 2)
    with runtime(args) as session:
        session.configure(banks)
        session.select(0)
        tree = session.step(seeds)
    np.testing.assert_array_equal(tree.nodes, structure.nodes)
    ids = route(tree, banks[1])
    chunk = args['leaf_estimation_iterations'] + 1
    values, weights = oracle_leaves(args, args['initial_predictions'], ids, len(tree.leaf_values),
                                    seeds[1 + chunk:1 + chunk + args['leaf_estimation_iterations']])
    empty = weights == 0
    assert empty.any() and (~empty).sum() >= 2
    assert np.max(np.abs(values[empty])) > 1e-7
    np.testing.assert_allclose(tree.leaf_values, values, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(tree.leaf_weights, weights, rtol=4e-6, atol=1e-5)
    assert abs(tree.leaf_values.mean(dtype=float)) < 1e-7


@pytest.mark.parametrize('policy', POLICIES)
def test_final_evaluation_seed_draw_is_consumed_without_changing_leaf_walk(runtime, policy):
    args = inputs(policy, iterations=1)
    seeds = seed_packet(args, 0)
    changed = seeds[:-1] + [seeds[-1] ^ 0xfedcba9876543210]
    with runtime(args) as first:
        expected = first.step(seeds)
        cursor = first.cursors()
    with runtime(args) as second:
        assert_same_tree(second.step(changed, staged=True), expected)
        np.testing.assert_array_equal(second.cursors(), cursor)
