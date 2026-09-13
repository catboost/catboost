"""Private additive Ordered ABI: retained trees, bank growth and feature metadata."""
import ctypes as ct

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._native import StepInfo, _u8, _u32, _f32
from test_ordered_feature_banks import bank_problem
from test_ordered_training import apple_silicon, prohibit_cpu_training


class Structure(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ('depth', 'finished', 'has_split', 'feature', 'bin', 'type')]
    _fields_ += [('score', ct.c_float), ('gain', ct.c_float)]


class Append(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ('permutation_count', 'features', 'candidates', 'bins_per_feature')]
    _fields_ += [('reserved', ct.c_uint32 * 4)]


class Runtime:
    def __init__(self, session):
        self.session, self.lib, self.features = session, session._lib, session._params.features
        u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
        error = [ct.c_char_p, ct.c_size_t]
        signatures = {
            'begin_tree': [ct.c_void_p, ct.c_uint32],
            'grow_tree': [ct.c_void_p, ct.POINTER(Structure)],
            'finish_tree': [ct.c_void_p, ct.POINTER(StepInfo), u32, u32, u32, u8, f32, f32],
            'append_features': [ct.c_void_p, ct.POINTER(Append), ct.POINTER(u8), u32, u32, u8, u32, f32, u8, u8, u32],
            'set_feature_activity': [ct.c_void_p, ct.c_uint32, u8],
            'restore_feature_metadata': [ct.c_void_p, ct.c_uint32, u8, u8, u8],
            'copy_feature_metadata': [ct.c_void_p, ct.c_uint32, u32, f32, u8, u8, u8],
        }
        for name, signature in signatures.items():
            getattr(self.lib, 'cbm_ordered_session_' + name).argtypes = signature + error

    def call(self, name, *args):
        error = ct.create_string_buffer(4096)
        result = getattr(self.lib, 'cbm_ordered_session_' + name)(self.session._handle, *args, error, len(error))
        if result:
            raise RuntimeError(error.value.decode())

    def begin(self, selected=0):
        self.call('begin_tree', selected)

    def grow(self):
        info = Structure()
        self.call('grow_tree', ct.byref(info))
        return info

    def finish(self):
        p = self.session._params
        depth, info = ct.c_uint32(), StepInfo()
        features, borders, types = np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint8)
        values, weights = np.zeros(1 << p.depth, np.float32), np.zeros(1 << p.depth, np.float32)
        self.call('finish_tree', ct.byref(info), ct.byref(depth), _u32(features), _u32(borders), _u8(types), _f32(values), _f32(weights))
        d = depth.value
        return dict(depth=d, split_features=features[:d], split_bins=borders[:d], split_types=types[:d],
                    leaf_values=values[:1 << d], leaf_weights=weights[:1 << d], loss=info.loss)

    def activity(self, active):
        active = np.asarray(active, np.uint8)
        self.call('set_feature_activity', len(active), _u8(active))

    def append(self, banks=None, counts=None, flags=None, used=None):
        if banks is None:
            options, first = Append(self.session._params.permutations, 0, 0, 256), ct.c_uint32()
            self.call('append_features', ct.byref(options), None, None, None, None, None, None, None, None, ct.byref(first))
            return first.value
        banks = np.ascontiguousarray(banks, np.uint8)
        count, features, rows = banks.shape
        cf = np.repeat(np.arange(features, dtype=np.uint32), 7)
        cb = np.tile(np.arange(7, dtype=np.uint32), features)
        pointers = (ct.POINTER(ct.c_uint8) * count)(*[_u8(bank) for bank in banks])
        options, first = Append(count, features, len(cf), 256), ct.c_uint32()
        counts = None if counts is None else np.asarray(counts, np.uint32)
        flags = None if flags is None else np.asarray(flags, np.uint8)
        used = None if used is None else np.asarray(used, np.uint8)
        self.call('append_features', ct.byref(options), pointers, _u32(cf), _u32(cb), None,
                  None if counts is None else _u32(counts), None, None if flags is None else _u8(flags),
                  None if used is None else _u8(used), ct.byref(first))
        self.features += features
        return first.value

    def metadata(self):
        counts, weights = np.empty(self.features, np.uint32), np.empty(self.features, np.float32)
        flags, used, active = (np.empty(self.features, np.uint8) for _ in range(3))
        self.call('copy_feature_metadata', self.features, _u32(counts), _f32(weights), _u8(flags), _u8(used), _u8(active))
        return counts, weights, flags, used, active


def complete(runtime):
    for _ in range(runtime.session._params.depth + 1):
        if runtime.grow().finished:
            return runtime.finish()
    pytest.fail('Ordered incremental structure did not terminate')


@pytest.mark.parametrize('sampler', ['No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'])
@pytest.mark.parametrize('backtracking', ['No', 'AnyImprovement', 'Armijo'])
@pytest.mark.parametrize('grouped', [False, True])
def test_incremental_simple_ordered_is_exact(sampler, backtracking, grouped):
    bins, targets, cf, cb, config = bank_problem(7, grouped, 'Logloss', bootstrap_type=sampler, subsample=.7,
        random_strength=.8, leaf_estimation_backtracking=backtracking, leaf_estimation_iterations=3,
        ctr_unique_values=[19, 3, 0], model_size_reg=2.)
    with _ordered.Session(bins, targets, cf, cb, **config) as legacy:
        expected = legacy.step()
        cursors, predictions = legacy.state()['cursors'], legacy.predictions()
    with _ordered.Session(bins, targets, cf, cb, **config) as session:
        runtime = Runtime(session)
        runtime.begin(expected.stats['search_permutation'])
        actual = complete(runtime)
        for name, value in actual.items():
            np.testing.assert_array_equal(value, getattr(expected, name))
        np.testing.assert_array_equal(session.predictions(), predictions)
        np.testing.assert_array_equal(session.state()['cursors'], cursors)


@pytest.mark.parametrize('sampler', ['No', 'Bayesian', 'MVS'])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('append_depth', [0, 1, 2, 4])
def test_mid_tree_append_preserves_selected_splits_and_all_banks(sampler, grouped, append_depth):
    bins, targets, cf, cb, config = bank_problem(7, grouped, 'Logloss', bootstrap_type=sampler, subsample=.7,
        random_strength=.5, leaf_estimation_backtracking='Armijo', leaf_estimation_iterations=3)
    config['depth'] = 4
    # Keep equality candidates for the existing second column; only column 2 is appended.
    with _ordered.Session(bins, targets, cf, cb, **config) as baseline:
        expected = Runtime(baseline)
        expected.activity([1, 1, 0]); expected.begin(2)
        for _ in range(append_depth):
            expected.grow()
        expected.activity([1, 1, 1]); tree = complete(expected)
        final_cursors, predictions = baseline.state()['cursors'], baseline.predictions()
    reduced = dict(config, candidate_types=config['candidate_types'][:14],
                   permutation_bins=np.ascontiguousarray(config['permutation_bins'][:, :2]))
    with _ordered.Session(np.ascontiguousarray(bins[:2]), targets, cf[:14], cb[:14], **reduced) as session:
        runtime = Runtime(session)
        runtime.append(); runtime.begin(2)
        for _ in range(append_depth):
            runtime.grow()
        assert runtime.append(config['permutation_bins'][:, 2:]) == 2
        actual = complete(runtime)
        for name in actual:
            np.testing.assert_array_equal(actual[name], tree[name])
        np.testing.assert_array_equal(session.predictions(), predictions)
        np.testing.assert_array_equal(session.state()['cursors'], final_cursors)


def test_exhaustion_reopens_but_state_cannot_be_copied_mid_tree():
    bins, targets, cf, cb, config = bank_problem()
    with _ordered.Session(bins, targets, cf, cb, **config) as session:
        runtime = Runtime(session)
        runtime.activity([0, 0, 0]); runtime.begin()
        assert runtime.grow().finished
        with pytest.raises(RuntimeError, match='between completed trees'):
            session.state()
        with pytest.raises(RuntimeError, match='between-tree'):
            runtime.metadata()
        runtime.activity([1, 1, 1])
        assert runtime.grow().has_split
        assert complete(runtime)['depth'] > 0


def test_invalid_append_keeps_session_usable_and_metadata_restores():
    bins, targets, cf, cb, config = bank_problem(ctr_unique_values=[19, 3, 0])
    with _ordered.Session(bins, targets, cf, cb, **config) as session:
        runtime = Runtime(session)
        with pytest.raises(RuntimeError, match='flags'):
            runtime.append(config['permutation_bins'][:, 2:], counts=[7], flags=[4])
        assert runtime.features == 3
        with pytest.raises(RuntimeError, match='globally registered'):
            runtime.append(config['permutation_bins'][:, 2:], counts=[7], flags=[1], used=[1])
        runtime.append(config['permutation_bins'][:, 2:], counts=[7], flags=[3], used=[1])
        metadata = runtime.metadata()
        np.testing.assert_array_equal(metadata[0], [19, 3, 0, 7])
        np.testing.assert_array_equal(metadata[2], [2, 2, 2, 3])
        np.testing.assert_array_equal(metadata[3], [0, 0, 0, 1])
        flags, used, active = (np.asarray(x, np.uint8) for x in ([2, 2, 2, 1], [0, 0, 0, 0], [1, 1, 1, 0]))
        runtime.call('restore_feature_metadata', 4, _u8(flags), _u8(used), _u8(active))
        np.testing.assert_array_equal(runtime.metadata()[2], flags)
        runtime.begin(); complete(runtime)
        # Static/simple CTR winners are never marked used by FeatureParallel.
        np.testing.assert_array_equal(runtime.metadata()[3], [0, 0, 0, 0])


@pytest.mark.parametrize('objective', ['Quantile', 'MAPE'])
def test_incremental_exact_finish_and_shared_bank_promotion(objective):
    bins, targets, cf, cb, config = bank_problem(4, True, objective, .6 if objective == 'Quantile' else None, 'Exact')
    banks = np.repeat(bins[None], 4, axis=0)
    config.pop('permutation_bins')
    with _ordered.Session(bins, targets, cf, cb, **config) as baseline:
        expected = baseline.step()
    with _ordered.Session(bins[:2].copy(), targets, cf[:14], cb[:14],
            **(config | dict(candidate_types=config['candidate_types'][:14]))) as session:
        runtime = Runtime(session)
        runtime.append(banks[:, 2:]); runtime.begin(expected.stats['search_permutation'])
        actual = complete(runtime)
        for name, value in actual.items():
            np.testing.assert_array_equal(value, getattr(expected, name))


@pytest.mark.parametrize('inactive_registered', [False, True])
def test_dynamic_denominators_and_retained_penalty_match_independent_scores(inactive_registered):
    from cuda_ordered_reference import _score_candidate
    rng = np.random.default_rng(7981)
    rows = 263
    bins = rng.integers(0, 2, (3, rows), dtype=np.uint8)
    targets = np.float32(1.2 * bins[0] + .9 * bins[1] + .01 * rng.normal(size=rows))
    cf, cb = np.arange(3, dtype=np.uint32), np.zeros(3, np.uint32)
    config = dict(iterations=1, depth=2, learning_rate=.1, l2_leaf_reg=3., bias=0.,
        min_fold_size=32, permutations=np.arange(rows, dtype=np.uint32)[None],
        ctr_unique_values=[100, 7, 1000], model_size_reg=.5)
    with _ordered.Session(bins, targets, cf, cb, **config) as session:
        runtime = Runtime(session)
        flags = np.asarray([2, 1, 3 if inactive_registered else 1], np.uint8)
        used, active = np.zeros(3, np.uint8), np.asarray([0, 1, 0], np.uint8)
        runtime.call('restore_feature_metadata', 3, _u8(flags), _u8(used), _u8(active))
        descriptors = session.state()['descriptors'][:-1]
        derivatives = [(targets[:int(end)].astype(np.float64), np.ones(int(end))) for _, end, _, _ in descriptors]
        partitions, previous = np.zeros(rows, np.int64), 0.
        runtime.begin()
        for level in range(2):
            # The inactive, unregistered transient never enters either maximum.
            # A used dynamic CTR remains penalized but exits dynamic_max.
            dynamic_max = 7 if level == 0 else 1
            static_max = 1000 if inactive_registered else 100
            candidate_scores = []
            for feature in ([1] if level == 0 else [0, 1]):
                raw = _score_candidate(bins, feature, 0, partitions, derivatives, descriptors,
                    config['permutations'], 3., False, 1 << level)
                maximum = static_max if feature == 0 else dynamic_max
                count = 100 if feature == 0 else 7
                factor = np.float32(float(np.float32(1 + np.float32(count) / np.float32(maximum))) ** -.5)
                score = np.float32(raw) * factor
                candidate_scores.append((float(score - previous), feature, float(score)))
            _, expected_feature, expected_score = min(candidate_scores)
            winner = runtime.grow()
            assert winner.has_split and winner.feature == expected_feature
            np.testing.assert_allclose(winner.score, expected_score, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(winner.gain, expected_score - previous, rtol=1e-5, atol=1e-5)
            partitions |= (bins[winner.feature] > 0).astype(np.int64) << level
            previous = winner.score
            if level == 0:
                runtime.activity([1, 1, 0])
        runtime.finish()
        np.testing.assert_array_equal(runtime.metadata()[3], [0, 1, 0])
