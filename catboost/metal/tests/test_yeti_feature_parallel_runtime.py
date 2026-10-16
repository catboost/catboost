"""FeatureParallel YetiRank packets use evaluation-major, then history-major seeds.

The resident leaf walker still estimates one complete history at a time. These
tests check that activating the dynamic feature path transposes caller packets
at both seed handoffs, while the legacy DocParallel packet remains unchanged.
"""
import platform

import numpy as np
import pytest

from catboost_metal import _yeti
from test_pairwise_training import leaf_ids
from test_yeti_rank_training import oracle_leaves, problem


pytestmark = pytest.mark.skipif(
    platform.system() != 'Darwin' or platform.machine() != 'arm64',
    reason='requires Apple Silicon Metal')


@pytest.fixture(autouse=True)
def no_cpu_fit(monkeypatch):
    from catboost import CatBoost

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU CatBoost fitting is forbidden')

    monkeypatch.setattr(CatBoost, '_fit', forbidden)


def inputs(leaf_iterations=3, iterations=3):
    args = problem(iterations=iterations, leaf_estimation_iterations=leaf_iterations)
    rng = np.random.default_rng(44173)
    banks = np.stack([args['bins'], *[rng.integers(0, 4, args['bins'].shape, dtype=np.uint8)
                                     for _ in range(3)]])
    cursors = np.stack([np.float32(args['initial_predictions'] + .13 * history)
                        for history in range(4)])
    return args, banks, cursors


def packet(leaf_iterations, iteration):
    evaluations = leaf_iterations + int(leaf_iterations > 1)
    base = 0xdef0123400000011 + iteration * 4099
    weak = base
    # Each row is one batched oracle evaluation across all four histories.
    leaves = np.array([[base + 1 + evaluation * 131 + history * 17 for history in range(4)]
                        for evaluation in range(evaluations)], np.uint64)
    return weak, leaves


def advance(session, weak, leaves, staged):
    if not staged:
        return session.step([weak, *leaves])
    session.begin_tree([weak])
    while not session.grow_tree()['finished']:
        pass
    return session.finish_tree(leaves)


def same_structure(actual, expected):
    assert actual.depth == expected.depth
    for key in ('split_features', 'split_bins', 'split_types'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))


@pytest.mark.parametrize('leaf_iterations', [1, 3])
@pytest.mark.parametrize('selected', [1, 3])
@pytest.mark.parametrize('staged', [False, True])
def test_dynamic_yeti_packets_match_each_history_oracle_and_legacy_doc_order(leaf_iterations, selected, staged):
    args, banks, cursors = inputs(leaf_iterations)
    with _yeti.Session(**args) as dynamic, _yeti.Session(**args) as doc:
        dynamic.configure_permutations(banks, cursors)
        doc.configure_permutations(banks, cursors)
        # Activity enables FeatureParallel before either setter sees a packet.
        dynamic.set_feature_activity(np.ones(banks.shape[1], np.uint8))
        for iteration in range(args['iterations']):
            weak, evaluations = packet(leaf_iterations, iteration)
            dynamic.select_permutation(selected)
            doc.select_permutation(selected)
            feature_packet = evaluations.ravel().tolist()
            doc_packet = [int(evaluations[evaluation, history]) for history in range(4)
                          for evaluation in range(len(evaluations))]
            actual = advance(dynamic, weak, feature_packet, staged)
            legacy = advance(doc, weak, doc_packet, staged)
            same_structure(actual, legacy)
            np.testing.assert_allclose(actual.leaf_values, legacy.leaf_values, rtol=2e-5, atol=3e-7)
            np.testing.assert_allclose(dynamic.permutation_state['predictions'],
                                       doc.permutation_state['predictions'], rtol=2e-5, atol=3e-7)
            if leaf_iterations > 1:
                assert feature_packet != doc_packet
            wrong_order_difference = 0.
            for history in range(4):
                ids = leaf_ids(actual, banks[history])
                values, weights = oracle_leaves(args, cursors[history], ids, 1 << actual.depth,
                                                evaluations[:leaf_iterations, history])
                if iteration == 0 and leaf_iterations > 1:
                    # Prove this fixture detects a missing transpose. Reading
                    # evaluation-major input as contiguous history chunks must
                    # produce observably different leaf values.
                    start = history * len(evaluations)
                    wrong, _ = oracle_leaves(args, cursors[history], ids, 1 << actual.depth,
                                              feature_packet[start:start + leaf_iterations])
                    wrong_order_difference = max(wrong_order_difference, float(np.max(np.abs(values - wrong))))
                cursors[history] = np.float32(cursors[history] + values[ids])
                if history == 3:
                    np.testing.assert_allclose(actual.leaf_values, values, rtol=2e-4, atol=2e-6)
                    np.testing.assert_allclose(actual.leaf_weights, weights, rtol=4e-6, atol=1e-5)
            if iteration == 0 and leaf_iterations > 1:
                assert wrong_order_difference > 1e-4
            np.testing.assert_allclose(dynamic.permutation_state['predictions'], cursors, rtol=3e-4, atol=3e-6)
            assert actual.loss == legacy.loss == 0.


@pytest.mark.parametrize('staged', [False, True])
def test_dynamic_final_evaluation_row_is_unused_for_every_history(staged):
    args, banks, cursors = inputs(iterations=1)
    weak, evaluations = packet(args['leaf_estimation_iterations'], 0)
    changed = evaluations.copy()
    changed[-1] ^= np.uint64(0xfedcba9876543210)
    with _yeti.Session(**args) as first, _yeti.Session(**args) as second:
        for session in (first, second):
            session.configure_permutations(banks, cursors)
            session.set_feature_activity(np.ones(banks.shape[1], np.uint8))
            session.select_permutation(2)
        expected = advance(first, weak, evaluations.ravel().tolist(), staged)
        actual = advance(second, weak, changed.ravel().tolist(), staged)
        same_structure(actual, expected)
        for key in ('leaf_values', 'leaf_weights'):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
        np.testing.assert_array_equal(second.permutation_state['predictions'],
                                      first.permutation_state['predictions'])
