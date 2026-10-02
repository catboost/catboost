"""Check actual Ordered sessions consume the exact CUDA numeric host stream."""

import numpy as np
import pytest

from catboost_metal import _ordered
from test_ordered_rng import ReferenceMt64, reference_order, reference_steps
from test_ordered_training import apple_silicon, prohibit_cpu_training, dataset, options


def problem(mode):
    bins, targets, features, borders, weights = dataset("Logloss")
    config = options("Logloss", iterations=5, depth=3, sample_weight=weights)
    if mode == "no_candidates":
        features, borders = np.empty(0, np.uint32), np.empty(0, np.uint32)
    elif mode == "depth_zero":
        config["depth"] = 0
    elif mode == "repeated_split":
        bins = np.asarray([[0, 0, 1, 0, 1, 0, 1, 1, 1]], np.uint8)
        targets = bins[0].astype(np.float32)
        features, borders = np.asarray([0], np.uint32), np.asarray([0], np.uint32)
        config = options("Logloss", iterations=5, depth=4, bias=0)
    return bins, targets, features, borders, config


@pytest.mark.parametrize("mode", ["no_candidates", "depth_zero", "repeated_split", "normal"])
@pytest.mark.parametrize("permutation_count", [1, 4, 7])
def test_actual_session_mt_state_and_search_permutations_match_cuda_draws(mode, permutation_count):
    bins, targets, features, borders, config = problem(mode)
    config["permutation_count"] = permutation_count
    snapshots, selected, attempts, depths = [], [], [], []
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        initial = session.state()["selection_rng"]
        assert initial["words"] == ReferenceMt64(config["random_seed"]).words
        assert initial["index"] == 312 and not initial["bootstrap_initialized"]
        for _ in range(config["iterations"]):
            result = session.step()
            selected.append(result.stats["search_permutation"])
            depths.append(result.depth)
            count = min(config["depth"], result.depth + 1) if len(features) and config["depth"] else 0
            attempts.append(count)
            snapshots.append(session.state()["selection_rng"])
        np.testing.assert_array_equal(session.result().stats["search_permutations"], selected)
    expected = reference_steps(config["random_seed"], permutation_count, attempts)
    for index, (snapshot, (choice, words, position)) in enumerate(zip(snapshots, expected)):
        assert selected[index] == choice
        assert snapshot["words"] == words
        assert snapshot["index"] == position
        assert snapshot["completed_iterations"] == index + 1
        assert snapshot["bootstrap_initialized"] is True
    if mode in ("no_candidates", "depth_zero"):
        assert attempts == [0] * config["iterations"]
    if mode == "repeated_split":
        assert depths == [1] * config["iterations"]
        assert attempts == [2] * config["iterations"]
    if mode == "normal" and permutation_count in (1, 4):
        assert len(set(depths)) > 1  # Known first-tree repeat and later deeper searches.


@pytest.mark.parametrize("mode", ["no_candidates", "repeated_split", "normal"])
@pytest.mark.parametrize("permutation_count", [4, 7])
def test_session_snapshot_resumes_identical_mt_state_after_each_completed_tree(mode, permutation_count):
    bins, targets, features, borders, config = problem(mode)
    config["permutation_count"] = permutation_count
    with _ordered.Session(bins, targets, features, borders, **config) as whole:
        whole.step()
        whole.step()
        checkpoint = whole.state()
        with _ordered.Session(bins, targets, features, borders,
                              **dict(config, iterations=3, initial_state=checkpoint)) as restored:
            assert restored.state()["selection_rng"] == checkpoint["selection_rng"]
            for _ in range(3):
                first, second = whole.step(), restored.step()
                assert first.stats["search_permutation"] == second.stats["search_permutation"]
                np.testing.assert_array_equal(first.leaf_values, second.leaf_values)
                assert whole.state()["selection_rng"] == restored.state()["selection_rng"]
                np.testing.assert_array_equal(whole.predictions(), restored.predictions())


@pytest.mark.parametrize("configured_block", [64, 300])
def test_large_numeric_session_maps_block_permutations_into_every_prefix_cursor(configured_block):
    rows, permutations = 50003, 3
    bins = np.zeros((1, rows), np.uint8)
    targets = np.linspace(-2, 2, rows, dtype=np.float32)
    baseline = (.3 * np.sin(np.arange(rows) * .013)).astype(np.float32)
    empty = np.empty(0, np.uint32)
    config = options(iterations=1, depth=0, learning_rate=.1, l2_leaf_reg=2,
                     permutation_count=permutations, fold_permutation_block=configured_block,
                     initial_predictions=baseline)
    orders = [reference_order(rows, permutation, configured_block) for permutation in range(permutations)]
    with _ordered.Session(bins, targets, empty, empty, **config) as session:
        initial = session.state()
        for _, end, offset, permutation in initial["descriptors"]:
            np.testing.assert_array_equal(initial["cursors"][offset:offset + end], baseline[orders[permutation][:end]])
        np.testing.assert_array_equal(session.predictions(), baseline)
        session.step()
        final = session.state()
        for prefix, end, offset, permutation in final["descriptors"]:
            order = orders[permutation]
            gradient = (targets[order[:prefix]] - baseline[order[:prefix]]).astype(np.float32)
            update = float(np.float32(config["learning_rate"])) * gradient.sum(dtype=np.float64) / (int(prefix) + 2)
            expected = baseline[order[:end]].astype(np.float64) + update
            np.testing.assert_allclose(final["cursors"][offset:offset + end], expected, rtol=3e-6, atol=2e-7)
        reference = reference_steps(config["random_seed"], permutations, [0])[0]
        assert final["selection_rng"]["words"] == reference[1]
        assert final["selection_rng"]["index"] == reference[2]
