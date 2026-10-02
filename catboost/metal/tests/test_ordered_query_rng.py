"""Ordered ranking validation and independent FeatureParallel Yeti host draws.

No Metal library is loaded. The MT interpreter is independent of production;
the transcript follows dynamic_boosting.h, dynamic_structure_searcher.h,
gpu_random.cpp and batch_symmetrictree_estimator.h in the checked-in CUDA tree.
"""

import copy

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._ordered_yeti_rng import OrderedYetiRankRng
from test_ordered_rng import ReferenceMt64


@pytest.mark.parametrize("permutations,score_sets", [(1, 1), (2, 1), (4, 1), (4, 2), (7, 2)])
@pytest.mark.parametrize("leaf_iterations", [1, 3])
def test_yeti_ordered_transcript_and_resume(permutations, score_sets, leaf_iterations):
    seed = 0xF123456789ABCDEF
    expected = ReferenceMt64(seed)
    rng = OrderedYetiRankRng(seed, permutations, leaf_iterations, score_sets=score_sets)
    # No DocParallel BaseIterationSeed draw occurs before the first chooser.
    assert rng.state()["words"] == expected.words
    for iteration, (folds, attempts, active_tasks) in enumerate([(2, 3, 7), (1, 0, 4), (3, 2, 5)]):
        selected = expected.next() % (permutations - 2) if permutations > 2 else 0
        assert rng.select() == selected
        # Each selected fold contributes learn then quality, including an empty
        # quality range; seed_shape supplies precisely this doubled count.
        weak = [[expected.next(), expected.next()] for _ in range(folds)]
        assert rng.weak(2 * folds) == [value for pair in weak for value in pair]
        if iteration == 0:
            expected.advance(65537)  # MirrorMapping cache, even bootstrap No.
        expected.advance(attempts * score_sets)
        evaluations = leaf_iterations + int(leaf_iterations > 1)
        leaf = [[expected.next() for _ in range(active_tasks)] for _ in range(evaluations)]
        assert rng.leaves(attempts, active_tasks * evaluations) == [value for evaluation in leaf for value in evaluation]
        rng.finish()
        state = rng.state()
        assert state["words"] == expected.words
        assert state["index"] == expected.index
        assert state["completed_iterations"] == iteration + 1
        rng = OrderedYetiRankRng(seed, permutations, leaf_iterations, score_sets=score_sets,
                                initial_state=state, iteration_offset=iteration + 1)
        assert rng.state() == state


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("policy", "numeric_ordered_host_v1"), ("random_seed", 9),
    ("permutation_count", 3), ("leaf_iterations", 2), ("score_sets", 2),
    ("completed_iterations", 1), ("words", [0] * 312), ("index", 313),
    ("bootstrap_initialized", True),
])
def test_yeti_rng_rejects_incompatible_state(field, value):
    state = copy.deepcopy(OrderedYetiRankRng(7, 4, 1).state())
    state[field] = value
    with pytest.raises(ValueError):
        OrderedYetiRankRng(7, 4, 1, initial_state=state)


def test_yeti_rng_requires_complete_iteration_and_valid_counts():
    rng = OrderedYetiRankRng(7, 4, 2)
    with pytest.raises(RuntimeError):
        rng.weak(2)
    rng.select()
    with pytest.raises(ValueError):
        rng.weak(3)
    rng.weak(2)
    with pytest.raises(RuntimeError):
        rng.state()
    with pytest.raises(ValueError):
        rng.leaves(1, 2)  # I=2 needs three evaluation rounds.
    rng.leaves(1, 3)
    rng.finish()
    with pytest.raises(ValueError):
        OrderedYetiRankRng(7, 1, 1, score_sets=2)
    with pytest.raises(ValueError):
        OrderedYetiRankRng(7, 4, 1, iteration_offset=1)
    state = rng.state()
    state["completed_iterations"] = (1 << 32) - 1
    restored = OrderedYetiRankRng(7, 4, 2, initial_state=state, iteration_offset=(1 << 32) - 1)
    with pytest.raises(ValueError, match="uint32"):
        restored.select()


def _forbid_native(monkeypatch):
    monkeypatch.setattr(_ordered, "build_library", lambda: pytest.fail("invalid Ordered input reached Metal"))


@pytest.mark.parametrize("objective,options,message", [
    ("QueryRMSE", {"group_offsets": None}, "requires explicit group"),
    ("QueryRMSE", {"group_offsets": [0, 6, 12]}, "at least four"),
    ("QueryRMSE", {"group_sizes": [2, 4, 3, 3]}, "same groups"),
    ("QueryRMSE", {"leaf_estimation_method": "Exact"}, "Exact supports"),
    ("QuerySoftMax", {"query_beta": np.nan}, "finite"),
    ("QuerySoftMax", {"targets": np.full(12, -1.)}, "nonnegative"),
    ("QuerySoftMax", {"targets": np.zeros(12)}, "positive effective"),
    ("PairLogit", {}, "nonempty valid"),
    ("PairLogit", {"pair_winners": [0], "pair_losers": [3]}, "same query"),
    ("PairLogit", {"pair_winners": [0], "pair_losers": [0]}, "self-pairs"),
    ("PairLogit", {"pair_winners": [0], "pair_losers": [1], "pair_weights": [0]}, "positive finite"),
    ("PairLogit", {"sample_weight": np.ones(12)}, "incident pair mass"),
    ("QueryRMSE", {"pair_winners": [0], "pair_losers": [1]}, "Supplied pair"),
    ("YetiRank", {"leaf_estimation_method": "Gradient"}, "Newton leaves"),
    ("YetiRank", {"leaf_estimation_backtracking": "Armijo"}, "no backtracking"),
    ("YetiRank", {"yeti_permutations": True}, "integer"),
    ("YetiRank", {"yeti_permutations": 10001}, "integer"),
    ("YetiRank", {"decay": 1.1}, "decay"),
    ("YetiRank", {"legacy_prefix_centering": 1}, "boolean"),
    ("YetiRank", {"targets": np.full(12, 1.1)}, "relevance labels"),
    ("YetiRank", {"subgroup_hashes": np.arange(11)}, "subgroup"),
    ("YetiRank", {"group_offsets": np.arange(13)}, "multiple rows"),
])
def test_ordered_ranking_rejects_invalid_inputs_before_gpu(monkeypatch, objective, options, message):
    _forbid_native(monkeypatch)
    bins = np.asarray([np.arange(12) % 3], np.uint8)
    targets = np.linspace(0, 1, 12, dtype=np.float32)
    config = dict(group_offsets=[0, 3, 6, 9, 12], objective=objective, iterations=2, min_fold_size=2)
    config.update(options)
    targets = config.pop("targets", targets)
    with pytest.raises(ValueError, match=message):
        _ordered.Session(bins, targets, [0], [0], **config)


def test_ordered_ranking_rejects_split_query_history_before_gpu(monkeypatch):
    _forbid_native(monkeypatch)
    order = np.arange(12, dtype=np.uint32)
    order[[1, 3]] = order[[3, 1]]
    with pytest.raises(ValueError, match="whole groups"):
        _ordered.Session(np.asarray([np.arange(12) % 3], np.uint8), np.linspace(0, 1, 12), [0], [0],
                         objective="QueryRMSE", group_offsets=[0, 3, 6, 9, 12], permutations=order[None])


def test_ordered_yeti_rejects_query_over_kernel_limit_before_gpu(monkeypatch):
    _forbid_native(monkeypatch)
    with pytest.raises(ValueError, match="1023 rows"):
        _ordered.Session(np.zeros((1, 4096), np.uint8), np.zeros(4096), [0], [0],
                         objective="YetiRank", group_offsets=[0, 1024, 2048, 3072, 4096])
