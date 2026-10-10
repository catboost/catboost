"""Independent CUDA numeric Ordered host RNG and block permutation checks.

Source map:
* libs/helpers/cpu_random.h: TRandom(seed) consumes no initial draws.
* methods/dynamic_boosting.h:127-139,285-288: block policy and chooser.
* train_lib/train.cpp:118-121: zero/unset configured block becomes64.
* cuda_util/gpu_random.cpp:FillSeeds/CreateSeeds/GetGpuSeeds: first mirror seed
  allocation consumes 1+65536 HOST draws; the local TRandom(baseSeed) is unused.
* methods/oblivious_tree_structure_searcher.cpp:59-73,166-175: bootstrap seed
  allocation also occurs for No; every attempted numeric depth consumes a seed.
* data/permutation.h:GetSeed and data/data_utils.h:Shuffle: uint32 seed,
  ten warm-up draws and block-level Fisher-Yates using rejection-sampled Uniform.

The interpreter below does not import the production RNG engine. No GPU or
CPU CatBoost fit is needed to validate these deterministic host operations.
"""

import copy
import importlib
import json

import numpy as np
import pytest


class ReferenceMt64:
    """Standard MT19937-64 with explicit two-range in-place state transition."""

    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.words = [int(seed) & self.MASK]
        for index in range(1, 312):
            previous = self.words[-1]
            self.words.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & self.MASK)
        self.index = 312

    def next(self):
        if self.index == 312:
            for index in range(156):
                value = (self.words[index] & 0xFFFFFFFF80000000) | (self.words[index + 1] & 0x7FFFFFFF)
                self.words[index] = self.words[index + 156] ^ (value >> 1) ^ (0xB5026F5AA96619E9 if value & 1 else 0)
            for index in range(156, 311):
                value = (self.words[index] & 0xFFFFFFFF80000000) | (self.words[index + 1] & 0x7FFFFFFF)
                self.words[index] = self.words[index - 156] ^ (value >> 1) ^ (0xB5026F5AA96619E9 if value & 1 else 0)
            value = (self.words[311] & 0xFFFFFFFF80000000) | (self.words[0] & 0x7FFFFFFF)
            self.words[311] = self.words[155] ^ (value >> 1) ^ (0xB5026F5AA96619E9 if value & 1 else 0)
            self.index = 0
        value = self.words[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        return value ^ (value >> 43)

    def advance(self, count):
        for _ in range(count):
            self.next()

    def uniform(self, maximum):
        # util/random/common_ops.h excludes RandMax even for exact divisors.
        boundary = self.MASK - self.MASK % maximum
        value = self.next()
        while value >= boundary:
            value = self.next()
        return value % maximum


def reference_block_size(rows, configured=64):
    if rows < 50000:
        return 1
    block = 1 << (int(configured or 64) - 1).bit_length()
    while block * 128 > rows:
        block //= 2
    return block


def reference_order(rows, permutation_id, configured=64):
    if permutation_id == 0:
        return np.arange(rows, dtype=np.uint32)
    block = reference_block_size(rows, configured)
    seed = (1664525 * permutation_id + 1013904223 + block) & 0xFFFFFFFF
    random = ReferenceMt64(seed)
    random.advance(10)
    blocks = list(range((rows + block - 1) // block))
    for index in range(1, len(blocks)):
        other = random.uniform(index + 1)
        blocks[index], blocks[other] = blocks[other], blocks[index]
    return np.asarray([row for index in blocks for row in range(index * block, min((index + 1) * block, rows))], np.uint32)


def reference_steps(seed, permutations, attempts):
    random, initialized = ReferenceMt64(seed), False
    results = []
    for count in attempts:
        selected = random.next() % (permutations - 2) if permutations > 2 else 0
        if not initialized:
            random.advance(65537)
            initialized = True
        random.advance(count)
        results.append((selected, copy.deepcopy(random.words), random.index))
    return results


@pytest.fixture(scope="module")
def rng_module():
    return importlib.import_module("catboost_metal._ordered_rng")


def test_independent_reference_matches_published_mt19937_64_seed5489_vector():
    random = ReferenceMt64(5489)
    # First five values from the standard MT19937-64 reference sequence.
    assert [random.next() for _ in range(5)] == [
        14514284786278117030, 4620546740167642908, 13109570281517897720,
        17462938647148434322, 355488278567739596,
    ]


@pytest.mark.parametrize("seed", [0, 42, 2**64 - 1])
@pytest.mark.parametrize("permutations", [1, 2, 3, 4, 7])
def test_selector_and_exact_host_draw_consumption_match_cuda(rng_module, seed, permutations):
    attempted = [0, 1, 4, 2, 6, 1, 0, 3, 5]
    expected = reference_steps(seed, permutations, attempted)
    random = rng_module.OrderedSelectionRng(seed, permutations)
    initial = random.state()
    assert initial["index"] == 312
    assert initial["bootstrap_initialized"] is False
    assert initial["completed_iterations"] == 0
    assert initial["words"] == ReferenceMt64(seed).words
    for iteration, (attempts, (selected, words, index)) in enumerate(zip(attempted, expected)):
        assert random.select() == selected
        random.finish(attempts)
        actual = random.state()
        assert actual["version"] == 1
        assert actual["completed_iterations"] == iteration + 1
        assert actual["bootstrap_initialized"] is True
        assert actual["words"] == words
        assert actual["index"] == index


def test_repeated_split_attempt_consumes_a_seed_and_changes_later_selection(rng_module):
    ordinary = rng_module.OrderedSelectionRng(819, 7)
    repeated = rng_module.OrderedSelectionRng(819, 7)
    assert ordinary.select() == repeated.select()
    ordinary.finish(2)
    repeated.finish(3)  # Third search returns a previously selected split.
    choices = [[], []]
    for _ in range(8):
        for result, generator in zip(choices, (ordinary, repeated)):
            result.append(generator.select())
            generator.finish(2)
    assert choices[0] != choices[1]


@pytest.mark.parametrize("permutations", [1, 3, 4, 7])
def test_json_state_resume_continues_variable_depth_sequence_exactly(rng_module, permutations):
    original = rng_module.OrderedSelectionRng(781941, permutations)
    for attempts in (4, 1, 0):
        original.select()
        original.finish(attempts)
    state = json.loads(json.dumps(original.state()))
    restored = rng_module.OrderedSelectionRng(781941, permutations, initial_state=state,
                                             iteration_offset=state["completed_iterations"])
    for attempts in (2, 5, 1, 3, 0):
        assert original.select() == restored.select()
        original.finish(attempts)
        restored.finish(attempts)
        assert original.state() == restored.state()


def test_pending_selection_cannot_be_snapshotted_or_advanced_twice(rng_module):
    random = rng_module.OrderedSelectionRng(814, 4)
    with pytest.raises((ValueError, RuntimeError)):
        random.finish(1)
    selected = random.select()
    with pytest.raises((ValueError, RuntimeError)):
        random.state()
    # Either explicitly reject repeated selection, or return the same pending
    # choice without consuming another host word. Both keep transactions safe.
    try:
        assert random.select() == selected
    except (ValueError, RuntimeError):
        pass
    random.finish(3)
    expected = reference_steps(814, 4, [3])[0]
    assert random.state()["words"] == expected[1]
    assert random.state()["index"] == expected[2]


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("words", [0] * 311), ("words", [-1] + [0] * 311),
    ("words", [2**64] + [0] * 311), ("index", -1), ("index", 313),
    ("bootstrap_initialized", "yes"), ("completed_iterations", -1),
])
def test_invalid_saved_rng_state_is_rejected(rng_module, field, value):
    random = rng_module.OrderedSelectionRng(42, 4)
    state = random.state()
    state[field] = value
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        rng_module.OrderedSelectionRng(42, 4, initial_state=state)


@pytest.mark.parametrize("rows,configured,expected", [
    (4, 64, 1), (49999, 4096, 1), (50000, 0, 64), (50000, 1, 1), (50000, 7, 8),
    (50000, 64, 64), (50000, 129, 256), (50000, 257, 256),
    (65536, 513, 512), (100000, 513, 512), (1 << 24, 64, 64),
])
def test_numeric_block_policy_matches_cuda_boundaries(rng_module, rows, configured, expected):
    assert reference_block_size(rows, configured) == expected
    assert rng_module.cuda_ordered_block_size(rows, configured) == expected


@pytest.mark.parametrize("rows,configured", [(33, 64), (49999, 64), (50000, 64), (50003, 300), (65539, 513)])
@pytest.mark.parametrize("permutation_id", [0, 1, 3, 2**32 - 1])
def test_numeric_block_shuffle_matches_independent_cuda_rng(rng_module, rows, configured, permutation_id):
    expected = reference_order(rows, permutation_id, configured)
    actual = rng_module.cuda_ordered_history_order(rows, permutation_id, block_size=configured)
    assert actual.dtype == np.uint32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.sort(actual), np.arange(rows))


def test_shuffled_blocks_keep_internal_row_order_including_short_final_block(rng_module):
    rows, configured = 50003, 64
    order = rng_module.cuda_ordered_history_order(rows, 1, block_size=configured)
    block = reference_block_size(rows, configured)
    starts = np.flatnonzero(np.r_[True, order[1:] // block != order[:-1] // block])
    stops = np.r_[starts[1:], rows]
    assert len(starts) == (rows + block - 1) // block
    for start, stop in zip(starts, stops):
        original_start = int(order[start])
        assert original_start % block == 0
        np.testing.assert_array_equal(order[start:stop], np.arange(original_start, min(original_start + block, rows)))
    assert 19 in (stops - starts)


@pytest.mark.parametrize("rows,configured", [(-1, 64), (1 << 25, 64), (50000, -1), (50000, 1.5), (True, 64)])
def test_invalid_block_options_are_rejected(rng_module, rows, configured):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        rng_module.cuda_ordered_block_size(rows, configured)
