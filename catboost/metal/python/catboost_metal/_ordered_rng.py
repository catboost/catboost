"""Numeric FeatureParallel Ordered's host RNG and history permutation protocol.

These host decisions reproduce checked-in CUDA orchestration. GPU bootstrap
and feature-noise streams still use the separately documented Metal adaptation.
"""

import numpy as np

from ._data import _CudaMersenne64, _unsigned_integer


def cuda_ordered_block_size(rows, configured=64):
    """dynamic_boosting.h::GetPermutationBlockSize, for trivial row groups."""
    rows = _unsigned_integer(rows, "rows", 1 << 24)
    configured = _unsigned_integer(configured, "fold_permutation_block", (1 << 32) - 1)
    if not rows:
        raise ValueError("rows must be positive.")
    if rows < 50000:
        return 1
    configured = configured or 64  # GPU option normalization replaces unset/zero.
    block = 1 << (configured - 1).bit_length()
    while block * 128 > rows:
        block >>= 1
    return block


def cuda_ordered_history_order(rows, permutation_id, block_size=64):
    """TDataPermutation::FillOrder and data_utils.h::Shuffle for numeric rows."""
    block = cuda_ordered_block_size(rows, block_size)
    permutation_id = _unsigned_integer(permutation_id, "permutation_id", (1 << 32) - 1)
    order = np.arange(rows, dtype=np.uint32)
    if not permutation_id:
        return order
    seed = (1664525 * permutation_id + 1013904223 + block) & 0xffffffff
    random = _CudaMersenne64(seed)
    random.advance(10)
    blocks = np.arange((rows + block - 1) // block, dtype=np.uint32)
    for index in range(1, len(blocks)):
        other = random.uniform(index + 1)
        blocks[index], blocks[other] = blocks[other], blocks[index]
    if block == 1:
        return blocks
    cursor = 0
    for block_id in blocks:
        begin = int(block_id) * block
        end = min(begin + block, rows)
        order[cursor:cursor + end - begin] = np.arange(begin, end, dtype=np.uint32)
        cursor += end - begin
    return order


def cuda_ordered_group_history_order(group_sizes, permutation_id, block_size=64):
    """CUDA shuffles group blocks using the block policy for DOCUMENT count."""
    sizes = np.asarray(group_sizes)
    if (sizes.ndim != 1 or not sizes.size or sizes.dtype.kind not in "iu"
            or (sizes <= 0).any() or (sizes > 1 << 24).any()):
        raise ValueError("Ordered group sizes must be positive integers.")
    rows = int(sizes.sum(dtype=np.uint64))
    block = cuda_ordered_block_size(rows, block_size)
    permutation_id = _unsigned_integer(permutation_id, "permutation_id", (1 << 32) - 1)
    if not permutation_id:
        return np.arange(rows, dtype=np.uint32)
    seed = (1664525 * permutation_id + 1013904223 + block) & 0xffffffff
    random = _CudaMersenne64(seed); random.advance(10)
    blocks = np.arange((len(sizes) + block - 1) // block, dtype=np.uint32)
    for index in range(1, len(blocks)):
        other = random.uniform(index + 1)
        blocks[index], blocks[other] = blocks[other], blocks[index]
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.uint64)]
    order = np.empty(rows, np.uint32); cursor = 0
    for block_id in blocks:
        first = int(block_id) * block
        for group in range(first, min(first + block, len(sizes))):
            size = int(sizes[group]); begin = int(offsets[group])
            order[cursor:cursor + size] = np.arange(begin, begin + size, dtype=np.uint32)
            cursor += size
    return order


class OrderedSelectionRng:
    """Persistent single-device numeric Ordered host stream.

    The chooser runs before bootstrap. The first MirrorMapping seed-cache
    creation consumes one base draw plus 65,536 FillSeeds draws, even for No.
    Each attempted numeric split search then consumes one host draw regardless
    of random_strength. No target or constructor draws precede the chooser.
    """

    def __init__(self, seed, permutations, initial_state=None, iteration_offset=0):
        seed = _unsigned_integer(seed, "random_seed", (1 << 64) - 1)
        self.permutations = _unsigned_integer(permutations, "permutation_count", 64)
        if not self.permutations:
            raise ValueError("permutation_count must be positive.")
        self.completed_iterations = _unsigned_integer(iteration_offset, "iteration_offset", (1 << 32) - 1)
        self.random = _CudaMersenne64(seed)
        self.bootstrap_initialized = False
        self._pending = False
        if initial_state is not None:
            if (not isinstance(initial_state, dict) or isinstance(initial_state.get("version"), bool)
                    or initial_state.get("version") != 1):
                raise ValueError("Invalid Ordered selection RNG state version.")
            words = initial_state.get("words")
            if not isinstance(words, (list, tuple, np.ndarray)) or len(words) != 312:
                raise ValueError("Ordered selection RNG words must contain 312 uint64 integers.")
            words = [_unsigned_integer(value, "Ordered selection RNG word", (1 << 64) - 1) for value in words]
            if not any(words):
                raise ValueError("Ordered selection RNG words cannot all be zero.")
            index = _unsigned_integer(initial_state.get("index"), "Ordered selection RNG index", 312)
            initialized = initial_state.get("bootstrap_initialized")
            if not isinstance(initialized, bool):
                raise ValueError("Ordered selection RNG bootstrap_initialized must be boolean.")
            completed = _unsigned_integer(initial_state.get("completed_iterations"),
                                          "Ordered selection RNG completed_iterations", (1 << 32) - 1)
            if completed != self.completed_iterations:
                raise ValueError("Ordered selection RNG completed_iterations conflicts with iteration_offset.")
            self.random.state, self.random.index = words, index
            self.bootstrap_initialized = initialized

    def select(self):
        if self._pending:
            raise RuntimeError("Finish the pending Ordered RNG iteration before selecting again.")
        self._pending = True
        learn_count = self.permutations - 1 if self.permutations > 1 else 1
        return self.random.next() % (learn_count - 1) if learn_count > 1 else 0

    def finish(self, search_attempts):
        search_attempts = _unsigned_integer(search_attempts, "Ordered search_attempts", 16)
        if not self._pending:
            raise RuntimeError("Select the Ordered permutation before finishing its iteration.")
        if not self.bootstrap_initialized:
            self.random.advance(65537)
            self.bootstrap_initialized = True
        self.random.advance(search_attempts)
        self.completed_iterations += 1
        self._pending = False

    def state(self):
        if self._pending:
            raise RuntimeError("Cannot snapshot an incomplete Ordered RNG iteration.")
        return {"version": 1, "words": list(self.random.state), "index": self.random.index,
                "bootstrap_initialized": self.bootstrap_initialized,
                "completed_iterations": self.completed_iterations}
