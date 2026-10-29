"""Independent CUDA group permutation and Ordered fold boundary reference.

Source: cuda/data/{permutation.cpp,permutation.h,data_utils.h},
cuda/gpu_data/samples_grouping.h and cuda/methods/dynamic_boosting.h.
Only host combinatorics are evaluated here; no CatBoost fitting is performed.
"""

import bisect
import math
import operator

import numpy as np

from test_ordered_rng import ReferenceMt64, reference_block_size


def checked_group_sizes(group_sizes):
    values = []
    for value in group_sizes:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("group sizes must be positive integers")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise ValueError("group sizes must be positive integers") from error
        if value <= 0:
            raise ValueError("group sizes must be positive integers")
        values.append(value)
    if not values or sum(values) > 2**24:
        raise ValueError("invalid grouped row count")
    return values


def reference_next_group_end(group_sizes, line):
    """Return the end of the containing group; a boundary starts a new group."""
    sizes = checked_group_sizes(group_sizes)
    ends = np.cumsum(sizes, dtype=np.int64).tolist()
    if line < 0:
        raise ValueError("line must be nonnegative")
    group = bisect.bisect_right(ends, int(line))
    return ends[group] if group < len(ends) else ends[-1]


def reference_min_estimation_size(rows, min_fold_size=100):
    if min_fold_size <= 0:
        raise ValueError("minimum fold size must be positive")
    if rows < 500:
        return 1
    ratio = (rows + min_fold_size - 1) // min_fold_size
    if (ratio - 1).bit_length() >= 18:
        return (rows + (1 << 18) - 1) // (1 << 18)
    return min(min_fold_size, rows // 50)


def reference_group_folds(group_sizes, growth=2.0, min_fold_size=100):
    """CUDA's one-device Ordered folds as (estimate end, quality end)."""
    sizes = checked_group_sizes(group_sizes)
    if len(sizes) < 4:
        raise ValueError("Ordered training requires at least four groups")
    if not math.isfinite(growth) or growth <= 1:
        raise ValueError("fold growth must exceed one")
    rows = sum(sizes)
    prefix = reference_next_group_end(sizes, reference_min_estimation_size(rows, min_fold_size))
    result = []
    while True:
        requested = min(int(prefix * growth), rows)
        end = reference_next_group_end(sizes, requested)
        result.append((prefix, end))
        if end == rows:
            return result
        prefix = end


def reference_group_order(group_sizes, permutation_id, configured_block=64):
    """Shuffle group blocks, then expand each group in original row order.

    The block policy uses DOCUMENT count; the shuffle applies that block size
    to GROUP indices. Recomputing the block from group count changes CUDA's
    stream and can incorrectly split a retained group block.
    """
    sizes = checked_group_sizes(group_sizes)
    groups = list(range(len(sizes)))
    rows = sum(sizes)
    if permutation_id:
        block = reference_block_size(rows, configured_block)
        seed = (1664525 * permutation_id + 1013904223 + block) & 0xFFFFFFFF
        random = ReferenceMt64(seed)
        random.advance(10)
        blocks = list(range((len(groups) + block - 1) // block))
        for index in range(1, len(blocks)):
            other = random.uniform(index + 1)
            blocks[index], blocks[other] = blocks[other], blocks[index]
        groups = [group for first in blocks for group in range(first * block, min((first + 1) * block, len(sizes)))]
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    order = np.asarray([row for group in groups for row in range(offsets[group], offsets[group + 1])], np.uint32)
    return np.asarray(groups, np.uint32), order


def reference_grouped_descriptors(group_sizes, permutation_count=1, *, configured_block=64,
                                  has_time=False, growth=2.0, min_fold_size=100):
    """Pack all variable learning-fold lists, then a separate full task."""
    sizes = checked_group_sizes(group_sizes)
    count = 1 if has_time else permutation_count
    if count < 1:
        raise ValueError("permutation count must be positive")
    rows = sum(sizes)
    maps, group_maps = [], []
    for permutation in range(count):
        group_order, row_order = reference_group_order(sizes, permutation, configured_block)
        group_maps.append(group_order)
        maps.append(row_order)
    descriptors, per_permutation, offset = [], [], 0
    for permutation in range(max(count - 1, 1)):
        folds = reference_group_folds([sizes[index] for index in group_maps[permutation]], growth, min_fold_size)
        per_permutation.append(folds)
        for estimate, quality in folds:
            descriptors.append([estimate, quality, offset, permutation])
            offset += quality
    descriptors.append([rows, rows, offset, count - 1])
    return dict(descriptors=np.asarray(descriptors, np.uint32), permutations=np.asarray(maps, np.uint32),
                group_permutations=np.asarray(group_maps, np.uint32), folds=per_permutation,
                cursor_count=offset + rows)
