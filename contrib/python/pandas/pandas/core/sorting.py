""" miscellaneous sorting / groupby utilities """
import warnings

import numpy as np

from pandas._libs import algos, hashtable, lib
from pandas._libs.hashtable import unique_label_indices
from pandas.compat import PY3, long, string_types

from pandas.core.dtypes.cast import infer_dtype_from_array
from pandas.core.dtypes.common import (
    ensure_int64, ensure_platform_int, is_categorical_dtype, is_list_like)
from pandas.core.dtypes.missing import isna

import pandas.core.algorithms as algorithms

_INT64_MAX = np.iinfo(np.int64).max


def get_group_index(labels, shape, sort, xnull):
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels: sequence of arrays
        Integers identifying levels at each location
    shape: sequence of ints same length as labels
        Number of unique levels at each location
    sort: boolean
        If the ranks of returned ids should match lexical ranks of labels
    xnull: boolean
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through
    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.
    """
    def _int64_cut_off(shape):
        acc = long(1)
        for i, mul in enumerate(shape):
            acc *= long(mul)
            if not acc < _INT64_MAX:
                return i
        return len(shape)

    def maybe_lift(lab, size):
        # promote nan values (assigned -1 label in lab array)
        # so that all output values are non-negative
        return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)

    labels = map(ensure_int64, labels)
    if not xnull:
        labels, shape = map(list, zip(*map(maybe_lift, labels, shape)))

    labels = list(labels)
    shape = list(shape)

    # Iteratively process all the labels in chunks sized so less
    # than _INT64_MAX unique int ids will be required for each chunk
    while True:
        # how many levels can be done without overflow:
        nlev = _int64_cut_off(shape)

        # compute flat ids for the first `nlev` levels
        stride = np.prod(shape[1:nlev], dtype='i8')
        out = stride * labels[0].astype('i8', subok=False, copy=False)

        for i in range(1, nlev):
            if shape[i] == 0:
                stride = 0
            else:
                stride //= shape[i]
            out += labels[i] * stride

        if xnull:  # exclude nulls
            mask = labels[0] == -1
            for lab in labels[1:nlev]:
                mask |= lab == -1
            out[mask] = -1

        if nlev == len(shape):  # all levels done!
            break

        # compress what has been done so far in order to avoid overflow
        # to retain lexical ranks, obs_ids should be sorted
        comp_ids, obs_ids = compress_group_index(out, sort=sort)

        labels = [comp_ids] + labels[nlev:]
        shape = [len(obs_ids)] + shape[nlev:]

    return out


def get_compressed_ids(labels, sizes):
    """

    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : list of size of the levels

    Returns
    -------
    tuple of (comp_ids, obs_group_ids)

    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)


def is_int64_overflow_possible(shape):
    the_prod = long(1)
    for x in shape:
        the_prod *= long(x)

    return the_prod >= _INT64_MAX


def decons_group_index(comp_labels, shape):
    # reconstruct labels
    if is_int64_overflow_possible(shape):
        # at some point group indices are factorized,
        # and may not be deconstructed here! wrong path!
        raise ValueError('cannot deconstruct factorized group indices!')

    label_list = []
    factor = 1
    y = 0
    x = comp_labels
    for i in reversed(range(len(shape))):
        labels = (x - y) % (factor * shape[i]) // factor
        np.putmask(labels, comp_labels < 0, -1)
        label_list.append(labels)
        y = labels * factor
        factor *= shape[i]
    return label_list[::-1]


def decons_obs_group_ids(comp_ids, obs_ids, shape, labels, xnull):
    """
    reconstruct labels from observed group ids

    Parameters
    ----------
    xnull: boolean,
        if nulls are excluded; i.e. -1 labels are passed through
    """

    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype='i8')
        shape = np.asarray(shape, dtype='i8') + lift

    if not is_int64_overflow_possible(shape):
        # obs ids are deconstructable! take the fast route!
        out = decons_group_index(obs_ids, shape)
        return out if xnull or not lift.any() \
            else [x - y for x, y in zip(out, lift)]

    i = unique_label_indices(comp_ids)
    i8copy = lambda a: a.astype('i8', subok=False, copy=True)
    return [i8copy(lab[i]) for lab in labels]


def indexer_from_factorized(labels, shape, compress=True):
    ids = get_group_index(labels, shape, sort=True, xnull=False)

    if not compress:
        ngroups = (ids.size and ids.max()) + 1
    else:
        ids, obs = compress_group_index(ids, sort=True)
        ngroups = len(obs)

    return get_group_index_sorter(ids, ngroups)


def lexsort_indexer(keys, orders=None, na_position='last'):
    from pandas.core.arrays import Categorical

    labels = []
    shape = []
    if isinstance(orders, bool):
        orders = [orders] * len(keys)
    elif orders is None:
        orders = [True] * len(keys)

    for key, order in zip(keys, orders):

        # we are already a Categorical
        if is_categorical_dtype(key):
            c = key

        # create the Categorical
        else:
            c = Categorical(key, ordered=True)

        if na_position not in ['last', 'first']:
            raise ValueError('invalid na_position: {!r}'.format(na_position))

        n = len(c.categories)
        codes = c.codes.copy()

        mask = (c.codes == -1)
        if order:  # ascending
            if na_position == 'last':
                codes = np.where(mask, n, codes)
            elif na_position == 'first':
                codes += 1
        else:  # not order means descending
            if na_position == 'last':
                codes = np.where(mask, n, n - codes - 1)
            elif na_position == 'first':
                codes = np.where(mask, 0, n - codes)
        if mask.any():
            n += 1

        shape.append(n)
        labels.append(codes)

    return indexer_from_factorized(labels, shape)


def nargsort(items, kind='quicksort', ascending=True, na_position='last'):
    """
    This is intended to be a drop-in replacement for np.argsort which
    handles NaNs. It adds ascending and na_position parameters.
    GH #6399, #5231
    """

    # specially handle Categorical
    if is_categorical_dtype(items):
        if na_position not in {'first', 'last'}:
            raise ValueError('invalid na_position: {!r}'.format(na_position))

        mask = isna(items)
        cnt_null = mask.sum()
        sorted_idx = items.argsort(ascending=ascending, kind=kind)
        if ascending and na_position == 'last':
            # NaN is coded as -1 and is listed in front after sorting
            sorted_idx = np.roll(sorted_idx, -cnt_null)
        elif not ascending and na_position == 'first':
            # NaN is coded as -1 and is listed in the end after sorting
            sorted_idx = np.roll(sorted_idx, cnt_null)
        return sorted_idx

    with warnings.catch_warnings():
        # https://github.com/pandas-dev/pandas/issues/25439
        # can be removed once ExtensionArrays are properly handled by nargsort
        warnings.filterwarnings(
            "ignore", category=FutureWarning,
            message="Converting timezone-aware DatetimeArray to")
        items = np.asanyarray(items)
    idx = np.arange(len(items))
    mask = isna(items)
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]
    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    # Finally, place the NaNs at the end or the beginning according to
    # na_position
    if na_position == 'last':
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == 'first':
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError('invalid na_position: {!r}'.format(na_position))
    return indexer


class _KeyMapper(object):

    """
    Ease my suffering. Map compressed group id -> key tuple
    """

    def __init__(self, comp_ids, ngroups, levels, labels):
        self.levels = levels
        self.labels = labels
        self.comp_ids = comp_ids.astype(np.int64)

        self.k = len(labels)
        self.tables = [hashtable.Int64HashTable(ngroups)
                       for _ in range(self.k)]

        self._populate_tables()

    def _populate_tables(self):
        for labs, table in zip(self.labels, self.tables):
            table.map(self.comp_ids, labs.astype(np.int64))

    def get_key(self, comp_id):
        return tuple(level[table.get_item(comp_id)]
                     for table, level in zip(self.tables, self.levels))


def get_flattened_iterator(comp_ids, ngroups, levels, labels):
    # provide "flattened" iterator for multi-group setting
    mapper = _KeyMapper(comp_ids, ngroups, levels, labels)
    return [mapper.get_key(i) for i in range(ngroups)]


def get_indexer_dict(label_list, keys):
    """ return a diction of {labels} -> {indexers} """
    shape = list(map(len, keys))

    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    ngroups = ((group_index.size and group_index.max()) + 1) \
        if is_int64_overflow_possible(shape) \
        else np.prod(shape, dtype='i8')

    sorter = get_group_index_sorter(group_index, ngroups)

    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)

    return lib.indices_fast(sorter, group_index, keys, sorted_labels)


# ----------------------------------------------------------------------
# sorting levels...cleverly?

def get_group_index_sorter(group_index, ngroups):
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')
    """
    count = len(group_index)
    alpha = 0.0  # taking complexities literally; there may be
    beta = 1.0  # some room for fine-tuning these parameters
    do_groupsort = (count > 0 and ((alpha + beta * ngroups) <
                                   (count * np.log(count))))
    if do_groupsort:
        sorter, _ = algos.groupsort_indexer(ensure_int64(group_index),
                                            ngroups)
        return ensure_platform_int(sorter)
    else:
        return group_index.argsort(kind='mergesort')


def compress_group_index(group_index, sort=True):
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """

    size_hint = min(len(group_index), hashtable._SIZE_HINT_LIMIT)
    table = hashtable.Int64HashTable(size_hint)

    group_index = ensure_int64(group_index)

    # note, group labels come out ascending (ie, 1,2,3 etc)
    comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

    if sort and len(obs_group_ids) > 0:
        obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    return comp_ids, obs_group_ids


def _reorder_by_uniques(uniques, labels):
    # sorter is index where elements ought to go
    sorter = uniques.argsort()

    # reverse_indexer is where elements came from
    reverse_indexer = np.empty(len(sorter), dtype=np.int64)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    mask = labels < 0

    # move labels to right locations (ie, unsort ascending labels)
    labels = algorithms.take_nd(reverse_indexer, labels, allow_fill=False)
    np.putmask(labels, mask, -1)

    # sort observed ids
    uniques = algorithms.take_nd(uniques, sorter, allow_fill=False)

    return uniques, labels


def safe_sort(values, labels=None, na_sentinel=-1, assume_unique=False):
    """
    Sort ``values`` and reorder corresponding ``labels``.
    ``values`` should be unique if ``labels`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    .. versionadded:: 0.19.0

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``labels`` is not None.
    labels : list_like
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``na_sentinel``.
    na_sentinel : int, default -1
        Value in ``labels`` to mark "not found".
        Ignored when ``labels`` is None.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``labels`` is None.

    Returns
    -------
    ordered : ndarray
        Sorted ``values``
    new_labels : ndarray
        Reordered ``labels``; returned when ``labels`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``labels`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``labels`` is not None and ``values`` contain duplicates.
    """
    if not is_list_like(values):
        raise TypeError("Only list-like objects are allowed to be passed to"
                        "safe_sort as values")

    if not isinstance(values, np.ndarray):

        # don't convert to string types
        dtype, _ = infer_dtype_from_array(values)
        values = np.asarray(values, dtype=dtype)

    def sort_mixed(values):
        # order ints before strings, safe in py3
        str_pos = np.array([isinstance(x, string_types) for x in values],
                           dtype=bool)
        nums = np.sort(values[~str_pos])
        strs = np.sort(values[str_pos])
        return np.concatenate([nums, np.asarray(strs, dtype=object)])

    sorter = None
    if PY3 and lib.infer_dtype(values, skipna=False) == 'mixed-integer':
        # unorderable in py3 if mixed str/int
        ordered = sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except TypeError:
            # try this anyway
            ordered = sort_mixed(values)

    # labels:

    if labels is None:
        return ordered

    if not is_list_like(labels):
        raise TypeError("Only list-like objects or None are allowed to be"
                        "passed to safe_sort as labels")
    labels = ensure_platform_int(np.asarray(labels))

    from pandas import Index
    if not assume_unique and not Index(values).is_unique:
        raise ValueError("values should be unique if labels is not None")

    if sorter is None:
        # mixed types
        (hash_klass, _), values = algorithms._get_data_algo(
            values, algorithms._hashtables)
        t = hash_klass(len(values))
        t.map_locations(values)
        sorter = ensure_platform_int(t.lookup(ordered))

    reverse_indexer = np.empty(len(sorter), dtype=np.int_)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    mask = (labels < -len(values)) | (labels >= len(values)) | \
        (labels == na_sentinel)

    # (Out of bound indices will be masked with `na_sentinel` next, so we may
    # deal with them here without performance loss using `mode='wrap'`.)
    new_labels = reverse_indexer.take(labels, mode='wrap')
    np.putmask(new_labels, mask, na_sentinel)

    return ordered, ensure_platform_int(new_labels)
