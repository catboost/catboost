# pylint: disable=E1101,E1103
# pylint: disable=W0703,W0622,W0613,W0201
from functools import partial
import itertools

import numpy as np

from pandas._libs import algos as _algos, reshape as _reshape
from pandas._libs.sparse import IntIndex
from pandas.compat import PY2, range, text_type, u, zip

from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
    ensure_platform_int, is_bool_dtype, is_extension_array_dtype,
    is_integer_dtype, is_list_like, is_object_dtype, needs_i8_conversion)
from pandas.core.dtypes.missing import notna

from pandas import compat
import pandas.core.algorithms as algos
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import _factorize_from_iterable
from pandas.core.frame import DataFrame
from pandas.core.index import Index, MultiIndex
from pandas.core.internals.arrays import extract_array
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index, decons_obs_group_ids, get_compressed_ids,
    get_group_index)


class _Unstacker(object):
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    values : ndarray
        Values of DataFrame to "Unstack"
    index : object
        Pandas ``Index``
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    value_columns : Index, optional
        Pandas ``Index`` or ``MultiIndex`` object if unstacking a DataFrame
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame or SparseDataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
    ...                                    ('two', 'a'), ('two', 'b')])
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(self, values, index, level=-1, value_columns=None,
                 fill_value=None, constructor=None):

        if values.ndim == 1:
            values = values[:, np.newaxis]
        self.values = values
        self.value_columns = value_columns
        self.fill_value = fill_value

        if constructor is None:
            constructor = DataFrame
        self.constructor = constructor

        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError('must pass column labels for multi-column data')

        self.index = index.remove_unused_levels()

        self.level = self.index._get_level_number(level)

        # when index includes `nan`, need to lift levels/strides by 1
        self.lift = 1 if -1 in self.index.codes[self.level] else 0

        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]

        # Bug fix GH 20601
        # If the data frame is too big, the number of unique index combination
        # will cause int32 overflow on windows environments.
        # We want to check and raise an error before this happens
        num_rows = np.max([index_level.size for index_level
                           in self.new_index_levels])
        num_columns = self.removed_level.size

        # GH20601: This forces an overflow if the number of cells is too high.
        num_cells = np.multiply(num_rows, num_columns, dtype=np.int32)

        if num_rows > 0 and num_columns > 0 and num_cells <= 0:
            raise ValueError('Unstacked DataFrame is too big, '
                             'causing int32 overflow')

        self._make_sorted_values_labels()
        self._make_selectors()

    def _make_sorted_values_labels(self):
        v = self.level

        codes = list(self.index.codes)
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1:] + [codes[v]]
        sizes = [len(x) for x in levs[:v] + levs[v + 1:] + [levs[v]]]

        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        indexer = _algos.groupsort_indexer(comp_index, ngroups)[0]
        indexer = ensure_platform_int(indexer)

        self.sorted_values = algos.take_nd(self.values, indexer, axis=0)
        self.sorted_labels = [l.take(indexer) for l in to_sort]

    def _make_selectors(self):
        new_levels = self.new_index_levels

        # make the mask
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = [len(x) for x in new_levels]

        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = ngroups, stride

        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        if mask.sum() < len(self.index):
            raise ValueError('Index contains duplicate entries, '
                             'cannot reshape')

        self.group_index = comp_index
        self.mask = mask
        self.unique_groups = obs_ids
        self.compressor = comp_index.searchsorted(np.arange(ngroups))

    def get_result(self):
        values, _ = self.get_new_values()
        columns = self.get_new_columns()
        index = self.get_new_index()

        return self.constructor(values, index=index, columns=columns)

    def get_new_values(self):
        values = self.values

        # place the values
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = mask.all()

        # we can simply reshape if we don't have a mask
        if mask_all and len(values):
            new_values = (self.sorted_values
                              .reshape(length, width, stride)
                              .swapaxes(1, 2)
                              .reshape(result_shape)
                          )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        # if our mask is all True, then we can use our existing dtype
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        else:
            dtype, fill_value = maybe_promote(values.dtype, self.fill_value)
            new_values = np.empty(result_shape, dtype=dtype)
            new_values.fill(fill_value)

        new_mask = np.zeros(result_shape, dtype=bool)

        name = np.dtype(dtype).name
        sorted_values = self.sorted_values

        # we need to convert to a basic dtype
        # and possibly coerce an input to our output dtype
        # e.g. ints -> floats
        if needs_i8_conversion(values):
            sorted_values = sorted_values.view('i8')
            new_values = new_values.view('i8')
            name = 'int64'
        elif is_bool_dtype(values):
            sorted_values = sorted_values.astype('object')
            new_values = new_values.astype('object')
            name = 'object'
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # fill in our values & mask
        f = getattr(_reshape, "unstack_{name}".format(name=name))
        f(sorted_values,
          mask.view('u1'),
          stride,
          length,
          width,
          new_values,
          new_mask.view('u1'))

        # reconstruct dtype if needed
        if needs_i8_conversion(values):
            new_values = new_values.view(values.dtype)

        return new_values, new_mask

    def get_new_columns(self):
        if self.value_columns is None:
            if self.lift == 0:
                return self.removed_level

            lev = self.removed_level
            return lev.insert(0, lev._na_value)

        stride = len(self.removed_level) + self.lift
        width = len(self.value_columns)
        propagator = np.repeat(np.arange(width), stride)
        if isinstance(self.value_columns, MultiIndex):
            new_levels = self.value_columns.levels + (self.removed_level_full,)
            new_names = self.value_columns.names + (self.removed_name,)

            new_codes = [lab.take(propagator)
                         for lab in self.value_columns.codes]
        else:
            new_levels = [self.value_columns, self.removed_level_full]
            new_names = [self.value_columns.name, self.removed_name]
            new_codes = [propagator]

        # The two indices differ only if the unstacked level had unused items:
        if len(self.removed_level_full) != len(self.removed_level):
            # In this case, we remap the new codes to the original level:
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            # Otherwise, we just use each level item exactly once:
            repeater = np.arange(stride) - self.lift

        # The entire level is then just a repetition of the single chunk:
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(levels=new_levels, codes=new_codes,
                          names=new_names, verify_integrity=False)

    def get_new_index(self):
        result_codes = [lab.take(self.compressor)
                        for lab in self.sorted_labels[:-1]]

        # construct the new index
        if len(self.new_index_levels) == 1:
            lev, lab = self.new_index_levels[0], result_codes[0]
            if (lab == -1).any():
                lev = lev.insert(len(lev), lev._na_value)
            return lev.take(lab)

        return MultiIndex(levels=self.new_index_levels, codes=result_codes,
                          names=self.new_index_names, verify_integrity=False)


def _unstack_multiple(data, clocs, fill_value=None):
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet

    index = data.index

    clocs = [index._get_level_number(i) for i in clocs]

    rlocs = [i for i in range(index.nlevels) if i not in clocs]

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]

    shape = [len(x) for x in clevels]
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes,
                                        xnull=False)

    if rlocs == []:
        # Everything is in clocs, so the dummy df has a regular index
        dummy_index = Index(obs_ids, name='__placeholder__')
    else:
        dummy_index = MultiIndex(levels=rlevels + [obs_ids],
                                 codes=rcodes + [comp_ids],
                                 names=rnames + ['__placeholder__'],
                                 verify_integrity=False)

    if isinstance(data, Series):
        dummy = data.copy()
        dummy.index = dummy_index

        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            for i in range(len(clocs)):
                val = clocs[i]
                result = result.unstack(val)
                clocs = [v if i > v else v - 1 for v in clocs]

            return result

        dummy = data.copy()
        dummy.index = dummy_index

        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value)
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        new_levels = [unstcols.levels[0]] + clevels
        new_names = [data.columns.name] + cnames

        new_codes = [unstcols.codes[0]]
        for rec in recons_codes:
            new_codes.append(rec.take(unstcols.codes[-1]))

    new_columns = MultiIndex(levels=new_levels, codes=new_codes,
                             names=new_names, verify_integrity=False)

    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns

    return unstacked


def unstack(obj, level, fill_value=None):
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            # _unstack_multiple only handles MultiIndexes,
            # and isn't needed for a single level
            return _unstack_multiple(obj, level, fill_value=fill_value)
        else:
            level = level[0]

    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value)
        else:
            return obj.T.stack(dropna=False)
    else:
        if is_extension_array_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value)
        unstacker = _Unstacker(obj.values, obj.index, level=level,
                               fill_value=fill_value,
                               constructor=obj._constructor_expanddim)
        return unstacker.get_result()


def _unstack_frame(obj, level, fill_value=None):
    if obj._is_mixed_type:
        unstacker = partial(_Unstacker, index=obj.index,
                            level=level, fill_value=fill_value)
        blocks = obj._data.unstack(unstacker,
                                   fill_value=fill_value)
        return obj._constructor(blocks)
    else:
        unstacker = _Unstacker(obj.values, obj.index, level=level,
                               value_columns=obj.columns,
                               fill_value=fill_value,
                               constructor=obj._constructor)
        return unstacker.get_result()


def _unstack_extension_series(series, level, fill_value):
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    # Implementation note: the basic idea is to
    # 1. Do a regular unstack on a dummy array of integers
    # 2. Followup with a columnwise take.
    # We use the dummy take to discover newly-created missing values
    # introduced by the reshape.
    from pandas.core.reshape.concat import concat

    dummy_arr = np.arange(len(series))
    # fill_value=-1, since we will do a series.values.take later
    result = _Unstacker(dummy_arr, series.index,
                        level=level, fill_value=-1).get_result()

    out = []
    values = extract_array(series, extract_numpy=False)

    for col, indices in result.iteritems():
        out.append(Series(values.take(indices.values,
                                      allow_fill=True,
                                      fill_value=fill_value),
                          name=col, index=result.index))
    return concat(out, axis='columns', copy=False, keys=result.columns)


def stack(frame, level=-1, dropna=True):
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series
    """
    def factorize(index):
        if index.is_unique:
            return index, np.arange(len(index))
        codes, categories = _factorize_from_iterable(index)
        return categories, codes

    N, K = frame.shape

    # Will also convert negative level numbers and check if out of bounds.
    level_num = frame.columns._get_level_number(level)

    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(frame, level_num=level_num, dropna=dropna)
    elif isinstance(frame.index, MultiIndex):
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]

        clev, clab = factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())

        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(levels=new_levels, codes=new_codes,
                               names=new_names, verify_integrity=False)
    else:
        levels, (ilab, clab) = zip(*map(factorize, (frame.index,
                                                    frame.columns)))
        codes = ilab.repeat(K), np.tile(clab, N).ravel()
        new_index = MultiIndex(levels=levels, codes=codes,
                               names=[frame.index.name, frame.columns.name],
                               verify_integrity=False)

    if frame._is_homogeneous_type:
        # For homogeneous EAs, frame.values will coerce to object. So
        # we concatenate instead.
        dtypes = list(frame.dtypes.values)
        dtype = dtypes[0]

        if is_extension_array_dtype(dtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type([
                col._values for _, col in frame.iteritems()
            ])
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            # homogeneous, non-EA
            new_values = frame.values.ravel()

    else:
        # non-homogeneous
        new_values = frame.values.ravel()

    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]

    return frame._constructor_sliced(new_values, index=new_index)


def stack_multiple(frame, level, dropna=True):
    # If all passed levels match up to column names, no
    # ambiguity about what to do
    if all(lev in frame.columns.names for lev in level):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna)

    # Otherwise, level numbers may change as each successive level is stacked
    elif all(isinstance(lev, int) for lev in level):
        # As each stack is done, the level numbers decrease, so we need
        #  to account for that when level is a sequence of ints
        result = frame
        # _get_level_number() checks level numbers are in range and converts
        # negative numbers to positive
        level = [frame.columns._get_level_number(lev) for lev in level]

        # Can't iterate directly through level as we might need to change
        # values as we go
        for index in range(len(level)):
            lev = level[index]
            result = stack(result, lev, dropna=dropna)
            # Decrement all level numbers greater than current, as these
            # have now shifted down by one
            updated_level = []
            for other in level:
                if other > lev:
                    updated_level.append(other - 1)
                else:
                    updated_level.append(other)
            level = updated_level

    else:
        raise ValueError("level should contain all level names or all level "
                         "numbers, not a mixture of the two.")

    return result


def _stack_multi_columns(frame, level_num=-1, dropna=True):
    def _convert_level_number(level_num, columns):
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel:

        We generally want to convert the level number into a level name, except
        when columns do not have names, in which case we must leave as a level
        number
        """
        if level_num in columns.names:
            return columns.names[level_num]
        else:
            if columns.names[level_num] is None:
                return level_num
            else:
                return columns.names[level_num]

    this = frame.copy()

    # this makes life much simpler
    if level_num != frame.columns.nlevels - 1:
        # roll levels to put selected level at end
        roll_columns = this.columns
        for i in range(level_num, frame.columns.nlevels - 1):
            # Need to check if the ints conflict with level names
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = roll_columns

    if not this.columns.is_lexsorted():
        # Workaround the edge case where 0 is one of the column names,
        # which interferes with trying to sort based on the first
        # level
        level_to_sort = _convert_level_number(0, this.columns)
        this = this.sort_index(level=level_to_sort, axis=1)

    # tuple list excluding level for grouping columns
    if len(frame.columns.levels) > 2:
        tuples = list(zip(*[lev.take(level_codes) for lev, level_codes
                            in zip(this.columns.levels[:-1],
                                   this.columns.codes[:-1])]))
        unique_groups = [key for key, _ in itertools.groupby(tuples)]
        new_names = this.columns.names[:-1]
        new_columns = MultiIndex.from_tuples(unique_groups, names=new_names)
    else:
        new_columns = unique_groups = this.columns.levels[0]

    # time to ravel the values
    new_data = {}
    level_vals = this.columns.levels[-1]
    level_codes = sorted(set(this.columns.codes[-1]))
    level_vals_used = level_vals[level_codes]
    levsize = len(level_codes)
    drop_cols = []
    for key in unique_groups:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue

        # can make more efficient?
        # we almost always return a slice
        # but if unsorted can get a boolean
        # indexer
        if not isinstance(loc, slice):
            slice_len = len(loc)
        else:
            slice_len = loc.stop - loc.start

        if slice_len != levsize:
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals.take(chunk.columns.codes[-1])
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            if (frame._is_homogeneous_type and
                    is_extension_array_dtype(frame.dtypes.iloc[0])):
                dtype = this[this.columns[loc]].dtypes.iloc[0]
                subset = this[this.columns[loc]]

                value_slice = dtype.construct_array_type()._concat_same_type(
                    [x._values for _, x in subset.iteritems()]
                )
                N, K = this.shape
                idx = np.arange(N * K).reshape(K, N).T.ravel()
                value_slice = value_slice.take(idx)

            elif frame._is_mixed_type:
                value_slice = this[this.columns[loc]].values
            else:
                value_slice = this.values[:, loc]

        if value_slice.ndim > 1:
            # i.e. not extension
            value_slice = value_slice.ravel()

        new_data[key] = value_slice

    if len(drop_cols) > 0:
        new_columns = new_columns.difference(drop_cols)

    N = len(this)

    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        new_levels = [this.index]
        new_codes = [np.arange(N).repeat(levsize)]
        new_names = [this.index.name]  # something better?

    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])

    new_index = MultiIndex(levels=new_levels, codes=new_codes,
                           names=new_names, verify_integrity=False)

    result = frame._constructor(new_data, index=new_index, columns=new_columns)

    # more efficient way to go about this? can do the whole masking biz but
    # will only save a small amount of time...
    if dropna:
        result = result.dropna(axis=0, how='all')

    return result


def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False,
                columns=None, sparse=False, drop_first=False, dtype=None):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    data : array-like, Series, or DataFrame
    prefix : string, list of strings, or dict of strings, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : string, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix.`
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.

        .. versionadded:: 0.18.0

    dtype : dtype, default np.uint8
        Data type for new columns. Only a single dtype is allowed.

        .. versionadded:: 0.23.0

    Returns
    -------
    dummies : DataFrame

    See Also
    --------
    Series.str.get_dummies

    Examples
    --------
    >>> s = pd.Series(list('abca'))

    >>> pd.get_dummies(s)
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> s1 = ['a', 'b', np.nan]

    >>> pd.get_dummies(s1)
       a  b
    0  1  0
    1  0  1
    2  0  0

    >>> pd.get_dummies(s1, dummy_na=True)
       a  b  NaN
    0  1  0    0
    1  0  1    0
    2  0  0    1

    >>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]})

    >>> pd.get_dummies(df, prefix=['col1', 'col2'])
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1       1       0       0       1       0
    1  2       0       1       1       0       0
    2  3       1       0       0       0       1

    >>> pd.get_dummies(pd.Series(list('abcaa')))
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    4  1  0  0

    >>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
       b  c
    0  0  0
    1  1  0
    2  0  1
    3  0  0
    4  0  0

    >>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    from pandas.core.reshape.concat import concat
    from itertools import cycle

    dtypes_to_encode = ['object', 'category']

    if isinstance(data, DataFrame):
        # determine columns being encoded
        if columns is None:
            data_to_encode = data.select_dtypes(
                include=dtypes_to_encode)
        else:
            data_to_encode = data[columns]

        # validate prefixes and separator to avoid silently dropping cols
        def check_len(item, name):
            len_msg = ("Length of '{name}' ({len_item}) did not match the "
                       "length of the columns being encoded ({len_enc}).")

            if is_list_like(item):
                if not len(item) == data_to_encode.shape[1]:
                    len_msg = len_msg.format(name=name, len_item=len(item),
                                             len_enc=data_to_encode.shape[1])
                    raise ValueError(len_msg)

        check_len(prefix, 'prefix')
        check_len(prefix_sep, 'prefix_sep')

        if isinstance(prefix, compat.string_types):
            prefix = cycle([prefix])
        if isinstance(prefix, dict):
            prefix = [prefix[col] for col in data_to_encode.columns]

        if prefix is None:
            prefix = data_to_encode.columns

        # validate separators
        if isinstance(prefix_sep, compat.string_types):
            prefix_sep = cycle([prefix_sep])
        elif isinstance(prefix_sep, dict):
            prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]

        if data_to_encode.shape == data.shape:
            # Encoding the entire df, do not prepend any dropped columns
            with_dummies = []
        elif columns is not None:
            # Encoding only cols specified in columns. Get all cols not in
            # columns to prepend to result.
            with_dummies = [data.drop(columns, axis=1)]
        else:
            # Encoding only object and category dtype columns. Get remaining
            # columns to prepend to result.
            with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]

        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (column_name, column), use just column data here
            dummy = _get_dummies_1d(col[1], prefix=pre, prefix_sep=sep,
                                    dummy_na=dummy_na, sparse=sparse,
                                    drop_first=drop_first, dtype=dtype)
            with_dummies.append(dummy)
        result = concat(with_dummies, axis=1)
    else:
        result = _get_dummies_1d(data, prefix, prefix_sep, dummy_na,
                                 sparse=sparse,
                                 drop_first=drop_first,
                                 dtype=dtype)
    return result


def _get_dummies_1d(data, prefix, prefix_sep='_', dummy_na=False,
                    sparse=False, drop_first=False, dtype=None):
    from pandas.core.reshape.concat import concat
    # Series avoids inconsistent NaN handling
    codes, levels = _factorize_from_iterable(Series(data))

    if dtype is None:
        dtype = np.uint8
    dtype = np.dtype(dtype)

    if is_object_dtype(dtype):
        raise ValueError("dtype=object is not a valid dtype for get_dummies")

    def get_empty_frame(data):
        if isinstance(data, Series):
            index = data.index
        else:
            index = np.arange(len(data))
        return DataFrame(index=index)

    # if all NaN
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)

    codes = codes.copy()
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = np.append(levels, np.nan)

    # if dummy_na, we just fake a nan level. drop_first will drop it again
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)

    number_of_cols = len(levels)

    if prefix is None:
        dummy_cols = levels
    else:

        # PY2 embedded unicode, gh-22084
        def _make_col_name(prefix, prefix_sep, level):
            fstr = '{prefix}{prefix_sep}{level}'
            if PY2 and (isinstance(prefix, text_type) or
                        isinstance(prefix_sep, text_type) or
                        isinstance(level, text_type)):
                fstr = u(fstr)
            return fstr.format(prefix=prefix,
                               prefix_sep=prefix_sep,
                               level=level)

        dummy_cols = [_make_col_name(prefix, prefix_sep, level)
                      for level in levels]

    if isinstance(data, Series):
        index = data.index
    else:
        index = None

    if sparse:

        if is_integer_dtype(dtype):
            fill_value = 0
        elif dtype == bool:
            fill_value = False
        else:
            fill_value = 0.0

        sparse_series = []
        N = len(data)
        sp_indices = [[] for _ in range(len(dummy_cols))]
        mask = codes != -1
        codes = codes[mask]
        n_idx = np.arange(N)[mask]

        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)

        if drop_first:
            # remove first categorical level to avoid perfect collinearity
            # GH12042
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]
        for col, ixs in zip(dummy_cols, sp_indices):
            sarr = SparseArray(np.ones(len(ixs), dtype=dtype),
                               sparse_index=IntIndex(N, ixs),
                               fill_value=fill_value,
                               dtype=dtype)
            sparse_series.append(Series(data=sarr, index=index, name=col))

        out = concat(sparse_series, axis=1, copy=False)
        return out

    else:
        dummy_mat = np.eye(number_of_cols, dtype=dtype).take(codes, axis=0)

        if not dummy_na:
            # reset NaN GH4446
            dummy_mat[codes == -1] = 0

        if drop_first:
            # remove first GH12042
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]
        return DataFrame(dummy_mat, index=index, columns=dummy_cols)


def make_axis_dummies(frame, axis='minor', transform=None):
    """
    Construct 1-0 dummy variables corresponding to designated axis
    labels

    Parameters
    ----------
    frame : DataFrame
    axis : {'major', 'minor'}, default 'minor'
    transform : function, default None
        Function to apply to axis labels first. For example, to
        get "day of week" dummies in a time series regression
        you might call::

            make_axis_dummies(panel, axis='major',
                              transform=lambda d: d.weekday())
    Returns
    -------
    dummies : DataFrame
        Column names taken from chosen axis
    """
    numbers = {'major': 0, 'minor': 1}
    num = numbers.get(axis, axis)

    items = frame.index.levels[num]
    codes = frame.index.codes[num]
    if transform is not None:
        mapped_items = items.map(transform)
        codes, items = _factorize_from_iterable(mapped_items.take(codes))

    values = np.eye(len(items), dtype=float)
    values = values.take(codes, axis=0)

    return DataFrame(values, columns=items, index=frame.index)


def _reorder_for_extension_array_stack(arr, n_rows, n_columns):
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    # final take to get the order correct.
    # idx is an indexer like
    # [c0r0, c1r0, c2r0, ...,
    #  c0r1, c1r1, c2r1, ...]
    idx = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.ravel()
    return arr.take(idx)
