"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""
from __future__ import division

from textwrap import dedent
from warnings import catch_warnings, simplefilter, warn

import numpy as np

from pandas._libs import algos, hashtable as htable, lib
from pandas._libs.tslib import iNaT
from pandas.util._decorators import Appender, Substitution, deprecate_kwarg

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike, maybe_promote)
from pandas.core.dtypes.common import (
    ensure_float64, ensure_int64, ensure_object, ensure_platform_int,
    ensure_uint64, is_array_like, is_bool_dtype, is_categorical_dtype,
    is_complex_dtype, is_datetime64_any_dtype, is_datetime64tz_dtype,
    is_datetimelike, is_extension_array_dtype, is_float_dtype,
    is_integer_dtype, is_interval_dtype, is_list_like, is_numeric_dtype,
    is_object_dtype, is_period_dtype, is_scalar, is_signed_integer_dtype,
    is_sparse, is_timedelta64_dtype, is_unsigned_integer_dtype,
    needs_i8_conversion)
from pandas.core.dtypes.generic import ABCIndex, ABCIndexClass, ABCSeries
from pandas.core.dtypes.missing import isna, na_value_for_dtype

from pandas.core import common as com

_shared_docs = {}


# --------------- #
# dtype access    #
# --------------- #
def _ensure_data(values, dtype=None):
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint64 (TODO this should be uint8)
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : array-like
    dtype : pandas_dtype, optional
        coerce to this dtype

    Returns
    -------
    (ndarray, pandas_dtype, algo dtype as a string)

    """

    # we check some simple dtypes first
    try:
        if is_object_dtype(dtype):
            return ensure_object(np.asarray(values)), 'object', 'object'
        if is_bool_dtype(values) or is_bool_dtype(dtype):
            # we are actually coercing to uint64
            # until our algos support uint8 directly (see TODO)
            return np.asarray(values).astype('uint64'), 'bool', 'uint64'
        elif is_signed_integer_dtype(values) or is_signed_integer_dtype(dtype):
            return ensure_int64(values), 'int64', 'int64'
        elif (is_unsigned_integer_dtype(values) or
              is_unsigned_integer_dtype(dtype)):
            return ensure_uint64(values), 'uint64', 'uint64'
        elif is_float_dtype(values) or is_float_dtype(dtype):
            return ensure_float64(values), 'float64', 'float64'
        elif is_object_dtype(values) and dtype is None:
            return ensure_object(np.asarray(values)), 'object', 'object'
        elif is_complex_dtype(values) or is_complex_dtype(dtype):

            # ignore the fact that we are casting to float
            # which discards complex parts
            with catch_warnings():
                simplefilter("ignore", np.ComplexWarning)
                values = ensure_float64(values)
            return values, 'float64', 'float64'

    except (TypeError, ValueError, OverflowError):
        # if we are trying to coerce to a dtype
        # and it is incompat this will fall thru to here
        return ensure_object(values), 'object', 'object'

    # datetimelike
    if (needs_i8_conversion(values) or
            is_period_dtype(dtype) or
            is_datetime64_any_dtype(dtype) or
            is_timedelta64_dtype(dtype)):
        if is_period_dtype(values) or is_period_dtype(dtype):
            from pandas import PeriodIndex
            values = PeriodIndex(values)
            dtype = values.dtype
        elif is_timedelta64_dtype(values) or is_timedelta64_dtype(dtype):
            from pandas import TimedeltaIndex
            values = TimedeltaIndex(values)
            dtype = values.dtype
        else:
            # Datetime
            from pandas import DatetimeIndex
            values = DatetimeIndex(values)
            dtype = values.dtype

        return values.asi8, dtype, 'int64'

    elif (is_categorical_dtype(values) and
          (is_categorical_dtype(dtype) or dtype is None)):
        values = getattr(values, 'values', values)
        values = values.codes
        dtype = 'category'

        # we are actually coercing to int64
        # until our algos support int* directly (not all do)
        values = ensure_int64(values)

        return values, dtype, 'int64'

    # we have failed, return object
    values = np.asarray(values, dtype=np.object)
    return ensure_object(values), 'object', 'object'


def _reconstruct_data(values, dtype, original):
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : ndarray
    dtype : pandas_dtype
    original : ndarray-like

    Returns
    -------
    Index for extension types, otherwise ndarray casted to dtype
    """
    from pandas import Index
    if is_extension_array_dtype(dtype):
        values = dtype.construct_array_type()._from_sequence(values)
    elif is_datetime64tz_dtype(dtype) or is_period_dtype(dtype):
        values = Index(original)._shallow_copy(values, name=None)
    elif is_bool_dtype(dtype):
        values = values.astype(dtype)

        # we only support object dtypes bool Index
        if isinstance(original, Index):
            values = values.astype(object)
    elif dtype is not None:
        values = values.astype(dtype)

    return values


def _ensure_arraylike(values):
    """
    ensure that we are arraylike if not already
    """
    if not is_array_like(values):
        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ['mixed', 'string', 'unicode']:
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    'float64': (htable.Float64HashTable, htable.Float64Vector),
    'uint64': (htable.UInt64HashTable, htable.UInt64Vector),
    'int64': (htable.Int64HashTable, htable.Int64Vector),
    'string': (htable.StringHashTable, htable.ObjectVector),
    'object': (htable.PyObjectHashTable, htable.ObjectVector)
}


def _get_hashtable_algo(values):
    """
    Parameters
    ----------
    values : arraylike

    Returns
    -------
    tuples(hashtable class,
           vector class,
           values,
           dtype,
           ndtype)
    """
    values, dtype, ndtype = _ensure_data(values)

    if ndtype == 'object':

        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.infer_dtype(values, skipna=False) in ['string']:
            ndtype = 'string'
        else:
            ndtype = 'object'

    htable, table = _hashtables[ndtype]
    return (htable, table, values, dtype, ndtype)


def _get_data_algo(values, func_map):

    if is_categorical_dtype(values):
        values = values._values_for_rank()

    values, dtype, ndtype = _ensure_data(values)
    if ndtype == 'object':

        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.infer_dtype(values, skipna=False) in ['string']:
            ndtype = 'string'

    f = func_map.get(ndtype, func_map['object'])

    return f, values


# --------------- #
# top-level algos #
# --------------- #

def match(to_match, values, na_sentinel=-1):
    """
    Compute locations of to_match into values

    Parameters
    ----------
    to_match : array-like
        values to find positions of
    values : array-like
        Unique set of values
    na_sentinel : int, default -1
        Value to mark "not found"

    Examples
    --------

    Returns
    -------
    match : ndarray of integers
    """
    values = com.asarray_tuplesafe(values)
    htable, _, values, dtype, ndtype = _get_hashtable_algo(values)
    to_match, _, _ = _ensure_data(to_match, dtype)
    table = htable(min(len(to_match), 1000000))
    table.map_locations(values)
    result = table.lookup(to_match)

    if na_sentinel != -1:

        # replace but return a numpy array
        # use a Series because it handles dtype conversions properly
        from pandas import Series
        result = Series(result.ravel()).replace(-1, na_sentinel)
        result = result.values.reshape(result.shape)

    return result


def unique(values):
    """
    Hash table-based unique. Uniques are returned in order
    of appearance. This does NOT sort.

    Significantly faster than numpy.unique. Includes NA values.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    unique values.
      - If the input is an Index, the return is an Index
      - If the input is a Categorical dtype, the return is a Categorical
      - If the input is a Series/ndarray, the return will be an ndarray

    See Also
    --------
    pandas.Index.unique
    pandas.Series.unique

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp('20160101'),
    ...                     pd.Timestamp('20160101')]))
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> pd.unique(pd.Series([pd.Timestamp('20160101', tz='US/Eastern'),
    ...                      pd.Timestamp('20160101', tz='US/Eastern')]))
    array([Timestamp('2016-01-01 00:00:00-0500', tz='US/Eastern')],
          dtype=object)

    >>> pd.unique(pd.Index([pd.Timestamp('20160101', tz='US/Eastern'),
    ...                     pd.Timestamp('20160101', tz='US/Eastern')]))
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
    ...           dtype='datetime64[ns, US/Eastern]', freq=None)

    >>> pd.unique(list('baabc'))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'))))
    [b, a, c]
    Categories (3, object): [b, a, c]

    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'),
    ...                                    categories=list('abc'))))
    [b, a, c]
    Categories (3, object): [b, a, c]

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(pd.Series(pd.Categorical(list('baabc'),
    ...                                    categories=list('abc'),
    ...                                    ordered=True)))
    [b, a, c]
    Categories (3, object): [a < b < c]

    An array of tuples

    >>> pd.unique([('a', 'b'), ('b', 'a'), ('a', 'c'), ('b', 'a')])
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """

    values = _ensure_arraylike(values)

    if is_extension_array_dtype(values):
        # Dispatch to extension dtype's unique.
        return values.unique()

    original = values
    htable, _, values, dtype, ndtype = _get_hashtable_algo(values)

    table = htable(len(values))
    uniques = table.unique(values)
    uniques = _reconstruct_data(uniques, dtype, original)
    return uniques


unique1d = unique


def isin(comps, values):
    """
    Compute the isin boolean array

    Parameters
    ----------
    comps : array-like
    values : array-like

    Returns
    -------
    boolean array same length as comps
    """

    if not is_list_like(comps):
        raise TypeError("only list-like objects are allowed to be passed"
                        " to isin(), you passed a [{comps_type}]"
                        .format(comps_type=type(comps).__name__))
    if not is_list_like(values):
        raise TypeError("only list-like objects are allowed to be passed"
                        " to isin(), you passed a [{values_type}]"
                        .format(values_type=type(values).__name__))

    if not isinstance(values, (ABCIndex, ABCSeries, np.ndarray)):
        values = construct_1d_object_array_from_listlike(list(values))

    if is_categorical_dtype(comps):
        # TODO(extension)
        # handle categoricals
        return comps._values.isin(values)

    comps = com.values_from_object(comps)

    comps, dtype, _ = _ensure_data(comps)
    values, _, _ = _ensure_data(values, dtype=dtype)

    # faster for larger cases to use np.in1d
    f = lambda x, y: htable.ismember_object(x, values)

    # GH16012
    # Ensure np.in1d doesn't get object types or it *may* throw an exception
    if len(comps) > 1000000 and not is_object_dtype(comps):
        f = lambda x, y: np.in1d(x, y)
    elif is_integer_dtype(comps):
        try:
            values = values.astype('int64', copy=False)
            comps = comps.astype('int64', copy=False)
            f = lambda x, y: htable.ismember_int64(x, y)
        except (TypeError, ValueError, OverflowError):
            values = values.astype(object)
            comps = comps.astype(object)

    elif is_float_dtype(comps):
        try:
            values = values.astype('float64', copy=False)
            comps = comps.astype('float64', copy=False)
            f = lambda x, y: htable.ismember_float64(x, y)
        except (TypeError, ValueError):
            values = values.astype(object)
            comps = comps.astype(object)

    return f(comps, values)


def _factorize_array(values, na_sentinel=-1, size_hint=None,
                     na_value=None):
    """Factorize an array-like to labels and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    na_sentinel : int, default -1
    size_hint : int, optional
        Passsed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).

    Returns
    -------
    labels, uniques : ndarray
    """
    (hash_klass, _), values = _get_data_algo(values, _hashtables)

    table = hash_klass(size_hint or len(values))
    uniques, labels = table.factorize(values, na_sentinel=na_sentinel,
                                      na_value=na_value)

    labels = ensure_platform_int(labels)
    return labels, uniques


_shared_docs['factorize'] = """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    %(values)s%(sort)s%(order)s
    na_sentinel : int, default -1
        Value to mark "not found".
    %(size_hint)s\

    Returns
    -------
    labels : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(labels)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note ::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> labels, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
    >>> labels
    array([0, 0, 1, 2, 0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `labels` will be
    shuffled so that the relationship is the maintained.

    >>> labels, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'], sort=True)
    >>> labels
    array([1, 1, 0, 2, 1])
    >>> uniques
    array(['a', 'b', 'c'], dtype=object)

    Missing values are indicated in `labels` with `na_sentinel`
    (``-1`` by default). Note that missing values are never
    included in `uniques`.

    >>> labels, uniques = pd.factorize(['b', None, 'a', 'c', 'b'])
    >>> labels
    array([ 0, -1,  1,  2,  0])
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = pd.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> labels, uniques = pd.factorize(cat)
    >>> labels
    array([0, 0, 1])
    >>> uniques
    [a, c]
    Categories (3, object): [a, b, c]

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.

    For all other pandas objects, an Index of the appropriate type is
    returned.

    >>> cat = pd.Series(['a', 'a', 'c'])
    >>> labels, uniques = pd.factorize(cat)
    >>> labels
    array([0, 0, 1])
    >>> uniques
    Index(['a', 'c'], dtype='object')
    """


@Substitution(
    values=dedent("""\
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    """),
    order=dedent("""\
    order
        .. deprecated:: 0.23.0

           This parameter has no effect and is deprecated.
    """),
    sort=dedent("""\
    sort : bool, default False
        Sort `uniques` and shuffle `labels` to maintain the
        relationship.
    """),
    size_hint=dedent("""\
    size_hint : int, optional
        Hint to the hashtable sizer.
    """),
)
@Appender(_shared_docs['factorize'])
@deprecate_kwarg(old_arg_name='order', new_arg_name=None)
def factorize(values, sort=False, order=None, na_sentinel=-1, size_hint=None):
    # Implementation notes: This method is responsible for 3 things
    # 1.) coercing data to array-like (ndarray, Index, extension array)
    # 2.) factorizing labels and uniques
    # 3.) Maybe boxing the output in an Index
    #
    # Step 2 is dispatched to extension types (like Categorical). They are
    # responsible only for factorization. All data coercion, sorting and boxing
    # should happen here.

    values = _ensure_arraylike(values)
    original = values

    if is_extension_array_dtype(values):
        values = getattr(values, '_values', values)
        labels, uniques = values.factorize(na_sentinel=na_sentinel)
        dtype = original.dtype
    else:
        values, dtype, _ = _ensure_data(values)

        if (is_datetime64_any_dtype(original) or
                is_timedelta64_dtype(original) or
                is_period_dtype(original)):
            na_value = na_value_for_dtype(original.dtype)
        else:
            na_value = None

        labels, uniques = _factorize_array(values,
                                           na_sentinel=na_sentinel,
                                           size_hint=size_hint,
                                           na_value=na_value)

    if sort and len(uniques) > 0:
        from pandas.core.sorting import safe_sort
        if na_sentinel == -1:
            # GH-25409 take_1d only works for na_sentinels of -1
            try:
                order = uniques.argsort()
                order2 = order.argsort()
                labels = take_1d(order2, labels, fill_value=na_sentinel)
                uniques = uniques.take(order)
            except TypeError:
                # Mixed types, where uniques.argsort fails.
                uniques, labels = safe_sort(uniques, labels,
                                            na_sentinel=na_sentinel,
                                            assume_unique=True)
        else:
            uniques, labels = safe_sort(uniques, labels,
                                        na_sentinel=na_sentinel,
                                        assume_unique=True)

    uniques = _reconstruct_data(uniques, dtype, original)

    # return original tenor
    if isinstance(original, ABCIndexClass):
        uniques = original._shallow_copy(uniques, name=None)
    elif isinstance(original, ABCSeries):
        from pandas import Index
        uniques = Index(uniques)

    return labels, uniques


def value_counts(values, sort=True, ascending=False, normalize=False,
                 bins=None, dropna=True):
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
    sort : boolean, default True
        Sort by values
    ascending : boolean, default False
        Sort in ascending order
    normalize: boolean, default False
        If True then compute a relative histogram
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data
    dropna : boolean, default True
        Don't include counts of NaN

    Returns
    -------
    value_counts : Series

    """
    from pandas.core.series import Series, Index
    name = getattr(values, 'name', None)

    if bins is not None:
        try:
            from pandas.core.reshape.tile import cut
            values = Series(values)
            ii = cut(values, bins, include_lowest=True)
        except TypeError:
            raise TypeError("bins argument only works with numeric data.")

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result = result[result.index.notna()]
        result.index = result.index.astype('interval')
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result.values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        counts = np.array([len(ii)])

    else:

        if is_extension_array_dtype(values) or is_sparse(values):

            # handle Categorical and sparse,
            result = Series(values)._values.value_counts(dropna=dropna)
            result.name = name
            counts = result.values

        else:
            keys, counts = _value_counts_arraylike(values, dropna)

            if not isinstance(keys, Index):
                keys = Index(keys)
            result = Series(counts, index=keys, name=name)

    if sort:
        result = result.sort_values(ascending=ascending)

    if normalize:
        result = result / float(counts.sum())

    return result


def _value_counts_arraylike(values, dropna):
    """
    Parameters
    ----------
    values : arraylike
    dropna : boolean

    Returns
    -------
    (uniques, counts)

    """
    values = _ensure_arraylike(values)
    original = values
    values, dtype, ndtype = _ensure_data(values)

    if needs_i8_conversion(dtype):
        # i8

        keys, counts = htable.value_count_int64(values, dropna)

        if dropna:
            msk = keys != iNaT
            keys, counts = keys[msk], counts[msk]

    else:
        # ndarray like

        # TODO: handle uint8
        f = getattr(htable, "value_count_{dtype}".format(dtype=ndtype))
        keys, counts = f(values, dropna)

        mask = isna(values)
        if not dropna and mask.any():
            if not isna(keys).any():
                keys = np.insert(keys, 0, np.NaN)
                counts = np.insert(counts, 0, mask.sum())

    keys = _reconstruct_data(keys, original.dtype, original)

    return keys, counts


def duplicated(values, keep='first'):
    """
    Return boolean ndarray denoting duplicate values.

    .. versionadded:: 0.19.0

    Parameters
    ----------
    values : ndarray-like
        Array over which to check for duplicate values.
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : Mark duplicates as ``True`` except for the first
          occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last
          occurrence.
        - False : Mark all duplicates as ``True``.

    Returns
    -------
    duplicated : ndarray
    """

    values, dtype, ndtype = _ensure_data(values)
    f = getattr(htable, "duplicated_{dtype}".format(dtype=ndtype))
    return f(values, keep=keep)


def mode(values, dropna=True):
    """
    Returns the mode(s) of an array.

    Parameters
    ----------
    values : array-like
        Array over which to check for duplicate values.
    dropna : boolean, default True
        Don't consider counts of NaN/NaT.

        .. versionadded:: 0.24.0

    Returns
    -------
    mode : Series
    """
    from pandas import Series

    values = _ensure_arraylike(values)
    original = values

    # categorical is a fast-path
    if is_categorical_dtype(values):
        if isinstance(values, Series):
            return Series(values.values.mode(dropna=dropna), name=values.name)
        return values.mode(dropna=dropna)

    if dropna and is_datetimelike(values):
        mask = values.isnull()
        values = values[~mask]

    values, dtype, ndtype = _ensure_data(values)

    f = getattr(htable, "mode_{dtype}".format(dtype=ndtype))
    result = f(values, dropna=dropna)
    try:
        result = np.sort(result)
    except TypeError as e:
        warn("Unable to sort modes: {error}".format(error=e))

    result = _reconstruct_data(result, original.dtype, original)
    return Series(result)


def rank(values, axis=0, method='average', na_option='keep',
         ascending=True, pct=False):
    """
    Rank the values along a given axis.

    Parameters
    ----------
    values : array-like
        Array whose values will be ranked. The number of dimensions in this
        array must not exceed 2.
    axis : int, default 0
        Axis over which to perform rankings.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        The method by which tiebreaks are broken during the ranking.
    na_option : {'keep', 'top'}, default 'keep'
        The method by which NaNs are placed in the ranking.
        - ``keep``: rank each NaN value with a NaN ranking
        - ``top``: replace each NaN with either +/- inf so that they
                   there are ranked at the top
    ascending : boolean, default True
        Whether or not the elements should be ranked in ascending order.
    pct : boolean, default False
        Whether or not to the display the returned rankings in integer form
        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).
    """
    if values.ndim == 1:
        f, values = _get_data_algo(values, _rank1d_functions)
        ranks = f(values, ties_method=method, ascending=ascending,
                  na_option=na_option, pct=pct)
    elif values.ndim == 2:
        f, values = _get_data_algo(values, _rank2d_functions)
        ranks = f(values, axis=axis, ties_method=method,
                  ascending=ascending, na_option=na_option, pct=pct)
    else:
        raise TypeError("Array with ndim > 2 are not supported.")

    return ranks


def checked_add_with_arr(arr, b, arr_mask=None, b_mask=None):
    """
    Perform array addition that checks for underflow and overflow.

    Performs the addition of an int64 array and an int64 integer (or array)
    but checks that they do not result in overflow first. For elements that
    are indicated to be NaN, whether or not there is overflow for that element
    is automatically ignored.

    Parameters
    ----------
    arr : array addend.
    b : array or scalar addend.
    arr_mask : boolean array or None
        array indicating which elements to exclude from checking
    b_mask : boolean array or boolean or None
        array or scalar indicating which element(s) to exclude from checking

    Returns
    -------
    sum : An array for elements x + b for each element x in arr if b is
          a scalar or an array for elements x + y for each element pair
          (x, y) in (arr, b).

    Raises
    ------
    OverflowError if any x + y exceeds the maximum or minimum int64 value.
    """
    # For performance reasons, we broadcast 'b' to the new array 'b2'
    # so that it has the same size as 'arr'.
    b2 = np.broadcast_to(b, arr.shape)
    if b_mask is not None:
        # We do the same broadcasting for b_mask as well.
        b2_mask = np.broadcast_to(b_mask, arr.shape)
    else:
        b2_mask = None

    # For elements that are NaN, regardless of their value, we should
    # ignore whether they overflow or not when doing the checked add.
    if arr_mask is not None and b2_mask is not None:
        not_nan = np.logical_not(arr_mask | b2_mask)
    elif arr_mask is not None:
        not_nan = np.logical_not(arr_mask)
    elif b_mask is not None:
        not_nan = np.logical_not(b2_mask)
    else:
        not_nan = np.empty(arr.shape, dtype=bool)
        not_nan.fill(True)

    # gh-14324: For each element in 'arr' and its corresponding element
    # in 'b2', we check the sign of the element in 'b2'. If it is positive,
    # we then check whether its sum with the element in 'arr' exceeds
    # np.iinfo(np.int64).max. If so, we have an overflow error. If it
    # it is negative, we then check whether its sum with the element in
    # 'arr' exceeds np.iinfo(np.int64).min. If so, we have an overflow
    # error as well.
    mask1 = b2 > 0
    mask2 = b2 < 0

    if not mask1.any():
        to_raise = ((np.iinfo(np.int64).min - b2 > arr) & not_nan).any()
    elif not mask2.any():
        to_raise = ((np.iinfo(np.int64).max - b2 < arr) & not_nan).any()
    else:
        to_raise = (((np.iinfo(np.int64).max -
                      b2[mask1] < arr[mask1]) & not_nan[mask1]).any() or
                    ((np.iinfo(np.int64).min -
                      b2[mask2] > arr[mask2]) & not_nan[mask2]).any())

    if to_raise:
        raise OverflowError("Overflow in int64 addition")
    return arr + b


_rank1d_functions = {
    'float64': algos.rank_1d_float64,
    'int64': algos.rank_1d_int64,
    'uint64': algos.rank_1d_uint64,
    'object': algos.rank_1d_object
}

_rank2d_functions = {
    'float64': algos.rank_2d_float64,
    'int64': algos.rank_2d_int64,
    'uint64': algos.rank_2d_uint64,
    'object': algos.rank_2d_object
}


def quantile(x, q, interpolation_method='fraction'):
    """
    Compute sample quantile or quantiles of the input array. For example, q=0.5
    computes the median.

    The `interpolation_method` parameter supports three values, namely
    `fraction` (default), `lower` and `higher`. Interpolation is done only,
    if the desired quantile lies between two data points `i` and `j`. For
    `fraction`, the result is an interpolated value between `i` and `j`;
    for `lower`, the result is `i`, for `higher` the result is `j`.

    Parameters
    ----------
    x : ndarray
        Values from which to extract score.
    q : scalar or array
        Percentile at which to extract score.
    interpolation_method : {'fraction', 'lower', 'higher'}, optional
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - fraction: `i + (j - i)*fraction`, where `fraction` is the
                    fractional part of the index surrounded by `i` and `j`.
        -lower: `i`.
        - higher: `j`.

    Returns
    -------
    score : float
        Score at percentile.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5

    """
    x = np.asarray(x)
    mask = isna(x)

    x = x[~mask]

    values = np.sort(x)

    def _interpolate(a, b, fraction):
        """Returns the point at the given fraction between a and b, where
        'fraction' must be between 0 and 1.
        """
        return a + (b - a) * fraction

    def _get_score(at):
        if len(values) == 0:
            return np.nan

        idx = at * (len(values) - 1)
        if idx % 1 == 0:
            score = values[int(idx)]
        else:
            if interpolation_method == 'fraction':
                score = _interpolate(values[int(idx)], values[int(idx) + 1],
                                     idx % 1)
            elif interpolation_method == 'lower':
                score = values[np.floor(idx)]
            elif interpolation_method == 'higher':
                score = values[np.ceil(idx)]
            else:
                raise ValueError("interpolation_method can only be 'fraction' "
                                 ", 'lower' or 'higher'")

        return score

    if is_scalar(q):
        return _get_score(q)
    else:
        q = np.asarray(q, np.float64)
        return algos.arrmap_float64(q, _get_score)


# --------------- #
# select n        #
# --------------- #

class SelectN(object):

    def __init__(self, obj, n, keep):
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ('first', 'last', 'all'):
            raise ValueError('keep must be either "first", "last" or "all"')

    def nlargest(self):
        return self.compute('nlargest')

    def nsmallest(self):
        return self.compute('nsmallest')

    @staticmethod
    def is_valid_dtype_n_method(dtype):
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        return ((is_numeric_dtype(dtype) and not is_complex_dtype(dtype)) or
                needs_i8_conversion(dtype))


class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method):

        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError("Cannot use method '{method}' with "
                            "dtype {dtype}".format(method=method,
                                                   dtype=dtype))

        if n <= 0:
            return self.obj[[]]

        dropped = self.obj.dropna()

        # slow method
        if n >= len(self.obj):

            reverse_it = (self.keep == 'last' or method == 'nlargest')
            ascending = method == 'nsmallest'
            slc = np.s_[::-1] if reverse_it else np.s_[:]
            return dropped[slc].sort_values(ascending=ascending).head(n)

        # fast method
        arr, pandas_dtype, _ = _ensure_data(dropped.values)
        if method == 'nlargest':
            arr = -arr
            if is_integer_dtype(pandas_dtype):
                # GH 21426: ensure reverse ordering at boundaries
                arr -= 1

        if self.keep == 'last':
            arr = arr[::-1]

        narr = len(arr)
        n = min(n, narr)

        kth_val = algos.kth_smallest(arr.copy(), n - 1)
        ns, = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind='mergesort')]

        if self.keep != 'all':
            inds = inds[:n]

        if self.keep == 'last':
            # reverse indices
            inds = narr - 1 - inds

        return dropped.iloc[inds]


class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj, n, keep, columns):
        super(SelectNFrame, self).__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]
        columns = list(columns)
        self.columns = columns

    def compute(self, method):

        from pandas import Int64Index
        n = self.n
        frame = self.obj
        columns = self.columns

        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError((
                    "Column {column!r} has dtype {dtype}, cannot use method "
                    "{method!r} with this dtype"
                ).format(column=column, dtype=dtype, method=method))

        def get_indexer(current_indexer, other_indexer):
            """Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == 'nsmallest':
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # Below we save and reset the index in case index contains duplicates
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Int64Index([])

        for i, column in enumerate(columns):
            # For each column we apply method to cur_frame[column].
            # If it's the last column or if we have the number of
            # results desired we are done.
            # Otherwise there are duplicates of the largest/smallest
            # value and we need to look at the rest of the columns
            # to determine which of the rows with the largest/smallest
            # value in the column to keep.
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(
                cur_n,
                keep=self.keep if is_last_column else 'all')

            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break

            # Now find all values which are equal to
            # the (nsmallest: largest)/(nlarrgest: smallest)
            # from our series.
            border_value = values == values[values.index[-1]]

            # Some of these values are among the top-n
            # some aren't.
            unsafe_values = values[border_value]

            # These values are definitely among the top-n
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # Go on and separate the unsafe_values on the remaining
            # columns.
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        frame = frame.take(indexer)

        # Restore the index on frame
        frame.index = original_index.take(indexer)

        # If there is only one column, the frame is already sorted.
        if len(columns) == 1:
            return frame

        ascending = method == 'nsmallest'

        return frame.sort_values(
            columns,
            ascending=ascending,
            kind='mergesort')


# ------- ## ---- #
# take #
# ---- #


def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):
    def wrapper(arr, indexer, out, fill_value=np.nan):
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        if out_dtype is not None:
            out = out.view(out_dtype)
        if fill_wrap is not None:
            fill_value = fill_wrap(fill_value)
        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


def _convert_wrapper(f, conv_dtype):
    def wrapper(arr, indexer, out, fill_value=np.nan):
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)

    return wrapper


def _take_2d_multi_object(arr, indexer, out, fill_value, mask_info):
    # this is not ideal, performance-wise, but it's better than raising
    # an exception (best to optimize in Cython to avoid getting here)
    row_idx, col_idx = indexer
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value
    for i in range(len(row_idx)):
        u_ = row_idx[i]
        for j in range(len(col_idx)):
            v = col_idx[j]
            out[i, j] = arr[u_, v]


def _take_nd_object(arr, indexer, out, axis, fill_value, mask_info):
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    if arr.shape[axis] > 0:
        arr.take(ensure_platform_int(indexer), axis=axis, out=out)
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value


_take_1d_dict = {
    ('int8', 'int8'): algos.take_1d_int8_int8,
    ('int8', 'int32'): algos.take_1d_int8_int32,
    ('int8', 'int64'): algos.take_1d_int8_int64,
    ('int8', 'float64'): algos.take_1d_int8_float64,
    ('int16', 'int16'): algos.take_1d_int16_int16,
    ('int16', 'int32'): algos.take_1d_int16_int32,
    ('int16', 'int64'): algos.take_1d_int16_int64,
    ('int16', 'float64'): algos.take_1d_int16_float64,
    ('int32', 'int32'): algos.take_1d_int32_int32,
    ('int32', 'int64'): algos.take_1d_int32_int64,
    ('int32', 'float64'): algos.take_1d_int32_float64,
    ('int64', 'int64'): algos.take_1d_int64_int64,
    ('int64', 'float64'): algos.take_1d_int64_float64,
    ('float32', 'float32'): algos.take_1d_float32_float32,
    ('float32', 'float64'): algos.take_1d_float32_float64,
    ('float64', 'float64'): algos.take_1d_float64_float64,
    ('object', 'object'): algos.take_1d_object_object,
    ('bool', 'bool'): _view_wrapper(algos.take_1d_bool_bool, np.uint8,
                                    np.uint8),
    ('bool', 'object'): _view_wrapper(algos.take_1d_bool_object, np.uint8,
                                      None),
    ('datetime64[ns]', 'datetime64[ns]'): _view_wrapper(
        algos.take_1d_int64_int64, np.int64, np.int64, np.int64)
}

_take_2d_axis0_dict = {
    ('int8', 'int8'): algos.take_2d_axis0_int8_int8,
    ('int8', 'int32'): algos.take_2d_axis0_int8_int32,
    ('int8', 'int64'): algos.take_2d_axis0_int8_int64,
    ('int8', 'float64'): algos.take_2d_axis0_int8_float64,
    ('int16', 'int16'): algos.take_2d_axis0_int16_int16,
    ('int16', 'int32'): algos.take_2d_axis0_int16_int32,
    ('int16', 'int64'): algos.take_2d_axis0_int16_int64,
    ('int16', 'float64'): algos.take_2d_axis0_int16_float64,
    ('int32', 'int32'): algos.take_2d_axis0_int32_int32,
    ('int32', 'int64'): algos.take_2d_axis0_int32_int64,
    ('int32', 'float64'): algos.take_2d_axis0_int32_float64,
    ('int64', 'int64'): algos.take_2d_axis0_int64_int64,
    ('int64', 'float64'): algos.take_2d_axis0_int64_float64,
    ('float32', 'float32'): algos.take_2d_axis0_float32_float32,
    ('float32', 'float64'): algos.take_2d_axis0_float32_float64,
    ('float64', 'float64'): algos.take_2d_axis0_float64_float64,
    ('object', 'object'): algos.take_2d_axis0_object_object,
    ('bool', 'bool'): _view_wrapper(algos.take_2d_axis0_bool_bool, np.uint8,
                                    np.uint8),
    ('bool', 'object'): _view_wrapper(algos.take_2d_axis0_bool_object,
                                      np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'):
    _view_wrapper(algos.take_2d_axis0_int64_int64, np.int64, np.int64,
                  fill_wrap=np.int64)
}

_take_2d_axis1_dict = {
    ('int8', 'int8'): algos.take_2d_axis1_int8_int8,
    ('int8', 'int32'): algos.take_2d_axis1_int8_int32,
    ('int8', 'int64'): algos.take_2d_axis1_int8_int64,
    ('int8', 'float64'): algos.take_2d_axis1_int8_float64,
    ('int16', 'int16'): algos.take_2d_axis1_int16_int16,
    ('int16', 'int32'): algos.take_2d_axis1_int16_int32,
    ('int16', 'int64'): algos.take_2d_axis1_int16_int64,
    ('int16', 'float64'): algos.take_2d_axis1_int16_float64,
    ('int32', 'int32'): algos.take_2d_axis1_int32_int32,
    ('int32', 'int64'): algos.take_2d_axis1_int32_int64,
    ('int32', 'float64'): algos.take_2d_axis1_int32_float64,
    ('int64', 'int64'): algos.take_2d_axis1_int64_int64,
    ('int64', 'float64'): algos.take_2d_axis1_int64_float64,
    ('float32', 'float32'): algos.take_2d_axis1_float32_float32,
    ('float32', 'float64'): algos.take_2d_axis1_float32_float64,
    ('float64', 'float64'): algos.take_2d_axis1_float64_float64,
    ('object', 'object'): algos.take_2d_axis1_object_object,
    ('bool', 'bool'): _view_wrapper(algos.take_2d_axis1_bool_bool, np.uint8,
                                    np.uint8),
    ('bool', 'object'): _view_wrapper(algos.take_2d_axis1_bool_object,
                                      np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'):
    _view_wrapper(algos.take_2d_axis1_int64_int64, np.int64, np.int64,
                  fill_wrap=np.int64)
}

_take_2d_multi_dict = {
    ('int8', 'int8'): algos.take_2d_multi_int8_int8,
    ('int8', 'int32'): algos.take_2d_multi_int8_int32,
    ('int8', 'int64'): algos.take_2d_multi_int8_int64,
    ('int8', 'float64'): algos.take_2d_multi_int8_float64,
    ('int16', 'int16'): algos.take_2d_multi_int16_int16,
    ('int16', 'int32'): algos.take_2d_multi_int16_int32,
    ('int16', 'int64'): algos.take_2d_multi_int16_int64,
    ('int16', 'float64'): algos.take_2d_multi_int16_float64,
    ('int32', 'int32'): algos.take_2d_multi_int32_int32,
    ('int32', 'int64'): algos.take_2d_multi_int32_int64,
    ('int32', 'float64'): algos.take_2d_multi_int32_float64,
    ('int64', 'int64'): algos.take_2d_multi_int64_int64,
    ('int64', 'float64'): algos.take_2d_multi_int64_float64,
    ('float32', 'float32'): algos.take_2d_multi_float32_float32,
    ('float32', 'float64'): algos.take_2d_multi_float32_float64,
    ('float64', 'float64'): algos.take_2d_multi_float64_float64,
    ('object', 'object'): algos.take_2d_multi_object_object,
    ('bool', 'bool'): _view_wrapper(algos.take_2d_multi_bool_bool, np.uint8,
                                    np.uint8),
    ('bool', 'object'): _view_wrapper(algos.take_2d_multi_bool_object,
                                      np.uint8, None),
    ('datetime64[ns]', 'datetime64[ns]'):
    _view_wrapper(algos.take_2d_multi_int64_int64, np.int64, np.int64,
                  fill_wrap=np.int64)
}


def _get_take_nd_function(ndim, arr_dtype, out_dtype, axis=0, mask_info=None):
    if ndim <= 2:
        tup = (arr_dtype.name, out_dtype.name)
        if ndim == 1:
            func = _take_1d_dict.get(tup, None)
        elif ndim == 2:
            if axis == 0:
                func = _take_2d_axis0_dict.get(tup, None)
            else:
                func = _take_2d_axis1_dict.get(tup, None)
        if func is not None:
            return func

        tup = (out_dtype.name, out_dtype.name)
        if ndim == 1:
            func = _take_1d_dict.get(tup, None)
        elif ndim == 2:
            if axis == 0:
                func = _take_2d_axis0_dict.get(tup, None)
            else:
                func = _take_2d_axis1_dict.get(tup, None)
        if func is not None:
            func = _convert_wrapper(func, out_dtype)
            return func

    def func(arr, indexer, out, fill_value=np.nan):
        indexer = ensure_int64(indexer)
        _take_nd_object(arr, indexer, out, axis=axis, fill_value=fill_value,
                        mask_info=mask_info)

    return func


def take(arr, indices, axis=0, allow_fill=False, fill_value=None):
    """
    Take elements from an array.

    .. versionadded:: 0.23.0

    Parameters
    ----------
    arr : sequence
        Non array-likes (sequences without a dtype) are coerced
        to an ndarray.
    indices : sequence of integers
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          other negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take

    Examples
    --------
    >>> from pandas.api.extensions import take

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True,
    ...      fill_value=-10)
    array([ 10,  10, -10])
    """
    from pandas.core.indexing import validate_indices

    if not is_array_like(arr):
        arr = np.asarray(arr)

    indices = np.asarray(indices, dtype=np.intp)

    if allow_fill:
        # Pandas style, -1 means NA
        validate_indices(indices, len(arr))
        result = take_1d(arr, indices, axis=axis, allow_fill=True,
                         fill_value=fill_value)
    else:
        # NumPy style
        result = arr.take(indices, axis=axis)
    return result


def take_nd(arr, indexer, axis=0, out=None, fill_value=np.nan, mask_info=None,
            allow_fill=True):
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays. It does not
    currently dispatch to ``SparseArray.take`` for sparse ``arr``.

    Parameters
    ----------
    arr : array-like
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    out : ndarray or None, default None
        Optional output array, must be appropriate type to hold input and
        fill_value together, if indexer has any -1 value entries; call
        _maybe_promote to determine this type for any fill_value
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    mask_info : tuple of (ndarray, boolean)
        If provided, value should correspond to:
            (indexer != -1, (indexer != -1).any())
        If not provided, it will be computed internally if necessary
    allow_fill : boolean, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : array-like
        May be the same type as the input, or cast to an ndarray.
    """

    # TODO(EA): Remove these if / elifs as datetimeTZ, interval, become EAs
    # dispatch to internal type takes
    if is_extension_array_dtype(arr):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
    elif is_datetime64tz_dtype(arr):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
    elif is_interval_dtype(arr):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    if is_sparse(arr):
        arr = arr.get_values()
    elif isinstance(arr, (ABCIndexClass, ABCSeries)):
        arr = arr.values

    arr = np.asarray(arr)

    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.int64)
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        indexer = ensure_int64(indexer, copy=False)
        if not allow_fill:
            dtype, fill_value = arr.dtype, arr.dtype.type()
            mask_info = None, False
        else:
            # check for promotion based on types only (do this first because
            # it's faster than computing a mask)
            dtype, fill_value = maybe_promote(arr.dtype, fill_value)
            if dtype != arr.dtype and (out is None or out.dtype != dtype):
                # check if promotion is actually required based on indexer
                if mask_info is not None:
                    mask, needs_masking = mask_info
                else:
                    mask = indexer == -1
                    needs_masking = mask.any()
                    mask_info = mask, needs_masking
                if needs_masking:
                    if out is not None and out.dtype != dtype:
                        raise TypeError('Incompatible type for fill_value')
                else:
                    # if not, then depromote, set fill_value to dummy
                    # (it won't be used but we don't want the cython code
                    # to crash when trying to cast it to dtype)
                    dtype, fill_value = arr.dtype, arr.dtype.type()

    flip_order = False
    if arr.ndim == 2:
        if arr.flags.f_contiguous:
            flip_order = True

    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1
        if out is not None:
            out = out.T

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    if out is None:
        out_shape = list(arr.shape)
        out_shape[axis] = len(indexer)
        out_shape = tuple(out_shape)
        if arr.flags.f_contiguous and axis == arr.ndim - 1:
            # minor tweak that can make an order-of-magnitude difference
            # for dataframes initialized directly from 2-d ndarrays
            # (s.t. df.values is c-contiguous and df._data.blocks[0] is its
            # f-contiguous transpose)
            out = np.empty(out_shape, dtype=dtype, order='F')
        else:
            out = np.empty(out_shape, dtype=dtype)

    func = _get_take_nd_function(arr.ndim, arr.dtype, out.dtype, axis=axis,
                                 mask_info=mask_info)
    func(arr, indexer, out, fill_value)

    if flip_order:
        out = out.T
    return out


take_1d = take_nd


def take_2d_multi(arr, indexer, out=None, fill_value=np.nan, mask_info=None,
                  allow_fill=True):
    """
    Specialized Cython take which sets NaN values in one pass
    """
    if indexer is None or (indexer[0] is None and indexer[1] is None):
        row_idx = np.arange(arr.shape[0], dtype=np.int64)
        col_idx = np.arange(arr.shape[1], dtype=np.int64)
        indexer = row_idx, col_idx
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        row_idx, col_idx = indexer
        if row_idx is None:
            row_idx = np.arange(arr.shape[0], dtype=np.int64)
        else:
            row_idx = ensure_int64(row_idx)
        if col_idx is None:
            col_idx = np.arange(arr.shape[1], dtype=np.int64)
        else:
            col_idx = ensure_int64(col_idx)
        indexer = row_idx, col_idx
        if not allow_fill:
            dtype, fill_value = arr.dtype, arr.dtype.type()
            mask_info = None, False
        else:
            # check for promotion based on types only (do this first because
            # it's faster than computing a mask)
            dtype, fill_value = maybe_promote(arr.dtype, fill_value)
            if dtype != arr.dtype and (out is None or out.dtype != dtype):
                # check if promotion is actually required based on indexer
                if mask_info is not None:
                    (row_mask, col_mask), (row_needs, col_needs) = mask_info
                else:
                    row_mask = row_idx == -1
                    col_mask = col_idx == -1
                    row_needs = row_mask.any()
                    col_needs = col_mask.any()
                    mask_info = (row_mask, col_mask), (row_needs, col_needs)
                if row_needs or col_needs:
                    if out is not None and out.dtype != dtype:
                        raise TypeError('Incompatible type for fill_value')
                else:
                    # if not, then depromote, set fill_value to dummy
                    # (it won't be used but we don't want the cython code
                    # to crash when trying to cast it to dtype)
                    dtype, fill_value = arr.dtype, arr.dtype.type()

    # at this point, it's guaranteed that dtype can hold both the arr values
    # and the fill_value
    if out is None:
        out_shape = len(row_idx), len(col_idx)
        out = np.empty(out_shape, dtype=dtype)

    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)
    if func is None:

        def func(arr, indexer, out, fill_value=np.nan):
            _take_2d_multi_object(arr, indexer, out, fill_value=fill_value,
                                  mask_info=mask_info)

    func(arr, indexer, out=out, fill_value=fill_value)
    return out


# ---- #
# diff #
# ---- #

_diff_special = {
    'float64': algos.diff_2d_float64,
    'float32': algos.diff_2d_float32,
    'int64': algos.diff_2d_int64,
    'int32': algos.diff_2d_int32,
    'int16': algos.diff_2d_int16,
    'int8': algos.diff_2d_int8,
}


def diff(arr, n, axis=0):
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray
    n : int
        number of periods
    axis : int
        axis to shift on

    Returns
    -------
    shifted

    """

    n = int(n)
    na = np.nan
    dtype = arr.dtype

    is_timedelta = False
    if needs_i8_conversion(arr):
        dtype = np.float64
        arr = arr.view('i8')
        na = iNaT
        is_timedelta = True

    elif is_bool_dtype(dtype):
        dtype = np.object_

    elif is_integer_dtype(dtype):
        dtype = np.float64

    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)

    na_indexer = [slice(None)] * arr.ndim
    na_indexer[axis] = slice(None, n) if n >= 0 else slice(n, None)
    out_arr[tuple(na_indexer)] = na

    if arr.ndim == 2 and arr.dtype.name in _diff_special:
        f = _diff_special[arr.dtype.name]
        f(arr, out_arr, n, axis)
    else:
        res_indexer = [slice(None)] * arr.ndim
        res_indexer[axis] = slice(n, None) if n >= 0 else slice(None, n)
        res_indexer = tuple(res_indexer)

        lag_indexer = [slice(None)] * arr.ndim
        lag_indexer[axis] = slice(None, -n) if n > 0 else slice(-n, None)
        lag_indexer = tuple(lag_indexer)

        # need to make sure that we account for na for datelike/timedelta
        # we don't actually want to subtract these i8 numbers
        if is_timedelta:
            res = arr[res_indexer]
            lag = arr[lag_indexer]

            mask = (arr[res_indexer] == na) | (arr[lag_indexer] == na)
            if mask.any():
                res = res.copy()
                res[mask] = 0
                lag = lag.copy()
                lag[mask] = 0

            result = res - lag
            result[mask] = na
            out_arr[res_indexer] = result
        else:
            out_arr[res_indexer] = arr[res_indexer] - arr[lag_indexer]

    if is_timedelta:
        from pandas import TimedeltaIndex
        out_arr = TimedeltaIndex(out_arr.ravel().astype('int64')).asi8.reshape(
            out_arr.shape).astype('timedelta64[ns]')

    return out_arr
