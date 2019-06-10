""" implement the TimedeltaIndex """
from datetime import datetime
import warnings

import numpy as np

from pandas._libs import (
    NaT, Timedelta, index as libindex, join as libjoin, lib)
import pandas.compat as compat
from pandas.util._decorators import Appender, Substitution

from pandas.core.dtypes.common import (
    _TD_DTYPE, ensure_int64, is_float, is_integer, is_list_like, is_scalar,
    is_timedelta64_dtype, is_timedelta64_ns_dtype, pandas_dtype)
import pandas.core.dtypes.concat as _concat
from pandas.core.dtypes.missing import isna

from pandas.core.accessor import delegate_names
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays.timedeltas import TimedeltaArray, _is_convertible_to_td
from pandas.core.base import _shared_docs
import pandas.core.common as com
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.datetimelike import (
    DatetimeIndexOpsMixin, DatetimelikeDelegateMixin, maybe_unwrap_index,
    wrap_arithmetic_op)
from pandas.core.indexes.numeric import Int64Index
from pandas.core.ops import get_op_result_name

from pandas.tseries.frequencies import to_offset


def _make_wrapped_arith_op(opname):

    meth = getattr(TimedeltaArray, opname)

    def method(self, other):
        result = meth(self._data, maybe_unwrap_index(other))
        return wrap_arithmetic_op(self, other, result)

    method.__name__ = opname
    return method


class TimedeltaDelegateMixin(DatetimelikeDelegateMixin):
    # Most attrs are dispatched via datetimelike_{ops,methods}
    # Some are "raw" methods, the result is not not re-boxed in an Index
    # We also have a few "extra" attrs, which may or may not be raw,
    # which we we dont' want to expose in the .dt accessor.
    _delegate_class = TimedeltaArray
    _delegated_properties = (TimedeltaArray._datetimelike_ops + [
        'components',
    ])
    _delegated_methods = TimedeltaArray._datetimelike_methods + [
        '_box_values',
    ]
    _raw_properties = {
        'components',
    }
    _raw_methods = {
        'to_pytimedelta',
    }


@delegate_names(TimedeltaArray,
                TimedeltaDelegateMixin._delegated_properties,
                typ="property")
@delegate_names(TimedeltaArray,
                TimedeltaDelegateMixin._delegated_methods,
                typ="method", overwrite=False)
class TimedeltaIndex(DatetimeIndexOpsMixin, dtl.TimelikeOps, Int64Index,
                     TimedeltaDelegateMixin):
    """
    Immutable ndarray of timedelta64 data, represented internally as int64, and
    which can be boxed to timedelta objects

    Parameters
    ----------
    data  : array-like (1-dimensional), optional
        Optional timedelta-like data to construct index with
    unit : unit of the arg (D,h,m,s,ms,us,ns) denote the unit, optional
        which is an integer/float number
    freq : string or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation
    copy  : bool
        Make a copy of input ndarray
    start : starting value, timedelta-like, optional
        If data is None, start is used as the start point in generating regular
        timedelta data.

        .. deprecated:: 0.24.0

    periods  : int, optional, > 0
        Number of periods to generate, if generating index. Takes precedence
        over end argument

        .. deprecated:: 0.24.0

    end : end time, timedelta-like, optional
        If periods is none, generated index will extend to first conforming
        time on or just past end argument

        .. deprecated:: 0.24. 0

    closed : string or None, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None)

        .. deprecated:: 0.24. 0

    name : object
        Name to be stored in the index

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    to_pytimedelta
    to_series
    round
    floor
    ceil
    to_frame

    See Also
    ---------
    Index : The base pandas Index type.
    Timedelta : Represents a duration between two dates or times.
    DatetimeIndex : Index of datetime64 data.
    PeriodIndex : Index of Period data.
    timedelta_range : Create a fixed-frequency TimedeltaIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__.

    Creating a TimedeltaIndex based on `start`, `periods`, and `end` has
    been deprecated in favor of :func:`timedelta_range`.
    """

    _typ = 'timedeltaindex'
    _join_precedence = 10

    def _join_i8_wrapper(joinf, **kwargs):
        return DatetimeIndexOpsMixin._join_i8_wrapper(
            joinf, dtype='m8[ns]', **kwargs)

    _inner_indexer = _join_i8_wrapper(libjoin.inner_join_indexer_int64)
    _outer_indexer = _join_i8_wrapper(libjoin.outer_join_indexer_int64)
    _left_indexer = _join_i8_wrapper(libjoin.left_join_indexer_int64)
    _left_indexer_unique = _join_i8_wrapper(
        libjoin.left_join_indexer_unique_int64, with_indexers=False)

    _engine_type = libindex.TimedeltaEngine

    _comparables = ['name', 'freq']
    _attributes = ['name', 'freq']
    _is_numeric_dtype = True
    _infer_as_myclass = True

    _freq = None

    _box_func = TimedeltaArray._box_func
    _bool_ops = TimedeltaArray._bool_ops
    _object_ops = TimedeltaArray._object_ops
    _field_ops = TimedeltaArray._field_ops
    _datetimelike_ops = TimedeltaArray._datetimelike_ops
    _datetimelike_methods = TimedeltaArray._datetimelike_methods
    _other_ops = TimedeltaArray._other_ops

    # -------------------------------------------------------------------
    # Constructors

    def __new__(cls, data=None, unit=None, freq=None, start=None, end=None,
                periods=None, closed=None, dtype=_TD_DTYPE, copy=False,
                name=None, verify_integrity=None):

        if verify_integrity is not None:
            warnings.warn("The 'verify_integrity' argument is deprecated, "
                          "will be removed in a future version.",
                          FutureWarning, stacklevel=2)
        else:
            verify_integrity = True

        if data is None:
            freq, freq_infer = dtl.maybe_infer_freq(freq)
            warnings.warn("Creating a TimedeltaIndex by passing range "
                          "endpoints is deprecated.  Use "
                          "`pandas.timedelta_range` instead.",
                          FutureWarning, stacklevel=2)
            result = TimedeltaArray._generate_range(start, end, periods, freq,
                                                    closed=closed)
            return cls._simple_new(result._data, freq=freq, name=name)

        if is_scalar(data):
            raise TypeError('{cls}() must be called with a '
                            'collection of some kind, {data} was passed'
                            .format(cls=cls.__name__, data=repr(data)))

        if isinstance(data, TimedeltaArray):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name, freq=freq)

        if (isinstance(data, TimedeltaIndex) and
                freq is None and name is None):
            if copy:
                return data.copy()
            else:
                return data._shallow_copy()

        # - Cases checked above all return/raise before reaching here - #

        tdarr = TimedeltaArray._from_sequence(data, freq=freq, unit=unit,
                                              dtype=dtype, copy=copy)
        return cls._simple_new(tdarr._data, freq=tdarr.freq, name=name)

    @classmethod
    def _simple_new(cls, values, name=None, freq=None, dtype=_TD_DTYPE):
        # `dtype` is passed by _shallow_copy in corner cases, should always
        #  be timedelta64[ns] if present
        if not isinstance(values, TimedeltaArray):
            values = TimedeltaArray._simple_new(values, dtype=dtype,
                                                freq=freq)
        else:
            if freq is None:
                freq = values.freq
        assert isinstance(values, TimedeltaArray), type(values)
        assert dtype == _TD_DTYPE, dtype
        assert values.dtype == 'm8[ns]', values.dtype

        tdarr = TimedeltaArray._simple_new(values._data, freq=freq)
        result = object.__new__(cls)
        result._data = tdarr
        result.name = name
        # For groupby perf. See note in indexes/base about _index_data
        result._index_data = tdarr._data

        result._reset_identity()
        return result

    # -------------------------------------------------------------------

    def __setstate__(self, state):
        """Necessary for making this object picklable"""
        if isinstance(state, dict):
            super(TimedeltaIndex, self).__setstate__(state)
        else:
            raise Exception("invalid pickle state")
    _unpickle_compat = __setstate__

    def _maybe_update_attributes(self, attrs):
        """ Update Index attributes (e.g. freq) depending on op """
        freq = attrs.get('freq', None)
        if freq is not None:
            # no need to infer if freq is None
            attrs['freq'] = 'infer'
        return attrs

    # -------------------------------------------------------------------
    # Rendering Methods

    @property
    def _formatter_func(self):
        from pandas.io.formats.format import _get_format_timedelta64
        return _get_format_timedelta64(self, box=True)

    def _format_native_types(self, na_rep='NaT', date_format=None, **kwargs):
        from pandas.io.formats.format import Timedelta64Formatter
        return Timedelta64Formatter(values=self,
                                    nat_rep=na_rep,
                                    justify='all').get_result()

    # -------------------------------------------------------------------
    # Wrapping TimedeltaArray

    __mul__ = _make_wrapped_arith_op("__mul__")
    __rmul__ = _make_wrapped_arith_op("__rmul__")
    __floordiv__ = _make_wrapped_arith_op("__floordiv__")
    __rfloordiv__ = _make_wrapped_arith_op("__rfloordiv__")
    __mod__ = _make_wrapped_arith_op("__mod__")
    __rmod__ = _make_wrapped_arith_op("__rmod__")
    __divmod__ = _make_wrapped_arith_op("__divmod__")
    __rdivmod__ = _make_wrapped_arith_op("__rdivmod__")
    __truediv__ = _make_wrapped_arith_op("__truediv__")
    __rtruediv__ = _make_wrapped_arith_op("__rtruediv__")
    if compat.PY2:
        __div__ = __truediv__
        __rdiv__ = __rtruediv__

    # Compat for frequency inference, see GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def _box_func(self):
        return lambda x: Timedelta(x, unit='ns')

    def __getitem__(self, key):
        result = self._data.__getitem__(key)
        if is_scalar(result):
            return result
        return type(self)(result, name=self.name)

    # -------------------------------------------------------------------

    @Appender(_index_shared_docs['astype'])
    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_timedelta64_dtype(dtype) and not is_timedelta64_ns_dtype(dtype):
            # Have to repeat the check for 'timedelta64' (not ns) dtype
            #  so that we can return a numeric index, since pandas will return
            #  a TimedeltaIndex when dtype='timedelta'
            result = self._data.astype(dtype, copy=copy)
            if self.hasnans:
                return Index(result, name=self.name)
            return Index(result.astype('i8'), name=self.name)
        return DatetimeIndexOpsMixin.astype(self, dtype, copy=copy)

    def union(self, other):
        """
        Specialized union for TimedeltaIndex objects. If combine
        overlapping ranges with the same DateOffset, will be much
        faster than Index.union

        Parameters
        ----------
        other : TimedeltaIndex or array-like

        Returns
        -------
        y : Index or TimedeltaIndex
        """
        self._assert_can_do_setop(other)

        if len(other) == 0 or self.equals(other) or len(self) == 0:
            return super(TimedeltaIndex, self).union(other)

        if not isinstance(other, TimedeltaIndex):
            try:
                other = TimedeltaIndex(other)
            except (TypeError, ValueError):
                pass
        this, other = self, other

        if this._can_fast_union(other):
            return this._fast_union(other)
        else:
            result = Index.union(this, other)
            if isinstance(result, TimedeltaIndex):
                if result.freq is None:
                    result.freq = to_offset(result.inferred_freq)
            return result

    def join(self, other, how='left', level=None, return_indexers=False,
             sort=False):
        """
        See Index.join
        """
        if _is_convertible_to_index(other):
            try:
                other = TimedeltaIndex(other)
            except (TypeError, ValueError):
                pass

        return Index.join(self, other, how=how, level=level,
                          return_indexers=return_indexers,
                          sort=sort)

    def _wrap_joined_index(self, joined, other):
        name = get_op_result_name(self, other)
        if (isinstance(other, TimedeltaIndex) and self.freq == other.freq and
                self._can_fast_union(other)):
            joined = self._shallow_copy(joined, name=name)
            return joined
        else:
            return self._simple_new(joined, name)

    def _can_fast_union(self, other):
        if not isinstance(other, TimedeltaIndex):
            return False

        freq = self.freq

        if freq is None or freq != other.freq:
            return False

        if not self.is_monotonic or not other.is_monotonic:
            return False

        if len(self) == 0 or len(other) == 0:
            return True

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left

    def _fast_union(self, other):
        if len(other) == 0:
            return self.view(type(self))

        if len(self) == 0:
            return other.view(type(self))

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side='right')
            right_chunk = right.values[loc:]
            dates = _concat._concat_compat((left.values, right_chunk))
            return self._shallow_copy(dates)
        else:
            return left

    def intersection(self, other):
        """
        Specialized intersection for TimedeltaIndex objects. May be much faster
        than Index.intersection

        Parameters
        ----------
        other : TimedeltaIndex or array-like

        Returns
        -------
        y : Index or TimedeltaIndex
        """
        self._assert_can_do_setop(other)

        if self.equals(other):
            return self._get_reconciled_name_object(other)

        if not isinstance(other, TimedeltaIndex):
            try:
                other = TimedeltaIndex(other)
            except (TypeError, ValueError):
                pass
            result = Index.intersection(self, other)
            return result

        if len(self) == 0:
            return self
        if len(other) == 0:
            return other
        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            return type(self)(data=[])
        else:
            lslice = slice(*left.slice_locs(start, end))
            left_chunk = left.values[lslice]
            return self._shallow_copy(left_chunk)

    def _maybe_promote(self, other):
        if other.inferred_type == 'timedelta':
            other = TimedeltaIndex(other)
        return self, other

    def get_value(self, series, key):
        """
        Fast lookup of value from 1-dimensional ndarray. Only use this if you
        know what you're doing
        """

        if _is_convertible_to_td(key):
            key = Timedelta(key)
            return self.get_value_maybe_box(series, key)

        try:
            return com.maybe_box(self, Index.get_value(self, series, key),
                                 series, key)
        except KeyError:
            try:
                loc = self._get_string_slice(key)
                return series[loc]
            except (TypeError, ValueError, KeyError):
                pass

            try:
                return self.get_value_maybe_box(series, key)
            except (TypeError, ValueError, KeyError):
                raise KeyError(key)

    def get_value_maybe_box(self, series, key):
        if not isinstance(key, Timedelta):
            key = Timedelta(key)
        values = self._engine.get_value(com.values_from_object(series), key)
        return com.maybe_box(self, values, series, key)

    def get_loc(self, key, method=None, tolerance=None):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        if is_list_like(key) or (isinstance(key, datetime) and key is not NaT):
            # GH#20464 datetime check here is to ensure we don't allow
            #   datetime objects to be incorrectly treated as timedelta
            #   objects; NaT is a special case because it plays a double role
            #   as Not-A-Timedelta
            raise TypeError

        if isna(key):
            key = NaT

        if tolerance is not None:
            # try converting tolerance now, so errors don't get swallowed by
            # the try/except clauses below
            tolerance = self._convert_tolerance(tolerance, np.asarray(key))

        if _is_convertible_to_td(key):
            key = Timedelta(key)
            return Index.get_loc(self, key, method, tolerance)

        try:
            return Index.get_loc(self, key, method, tolerance)
        except (KeyError, ValueError, TypeError):
            try:
                return self._get_string_slice(key)
            except (TypeError, KeyError, ValueError):
                pass

            try:
                stamp = Timedelta(key)
                return Index.get_loc(self, stamp, method, tolerance)
            except (KeyError, ValueError):
                raise KeyError(key)

    def _maybe_cast_slice_bound(self, label, side, kind):
        """
        If label is a string, cast it to timedelta according to resolution.


        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        label :  object

        """
        assert kind in ['ix', 'loc', 'getitem', None]

        if isinstance(label, compat.string_types):
            parsed = Timedelta(label)
            lbound = parsed.round(parsed.resolution)
            if side == 'left':
                return lbound
            else:
                return (lbound + to_offset(parsed.resolution) -
                        Timedelta(1, 'ns'))
        elif ((is_integer(label) or is_float(label)) and
              not is_timedelta64_dtype(label)):
            self._invalid_indexer('slice', label)

        return label

    def _get_string_slice(self, key):
        if is_integer(key) or is_float(key) or key is NaT:
            self._invalid_indexer('slice', key)
        loc = self._partial_td_slice(key)
        return loc

    def _partial_td_slice(self, key):

        # given a key, try to figure out a location for a partial slice
        if not isinstance(key, compat.string_types):
            return key

        raise NotImplementedError

    @Substitution(klass='TimedeltaIndex')
    @Appender(_shared_docs['searchsorted'])
    def searchsorted(self, value, side='left', sorter=None):
        if isinstance(value, (np.ndarray, Index)):
            value = np.array(value, dtype=_TD_DTYPE, copy=False)
        else:
            value = Timedelta(value).asm8.view(_TD_DTYPE)

        return self.values.searchsorted(value, side=side, sorter=sorter)

    def is_type_compatible(self, typ):
        return typ == self.inferred_type or typ == 'timedelta'

    @property
    def inferred_type(self):
        return 'timedelta64'

    @property
    def is_all_dates(self):
        return True

    def insert(self, loc, item):
        """
        Make new Index inserting new item at location

        Parameters
        ----------
        loc : int
        item : object
            if not either a Python datetime or a numpy integer-like, returned
            Index dtype will be object rather than datetime.

        Returns
        -------
        new_index : Index
        """
        # try to convert if possible
        if _is_convertible_to_td(item):
            try:
                item = Timedelta(item)
            except Exception:
                pass
        elif is_scalar(item) and isna(item):
            # GH 18295
            item = self._na_value

        freq = None
        if isinstance(item, Timedelta) or (is_scalar(item) and isna(item)):

            # check freq can be preserved on edge cases
            if self.freq is not None:
                if ((loc == 0 or loc == -len(self)) and
                        item + self.freq == self[0]):
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            item = Timedelta(item).asm8.view(_TD_DTYPE)

        try:
            new_tds = np.concatenate((self[:loc].asi8, [item.view(np.int64)],
                                      self[loc:].asi8))
            return self._shallow_copy(new_tds, freq=freq)

        except (AttributeError, TypeError):

            # fall back to object index
            if isinstance(item, compat.string_types):
                return self.astype(object).insert(loc, item)
            raise TypeError(
                "cannot insert TimedeltaIndex with incompatible label")

    def delete(self, loc):
        """
        Make a new TimedeltaIndex with passed location(s) deleted.

        Parameters
        ----------
        loc: int, slice or array of ints
            Indicate which sub-arrays to remove.

        Returns
        -------
        new_index : TimedeltaIndex
        """
        new_tds = np.delete(self.asi8, loc)

        freq = 'infer'
        if is_integer(loc):
            if loc in (0, -len(self), -1, len(self) - 1):
                freq = self.freq
        else:
            if is_list_like(loc):
                loc = lib.maybe_indices_to_slice(
                    ensure_int64(np.array(loc)), len(self))
            if isinstance(loc, slice) and loc.step in (1, None):
                if (loc.start in (0, None) or loc.stop in (len(self), None)):
                    freq = self.freq

        return TimedeltaIndex(new_tds, name=self.name, freq=freq)


TimedeltaIndex._add_comparison_ops()
TimedeltaIndex._add_numeric_methods_unary()
TimedeltaIndex._add_logical_methods_disabled()
TimedeltaIndex._add_datetimelike_methods()


def _is_convertible_to_index(other):
    """
    return a boolean whether I can attempt conversion to a TimedeltaIndex
    """
    if isinstance(other, TimedeltaIndex):
        return True
    elif (len(other) > 0 and
          other.inferred_type not in ('floating', 'mixed-integer', 'integer',
                                      'mixed-integer-float', 'mixed')):
        return True
    return False


def timedelta_range(start=None, end=None, periods=None, freq=None,
                    name=None, closed=None):
    """
    Return a fixed frequency TimedeltaIndex, with day as the default
    frequency

    Parameters
    ----------
    start : string or timedelta-like, default None
        Left bound for generating timedeltas
    end : string or timedelta-like, default None
        Right bound for generating timedeltas
    periods : integer, default None
        Number of periods to generate
    freq : string or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'
    name : string, default None
        Name of the resulting TimedeltaIndex
    closed : string, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None)

    Returns
    -------
    rng : TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__.

    Examples
    --------

    >>> pd.timedelta_range(start='1 day', periods=4)
    TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``closed`` parameter specifies which endpoint is included.  The default
    behavior is to include both endpoints.

    >>> pd.timedelta_range(start='1 day', periods=4, closed='right')
    TimedeltaIndex(['2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.
    Only fixed frequencies can be passed, non-fixed frequencies such as
    'M' (month end) will raise.

    >>> pd.timedelta_range(start='1 day', end='2 days', freq='6H')
    TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00',
                    '1 days 18:00:00', '2 days 00:00:00'],
                   dtype='timedelta64[ns]', freq='6H')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.timedelta_range(start='1 day', end='5 days', periods=4)
    TimedeltaIndex(['1 days 00:00:00', '2 days 08:00:00', '3 days 16:00:00',
                '5 days 00:00:00'],
               dtype='timedelta64[ns]', freq=None)
    """
    if freq is None and com._any_none(periods, start, end):
        freq = 'D'

    freq, freq_infer = dtl.maybe_infer_freq(freq)
    tdarr = TimedeltaArray._generate_range(start, end, periods, freq,
                                           closed=closed)
    return TimedeltaIndex._simple_new(tdarr._data, freq=tdarr.freq, name=name)
