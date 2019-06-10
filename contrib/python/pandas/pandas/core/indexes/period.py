# pylint: disable=E1101,E1103,W0232
from datetime import datetime, timedelta
import warnings

import numpy as np

from pandas._libs import index as libindex
from pandas._libs.tslibs import (
    NaT, frequencies as libfrequencies, iNaT, resolution)
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ, IncompatibleFrequency, Period)
from pandas.util._decorators import Appender, Substitution, cache_readonly

from pandas.core.dtypes.common import (
    is_bool_dtype, is_datetime64_any_dtype, is_float, is_float_dtype,
    is_integer, is_integer_dtype, pandas_dtype)

from pandas import compat
from pandas.core import common as com
from pandas.core.accessor import delegate_names
from pandas.core.algorithms import unique1d
from pandas.core.arrays.period import (
    PeriodArray, period_array, validate_dtype_freq)
from pandas.core.base import _shared_docs
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import _index_shared_docs, ensure_index
from pandas.core.indexes.datetimelike import (
    DatetimeIndexOpsMixin, DatetimelikeDelegateMixin)
from pandas.core.indexes.datetimes import DatetimeIndex, Index, Int64Index
from pandas.core.missing import isna
from pandas.core.ops import get_op_result_name
from pandas.core.tools.datetimes import DateParseError, parse_time_string

from pandas.tseries import frequencies
from pandas.tseries.offsets import DateOffset, Tick

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update(
    dict(target_klass='PeriodIndex or list of Periods'))


# --- Period index sketch


def _new_PeriodIndex(cls, **d):
    # GH13277 for unpickling
    values = d.pop('data')
    if values.dtype == 'int64':
        freq = d.pop('freq', None)
        values = PeriodArray(values, freq=freq)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


class PeriodDelegateMixin(DatetimelikeDelegateMixin):
    """
    Delegate from PeriodIndex to PeriodArray.
    """
    _delegate_class = PeriodArray
    _delegated_properties = PeriodArray._datetimelike_ops
    _delegated_methods = (
        set(PeriodArray._datetimelike_methods) | {'_addsub_int_array'}
    )
    _raw_properties = {'is_leap_year'}


@delegate_names(PeriodArray,
                PeriodDelegateMixin._delegated_properties,
                typ='property')
@delegate_names(PeriodArray,
                PeriodDelegateMixin._delegated_methods,
                typ="method",
                overwrite=True)
class PeriodIndex(DatetimeIndexOpsMixin, Int64Index, PeriodDelegateMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in
    time such as particular years, quarters, months, etc.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional period-like data to construct index with
    copy : bool
        Make a copy of input ndarray
    freq : string or period object, optional
        One of pandas period strings or corresponding objects
    start : starting value, period-like, optional
        If data is None, used as the start point in generating regular
        period data.

        .. deprecated:: 0.24.0

    periods : int, optional, > 0
        Number of periods to generate, if generating index. Takes precedence
        over end argument

        .. deprecated:: 0.24.0

    end : end value, period-like, optional
        If periods is none, generated index will extend to first conforming
        period on or just past end argument

        .. deprecated:: 0.24.0

    year : int, array, or Series, default None
    month : int, array, or Series, default None
    quarter : int, array, or Series, default None
    day : int, array, or Series, default None
    hour : int, array, or Series, default None
    minute : int, array, or Series, default None
    second : int, array, or Series, default None
    tz : object, default None
        Timezone for converting datetime64 data to Periods
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    dayofyear
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp

    Notes
    -----
    Creating a PeriodIndex based on `start`, `periods`, and `end` has
    been deprecated in favor of :func:`period_range`.

    Examples
    --------
    >>> idx = pd.PeriodIndex(year=year_arr, quarter=q_arr)

    See Also
    ---------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.
    """
    _typ = 'periodindex'
    _attributes = ['name', 'freq']

    # define my properties & methods for delegation
    _is_numeric_dtype = False
    _infer_as_myclass = True

    _data = None  # type: PeriodArray

    _engine_type = libindex.PeriodEngine

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(cls, data=None, ordinal=None, freq=None, start=None, end=None,
                periods=None, tz=None, dtype=None, copy=False, name=None,
                **fields):

        valid_field_set = {'year', 'month', 'day', 'quarter',
                           'hour', 'minute', 'second'}

        if not set(fields).issubset(valid_field_set):
            raise TypeError('__new__() got an unexpected keyword argument {}'.
                            format(list(set(fields) - valid_field_set)[0]))

        if name is None and hasattr(data, 'name'):
            name = data.name

        if data is None and ordinal is None:
            # range-based.
            data, freq2 = PeriodArray._generate_range(start, end, periods,
                                                      freq, fields)
            # PeriodArray._generate range does validate that fields is
            # empty when really using the range-based constructor.
            if not fields:
                msg = ("Creating a PeriodIndex by passing range "
                       "endpoints is deprecated.  Use "
                       "`pandas.period_range` instead.")
                # period_range differs from PeriodIndex for cases like
                # start="2000", periods=4
                # PeriodIndex interprets that as A-DEC freq.
                # period_range interprets it as 'D' freq.
                cond = (
                    freq is None and (
                        (start and not isinstance(start, Period)) or
                        (end and not isinstance(end, Period))
                    )
                )
                if cond:
                    msg += (
                        " Note that the default `freq` may differ. Pass "
                        "'freq=\"{}\"' to ensure the same output."
                    ).format(freq2.freqstr)
                warnings.warn(msg, FutureWarning, stacklevel=2)
            freq = freq2

            data = PeriodArray(data, freq=freq)
        else:
            freq = validate_dtype_freq(dtype, freq)

            # PeriodIndex allow PeriodIndex(period_index, freq=different)
            # Let's not encourage that kind of behavior in PeriodArray.

            if freq and isinstance(data, cls) and data.freq != freq:
                # TODO: We can do some of these with no-copy / coercion?
                # e.g. D -> 2D seems to be OK
                data = data.asfreq(freq)

            if data is None and ordinal is not None:
                # we strangely ignore `ordinal` if data is passed.
                ordinal = np.asarray(ordinal, dtype=np.int64)
                data = PeriodArray(ordinal, freq)
            else:
                # don't pass copy here, since we copy later.
                data = period_array(data=data, freq=freq)

        if copy:
            data = data.copy()

        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values, name=None, freq=None, **kwargs):
        """
        Create a new PeriodIndex.

        Parameters
        ----------
        values : PeriodArray, PeriodIndex, Index[int64], ndarray[int64]
            Values that can be converted to a PeriodArray without inference
            or coercion.

        """
        # TODO: raising on floats is tested, but maybe not useful.
        # Should the callers know not to pass floats?
        # At the very least, I think we can ensure that lists aren't passed.
        if isinstance(values, list):
            values = np.asarray(values)
        if is_float_dtype(values):
            raise TypeError("PeriodIndex._simple_new does not accept floats.")
        if freq:
            freq = Period._maybe_convert_freq(freq)
        values = PeriodArray(values, freq=freq)

        if not isinstance(values, PeriodArray):
            raise TypeError("PeriodIndex._simple_new only accepts PeriodArray")
        result = object.__new__(cls)
        result._data = values
        # For groupby perf. See note in indexes/base about _index_data
        result._index_data = values._data
        result.name = name
        result._reset_identity()
        return result

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self):
        return np.asarray(self)

    @property
    def freq(self):
        return self._data.freq

    @freq.setter
    def freq(self, value):
        value = Period._maybe_convert_freq(value)
        # TODO: When this deprecation is enforced, PeriodIndex.freq can
        # be removed entirely, and we'll just inherit.
        msg = ('Setting {cls}.freq has been deprecated and will be '
               'removed in a future version; use {cls}.asfreq instead. '
               'The {cls}.freq setter is not guaranteed to work.')
        warnings.warn(msg.format(cls=type(self).__name__),
                      FutureWarning, stacklevel=2)
        # PeriodArray._freq isn't actually mutable. We set the private _freq
        # here, but people shouldn't be doing this anyway.
        self._data._freq = value

    def _shallow_copy(self, values=None, **kwargs):
        # TODO: simplify, figure out type of values
        if values is None:
            values = self._data

        if isinstance(values, type(self)):
            values = values._values

        if not isinstance(values, PeriodArray):
            if (isinstance(values, np.ndarray) and
                    is_integer_dtype(values.dtype)):
                values = PeriodArray(values, freq=self.freq)
            else:
                # in particular, I would like to avoid period_array here.
                # Some people seem to be calling use with unexpected types
                # Index.difference -> ndarray[Period]
                # DatetimelikeIndexOpsMixin.repeat -> ndarray[ordinal]
                # I think that once all of Datetime* are EAs, we can simplify
                # this quite a bit.
                values = period_array(values, freq=self.freq)

        # We don't allow changing `freq` in _shallow_copy.
        validate_dtype_freq(self.dtype, kwargs.get('freq'))
        attributes = self._get_attributes_dict()

        attributes.update(kwargs)
        if not len(values) and 'dtype' not in kwargs:
            attributes['dtype'] = self.dtype
        return self._simple_new(values, **attributes)

    def _shallow_copy_with_infer(self, values=None, **kwargs):
        """ we always want to return a PeriodIndex """
        return self._shallow_copy(values=values, **kwargs)

    @property
    def _box_func(self):
        """Maybe box an ordinal or Period"""
        # TODO(DatetimeArray): Avoid double-boxing
        # PeriodArray takes care of boxing already, so we need to check
        # whether we're given an ordinal or a Period. It seems like some
        # places outside of indexes/period.py are calling this _box_func,
        # but passing data that's already boxed.
        def func(x):
            if isinstance(x, Period) or x is NaT:
                return x
            else:
                return Period._from_ordinal(ordinal=x, freq=self.freq)
        return func

    def _maybe_convert_timedelta(self, other):
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
        if isinstance(
                other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            offset = frequencies.to_offset(self.freq.rule_code)
            if isinstance(offset, Tick):
                # _check_timedeltalike_freq_compat will raise if incompatible
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, DateOffset):
            freqstr = other.rule_code
            base = libfrequencies.get_base_alias(freqstr)
            if base == self.freq.rule_code:
                return other.n

            msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                        own_freq=self.freqstr,
                                        other_freq=other.freqstr)
            raise IncompatibleFrequency(msg)
        elif is_integer(other):
            # integer is passed to .shift via
            # _add_datetimelike_methods basically
            # but ufunc may pass integer to _add_delta
            return other

        # raise when input doesn't have freq
        msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                    own_freq=self.freqstr,
                                    other_freq=None)
        raise IncompatibleFrequency(msg)

    # ------------------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(self, na_rep=u'NaT', quoting=None, **kwargs):
        # just dispatch, return ndarray
        return self._data._format_native_types(na_rep=na_rep,
                                               quoting=quoting,
                                               **kwargs)

    def _mpl_repr(self):
        # how to represent ourselves to matplotlib
        return self.astype(object).values

    @property
    def _formatter_func(self):
        return self.array._formatter(boxed=False)

    # ------------------------------------------------------------------------
    # Indexing

    @cache_readonly
    def _engine(self):
        return self._engine_type(lambda: self, len(self))

    @Appender(_index_shared_docs['contains'])
    def __contains__(self, key):
        if isinstance(key, Period):
            if key.freq != self.freq:
                return False
            else:
                return key.ordinal in self._engine
        else:
            try:
                self.get_loc(key)
                return True
            except Exception:
                return False

    contains = __contains__

    @cache_readonly
    def _int64index(self):
        return Int64Index._simple_new(self.asi8, name=self.name)

    # ------------------------------------------------------------------------
    # Index Methods

    def _coerce_scalar_to_index(self, item):
        """
        we need to coerce a scalar to a compat for our index type

        Parameters
        ----------
        item : scalar item to coerce
        """
        return PeriodIndex([item], **self._get_attributes_dict())

    def __array__(self, dtype=None):
        if is_integer_dtype(dtype):
            return self.asi8
        else:
            return self.astype(object).values

    def __array_wrap__(self, result, context=None):
        """
        Gets called after a ufunc. Needs additional handling as
        PeriodIndex stores internal data as int dtype

        Replace this to __numpy_ufunc__ in future version
        """
        if isinstance(context, tuple) and len(context) > 0:
            func = context[0]
            if func is np.add:
                pass
            elif func is np.subtract:
                name = self.name
                left = context[1][0]
                right = context[1][1]
                if (isinstance(left, PeriodIndex) and
                        isinstance(right, PeriodIndex)):
                    name = left.name if left.name == right.name else None
                    return Index(result, name=name)
                elif isinstance(left, Period) or isinstance(right, Period):
                    return Index(result, name=name)
            elif isinstance(func, np.ufunc):
                if 'M->M' not in func.types:
                    msg = "ufunc '{0}' not supported for the PeriodIndex"
                    # This should be TypeError, but TypeError cannot be raised
                    # from here because numpy catches.
                    raise ValueError(msg.format(func.__name__))

        if is_bool_dtype(result):
            return result
        # the result is object dtype array of Period
        # cannot pass _simple_new as it is
        return type(self)(result, freq=self.freq, name=self.name)

    def asof_locs(self, where, mask):
        """
        where : array of timestamps
        mask : array of booleans where data is not NA

        """
        where_idx = where
        if isinstance(where_idx, DatetimeIndex):
            where_idx = PeriodIndex(where_idx.values, freq=self.freq)

        locs = self._ndarray_values[mask].searchsorted(
            where_idx._ndarray_values, side='right')

        locs = np.where(locs > 0, locs - 1, 0)
        result = np.arange(len(self))[mask].take(locs)

        first = mask.argmax()
        result[(locs == 0) & (where_idx._ndarray_values <
                              self._ndarray_values[first])] = -1

        return result

    @Appender(_index_shared_docs['astype'])
    def astype(self, dtype, copy=True, how='start'):
        dtype = pandas_dtype(dtype)

        if is_datetime64_any_dtype(dtype):
            # 'how' is index-specific, isn't part of the EA interface.
            tz = getattr(dtype, 'tz', None)
            return self.to_timestamp(how=how).tz_localize(tz)

        # TODO: should probably raise on `how` here, so we don't ignore it.
        return super(PeriodIndex, self).astype(dtype, copy=copy)

    @Substitution(klass='PeriodIndex')
    @Appender(_shared_docs['searchsorted'])
    def searchsorted(self, value, side='left', sorter=None):
        if isinstance(value, Period):
            if value.freq != self.freq:
                msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                            own_freq=self.freqstr,
                                            other_freq=value.freqstr)
                raise IncompatibleFrequency(msg)
            value = value.ordinal
        elif isinstance(value, compat.string_types):
            try:
                value = Period(value, freq=self.freq).ordinal
            except DateParseError:
                raise KeyError("Cannot interpret '{}' as period".format(value))

        return self._ndarray_values.searchsorted(value, side=side,
                                                 sorter=sorter)

    @property
    def is_all_dates(self):
        return True

    @property
    def is_full(self):
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic:
            raise ValueError('Index is not monotonic')
        values = self.asi8
        return ((values[1:] - values[:-1]) < 2).all()

    @property
    def inferred_type(self):
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return 'period'

    def get_value(self, series, key):
        """
        Fast lookup of value from 1-dimensional ndarray. Only use this if you
        know what you're doing
        """
        s = com.values_from_object(series)
        try:
            return com.maybe_box(self,
                                 super(PeriodIndex, self).get_value(s, key),
                                 series, key)
        except (KeyError, IndexError):
            try:
                asdt, parsed, reso = parse_time_string(key, self.freq)
                grp = resolution.Resolution.get_freq_group(reso)
                freqn = resolution.get_freq_group(self.freq)

                vals = self._ndarray_values

                # if our data is higher resolution than requested key, slice
                if grp < freqn:
                    iv = Period(asdt, freq=(grp, 1))
                    ord1 = iv.asfreq(self.freq, how='S').ordinal
                    ord2 = iv.asfreq(self.freq, how='E').ordinal

                    if ord2 < vals[0] or ord1 > vals[-1]:
                        raise KeyError(key)

                    pos = np.searchsorted(self._ndarray_values, [ord1, ord2])
                    key = slice(pos[0], pos[1] + 1)
                    return series[key]
                elif grp == freqn:
                    key = Period(asdt, freq=self.freq).ordinal
                    return com.maybe_box(self, self._engine.get_value(s, key),
                                         series, key)
                else:
                    raise KeyError(key)
            except TypeError:
                pass

            period = Period(key, self.freq)
            key = period.value if isna(period) else period.ordinal
            return com.maybe_box(self, self._engine.get_value(s, key),
                                 series, key)

    @Appender(_index_shared_docs['get_indexer'] % _index_doc_kwargs)
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        target = ensure_index(target)

        if hasattr(target, 'freq') and target.freq != self.freq:
            msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                        own_freq=self.freqstr,
                                        other_freq=target.freqstr)
            raise IncompatibleFrequency(msg)

        if isinstance(target, PeriodIndex):
            target = target.asi8

        if tolerance is not None:
            tolerance = self._convert_tolerance(tolerance, target)
        return Index.get_indexer(self._int64index, target, method,
                                 limit, tolerance)

    def _get_unique_index(self, dropna=False):
        """
        wrap Index._get_unique_index to handle NaT
        """
        res = super(PeriodIndex, self)._get_unique_index(dropna=dropna)
        if dropna:
            res = res.dropna()
        return res

    @Appender(Index.unique.__doc__)
    def unique(self, level=None):
        # override the Index.unique method for performance GH#23083
        if level is not None:
            # this should never occur, but is retained to make the signature
            # match Index.unique
            self._validate_index_level(level)

        values = self._ndarray_values
        result = unique1d(values)
        return self._shallow_copy(result)

    def get_loc(self, key, method=None, tolerance=None):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        try:
            return self._engine.get_loc(key)
        except KeyError:
            if is_integer(key):
                raise

            try:
                asdt, parsed, reso = parse_time_string(key, self.freq)
                key = asdt
            except TypeError:
                pass
            except DateParseError:
                # A string with invalid format
                raise KeyError("Cannot interpret '{}' as period".format(key))

            try:
                key = Period(key, freq=self.freq)
            except ValueError:
                # we cannot construct the Period
                # as we have an invalid type
                raise KeyError(key)

            try:
                ordinal = iNaT if key is NaT else key.ordinal
                if tolerance is not None:
                    tolerance = self._convert_tolerance(tolerance,
                                                        np.asarray(key))
                return self._int64index.get_loc(ordinal, method, tolerance)

            except KeyError:
                raise KeyError(key)

    def _maybe_cast_slice_bound(self, label, side, kind):
        """
        If label is a string or a datetime, cast it to Period.ordinal according
        to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        bound : Period or object

        Notes
        -----
        Value of `side` parameter should be validated in caller.

        """
        assert kind in ['ix', 'loc', 'getitem']

        if isinstance(label, datetime):
            return Period(label, freq=self.freq)
        elif isinstance(label, compat.string_types):
            try:
                _, parsed, reso = parse_time_string(label, self.freq)
                bounds = self._parsed_string_to_bounds(reso, parsed)
                return bounds[0 if side == 'left' else 1]
            except Exception:
                raise KeyError(label)
        elif is_integer(label) or is_float(label):
            self._invalid_indexer('slice', label)

        return label

    def _parsed_string_to_bounds(self, reso, parsed):
        if reso == 'year':
            t1 = Period(year=parsed.year, freq='A')
        elif reso == 'month':
            t1 = Period(year=parsed.year, month=parsed.month, freq='M')
        elif reso == 'quarter':
            q = (parsed.month - 1) // 3 + 1
            t1 = Period(year=parsed.year, quarter=q, freq='Q-DEC')
        elif reso == 'day':
            t1 = Period(year=parsed.year, month=parsed.month, day=parsed.day,
                        freq='D')
        elif reso == 'hour':
            t1 = Period(year=parsed.year, month=parsed.month, day=parsed.day,
                        hour=parsed.hour, freq='H')
        elif reso == 'minute':
            t1 = Period(year=parsed.year, month=parsed.month, day=parsed.day,
                        hour=parsed.hour, minute=parsed.minute, freq='T')
        elif reso == 'second':
            t1 = Period(year=parsed.year, month=parsed.month, day=parsed.day,
                        hour=parsed.hour, minute=parsed.minute,
                        second=parsed.second, freq='S')
        else:
            raise KeyError(reso)
        return (t1.asfreq(self.freq, how='start'),
                t1.asfreq(self.freq, how='end'))

    def _get_string_slice(self, key):
        if not self.is_monotonic:
            raise ValueError('Partial indexing only valid for '
                             'ordered time series')

        key, parsed, reso = parse_time_string(key, self.freq)
        grp = resolution.Resolution.get_freq_group(reso)
        freqn = resolution.get_freq_group(self.freq)
        if reso in ['day', 'hour', 'minute', 'second'] and not grp < freqn:
            raise KeyError(key)

        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        return slice(self.searchsorted(t1.ordinal, side='left'),
                     self.searchsorted(t2.ordinal, side='right'))

    def _convert_tolerance(self, tolerance, target):
        tolerance = DatetimeIndexOpsMixin._convert_tolerance(self, tolerance,
                                                             target)
        if target.size != tolerance.size and tolerance.size > 1:
            raise ValueError('list-like tolerance size must match '
                             'target index size')
        return self._maybe_convert_timedelta(tolerance)

    def insert(self, loc, item):
        if not isinstance(item, Period) or self.freq != item.freq:
            return self.astype(object).insert(loc, item)

        idx = np.concatenate((self[:loc].asi8, np.array([item.ordinal]),
                              self[loc:].asi8))
        return self._shallow_copy(idx)

    def join(self, other, how='left', level=None, return_indexers=False,
             sort=False):
        """
        See Index.join
        """
        self._assert_can_do_setop(other)

        result = Int64Index.join(self, other, how=how, level=level,
                                 return_indexers=return_indexers,
                                 sort=sort)

        if return_indexers:
            result, lidx, ridx = result
            return self._apply_meta(result), lidx, ridx
        return self._apply_meta(result)

    def _assert_can_do_setop(self, other):
        super(PeriodIndex, self)._assert_can_do_setop(other)

        if not isinstance(other, PeriodIndex):
            raise ValueError('can only call with other PeriodIndex-ed objects')

        if self.freq != other.freq:
            msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                        own_freq=self.freqstr,
                                        other_freq=other.freqstr)
            raise IncompatibleFrequency(msg)

    def _wrap_setop_result(self, other, result):
        name = get_op_result_name(self, other)
        result = self._apply_meta(result)
        result.name = name
        return result

    def _apply_meta(self, rawarr):
        if not isinstance(rawarr, PeriodIndex):
            rawarr = PeriodIndex._simple_new(rawarr, freq=self.freq,
                                             name=self.name)
        return rawarr

    def __setstate__(self, state):
        """Necessary for making this object picklable"""

        if isinstance(state, dict):
            super(PeriodIndex, self).__setstate__(state)

        elif isinstance(state, tuple):

            # < 0.15 compat
            if len(state) == 2:
                nd_state, own_state = state
                data = np.empty(nd_state[1], dtype=nd_state[2])
                np.ndarray.__setstate__(data, nd_state)

                # backcompat
                freq = Period._maybe_convert_freq(own_state[1])

            else:  # pragma: no cover
                data = np.empty(state)
                np.ndarray.__setstate__(self, state)
                freq = None  # ?

            data = PeriodArray(data, freq=freq)
            self._data = data

        else:
            raise Exception("invalid pickle state")

    _unpickle_compat = __setstate__

    @property
    def flags(self):
        """ return the ndarray.flags for the underlying data """
        warnings.warn("{obj}.flags is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return self._ndarray_values.flags

    def item(self):
        """
        return the first element of the underlying data as a python
        scalar
        """
        # TODO(DatetimeArray): remove
        if len(self) == 1:
            return self[0]
        else:
            # copy numpy's message here because Py26 raises an IndexError
            raise ValueError('can only convert an array of size 1 to a '
                             'Python scalar')

    @property
    def data(self):
        """ return the data pointer of the underlying data """
        warnings.warn("{obj}.data is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return np.asarray(self._data).data

    @property
    def base(self):
        """ return the base object if the memory of the underlying data is
        shared
        """
        warnings.warn("{obj}.base is deprecated and will be removed "
                      "in a future version".format(obj=type(self).__name__),
                      FutureWarning, stacklevel=2)
        return np.asarray(self._data)


PeriodIndex._add_comparison_ops()
PeriodIndex._add_numeric_methods_disabled()
PeriodIndex._add_logical_methods_disabled()
PeriodIndex._add_datetimelike_methods()


def period_range(start=None, end=None, periods=None, freq=None, name=None):
    """
    Return a fixed frequency PeriodIndex, with day (calendar) as the default
    frequency

    Parameters
    ----------
    start : string or period-like, default None
        Left bound for generating periods
    end : string or period-like, default None
        Right bound for generating periods
    periods : integer, default None
        Number of periods to generate
    freq : string or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.

    name : string, default None
        Name of the resulting PeriodIndex

    Returns
    -------
    prng : PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases>`__.

    Examples
    --------

    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05',
                 '2017-06', '2017-06', '2017-07', '2017-08', '2017-09',
                 '2017-10', '2017-11', '2017-12', '2018-01'],
                dtype='period[M]', freq='M')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]', freq='M')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, '
                         'exactly two must be specified')
    if freq is None and (not isinstance(start, Period)
                         and not isinstance(end, Period)):
        freq = 'D'

    data, freq = PeriodArray._generate_range(start, end, periods, freq,
                                             fields={})
    data = PeriodArray(data, freq=freq)
    return PeriodIndex(data, name=name)
