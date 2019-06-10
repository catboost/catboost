# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import operator
import warnings

import numpy as np

from pandas._libs import NaT, algos, iNaT, lib
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ, IncompatibleFrequency, Period)
from pandas._libs.tslibs.timedeltas import Timedelta, delta_to_nanoseconds
from pandas._libs.tslibs.timestamps import (
    RoundTo, maybe_integer_op_deprecated, round_nsint64)
import pandas.compat as compat
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError, NullFrequencyError, PerformanceWarning)
from pandas.util._decorators import Appender, Substitution
from pandas.util._validators import validate_fillna_kwargs

from pandas.core.dtypes.common import (
    is_categorical_dtype, is_datetime64_any_dtype, is_datetime64_dtype,
    is_datetime64tz_dtype, is_datetime_or_timedelta_dtype, is_dtype_equal,
    is_extension_array_dtype, is_float_dtype, is_integer_dtype, is_list_like,
    is_object_dtype, is_offsetlike, is_period_dtype, is_string_dtype,
    is_timedelta64_dtype, is_unsigned_integer_dtype, pandas_dtype)
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import isna

from pandas.core import missing, nanops
from pandas.core.algorithms import (
    checked_add_with_arr, take, unique1d, value_counts)
import pandas.core.common as com

from pandas.tseries import frequencies
from pandas.tseries.offsets import DateOffset, Tick

from .base import ExtensionArray, ExtensionOpsMixin


class AttributesMixin(object):

    @property
    def _attributes(self):
        # Inheriting subclass should implement _attributes as a list of strings
        raise AbstractMethodError(self)

    @classmethod
    def _simple_new(cls, values, **kwargs):
        raise AbstractMethodError(cls)

    def _get_attributes_dict(self):
        """
        return an attributes dict for my class
        """
        return {k: getattr(self, k, None) for k in self._attributes}

    @property
    def _scalar_type(self):
        # type: () -> Union[type, Tuple[type]]
        """The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value):
        # type: (str) -> Union[Period, Timestamp, Timedelta, NaTType]
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(self, value):
        # type: (Union[Period, Timestamp, Timedelta, NaTType]) -> int
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Union[Period, Timestamp, Timedelta]

        Returns
        -------
        int

        Examples
        --------
        >>> self._unbox_scalar(Timedelta('10s'))  # DOCTEST: +SKIP
        10000000000
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other):
        # type: (Union[Period, Timestamp, Timedelta, NaTType]) -> None
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)


class DatelikeOps(object):
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    @Substitution(URL="https://docs.python.org/3/library/datetime.html"
                      "#strftime-and-strptime-behavior")
    def strftime(self, date_format):
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <%(URL)s>`__

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%%Y-%%m-%%d").

        Returns
        -------
        Index
            Index of formatted strings

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),
        ...                     periods=3, freq='s')
        >>> rng.strftime('%%B %%d, %%Y, %%r')
        Index(['March 10, 2018, 09:00:00 AM', 'March 10, 2018, 09:00:01 AM',
               'March 10, 2018, 09:00:02 AM'],
              dtype='object')
        """
        from pandas import Index
        return Index(self._format_native_types(date_format=date_format))


class TimelikeOps(object):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """

    _round_doc = (
        """
        Perform {op} operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to {op} the index to. Must be a fixed
            frequency like 'S' (second) not 'ME' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
            Only relevant for DatetimeIndex:

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times

            .. versionadded:: 0.24.0

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta,
                      default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise an NonExistentTimeError if there are
              nonexistent times

            .. versionadded:: 0.24.0

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range('1/1/2018 11:59:00', periods=3, freq='min')
        >>> rng
        DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:01:00'],
                      dtype='datetime64[ns]', freq='T')
        """)

    _round_example = (
        """>>> rng.round('H')
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[ns]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.round("H")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[ns]
        """)

    _floor_example = (
        """>>> rng.floor('H')
        DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[ns]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.floor("H")
        0   2018-01-01 11:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[ns]
        """
    )

    _ceil_example = (
        """>>> rng.ceil('H')
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 13:00:00'],
                      dtype='datetime64[ns]', freq=None)

        **Series**

        >>> pd.Series(rng).dt.ceil("H")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 13:00:00
        dtype: datetime64[ns]
        """
    )

    def _round(self, freq, mode, ambiguous, nonexistent):
        # round the local times
        values = _ensure_datetimelike_to_i8(self)
        result = round_nsint64(values, mode, freq)
        result = self._maybe_mask_results(result, fill_value=NaT)

        dtype = self.dtype
        if is_datetime64tz_dtype(self):
            dtype = None
        return self._ensure_localized(
            self._simple_new(result, dtype=dtype), ambiguous, nonexistent
        )

    @Appender((_round_doc + _round_example).format(op="round"))
    def round(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(
            freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent
        )

    @Appender((_round_doc + _floor_example).format(op="floor"))
    def floor(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op="ceil"))
    def ceil(self, freq, ambiguous='raise', nonexistent='raise'):
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)


class DatetimeLikeArrayMixin(ExtensionOpsMixin,
                             AttributesMixin,
                             ExtensionArray):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _data
        _freq

    and that the inheriting class has methods:
        _generate_range
    """

    @property
    def _box_func(self):
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values):
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func)

    def __iter__(self):
        return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self):
        # type: () -> ndarray
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        # do not cache or you'll create a memory leak
        return self._data.view('i8')

    @property
    def _ndarray_values(self):
        return self._data

    # ----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(self, na_rep='NaT', date_format=None):
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed=False):
        # TODO: Remove Datetime & DatetimeTZ formatters.
        return "'{}'".format

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    @property
    def nbytes(self):
        return self._data.nbytes

    def __array__(self, dtype=None):
        # used for Timedelta/DatetimeArray, overwritten by PeriodArray
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._data

    @property
    def shape(self):
        return (len(self),)

    @property
    def size(self):
        # type: () -> int
        """The number of elements in this array."""
        return np.prod(self.shape)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """

        is_int = lib.is_integer(key)
        if lib.is_scalar(key) and not is_int:
            raise IndexError("only integers, slices (`:`), ellipsis (`...`), "
                             "numpy.newaxis (`None`) and integer or boolean "
                             "arrays are valid indices")

        getitem = self._data.__getitem__
        if is_int:
            val = getitem(key)
            return self._box_func(val)

        if com.is_bool_indexer(key):
            key = np.asarray(key, dtype=bool)
            if key.all():
                key = slice(0, None, None)
            else:
                key = lib.maybe_booleans_to_slice(key.view(np.uint8))

        is_period = is_period_dtype(self)
        if is_period:
            freq = self.freq
        else:
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                # GH#21282 indexing with Ellipsis is similar to a full slice,
                #  should preserve `freq` attribute
                freq = self.freq

        result = getitem(key)
        if result.ndim > 1:
            # To support MPL which performs slicing with 2 dim
            # even though it only has 1 dim by definition
            if is_period:
                return self._simple_new(result, dtype=self.dtype, freq=freq)
            return result

        return self._simple_new(result, dtype=self.dtype, freq=freq)

    def __setitem__(
            self,
            key,    # type: Union[int, Sequence[int], Sequence[bool], slice]
            value,  # type: Union[NaTType, Scalar, Sequence[Scalar]]
    ):
        # type: (...) -> None
        # I'm fudging the types a bit here. The "Scalar" above really depends
        # on type(self). For PeriodArray, it's Period (or stuff coercible
        # to a period in from_sequence). For DatetimeArray, it's Timestamp...
        # I don't know if mypy can do that, possibly with Generics.
        # https://mypy.readthedocs.io/en/latest/generics.html

        if is_list_like(value):
            is_slice = isinstance(key, slice)

            if lib.is_scalar(key):
                raise ValueError("setting an array element with a sequence.")

            if (not is_slice
                    and len(key) != len(value)
                    and not com.is_bool_indexer(key)):
                msg = ("shape mismatch: value array of length '{}' does not "
                       "match indexing result of length '{}'.")
                raise ValueError(msg.format(len(key), len(value)))
            if not is_slice and len(key) == 0:
                return

            value = type(self)._from_sequence(value, dtype=self.dtype)
            self._check_compatible_with(value)
            value = value.asi8
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            value = self._unbox_scalar(value)
        elif isna(value) or value == iNaT:
            value = iNaT
        else:
            msg = (
                "'value' should be a '{scalar}', 'NaT', or array of those. "
                "Got '{typ}' instead."
            )
            raise TypeError(msg.format(scalar=self._scalar_type.__name__,
                                       typ=type(value).__name__))
        self._data[key] = value
        self._maybe_clear_freq()

    def _maybe_clear_freq(self):
        # inplace operations like __setitem__ may invalidate the freq of
        # DatetimeArray and TimedeltaArray
        pass

    def astype(self, dtype, copy=True):
        # Some notes on cases we don't have to handle here in the base class:
        #   1. PeriodArray.astype handles period -> period
        #   2. DatetimeArray.astype handles conversion between tz.
        #   3. DatetimeArray.astype handles datetime -> period
        from pandas import Categorical
        dtype = pandas_dtype(dtype)

        if is_object_dtype(dtype):
            return self._box_values(self.asi8)
        elif is_string_dtype(dtype) and not is_categorical_dtype(dtype):
            return self._format_native_types()
        elif is_integer_dtype(dtype):
            # we deliberately ignore int32 vs. int64 here.
            # See https://github.com/pandas-dev/pandas/issues/24381 for more.
            values = self.asi8

            if is_unsigned_integer_dtype(dtype):
                # Again, we ignore int32 vs. int64
                values = values.view("uint64")

            if copy:
                values = values.copy()
            return values
        elif (is_datetime_or_timedelta_dtype(dtype) and
              not is_dtype_equal(self.dtype, dtype)) or is_float_dtype(dtype):
            # disallow conversion between datetime/timedelta,
            # and conversions for any datetimelike to float
            msg = 'Cannot cast {name} to dtype {dtype}'
            raise TypeError(msg.format(name=type(self).__name__, dtype=dtype))
        elif is_categorical_dtype(dtype):
            return Categorical(self, dtype=dtype)
        else:
            return np.asarray(self, dtype=dtype)

    def view(self, dtype=None):
        """
        New view on this array with the same data.

        Parameters
        ----------
        dtype : numpy dtype, optional

        Returns
        -------
        ndarray
            With the specified `dtype`.
        """
        return self._data.view(dtype=dtype)

    # ------------------------------------------------------------------
    # ExtensionArray Interface

    def unique(self):
        result = unique1d(self.asi8)
        return type(self)(result, dtype=self.dtype)

    def _validate_fill_value(self, fill_value):
        """
        If a fill_value is passed to `take` convert it to an i8 representation,
        raising ValueError if this is not possible.

        Parameters
        ----------
        fill_value : object

        Returns
        -------
        fill_value : np.int64

        Raises
        ------
        ValueError
        """
        raise AbstractMethodError(self)

    def take(self, indices, allow_fill=False, fill_value=None):
        if allow_fill:
            fill_value = self._validate_fill_value(fill_value)

        new_values = take(self.asi8,
                          indices,
                          allow_fill=allow_fill,
                          fill_value=fill_value)

        return type(self)(new_values, dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        dtypes = {x.dtype for x in to_concat}
        assert len(dtypes) == 1
        dtype = list(dtypes)[0]

        values = np.concatenate([x.asi8 for x in to_concat])
        return cls(values, dtype=dtype)

    def copy(self, deep=False):
        values = self.asi8.copy()
        return type(self)._simple_new(values, dtype=self.dtype, freq=self.freq)

    def _values_for_factorize(self):
        return self.asi8, iNaT

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def _values_for_argsort(self):
        return self._data

    # ------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    def searchsorted(self, value, side='left', sorter=None):
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Parameters
        ----------
        value : array_like
            Values to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort `self` into ascending
            order. They are typically the result of ``np.argsort``.

        Returns
        -------
        indices : array of ints
            Array of insertion points with the same shape as `value`.
        """
        if isinstance(value, compat.string_types):
            value = self._scalar_from_string(value)

        if not (isinstance(value, (self._scalar_type, type(self)))
                or isna(value)):
            raise ValueError("Unexpected type for 'value': {valtype}"
                             .format(valtype=type(value)))

        self._check_compatible_with(value)
        if isinstance(value, type(self)):
            value = value.asi8
        else:
            value = self._unbox_scalar(value)

        return self.asi8.searchsorted(value, side=side, sorter=sorter)

    def repeat(self, repeats, *args, **kwargs):
        """
        Repeat elements of an array.

        See Also
        --------
        numpy.ndarray.repeat
        """
        nv.validate_repeat(args, kwargs)
        values = self._data.repeat(repeats)
        return type(self)(values.view('i8'), dtype=self.dtype)

    def value_counts(self, dropna=False):
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaT values.

        Returns
        -------
        Series
        """
        from pandas import Series, Index

        if dropna:
            values = self[~self.isna()]._data
        else:
            values = self._data

        cls = type(self)

        result = value_counts(values, sort=False, dropna=dropna)
        index = Index(cls(result.index.view('i8'), dtype=self.dtype),
                      name=result.index.name)
        return Series(result.values, index=index, name=result.name)

    def map(self, mapper):
        # TODO(GH-23179): Add ExtensionArray.map
        # Need to figure out if we want ExtensionArray.map first.
        # If so, then we can refactor IndexOpsMixin._map_values to
        # a standalone function and call from here..
        # Else, just rewrite _map_infer_values to do the right thing.
        from pandas import Index

        return Index(self).map(mapper).array

    # ------------------------------------------------------------------
    # Null Handling

    def isna(self):
        return self._isnan

    @property  # NB: override with cache_readonly in immutable subclasses
    def _isnan(self):
        """
        return if each value is nan
        """
        return (self.asi8 == iNaT)

    @property  # NB: override with cache_readonly in immutable subclasses
    def _hasnans(self):
        """
        return if I have any nans; enables various perf speedups
        """
        return bool(self._isnan.any())

    def _maybe_mask_results(self, result, fill_value=iNaT, convert=None):
        """
        Parameters
        ----------
        result : a ndarray
        fill_value : object, default iNaT
        convert : string/dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine
        """

        if self._hasnans:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            result[self._isnan] = fill_value
        return result

    def fillna(self, value=None, method=None, limit=None):
        # TODO(GH-20300): remove this
        # Just overriding to ensure that we avoid an astype(object).
        # Either 20300 or a `_values_for_fillna` would avoid this duplication.
        if isinstance(value, ABCSeries):
            value = value.array

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if is_array_like(value):
            if len(value) != len(self):
                raise ValueError("Length of 'value' does not match. Got ({}) "
                                 " expected {}".format(len(value), len(self)))
            value = value[mask]

        if mask.any():
            if method is not None:
                if method == 'pad':
                    func = missing.pad_1d
                else:
                    func = missing.backfill_1d

                values = self._data
                if not is_period_dtype(self):
                    # For PeriodArray self._data is i8, which gets copied
                    #  by `func`.  Otherwise we need to make a copy manually
                    # to avoid modifying `self` in-place.
                    values = values.copy()

                new_values = func(values, limit=limit,
                                  mask=mask)
                if is_datetime64tz_dtype(self):
                    # we need to pass int64 values to the constructor to avoid
                    #  re-localizing incorrectly
                    new_values = new_values.view("i8")
                new_values = type(self)(new_values, dtype=self.dtype)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    # ------------------------------------------------------------------
    # Frequency Properties/Methods

    @property
    def freq(self):
        """
        Return the frequency object if it is set, otherwise None.
        """
        return self._freq

    @freq.setter
    def freq(self, value):
        if value is not None:
            value = frequencies.to_offset(value)
            self._validate_frequency(self, value)

        self._freq = value

    @property
    def freqstr(self):
        """
        Return the frequency object as a string if its set, otherwise None
        """
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property  # NB: override with cache_readonly in immutable subclasses
    def inferred_freq(self):
        """
        Tryies to return a string representing a frequency guess,
        generated by infer_freq.  Returns None if it can't autodetect the
        frequency.
        """
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution(self):
        return frequencies.Resolution.get_reso_from_freq(self.freqstr)

    @property  # NB: override with cache_readonly in immutable subclasses
    def resolution(self):
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        return frequencies.Resolution.get_str(self._resolution)

    @classmethod
    def _validate_frequency(cls, index, freq, **kwargs):
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        if is_period_dtype(cls):
            # Frequency validation is not meaningful for Period Array/Index
            return None

        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None

        try:
            on_freq = cls._generate_range(start=index[0], end=None,
                                          periods=len(index), freq=freq,
                                          **kwargs)
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as e:
            if "non-fixed" in str(e):
                # non-fixed frequencies are not meaningful for timedelta64;
                #  we retain that error message
                raise e
            # GH#11587 the main way this is reached is if the `np.array_equal`
            #  check above is False.  This can also be reached if index[0]
            #  is `NaT`, in which case the call to `cls._generate_range` will
            #  raise a ValueError, which we re-raise with a more targeted
            #  message.
            raise ValueError('Inferred frequency {infer} from passed values '
                             'does not conform to passed frequency {passed}'
                             .format(infer=inferred, passed=freq.freqstr))

    # monotonicity/uniqueness properties are called via frequencies.infer_freq,
    #  see GH#23789

    @property
    def _is_monotonic_increasing(self):
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self):
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self):
        return len(unique1d(self.asi8)) == len(self)

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _add_datetimelike_scalar(self, other):
        # Overriden by TimedeltaArray
        raise TypeError("cannot add {cls} and {typ}"
                        .format(cls=type(self).__name__,
                                typ=type(other).__name__))

    _add_datetime_arraylike = _add_datetimelike_scalar

    def _sub_datetimelike_scalar(self, other):
        # Overridden by DatetimeArray
        assert other is not NaT
        raise TypeError("cannot subtract a datelike from a {cls}"
                        .format(cls=type(self).__name__))

    _sub_datetime_arraylike = _sub_datetimelike_scalar

    def _sub_period(self, other):
        # Overriden by PeriodArray
        raise TypeError("cannot subtract Period from a {cls}"
                        .format(cls=type(self).__name__))

    def _add_offset(self, offset):
        raise AbstractMethodError(self)

    def _add_delta(self, other):
        """
        Add a timedelta-like, Tick or TimedeltaIndex-like object
        to self, yielding an int64 numpy array

        Parameters
        ----------
        delta : {timedelta, np.timedelta64, Tick,
                 TimedeltaIndex, ndarray[timedelta64]}

        Returns
        -------
        result : ndarray[int64]

        Notes
        -----
        The result's name is set outside of _add_delta by the calling
        method (__add__ or __sub__), if necessary (i.e. for Indexes).
        """
        if isinstance(other, (Tick, timedelta, np.timedelta64)):
            new_values = self._add_timedeltalike_scalar(other)
        elif is_timedelta64_dtype(other):
            # ndarray[timedelta64] or TimedeltaArray/index
            new_values = self._add_delta_tdi(other)

        return new_values

    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike
        return the i8 result view
        """
        if isna(other):
            # i.e np.timedelta64("NaT"), not recognized by delta_to_nanoseconds
            new_values = np.empty(len(self), dtype='i8')
            new_values[:] = iNaT
            return new_values

        inc = delta_to_nanoseconds(other)
        new_values = checked_add_with_arr(self.asi8, inc,
                                          arr_mask=self._isnan).view('i8')
        new_values = self._maybe_mask_results(new_values)
        return new_values.view('i8')

    def _add_delta_tdi(self, other):
        """
        Add a delta of a TimedeltaIndex
        return the i8 result view
        """
        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        if isinstance(other, np.ndarray):
            # ndarray[timedelta64]; wrap in TimedeltaIndex for op
            from pandas import TimedeltaIndex
            other = TimedeltaIndex(other)

        self_i8 = self.asi8
        other_i8 = other.asi8
        new_values = checked_add_with_arr(self_i8, other_i8,
                                          arr_mask=self._isnan,
                                          b_mask=other._isnan)
        if self._hasnans or other._hasnans:
            mask = (self._isnan) | (other._isnan)
            new_values[mask] = iNaT
        return new_values.view('i8')

    def _add_nat(self):
        """
        Add pd.NaT to self
        """
        if is_period_dtype(self):
            raise TypeError('Cannot add {cls} and {typ}'
                            .format(cls=type(self).__name__,
                                    typ=type(NaT).__name__))

        # GH#19124 pd.NaT is treated like a timedelta for both timedelta
        # and datetime dtypes
        result = np.zeros(len(self), dtype=np.int64)
        result.fill(iNaT)
        return type(self)(result, dtype=self.dtype, freq=None)

    def _sub_nat(self):
        """
        Subtract pd.NaT from self
        """
        # GH#19124 Timedelta - datetime is not in general well-defined.
        # We make an exception for pd.NaT, which in this case quacks
        # like a timedelta.
        # For datetime64 dtypes by convention we treat NaT as a datetime, so
        # this subtraction returns a timedelta64 dtype.
        # For period dtype, timedelta64 is a close-enough return dtype.
        result = np.zeros(len(self), dtype=np.int64)
        result.fill(iNaT)
        return result.view('timedelta64[ns]')

    def _sub_period_array(self, other):
        """
        Subtract a Period Array/Index from self.  This is only valid if self
        is itself a Period Array/Index, raises otherwise.  Both objects must
        have the same frequency.

        Parameters
        ----------
        other : PeriodIndex or PeriodArray

        Returns
        -------
        result : np.ndarray[object]
            Array of DateOffset objects; nulls represented by NaT
        """
        if not is_period_dtype(self):
            raise TypeError("cannot subtract {dtype}-dtype from {cls}"
                            .format(dtype=other.dtype,
                                    cls=type(self).__name__))

        if len(self) != len(other):
            raise ValueError("cannot subtract arrays/indices of "
                             "unequal length")
        if self.freq != other.freq:
            msg = DIFFERENT_FREQ.format(cls=type(self).__name__,
                                        own_freq=self.freqstr,
                                        other_freq=other.freqstr)
            raise IncompatibleFrequency(msg)

        new_values = checked_add_with_arr(self.asi8, -other.asi8,
                                          arr_mask=self._isnan,
                                          b_mask=other._isnan)

        new_values = np.array([self.freq.base * x for x in new_values])
        if self._hasnans or other._hasnans:
            mask = (self._isnan) | (other._isnan)
            new_values[mask] = NaT
        return new_values

    def _addsub_int_array(self, other, op):
        """
        Add or subtract array-like of integers equivalent to applying
        `_time_shift` pointwise.

        Parameters
        ----------
        other : Index, ExtensionArray, np.ndarray
            integer-dtype
        op : {operator.add, operator.sub}

        Returns
        -------
        result : same class as self
        """
        # _addsub_int_array is overriden by PeriodArray
        assert not is_period_dtype(self)
        assert op in [operator.add, operator.sub]

        if self.freq is None:
            # GH#19123
            raise NullFrequencyError("Cannot shift with no freq")

        elif isinstance(self.freq, Tick):
            # easy case where we can convert to timedelta64 operation
            td = Timedelta(self.freq)
            return op(self, td * other)

        # We should only get here with DatetimeIndex; dispatch
        # to _addsub_offset_array
        assert not is_timedelta64_dtype(self)
        return op(self, np.array(other) * self.freq)

    def _addsub_offset_array(self, other, op):
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : Index, np.ndarray
            object-dtype containing pd.DateOffset objects
        op : {operator.add, operator.sub}

        Returns
        -------
        result : same class as self
        """
        assert op in [operator.add, operator.sub]
        if len(other) == 1:
            return op(self, other[0])

        warnings.warn("Adding/subtracting array of DateOffsets to "
                      "{cls} not vectorized"
                      .format(cls=type(self).__name__), PerformanceWarning)

        # For EA self.astype('O') returns a numpy array, not an Index
        left = lib.values_from_object(self.astype('O'))

        res_values = op(left, np.array(other))
        kwargs = {}
        if not is_period_dtype(self):
            kwargs['freq'] = 'infer'
        return self._from_sequence(res_values, **kwargs)

    def _time_shift(self, periods, freq=None):
        """
        Shift each value by `periods`.

        Note this is different from ExtensionArray.shift, which
        shifts the *position* of each element, padding the end with
        missing values.

        Parameters
        ----------
        periods : int
            Number of periods to shift by.
        freq : pandas.DateOffset, pandas.Timedelta, or string
            Frequency increment to shift by.
        """
        if freq is not None and freq != self.freq:
            if isinstance(freq, compat.string_types):
                freq = frequencies.to_offset(freq)
            offset = periods * freq
            result = self + offset
            return result

        if periods == 0:
            # immutable so OK
            return self.copy()

        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # Note: in the DatetimeTZ case, _generate_range will infer the
        #  appropriate timezone from `start` and `end`, so tz does not need
        #  to be passed explicitly.
        return self._generate_range(start=start, end=end, periods=None,
                                    freq=self.freq)

    def __add__(self, other):
        other = lib.item_from_zerodim(other)
        if isinstance(other, (ABCSeries, ABCDataFrame)):
            return NotImplemented

        # scalar others
        elif other is NaT:
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_delta(other)
        elif isinstance(other, DateOffset):
            # specifically _not_ a Tick
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not is_period_dtype(self):
                maybe_integer_op_deprecated(self)
            result = self._time_shift(other)

        # array-like others
        elif is_timedelta64_dtype(other):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_delta(other)
        elif is_offsetlike(other):
            # Array/Index of DateOffset objects
            result = self._addsub_offset_array(other, operator.add)
        elif is_datetime64_dtype(other) or is_datetime64tz_dtype(other):
            # DatetimeIndex, ndarray[datetime64]
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other):
            if not is_period_dtype(self):
                maybe_integer_op_deprecated(self)
            result = self._addsub_int_array(other, operator.add)
        elif is_float_dtype(other):
            # Explicitly catch invalid dtypes
            raise TypeError("cannot add {dtype}-dtype to {cls}"
                            .format(dtype=other.dtype,
                                    cls=type(self).__name__))
        elif is_period_dtype(other):
            # if self is a TimedeltaArray and other is a PeriodArray with
            #  a timedelta-like (i.e. Tick) freq, this operation is valid.
            #  Defer to the PeriodArray implementation.
            # In remaining cases, this will end up raising TypeError.
            return NotImplemented
        elif is_extension_array_dtype(other):
            # Categorical op will raise; defer explicitly
            return NotImplemented
        else:  # pragma: no cover
            return NotImplemented

        if is_timedelta64_dtype(result) and isinstance(result, np.ndarray):
            from pandas.core.arrays import TimedeltaArray
            # TODO: infer freq?
            return TimedeltaArray(result)
        return result

    def __radd__(self, other):
        # alias for __add__
        return self.__add__(other)

    def __sub__(self, other):
        other = lib.item_from_zerodim(other)
        if isinstance(other, (ABCSeries, ABCDataFrame)):
            return NotImplemented

        # scalar others
        elif other is NaT:
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_delta(-other)
        elif isinstance(other, DateOffset):
            # specifically _not_ a Tick
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not is_period_dtype(self):
                maybe_integer_op_deprecated(self)
            result = self._time_shift(-other)

        elif isinstance(other, Period):
            result = self._sub_period(other)

        # array-like others
        elif is_timedelta64_dtype(other):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_delta(-other)
        elif is_offsetlike(other):
            # Array/Index of DateOffset objects
            result = self._addsub_offset_array(other, operator.sub)
        elif is_datetime64_dtype(other) or is_datetime64tz_dtype(other):
            # DatetimeIndex, ndarray[datetime64]
            result = self._sub_datetime_arraylike(other)
        elif is_period_dtype(other):
            # PeriodIndex
            result = self._sub_period_array(other)
        elif is_integer_dtype(other):
            if not is_period_dtype(self):
                maybe_integer_op_deprecated(self)
            result = self._addsub_int_array(other, operator.sub)
        elif isinstance(other, ABCIndexClass):
            raise TypeError("cannot subtract {cls} and {typ}"
                            .format(cls=type(self).__name__,
                                    typ=type(other).__name__))
        elif is_float_dtype(other):
            # Explicitly catch invalid dtypes
            raise TypeError("cannot subtract {dtype}-dtype from {cls}"
                            .format(dtype=other.dtype,
                                    cls=type(self).__name__))
        elif is_extension_array_dtype(other):
            # Categorical op will raise; defer explicitly
            return NotImplemented
        else:  # pragma: no cover
            return NotImplemented

        if is_timedelta64_dtype(result) and isinstance(result, np.ndarray):
            from pandas.core.arrays import TimedeltaArray
            # TODO: infer freq?
            return TimedeltaArray(result)
        return result

    def __rsub__(self, other):
        if is_datetime64_dtype(other) and is_timedelta64_dtype(self):
            # ndarray[datetime64] cannot be subtracted from self, so
            # we need to wrap in DatetimeArray/Index and flip the operation
            if not isinstance(other, DatetimeLikeArrayMixin):
                # Avoid down-casting DatetimeIndex
                from pandas.core.arrays import DatetimeArray
                other = DatetimeArray(other)
            return other - self
        elif (is_datetime64_any_dtype(self) and hasattr(other, 'dtype') and
              not is_datetime64_any_dtype(other)):
            # GH#19959 datetime - datetime is well-defined as timedelta,
            # but any other type - datetime is not well-defined.
            raise TypeError("cannot subtract {cls} from {typ}"
                            .format(cls=type(self).__name__,
                                    typ=type(other).__name__))
        elif is_period_dtype(self) and is_timedelta64_dtype(other):
            # TODO: Can we simplify/generalize these cases at all?
            raise TypeError("cannot subtract {cls} from {dtype}"
                            .format(cls=type(self).__name__,
                                    dtype=other.dtype))
        return -(self - other)

    # FIXME: DTA/TDA/PA inplace methods should actually be inplace, GH#24115
    def __iadd__(self, other):
        # alias for __add__
        return self.__add__(other)

    def __isub__(self, other):
        # alias for __sub__
        return self.__sub__(other)

    # --------------------------------------------------------------
    # Comparison Methods

    def _ensure_localized(self, arg, ambiguous='raise', nonexistent='raise',
                          from_utc=False):
        """
        Ensure that we are re-localized.

        This is for compat as we can then call this on all datetimelike
        arrays generally (ignored for Period/Timedelta)

        Parameters
        ----------
        arg : Union[DatetimeLikeArray, DatetimeIndexOpsMixin, ndarray]
        ambiguous : str, bool, or bool-ndarray, default 'raise'
        nonexistent : str, default 'raise'
        from_utc : bool, default False
            If True, localize the i8 ndarray to UTC first before converting to
            the appropriate tz. If False, localize directly to the tz.

        Returns
        -------
        localized array
        """

        # reconvert to local tz
        tz = getattr(self, 'tz', None)
        if tz is not None:
            if not isinstance(arg, type(self)):
                arg = self._simple_new(arg)
            if from_utc:
                arg = arg.tz_localize('UTC').tz_convert(self.tz)
            else:
                arg = arg.tz_localize(
                    self.tz, ambiguous=ambiguous, nonexistent=nonexistent
                )
        return arg

    # --------------------------------------------------------------
    # Reductions

    def _reduce(self, name, axis=0, skipna=True, **kwargs):
        op = getattr(self, name, None)
        if op:
            return op(axis=axis, skipna=skipna, **kwargs)
        else:
            return super(DatetimeLikeArrayMixin, self)._reduce(
                name, skipna, **kwargs
            )

    def min(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)

        result = nanops.nanmin(self.asi8, skipna=skipna, mask=self.isna())
        if isna(result):
            # Period._from_ordinal does not handle np.nan gracefully
            return NaT
        return self._box_func(result)

    def max(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        # TODO: skipna is broken with max.
        # See https://github.com/pandas-dev/pandas/issues/24265
        nv.validate_max(args, kwargs)
        nv.validate_minmax_axis(axis)

        mask = self.isna()
        if skipna:
            values = self[~mask].asi8
        elif mask.any():
            return NaT
        else:
            values = self.asi8

        if not len(values):
            # short-circut for empty max / min
            return NaT

        result = nanops.nanmax(values, skipna=skipna)
        # Don't have to worry about NA `result`, since no NA went in.
        return self._box_func(result)


# -------------------------------------------------------------------
# Shared Constructor Helpers

def validate_periods(periods):
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, float, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is None, float, or int
    """
    if periods is not None:
        if lib.is_float(periods):
            periods = int(periods)
        elif not lib.is_integer(periods):
            raise TypeError('periods must be a number, got {periods}'
                            .format(periods=periods))
    return periods


def validate_endpoints(closed):
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}

    Returns
    -------
    left_closed : bool
    right_closed : bool

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


def validate_inferred_freq(freq, inferred_freq, freq_infer):
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None
    freq_infer : bool

    Returns
    -------
    freq : DateOffset or None
    freq_infer : bool

    Notes
    -----
    We assume at this point that `maybe_infer_freq` has been called, so
    `freq` is either a DateOffset object or None.
    """
    if inferred_freq is not None:
        if freq is not None and freq != inferred_freq:
            raise ValueError('Inferred frequency {inferred} from passed '
                             'values does not conform to passed frequency '
                             '{passed}'
                             .format(inferred=inferred_freq,
                                     passed=freq.freqstr))
        elif freq is None:
            freq = inferred_freq
        freq_infer = False

    return freq, freq_infer


def maybe_infer_freq(freq):
    """
    Comparing a DateOffset to the string "infer" raises, so we need to
    be careful about comparisons.  Make a dummy variable `freq_infer` to
    signify the case where the given freq is "infer" and set freq to None
    to avoid comparison trouble later on.

    Parameters
    ----------
    freq : {DateOffset, None, str}

    Returns
    -------
    freq : {DateOffset, None}
    freq_infer : bool
    """
    freq_infer = False
    if not isinstance(freq, DateOffset):
        # if a passed freq is None, don't infer automatically
        if freq != 'infer':
            freq = frequencies.to_offset(freq)
        else:
            freq_infer = True
            freq = None
    return freq, freq_infer


def _ensure_datetimelike_to_i8(other, to_utc=False):
    """
    Helper for coercing an input scalar or array to i8.

    Parameters
    ----------
    other : 1d array
    to_utc : bool, default False
        If True, convert the values to UTC before extracting the i8 values
        If False, extract the i8 values directly.

    Returns
    -------
    i8 1d array
    """
    from pandas import Index
    from pandas.core.arrays import PeriodArray

    if lib.is_scalar(other) and isna(other):
        return iNaT
    elif isinstance(other, (PeriodArray, ABCIndexClass,
                            DatetimeLikeArrayMixin)):
        # convert tz if needed
        if getattr(other, 'tz', None) is not None:
            if to_utc:
                other = other.tz_convert('UTC')
            else:
                other = other.tz_localize(None)
    else:
        try:
            return np.array(other, copy=False).view('i8')
        except TypeError:
            # period array cannot be coerced to int
            other = Index(other)
    return other.asi8
