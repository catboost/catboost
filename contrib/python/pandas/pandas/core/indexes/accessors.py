"""
datetimelike delegation
"""
import numpy as np

from pandas.core.dtypes.common import (
    is_categorical_dtype, is_datetime64_dtype, is_datetime64tz_dtype,
    is_datetime_arraylike, is_integer_dtype, is_list_like, is_period_arraylike,
    is_timedelta64_dtype)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.algorithms import take_1d
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):

    def __init__(self, data, orig):
        if not isinstance(data, ABCSeries):
            raise TypeError("cannot convert an object of type {0} to a "
                            "datetimelike index".format(type(data)))

        self._parent = data
        self.orig = orig
        self.name = getattr(data, 'name', None)
        self._freeze()

    def _get_values(self):
        data = self._parent
        if is_datetime64_dtype(data.dtype):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif is_datetime64tz_dtype(data.dtype):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif is_timedelta64_dtype(data.dtype):
            return TimedeltaIndex(data, copy=False, name=self.name)

        else:
            if is_period_arraylike(data):
                # TODO: use to_period_array
                return PeriodArray(data, copy=False)
            if is_datetime_arraylike(data):
                return DatetimeIndex(data, copy=False, name=self.name)

        raise TypeError("cannot convert an object of type {0} to a "
                        "datetimelike index".format(type(data)))

    def _delegate_property_get(self, name):
        from pandas import Series
        values = self._get_values()

        result = getattr(values, name)

        # maybe need to upcast (ints)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype('int64')
        elif not is_list_like(result):
            return result

        result = np.asarray(result)

        # blow up if we operate on categories
        if self.orig is not None:
            result = take_1d(result, self.orig.cat.codes)
            index = self.orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = Series(result, index=index, name=self.name)

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = ("modifications to a property of a datetimelike "
                           "object are not supported and are discarded. "
                           "Change values on the original.")

        return result

    def _delegate_property_set(self, name, value, *args, **kwargs):
        raise ValueError("modifications to a property of a datetimelike "
                         "object are not supported. Change values on the "
                         "original.")

    def _delegate_method(self, name, *args, **kwargs):
        from pandas import Series
        values = self._get_values()

        method = getattr(values, name)
        result = method(*args, **kwargs)

        if not is_list_like(result):
            return result

        result = Series(result, index=self._parent.index, name=self.name)

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = ("modifications to a method of a datetimelike "
                           "object are not supported and are discarded. "
                           "Change values on the original.")

        return result


@delegate_names(delegate=DatetimeArray,
                accessors=DatetimeArray._datetimelike_ops,
                typ="property")
@delegate_names(delegate=DatetimeArray,
                accessors=DatetimeArray._datetimelike_methods,
                typ="method")
class DatetimeProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> s.dt.hour
    >>> s.dt.second
    >>> s.dt.quarter

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pydatetime(self):
        """
        Return the data as an array of native Python datetime objects.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range('20180310', periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 11, 0, 0)], dtype=object)

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range('20180310', periods=2, freq='ns'))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 10, 0, 0)], dtype=object)
        """
        return self._get_values().to_pydatetime()

    @property
    def freq(self):
        return self._get_values().inferred_freq


@delegate_names(delegate=TimedeltaArray,
                accessors=TimedeltaArray._datetimelike_ops,
                typ="property")
@delegate_names(delegate=TimedeltaArray,
                accessors=TimedeltaArray._datetimelike_methods,
                typ="method")
class TimedeltaProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> s.dt.hours
    >>> s.dt.seconds

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pytimedelta(self):
        """
        Return an array of native `datetime.timedelta` objects.

        Python's standard `datetime` library uses a different representation
        timedelta's. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        a : numpy.ndarray
            1D array containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(1),
               datetime.timedelta(2), datetime.timedelta(3),
               datetime.timedelta(4)], dtype=object)
        """
        return self._get_values().to_pytimedelta()

    @property
    def components(self):
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))
        >>> s
        0   00:00:00
        1   00:00:01
        2   00:00:02
        3   00:00:03
        4   00:00:04
        dtype: timedelta64[ns]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """  # noqa: E501
        return self._get_values().components.set_index(self._parent.index)

    @property
    def freq(self):
        return self._get_values().inferred_freq


@delegate_names(delegate=PeriodArray,
                accessors=PeriodArray._datetimelike_ops,
                typ="property")
@delegate_names(delegate=PeriodArray,
                accessors=PeriodArray._datetimelike_methods,
                typ="method")
class PeriodProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> s.dt.hour
    >>> s.dt.second
    >>> s.dt.quarter

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """


class CombinedDatetimelikeProperties(DatetimeProperties,
                                     TimedeltaProperties, PeriodProperties):

    def __new__(cls, data):
        # CombinedDatetimelikeProperties isn't really instantiated. Instead
        # we need to choose which parent (datetime or timedelta) is
        # appropriate. Since we're checking the dtypes anyway, we'll just
        # do all the validation here.
        from pandas import Series

        if not isinstance(data, Series):
            raise TypeError("cannot convert an object of type {0} to a "
                            "datetimelike index".format(type(data)))

        orig = data if is_categorical_dtype(data) else None
        if orig is not None:
            data = Series(orig.values.categories,
                          name=orig.name,
                          copy=False)

        try:
            if is_datetime64_dtype(data.dtype):
                return DatetimeProperties(data, orig)
            elif is_datetime64tz_dtype(data.dtype):
                return DatetimeProperties(data, orig)
            elif is_timedelta64_dtype(data.dtype):
                return TimedeltaProperties(data, orig)
            elif is_period_arraylike(data):
                return PeriodProperties(data, orig)
            elif is_datetime_arraylike(data):
                return DatetimeProperties(data, orig)
        except Exception:
            pass  # we raise an attribute error anyway

        raise AttributeError("Can only use .dt accessor with datetimelike "
                             "values")
