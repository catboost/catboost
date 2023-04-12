import warnings

from pandas.util._exceptions import find_stack_level

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    datetime,
    import_datetime,
    timedelta,
)

import_datetime()
from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
)

import numpy as np

cimport numpy as cnp
from numpy cimport int64_t

cnp.import_array()

cimport pandas._libs.tslibs.util as util
from pandas._libs.tslibs.np_datetime cimport (
    get_datetime64_value,
    get_timedelta64_value,
)

# ----------------------------------------------------------------------
# Constants
nat_strings = {"NaT", "nat", "NAT", "nan", "NaN", "NAN"}
cdef set c_nat_strings = nat_strings

cdef int64_t NPY_NAT = util.get_nat()
iNaT = NPY_NAT  # python-visible constant

# ----------------------------------------------------------------------


def _make_nan_func(func_name: str, doc: str):
    def f(*args, **kwargs):
        return np.nan
    f.__name__ = func_name
    f.__doc__ = doc
    return f


def _make_nat_func(func_name: str, doc: str):
    def f(*args, **kwargs):
        return c_NaT
    f.__name__ = func_name
    f.__doc__ = doc
    return f


def _make_error_func(func_name: str, cls):
    def f(*args, **kwargs):
        raise ValueError(f"NaTType does not support {func_name}")

    f.__name__ = func_name
    if isinstance(cls, str):
        # passed the literal docstring directly
        f.__doc__ = cls
    elif cls is not None:
        f.__doc__ = getattr(cls, func_name).__doc__
    return f


cdef _nat_divide_op(self, other):
    if PyDelta_Check(other) or util.is_timedelta64_object(other) or other is c_NaT:
        return np.nan
    if util.is_integer_object(other) or util.is_float_object(other):
        return c_NaT
    return NotImplemented


cdef _nat_rdivide_op(self, other):
    if PyDelta_Check(other):
        return np.nan
    return NotImplemented


def __nat_unpickle(*args):
    # return constant defined in the module
    return c_NaT

# ----------------------------------------------------------------------


cdef class _NaT(datetime):
    # cdef readonly:
    #    int64_t value

    # higher than np.ndarray and np.matrix
    __array_priority__ = 100

    def __richcmp__(_NaT self, object other, int op):
        if util.is_datetime64_object(other) or PyDateTime_Check(other):
            # We treat NaT as datetime-like for this comparison
            return op == Py_NE

        elif util.is_timedelta64_object(other) or PyDelta_Check(other):
            # We treat NaT as timedelta-like for this comparison
            return op == Py_NE

        elif util.is_array(other):
            if other.dtype.kind in "mM":
                result = np.empty(other.shape, dtype=np.bool_)
                result.fill(op == Py_NE)
            elif other.dtype.kind == "O":
                result = np.array([PyObject_RichCompare(self, x, op) for x in other])
            elif op == Py_EQ:
                result = np.zeros(other.shape, dtype=bool)
            elif op == Py_NE:
                result = np.ones(other.shape, dtype=bool)
            else:
                return NotImplemented
            return result

        elif PyDate_Check(other):
            # GH#39151 don't defer to datetime.date object
            if op == Py_EQ:
                return False
            if op == Py_NE:
                return True
            warnings.warn(
                "Comparison of NaT with datetime.date is deprecated in "
                "order to match the standard library behavior. "
                "In a future version these will be considered non-comparable.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return False

        return NotImplemented

    def __add__(self, other):
        if self is not c_NaT:
            # TODO(cython3): remove this it moved to __radd__
            # cython __radd__ semantics
            self, other = other, self

        if PyDateTime_Check(other):
            return c_NaT
        elif PyDelta_Check(other):
            return c_NaT
        elif util.is_datetime64_object(other) or util.is_timedelta64_object(other):
            return c_NaT

        elif util.is_integer_object(other):
            # For Period compat
            return c_NaT

        elif util.is_array(other):
            if other.dtype.kind in "mM":
                # If we are adding to datetime64, we treat NaT as timedelta
                #  Either way, result dtype is datetime64
                result = np.empty(other.shape, dtype="datetime64[ns]")
                result.fill("NaT")
                return result
            raise TypeError(f"Cannot add NaT to ndarray with dtype {other.dtype}")

        # Includes Period, DateOffset going through here
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Duplicate some logic from _Timestamp.__sub__ to avoid needing
        # to subclass; allows us to @final(_Timestamp.__sub__)
        cdef:
            bint is_rsub = False

        if self is not c_NaT:
            # cython __rsub__ semantics
            # TODO(cython3): remove __rsub__ logic from here
            self, other = other, self
            is_rsub = True

        if PyDateTime_Check(other):
            return c_NaT
        elif PyDelta_Check(other):
            return c_NaT
        elif util.is_datetime64_object(other) or util.is_timedelta64_object(other):
            return c_NaT

        elif util.is_integer_object(other):
            # For Period compat
            return c_NaT

        elif util.is_array(other):
            if other.dtype.kind == "m":
                if not is_rsub:
                    # NaT - timedelta64 we treat NaT as datetime64, so result
                    #  is datetime64
                    result = np.empty(other.shape, dtype="datetime64[ns]")
                    result.fill("NaT")
                    return result

                # __rsub__ logic here
                # TODO(cython3): remove this, move above code out of ``if not is_rsub`` block
                # timedelta64 - NaT we have to treat NaT as timedelta64
                #  for this to be meaningful, and the result is timedelta64
                result = np.empty(other.shape, dtype="timedelta64[ns]")
                result.fill("NaT")
                return result

            elif other.dtype.kind == "M":
                # We treat NaT as a datetime, so regardless of whether this is
                #  NaT - other or other - NaT, the result is timedelta64
                result = np.empty(other.shape, dtype="timedelta64[ns]")
                result.fill("NaT")
                return result

            raise TypeError(
                f"Cannot subtract NaT from ndarray with dtype {other.dtype}"
            )

        # Includes Period, DateOffset going through here
        return NotImplemented

    def __rsub__(self, other):
        if util.is_array(other):
            if other.dtype.kind == "m":
                # timedelta64 - NaT we have to treat NaT as timedelta64
                #  for this to be meaningful, and the result is timedelta64
                result = np.empty(other.shape, dtype="timedelta64[ns]")
                result.fill("NaT")
                return result

            elif other.dtype.kind == "M":
                # We treat NaT as a datetime, so regardless of whether this is
                #  NaT - other or other - NaT, the result is timedelta64
                result = np.empty(other.shape, dtype="timedelta64[ns]")
                result.fill("NaT")
                return result
        # other cases are same, swap operands is allowed even though we subtract because this is NaT
        return self.__sub__(other)

    def __pos__(self):
        return NaT

    def __neg__(self):
        return NaT

    def __truediv__(self, other):
        return _nat_divide_op(self, other)

    def __floordiv__(self, other):
        return _nat_divide_op(self, other)

    def __mul__(self, other):
        if util.is_integer_object(other) or util.is_float_object(other):
            return NaT
        return NotImplemented

    @property
    def asm8(self) -> np.datetime64:
        return np.datetime64(NPY_NAT, "ns")

    def to_datetime64(self) -> np.datetime64:
        """
        Return a numpy.datetime64 object with 'ns' precision.
        """
        return np.datetime64('NaT', "ns")

    def to_numpy(self, dtype=None, copy=False) -> np.datetime64 | np.timedelta64:
        """
        Convert the Timestamp to a NumPy datetime64 or timedelta64.

        .. versionadded:: 0.25.0

        With the default 'dtype', this is an alias method for `NaT.to_datetime64()`.

        The copy parameter is available here only for compatibility. Its value
        will not affect the return value.

        Returns
        -------
        numpy.datetime64 or numpy.timedelta64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.to_numpy()
        numpy.datetime64('2020-03-14T15:32:52.192548651')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64('NaT')

        >>> pd.NaT.to_numpy("m8[ns]")
        numpy.timedelta64('NaT','ns')
        """
        if dtype is not None:
            # GH#44460
            dtype = np.dtype(dtype)
            if dtype.kind == "M":
                return np.datetime64("NaT").astype(dtype)
            elif dtype.kind == "m":
                return np.timedelta64("NaT").astype(dtype)
            else:
                raise ValueError(
                    "NaT.to_numpy dtype must be a datetime64 dtype, timedelta64 "
                    "dtype, or None."
                )
        return self.to_datetime64()

    def __repr__(self) -> str:
        return "NaT"

    def __str__(self) -> str:
        return "NaT"

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        # This allows Timestamp(ts.isoformat()) to always correctly roundtrip.
        return "NaT"

    def __hash__(self) -> int:
        return NPY_NAT

    @property
    def is_leap_year(self) -> bool:
        return False

    @property
    def is_month_start(self) -> bool:
        return False

    @property
    def is_quarter_start(self) -> bool:
        return False

    @property
    def is_year_start(self) -> bool:
        return False

    @property
    def is_month_end(self) -> bool:
        return False

    @property
    def is_quarter_end(self) -> bool:
        return False

    @property
    def is_year_end(self) -> bool:
        return False


class NaTType(_NaT):
    """
    (N)ot-(A)-(T)ime, the time equivalent of NaN.
    """

    def __new__(cls):
        cdef _NaT base

        base = _NaT.__new__(cls, 1, 1, 1)
        base.value = NPY_NAT

        return base

    @property
    def freq(self):
        warnings.warn(
            "NaT.freq is deprecated and will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return None

    def __reduce_ex__(self, protocol):
        # python 3.6 compat
        # https://bugs.python.org/issue28730
        # now __reduce_ex__ is defined and higher priority than __reduce__
        return self.__reduce__()

    def __reduce__(self):
        return (__nat_unpickle, (None, ))

    def __rtruediv__(self, other):
        return _nat_rdivide_op(self, other)

    def __rfloordiv__(self, other):
        return _nat_rdivide_op(self, other)

    def __rmul__(self, other):
        if util.is_integer_object(other) or util.is_float_object(other):
            return c_NaT
        return NotImplemented

    # ----------------------------------------------------------------------
    # inject the Timestamp field properties
    # these by definition return np.nan

    year = property(fget=lambda self: np.nan)
    quarter = property(fget=lambda self: np.nan)
    month = property(fget=lambda self: np.nan)
    day = property(fget=lambda self: np.nan)
    hour = property(fget=lambda self: np.nan)
    minute = property(fget=lambda self: np.nan)
    second = property(fget=lambda self: np.nan)
    millisecond = property(fget=lambda self: np.nan)
    microsecond = property(fget=lambda self: np.nan)
    nanosecond = property(fget=lambda self: np.nan)

    week = property(fget=lambda self: np.nan)
    dayofyear = property(fget=lambda self: np.nan)
    day_of_year = property(fget=lambda self: np.nan)
    weekofyear = property(fget=lambda self: np.nan)
    days_in_month = property(fget=lambda self: np.nan)
    daysinmonth = property(fget=lambda self: np.nan)
    dayofweek = property(fget=lambda self: np.nan)
    day_of_week = property(fget=lambda self: np.nan)

    # inject Timedelta properties
    days = property(fget=lambda self: np.nan)
    seconds = property(fget=lambda self: np.nan)
    microseconds = property(fget=lambda self: np.nan)
    nanoseconds = property(fget=lambda self: np.nan)

    # inject pd.Period properties
    qyear = property(fget=lambda self: np.nan)

    # ----------------------------------------------------------------------
    # GH9513 NaT methods (except to_datetime64) to raise, return np.nan, or
    # return NaT create functions that raise, for binding to NaTType
    # These are the ones that can get their docstrings from datetime.

    # nan methods
    weekday = _make_nan_func(
        "weekday",
        """
        Return the day of the week represented by the date.

        Monday == 0 ... Sunday == 6.
        """,
    )
    isoweekday = _make_nan_func(
        "isoweekday",
        """
        Return the day of the week represented by the date.

        Monday == 1 ... Sunday == 7.
        """,
    )
    total_seconds = _make_nan_func("total_seconds", timedelta.total_seconds.__doc__)
    month_name = _make_nan_func(
        "month_name",
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """,
    )
    day_name = _make_nan_func(
        "day_name",
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """,
    )
    # _nat_methods
    date = _make_nat_func("date", datetime.date.__doc__)

    utctimetuple = _make_error_func("utctimetuple", datetime)
    timetz = _make_error_func("timetz", datetime)
    timetuple = _make_error_func("timetuple", datetime)
    isocalendar = _make_error_func("isocalendar", datetime)
    dst = _make_error_func("dst", datetime)
    ctime = _make_error_func("ctime", datetime)
    time = _make_error_func("time", datetime)
    toordinal = _make_error_func("toordinal", datetime)
    tzname = _make_error_func("tzname", datetime)
    utcoffset = _make_error_func("utcoffset", datetime)

    # "fromisocalendar" was introduced in 3.8
    fromisocalendar = _make_error_func("fromisocalendar", datetime)

    # ----------------------------------------------------------------------
    # The remaining methods have docstrings copy/pasted from the analogous
    # Timestamp methods.

    strftime = _make_error_func(
        "strftime",
        """
        Return a formatted string of the Timestamp.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """,
    )

    strptime = _make_error_func(
        "strptime",
        """
        Timestamp.strptime(string, format)

        Function is not implemented. Use pd.to_datetime().
        """,
    )

    utcfromtimestamp = _make_error_func(
        "utcfromtimestamp",
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a naive UTC datetime from a POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52')
        """,
    )
    fromtimestamp = _make_error_func(
        "fromtimestamp",
        """
        Timestamp.fromtimestamp(ts)

        Transform timestamp[, tz] to tz's local time from POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time.
        """,
    )
    combine = _make_error_func(
        "combine",
        """
        Timestamp.combine(date, time)

        Combine date, time into datetime with same date and time fields.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """,
    )
    utcnow = _make_error_func(
        "utcnow",
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """,
    )

    timestamp = _make_error_func(
        "timestamp",
        """
        Return POSIX timestamp as float.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
    )

    # GH9513 NaT methods (except to_datetime64) to raise, return np.nan, or
    # return NaT create functions that raise, for binding to NaTType
    astimezone = _make_error_func(
        "astimezone",
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """,
    )
    fromordinal = _make_error_func(
        "fromordinal",
        """
        Construct a timestamp from a a proleptic Gregorian ordinal.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        freq : str, DateOffset
            Offset to apply to the Timestamp.
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Notes
        -----
        By definition there cannot be any tz info on the ordinal itself.

        Examples
        --------
        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')
        """,
    )

    # _nat_methods
    to_pydatetime = _make_nat_func(
        "to_pydatetime",
        """
        Convert a Timestamp object to a native Python datetime object.

        If warn=True, issue a warning if nanoseconds is nonzero.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """,
    )

    now = _make_nat_func(
        "now",
        """
        Return new Timestamp object representing current time local to tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """,
    )
    today = _make_nat_func(
        "today",
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """,
    )
    round = _make_nat_func(
        "round",
        """
        Round the Timestamp to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        a new Timestamp rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Notes
        -----
        If the Timestamp has a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be rounded using multiple frequency units:

        >>> ts.round(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.round(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.round(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.round(freq='L') # milliseconds
        Timestamp('2020-03-14 15:32:52.193000')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.round(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.round(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.round()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.round("H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.round("H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """,
    )
    floor = _make_nat_func(
        "floor",
        """
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq='H') # hour
        Timestamp('2020-03-14 15:00:00')

        >>> ts.floor(freq='T') # minute
        Timestamp('2020-03-14 15:32:00')

        >>> ts.floor(freq='S') # seconds
        Timestamp('2020-03-14 15:32:52')

        >>> ts.floor(freq='N') # nanoseconds
        Timestamp('2020-03-14 15:32:52.192548651')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.floor(freq='5T')
        Timestamp('2020-03-14 15:30:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq='1H30T')
        Timestamp('2020-03-14 15:00:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 03:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.floor("2H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.floor("2H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """,
    )
    ceil = _make_nat_func(
        "ceil",
        """
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {'raise', 'NaT'}, default 'raise'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {'raise', 'shift_forward', 'shift_backward, 'NaT', \
timedelta}, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq='H') # hour
        Timestamp('2020-03-14 16:00:00')

        >>> ts.ceil(freq='T') # minute
        Timestamp('2020-03-14 15:33:00')

        >>> ts.ceil(freq='S') # seconds
        Timestamp('2020-03-14 15:32:53')

        >>> ts.ceil(freq='U') # microseconds
        Timestamp('2020-03-14 15:32:52.192549')

        ``freq`` can also be a multiple of a single unit, like '5T' (i.e.  5 minutes):

        >>> ts.ceil(freq='5T')
        Timestamp('2020-03-14 15:35:00')

        or a combination of multiple units, like '1H30T' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq='1H30T')
        Timestamp('2020-03-14 16:30:00')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.ceil("H", ambiguous=False)
        Timestamp('2021-10-31 02:00:00+0100', tz='Europe/Amsterdam')

        >>> ts_tz.ceil("H", ambiguous=True)
        Timestamp('2021-10-31 02:00:00+0200', tz='Europe/Amsterdam')
        """,
    )

    tz_convert = _make_nat_func(
        "tz_convert",
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """,
    )
    tz_localize = _make_nat_func(
        "tz_localize",
        """
        Localize the Timestamp to a timezone.

        Convert naive Timestamp to local time zone or remove
        timezone from timezone-aware Timestamp.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """,
    )
    replace = _make_nat_func(
        "replace",
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """,
    )
    @property
    def tz(self) -> None:
        return None

    @property
    def tzinfo(self) -> None:
        return None


c_NaT = NaTType()  # C-visible
NaT = c_NaT        # Python-visible


# ----------------------------------------------------------------------

cdef inline bint checknull_with_nat(object val):
    """
    Utility to check if a value is a nat or not.
    """
    return val is None or util.is_nan(val) or val is c_NaT


cdef inline bint is_dt64nat(object val):
    """
    Is this a np.datetime64 object np.datetime64("NaT").
    """
    if util.is_datetime64_object(val):
        return get_datetime64_value(val) == NPY_NAT
    return False


cdef inline bint is_td64nat(object val):
    """
    Is this a np.timedelta64 object np.timedelta64("NaT").
    """
    if util.is_timedelta64_object(val):
        return get_timedelta64_value(val) == NPY_NAT
    return False
