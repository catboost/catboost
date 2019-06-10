from datetime import datetime, timedelta

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import iNaT
import pandas.compat as compat

from pandas import (
    DatetimeIndex, Index, NaT, Period, Series, Timedelta, TimedeltaIndex,
    Timestamp)
from pandas.core.arrays import PeriodArray
from pandas.util import testing as tm


@pytest.mark.parametrize("nat,idx", [(Timestamp("NaT"), DatetimeIndex),
                                     (Timedelta("NaT"), TimedeltaIndex),
                                     (Period("NaT", freq="M"), PeriodArray)])
def test_nat_fields(nat, idx):

    for field in idx._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(NaT, field)
        assert np.isnan(result)

        result = getattr(nat, field)
        assert np.isnan(result)

    for field in idx._bool_ops:

        result = getattr(NaT, field)
        assert result is False

        result = getattr(nat, field)
        assert result is False


def test_nat_vector_field_access():
    idx = DatetimeIndex(["1/1/2000", None, None, "1/4/2000"])

    for field in DatetimeIndex._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(idx, field)
        expected = Index([getattr(x, field) for x in idx])
        tm.assert_index_equal(result, expected)

    ser = Series(idx)

    for field in DatetimeIndex._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(ser.dt, field)
        expected = [getattr(x, field) for x in idx]
        tm.assert_series_equal(result, Series(expected))

    for field in DatetimeIndex._bool_ops:
        result = getattr(ser.dt, field)
        expected = [getattr(x, field) for x in idx]
        tm.assert_series_equal(result, Series(expected))


@pytest.mark.parametrize("klass", [Timestamp, Timedelta, Period])
@pytest.mark.parametrize("value", [None, np.nan, iNaT, float("nan"),
                                   NaT, "NaT", "nat"])
def test_identity(klass, value):
    assert klass(value) is NaT


@pytest.mark.parametrize("klass", [Timestamp, Timedelta, Period])
@pytest.mark.parametrize("value", ["", "nat", "NAT", None, np.nan])
def test_equality(klass, value):
    if klass is Period and value == "":
        pytest.skip("Period cannot parse empty string")

    assert klass(value).value == iNaT


@pytest.mark.parametrize("klass", [Timestamp, Timedelta])
@pytest.mark.parametrize("method", ["round", "floor", "ceil"])
@pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
def test_round_nat(klass, method, freq):
    # see gh-14940
    ts = klass("nat")

    round_method = getattr(ts, method)
    assert round_method(freq) is ts


@pytest.mark.parametrize("method", [
    "astimezone", "combine", "ctime", "dst", "fromordinal",
    "fromtimestamp", "isocalendar", "strftime", "strptime",
    "time", "timestamp", "timetuple", "timetz", "toordinal",
    "tzname", "utcfromtimestamp", "utcnow", "utcoffset",
    "utctimetuple", "timestamp"
])
def test_nat_methods_raise(method):
    # see gh-9513, gh-17329
    msg = "NaTType does not support {method}".format(method=method)

    with pytest.raises(ValueError, match=msg):
        getattr(NaT, method)()


@pytest.mark.parametrize("method", [
    "weekday", "isoweekday"
])
def test_nat_methods_nan(method):
    # see gh-9513, gh-17329
    assert np.isnan(getattr(NaT, method)())


@pytest.mark.parametrize("method", [
    "date", "now", "replace", "today",
    "tz_convert", "tz_localize"
])
def test_nat_methods_nat(method):
    # see gh-8254, gh-9513, gh-17329
    assert getattr(NaT, method)() is NaT


@pytest.mark.parametrize("get_nat", [
    lambda x: NaT,
    lambda x: Timedelta(x),
    lambda x: Timestamp(x)
])
def test_nat_iso_format(get_nat):
    # see gh-12300
    assert get_nat("NaT").isoformat() == "NaT"


@pytest.mark.parametrize("klass,expected", [
    (Timestamp, ["freqstr", "normalize", "to_julian_date", "to_period", "tz"]),
    (Timedelta, ["components", "delta", "is_populated", "to_pytimedelta",
                 "to_timedelta64", "view"])
])
def test_missing_public_nat_methods(klass, expected):
    # see gh-17327
    #
    # NaT should have *most* of the Timestamp and Timedelta methods.
    # Here, we check which public methods NaT does not have. We
    # ignore any missing private methods.
    nat_names = dir(NaT)
    klass_names = dir(klass)

    missing = [x for x in klass_names if x not in nat_names and
               not x.startswith("_")]
    missing.sort()

    assert missing == expected


def _get_overlap_public_nat_methods(klass, as_tuple=False):
    """
    Get overlapping public methods between NaT and another class.

    Parameters
    ----------
    klass : type
        The class to compare with NaT
    as_tuple : bool, default False
        Whether to return a list of tuples of the form (klass, method).

    Returns
    -------
    overlap : list
    """
    nat_names = dir(NaT)
    klass_names = dir(klass)

    overlap = [x for x in nat_names if x in klass_names and
               not x.startswith("_") and
               callable(getattr(klass, x))]

    # Timestamp takes precedence over Timedelta in terms of overlap.
    if klass is Timedelta:
        ts_names = dir(Timestamp)
        overlap = [x for x in overlap if x not in ts_names]

    if as_tuple:
        overlap = [(klass, method) for method in overlap]

    overlap.sort()
    return overlap


@pytest.mark.parametrize("klass,expected", [
    (Timestamp, ["astimezone", "ceil", "combine", "ctime", "date", "day_name",
                 "dst", "floor", "fromisoformat", "fromordinal",
                 "fromtimestamp", "isocalendar", "isoformat", "isoweekday",
                 "month_name", "now", "replace", "round", "strftime",
                 "strptime", "time", "timestamp", "timetuple", "timetz",
                 "to_datetime64", "to_pydatetime", "today", "toordinal",
                 "tz_convert", "tz_localize", "tzname", "utcfromtimestamp",
                 "utcnow", "utcoffset", "utctimetuple", "weekday"]),
    (Timedelta, ["total_seconds"])
])
def test_overlap_public_nat_methods(klass, expected):
    # see gh-17327
    #
    # NaT should have *most* of the Timestamp and Timedelta methods.
    # In case when Timestamp, Timedelta, and NaT are overlap, the overlap
    # is considered to be with Timestamp and NaT, not Timedelta.

    # "fromisoformat" was introduced in 3.7
    if klass is Timestamp and not compat.PY37:
        expected.remove("fromisoformat")

    assert _get_overlap_public_nat_methods(klass) == expected


@pytest.mark.parametrize("compare", (
    _get_overlap_public_nat_methods(Timestamp, True) +
    _get_overlap_public_nat_methods(Timedelta, True))
)
def test_nat_doc_strings(compare):
    # see gh-17327
    #
    # The docstrings for overlapping methods should match.
    klass, method = compare
    klass_doc = getattr(klass, method).__doc__

    nat_doc = getattr(NaT, method).__doc__
    assert klass_doc == nat_doc


_ops = {
    "left_plus_right": lambda a, b: a + b,
    "right_plus_left": lambda a, b: b + a,
    "left_minus_right": lambda a, b: a - b,
    "right_minus_left": lambda a, b: b - a,
    "left_times_right": lambda a, b: a * b,
    "right_times_left": lambda a, b: b * a,
    "left_div_right": lambda a, b: a / b,
    "right_div_left": lambda a, b: b / a,
}


@pytest.mark.parametrize("op_name", list(_ops.keys()))
@pytest.mark.parametrize("value,val_type", [
    (2, "scalar"),
    (1.5, "scalar"),
    (np.nan, "scalar"),
    (timedelta(3600), "timedelta"),
    (Timedelta("5s"), "timedelta"),
    (datetime(2014, 1, 1), "timestamp"),
    (Timestamp("2014-01-01"), "timestamp"),
    (Timestamp("2014-01-01", tz="UTC"), "timestamp"),
    (Timestamp("2014-01-01", tz="US/Eastern"), "timestamp"),
    (pytz.timezone("Asia/Tokyo").localize(datetime(2014, 1, 1)), "timestamp"),
])
def test_nat_arithmetic_scalar(op_name, value, val_type):
    # see gh-6873
    invalid_ops = {
        "scalar": {"right_div_left"},
        "timedelta": {"left_times_right", "right_times_left"},
        "timestamp": {"left_times_right", "right_times_left",
                      "left_div_right", "right_div_left"}
    }

    op = _ops[op_name]

    if op_name in invalid_ops.get(val_type, set()):
        if (val_type == "timedelta" and "times" in op_name and
                isinstance(value, Timedelta)):
            msg = "Cannot multiply"
        else:
            msg = "unsupported operand type"

        with pytest.raises(TypeError, match=msg):
            op(NaT, value)
    else:
        if val_type == "timedelta" and "div" in op_name:
            expected = np.nan
        else:
            expected = NaT

        assert op(NaT, value) is expected


@pytest.mark.parametrize("val,expected", [
    (np.nan, NaT),
    (NaT, np.nan),
    (np.timedelta64("NaT"), np.nan)
])
def test_nat_rfloordiv_timedelta(val, expected):
    # see gh-#18846
    #
    # See also test_timedelta.TestTimedeltaArithmetic.test_floordiv
    td = Timedelta(hours=3, minutes=4)
    assert td // val is expected


@pytest.mark.parametrize("op_name", [
    "left_plus_right", "right_plus_left",
    "left_minus_right", "right_minus_left"
])
@pytest.mark.parametrize("value", [
    DatetimeIndex(["2011-01-01", "2011-01-02"], name="x"),
    DatetimeIndex(["2011-01-01", "2011-01-02"], name="x"),
    TimedeltaIndex(["1 day", "2 day"], name="x"),
])
def test_nat_arithmetic_index(op_name, value):
    # see gh-11718
    exp_name = "x"
    exp_data = [NaT] * 2

    if isinstance(value, DatetimeIndex) and "plus" in op_name:
        expected = DatetimeIndex(exp_data, name=exp_name, tz=value.tz)
    else:
        expected = TimedeltaIndex(exp_data, name=exp_name)

    tm.assert_index_equal(_ops[op_name](NaT, value), expected)


@pytest.mark.parametrize("op_name", [
    "left_plus_right", "right_plus_left",
    "left_minus_right", "right_minus_left"
])
@pytest.mark.parametrize("box", [TimedeltaIndex, Series])
def test_nat_arithmetic_td64_vector(op_name, box):
    # see gh-19124
    vec = box(["1 day", "2 day"], dtype="timedelta64[ns]")
    box_nat = box([NaT, NaT], dtype="timedelta64[ns]")
    tm.assert_equal(_ops[op_name](vec, NaT), box_nat)


def test_nat_pinned_docstrings():
    # see gh-17327
    assert NaT.ctime.__doc__ == datetime.ctime.__doc__
