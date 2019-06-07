# -*- coding: utf-8 -*-

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
    CategoricalDtype, CategoricalDtypeType, DatetimeTZDtype, IntervalDtype,
    PeriodDtype)

import pandas as pd
from pandas.conftest import (
    ALL_EA_INT_DTYPES, ALL_INT_DTYPES, SIGNED_EA_INT_DTYPES, SIGNED_INT_DTYPES,
    UNSIGNED_EA_INT_DTYPES, UNSIGNED_INT_DTYPES)
from pandas.core.sparse.api import SparseDtype
import pandas.util.testing as tm


# EA & Actual Dtypes
def to_ea_dtypes(dtypes):
    """ convert list of string dtypes to EA dtype """
    return [getattr(pd, dt + 'Dtype') for dt in dtypes]


def to_numpy_dtypes(dtypes):
    """ convert list of string dtypes to numpy dtype """
    return [getattr(np, dt) for dt in dtypes if isinstance(dt, str)]


class TestPandasDtype(object):

    # Passing invalid dtype, both as a string or object, must raise TypeError
    # Per issue GH15520
    @pytest.mark.parametrize('box', [pd.Timestamp, 'pd.Timestamp', list])
    def test_invalid_dtype_error(self, box):
        with pytest.raises(TypeError, match='not understood'):
            com.pandas_dtype(box)

    @pytest.mark.parametrize('dtype', [
        object, 'float64', np.object_, np.dtype('object'), 'O',
        np.float64, float, np.dtype('float64')])
    def test_pandas_dtype_valid(self, dtype):
        assert com.pandas_dtype(dtype) == dtype

    @pytest.mark.parametrize('dtype', [
        'M8[ns]', 'm8[ns]', 'object', 'float64', 'int64'])
    def test_numpy_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == np.dtype(dtype)

    def test_numpy_string_dtype(self):
        # do not parse freq-like string as period dtype
        assert com.pandas_dtype('U') == np.dtype('U')
        assert com.pandas_dtype('S') == np.dtype('S')

    @pytest.mark.parametrize('dtype', [
        'datetime64[ns, US/Eastern]',
        'datetime64[ns, Asia/Tokyo]',
        'datetime64[ns, UTC]'])
    def test_datetimetz_dtype(self, dtype):
        assert (com.pandas_dtype(dtype) ==
                DatetimeTZDtype.construct_from_string(dtype))
        assert com.pandas_dtype(dtype) == dtype

    def test_categorical_dtype(self):
        assert com.pandas_dtype('category') == CategoricalDtype()

    @pytest.mark.parametrize('dtype', [
        'period[D]', 'period[3M]', 'period[U]',
        'Period[D]', 'Period[3M]', 'Period[U]'])
    def test_period_dtype(self, dtype):
        assert com.pandas_dtype(dtype) is PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == dtype


dtypes = dict(datetime_tz=com.pandas_dtype('datetime64[ns, US/Eastern]'),
              datetime=com.pandas_dtype('datetime64[ns]'),
              timedelta=com.pandas_dtype('timedelta64[ns]'),
              period=PeriodDtype('D'),
              integer=np.dtype(np.int64),
              float=np.dtype(np.float64),
              object=np.dtype(np.object),
              category=com.pandas_dtype('category'))


@pytest.mark.parametrize('name1,dtype1',
                         list(dtypes.items()),
                         ids=lambda x: str(x))
@pytest.mark.parametrize('name2,dtype2',
                         list(dtypes.items()),
                         ids=lambda x: str(x))
def test_dtype_equal(name1, dtype1, name2, dtype2):

    # match equal to self, but not equal to other
    assert com.is_dtype_equal(dtype1, dtype1)
    if name1 != name2:
        assert not com.is_dtype_equal(dtype1, dtype2)


@pytest.mark.parametrize("dtype1,dtype2", [
    (np.int8, np.int64),
    (np.int16, np.int64),
    (np.int32, np.int64),
    (np.float32, np.float64),
    (PeriodDtype("D"), PeriodDtype("2D")),  # PeriodType
    (com.pandas_dtype("datetime64[ns, US/Eastern]"),
     com.pandas_dtype("datetime64[ns, CET]")),  # Datetime
    (None, None)  # gh-15941: no exception should be raised.
])
def test_dtype_equal_strict(dtype1, dtype2):
    assert not com.is_dtype_equal(dtype1, dtype2)


def get_is_dtype_funcs():
    """
    Get all functions in pandas.core.dtypes.common that
    begin with 'is_' and end with 'dtype'

    """

    fnames = [f for f in dir(com) if (f.startswith('is_') and
                                      f.endswith('dtype'))]
    return [getattr(com, fname) for fname in fnames]


@pytest.mark.parametrize('func',
                         get_is_dtype_funcs(),
                         ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    # see gh-15941
    #
    # No exception should be raised.

    assert not func(None)


def test_is_object():
    assert com.is_object_dtype(object)
    assert com.is_object_dtype(np.array([], dtype=object))

    assert not com.is_object_dtype(int)
    assert not com.is_object_dtype(np.array([], dtype=int))
    assert not com.is_object_dtype([1, 2, 3])


@pytest.mark.parametrize("check_scipy", [
    False, pytest.param(True, marks=td.skip_if_no_scipy)
])
def test_is_sparse(check_scipy):
    assert com.is_sparse(pd.SparseArray([1, 2, 3]))
    assert com.is_sparse(pd.SparseSeries([1, 2, 3]))

    assert not com.is_sparse(np.array([1, 2, 3]))

    if check_scipy:
        import scipy.sparse
        assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))


@td.skip_if_no_scipy
def test_is_scipy_sparse():
    from scipy.sparse import bsr_matrix
    assert com.is_scipy_sparse(bsr_matrix([1, 2, 3]))

    assert not com.is_scipy_sparse(pd.SparseArray([1, 2, 3]))
    assert not com.is_scipy_sparse(pd.SparseSeries([1, 2, 3]))


def test_is_categorical():
    cat = pd.Categorical([1, 2, 3])
    assert com.is_categorical(cat)
    assert com.is_categorical(pd.Series(cat))
    assert com.is_categorical(pd.CategoricalIndex([1, 2, 3]))

    assert not com.is_categorical([1, 2, 3])


def test_is_datetimetz():
    with tm.assert_produces_warning(FutureWarning):
        assert not com.is_datetimetz([1, 2, 3])
        assert not com.is_datetimetz(pd.DatetimeIndex([1, 2, 3]))

        assert com.is_datetimetz(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))

        dtype = DatetimeTZDtype("ns", tz="US/Eastern")
        s = pd.Series([], dtype=dtype)
        assert com.is_datetimetz(s)


def test_is_period_deprecated():
    with tm.assert_produces_warning(FutureWarning):
        assert not com.is_period([1, 2, 3])
        assert not com.is_period(pd.Index([1, 2, 3]))
        assert com.is_period(pd.PeriodIndex(["2017-01-01"], freq="D"))


def test_is_datetime64_dtype():
    assert not com.is_datetime64_dtype(object)
    assert not com.is_datetime64_dtype([1, 2, 3])
    assert not com.is_datetime64_dtype(np.array([], dtype=int))

    assert com.is_datetime64_dtype(np.datetime64)
    assert com.is_datetime64_dtype(np.array([], dtype=np.datetime64))


def test_is_datetime64tz_dtype():
    assert not com.is_datetime64tz_dtype(object)
    assert not com.is_datetime64tz_dtype([1, 2, 3])
    assert not com.is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))
    assert com.is_datetime64tz_dtype(pd.DatetimeIndex(['2000'],
                                                      tz="US/Eastern"))


def test_is_timedelta64_dtype():
    assert not com.is_timedelta64_dtype(object)
    assert not com.is_timedelta64_dtype(None)
    assert not com.is_timedelta64_dtype([1, 2, 3])
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    assert not com.is_timedelta64_dtype('0 days')
    assert not com.is_timedelta64_dtype("0 days 00:00:00")
    assert not com.is_timedelta64_dtype(["0 days 00:00:00"])
    assert not com.is_timedelta64_dtype("NO DATE")

    assert com.is_timedelta64_dtype(np.timedelta64)
    assert com.is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    assert com.is_timedelta64_dtype(pd.to_timedelta(['0 days', '1 days']))


def test_is_period_dtype():
    assert not com.is_period_dtype(object)
    assert not com.is_period_dtype([1, 2, 3])
    assert not com.is_period_dtype(pd.Period("2017-01-01"))

    assert com.is_period_dtype(PeriodDtype(freq="D"))
    assert com.is_period_dtype(pd.PeriodIndex([], freq="A"))


def test_is_interval_dtype():
    assert not com.is_interval_dtype(object)
    assert not com.is_interval_dtype([1, 2, 3])

    assert com.is_interval_dtype(IntervalDtype())

    interval = pd.Interval(1, 2, closed="right")
    assert not com.is_interval_dtype(interval)
    assert com.is_interval_dtype(pd.IntervalIndex([interval]))


def test_is_categorical_dtype():
    assert not com.is_categorical_dtype(object)
    assert not com.is_categorical_dtype([1, 2, 3])

    assert com.is_categorical_dtype(CategoricalDtype())
    assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
    assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))


def test_is_string_dtype():
    assert not com.is_string_dtype(int)
    assert not com.is_string_dtype(pd.Series([1, 2]))

    assert com.is_string_dtype(str)
    assert com.is_string_dtype(object)
    assert com.is_string_dtype(np.array(['a', 'b']))


def test_is_period_arraylike():
    assert not com.is_period_arraylike([1, 2, 3])
    assert not com.is_period_arraylike(pd.Index([1, 2, 3]))
    assert com.is_period_arraylike(pd.PeriodIndex(["2017-01-01"], freq="D"))


def test_is_datetime_arraylike():
    assert not com.is_datetime_arraylike([1, 2, 3])
    assert not com.is_datetime_arraylike(pd.Index([1, 2, 3]))
    assert com.is_datetime_arraylike(pd.DatetimeIndex([1, 2, 3]))


def test_is_datetimelike():
    assert not com.is_datetimelike([1, 2, 3])
    assert not com.is_datetimelike(pd.Index([1, 2, 3]))

    assert com.is_datetimelike(pd.DatetimeIndex([1, 2, 3]))
    assert com.is_datetimelike(pd.PeriodIndex([], freq="A"))
    assert com.is_datetimelike(np.array([], dtype=np.datetime64))
    assert com.is_datetimelike(pd.Series([], dtype="timedelta64[ns]"))
    assert com.is_datetimelike(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_datetimelike(s)


@pytest.mark.parametrize(
    'dtype', [
        pd.Series([1, 2])] +
    ALL_INT_DTYPES + to_numpy_dtypes(ALL_INT_DTYPES) +
    ALL_EA_INT_DTYPES + to_ea_dtypes(ALL_EA_INT_DTYPES))
def test_is_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype', [str, float, np.datetime64, np.timedelta64,
              pd.Index([1, 2.]), np.array(['a', 'b']),
              np.array([], dtype=np.timedelta64)])
def test_is_not_integer_dtype(dtype):
    assert not com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype', [
        pd.Series([1, 2])] +
    SIGNED_INT_DTYPES + to_numpy_dtypes(SIGNED_INT_DTYPES) +
    SIGNED_EA_INT_DTYPES + to_ea_dtypes(SIGNED_EA_INT_DTYPES))
def test_is_signed_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype',
    [
        str, float, np.datetime64, np.timedelta64,
        pd.Index([1, 2.]), np.array(['a', 'b']),
        np.array([], dtype=np.timedelta64)] +
    UNSIGNED_INT_DTYPES + to_numpy_dtypes(UNSIGNED_INT_DTYPES) +
    UNSIGNED_EA_INT_DTYPES + to_ea_dtypes(UNSIGNED_EA_INT_DTYPES))
def test_is_not_signed_integer_dtype(dtype):
    assert not com.is_signed_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype',
    [pd.Series([1, 2], dtype=np.uint32)] +
    UNSIGNED_INT_DTYPES + to_numpy_dtypes(UNSIGNED_INT_DTYPES) +
    UNSIGNED_EA_INT_DTYPES + to_ea_dtypes(UNSIGNED_EA_INT_DTYPES))
def test_is_unsigned_integer_dtype(dtype):
    assert com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype',
    [
        str, float, np.datetime64, np.timedelta64,
        pd.Index([1, 2.]), np.array(['a', 'b']),
        np.array([], dtype=np.timedelta64)] +
    SIGNED_INT_DTYPES + to_numpy_dtypes(SIGNED_INT_DTYPES) +
    SIGNED_EA_INT_DTYPES + to_ea_dtypes(SIGNED_EA_INT_DTYPES))
def test_is_not_unsigned_integer_dtype(dtype):
    assert not com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    'dtype',
    [np.int64, np.array([1, 2], dtype=np.int64), 'Int64', pd.Int64Dtype])
def test_is_int64_dtype(dtype):
    assert com.is_int64_dtype(dtype)


@pytest.mark.parametrize(
    'dtype',
    [
        str, float, np.int32, np.uint64, pd.Index([1, 2.]),
        np.array(['a', 'b']), np.array([1, 2], dtype=np.uint32),
        'int8', 'Int8', pd.Int8Dtype])
def test_is_not_int64_dtype(dtype):
    assert not com.is_int64_dtype(dtype)


def test_is_datetime64_any_dtype():
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(['a', 'b']))

    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_any_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))


def test_is_datetime64_ns_dtype():
    assert not com.is_datetime64_ns_dtype(int)
    assert not com.is_datetime64_ns_dtype(str)
    assert not com.is_datetime64_ns_dtype(np.datetime64)
    assert not com.is_datetime64_ns_dtype(np.array([1, 2]))
    assert not com.is_datetime64_ns_dtype(np.array(['a', 'b']))
    assert not com.is_datetime64_ns_dtype(np.array([], dtype=np.datetime64))

    # This datetime array has the wrong unit (ps instead of ns)
    assert not com.is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))

    assert com.is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_ns_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype=np.dtype('datetime64[ns]')))


def test_is_timedelta64_ns_dtype():
    assert not com.is_timedelta64_ns_dtype(np.dtype('m8[ps]'))
    assert not com.is_timedelta64_ns_dtype(
        np.array([1, 2], dtype=np.timedelta64))

    assert com.is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))


def test_is_datetime_or_timedelta_dtype():
    assert not com.is_datetime_or_timedelta_dtype(int)
    assert not com.is_datetime_or_timedelta_dtype(str)
    assert not com.is_datetime_or_timedelta_dtype(pd.Series([1, 2]))
    assert not com.is_datetime_or_timedelta_dtype(np.array(['a', 'b']))

    # TODO(jreback), this is sligthly suspect
    assert not com.is_datetime_or_timedelta_dtype(
        DatetimeTZDtype("ns", "US/Eastern"))

    assert com.is_datetime_or_timedelta_dtype(np.datetime64)
    assert com.is_datetime_or_timedelta_dtype(np.timedelta64)
    assert com.is_datetime_or_timedelta_dtype(
        np.array([], dtype=np.timedelta64))
    assert com.is_datetime_or_timedelta_dtype(
        np.array([], dtype=np.datetime64))


def test_is_numeric_v_string_like():
    assert not com.is_numeric_v_string_like(1, 1)
    assert not com.is_numeric_v_string_like(1, "foo")
    assert not com.is_numeric_v_string_like("foo", "foo")
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    assert not com.is_numeric_v_string_like(
        np.array(["foo"]), np.array(["foo"]))

    assert com.is_numeric_v_string_like(np.array([1]), "foo")
    assert com.is_numeric_v_string_like("foo", np.array([1]))
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    assert com.is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))


def test_is_datetimelike_v_numeric():
    dt = np.datetime64(pd.datetime(2017, 1, 1))

    assert not com.is_datetimelike_v_numeric(1, 1)
    assert not com.is_datetimelike_v_numeric(dt, dt)
    assert not com.is_datetimelike_v_numeric(np.array([1]), np.array([2]))
    assert not com.is_datetimelike_v_numeric(np.array([dt]), np.array([dt]))

    assert com.is_datetimelike_v_numeric(1, dt)
    assert com.is_datetimelike_v_numeric(1, dt)
    assert com.is_datetimelike_v_numeric(np.array([dt]), 1)
    assert com.is_datetimelike_v_numeric(np.array([1]), dt)
    assert com.is_datetimelike_v_numeric(np.array([dt]), np.array([1]))


def test_is_datetimelike_v_object():
    obj = object()
    dt = np.datetime64(pd.datetime(2017, 1, 1))

    assert not com.is_datetimelike_v_object(dt, dt)
    assert not com.is_datetimelike_v_object(obj, obj)
    assert not com.is_datetimelike_v_object(np.array([dt]), np.array([1]))
    assert not com.is_datetimelike_v_object(np.array([dt]), np.array([dt]))
    assert not com.is_datetimelike_v_object(np.array([obj]), np.array([obj]))

    assert com.is_datetimelike_v_object(dt, obj)
    assert com.is_datetimelike_v_object(obj, dt)
    assert com.is_datetimelike_v_object(np.array([dt]), obj)
    assert com.is_datetimelike_v_object(np.array([obj]), dt)
    assert com.is_datetimelike_v_object(np.array([dt]), np.array([obj]))


def test_needs_i8_conversion():
    assert not com.needs_i8_conversion(str)
    assert not com.needs_i8_conversion(np.int64)
    assert not com.needs_i8_conversion(pd.Series([1, 2]))
    assert not com.needs_i8_conversion(np.array(['a', 'b']))

    assert com.needs_i8_conversion(np.datetime64)
    assert com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    assert com.needs_i8_conversion(pd.DatetimeIndex(
        ["2000"], tz="US/Eastern"))


def test_is_numeric_dtype():
    assert not com.is_numeric_dtype(str)
    assert not com.is_numeric_dtype(np.datetime64)
    assert not com.is_numeric_dtype(np.timedelta64)
    assert not com.is_numeric_dtype(np.array(['a', 'b']))
    assert not com.is_numeric_dtype(np.array([], dtype=np.timedelta64))

    assert com.is_numeric_dtype(int)
    assert com.is_numeric_dtype(float)
    assert com.is_numeric_dtype(np.uint64)
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    assert com.is_numeric_dtype(pd.Index([1, 2.]))


def test_is_string_like_dtype():
    assert not com.is_string_like_dtype(object)
    assert not com.is_string_like_dtype(pd.Series([1, 2]))

    assert com.is_string_like_dtype(str)
    assert com.is_string_like_dtype(np.array(['a', 'b']))


def test_is_float_dtype():
    assert not com.is_float_dtype(str)
    assert not com.is_float_dtype(int)
    assert not com.is_float_dtype(pd.Series([1, 2]))
    assert not com.is_float_dtype(np.array(['a', 'b']))

    assert com.is_float_dtype(float)
    assert com.is_float_dtype(pd.Index([1, 2.]))


def test_is_bool_dtype():
    assert not com.is_bool_dtype(int)
    assert not com.is_bool_dtype(str)
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    assert not com.is_bool_dtype(np.array(['a', 'b']))
    assert not com.is_bool_dtype(pd.Index(['a', 'b']))

    assert com.is_bool_dtype(bool)
    assert com.is_bool_dtype(np.bool)
    assert com.is_bool_dtype(np.array([True, False]))
    assert com.is_bool_dtype(pd.Index([True, False]))


@pytest.mark.parametrize("check_scipy", [
    False, pytest.param(True, marks=td.skip_if_no_scipy)
])
def test_is_extension_type(check_scipy):
    assert not com.is_extension_type([1, 2, 3])
    assert not com.is_extension_type(np.array([1, 2, 3]))
    assert not com.is_extension_type(pd.DatetimeIndex([1, 2, 3]))

    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_type(cat)
    assert com.is_extension_type(pd.Series(cat))
    assert com.is_extension_type(pd.SparseArray([1, 2, 3]))
    assert com.is_extension_type(pd.SparseSeries([1, 2, 3]))
    assert com.is_extension_type(pd.DatetimeIndex(['2000'], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_type(s)

    if check_scipy:
        import scipy.sparse
        assert not com.is_extension_type(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_complex_dtype():
    assert not com.is_complex_dtype(int)
    assert not com.is_complex_dtype(str)
    assert not com.is_complex_dtype(pd.Series([1, 2]))
    assert not com.is_complex_dtype(np.array(['a', 'b']))

    assert com.is_complex_dtype(np.complex)
    assert com.is_complex_dtype(np.array([1 + 1j, 5]))


def test_is_offsetlike():
    assert com.is_offsetlike(np.array([pd.DateOffset(month=3),
                                       pd.offsets.Nano()]))
    assert com.is_offsetlike(pd.offsets.MonthEnd())
    assert com.is_offsetlike(pd.Index([pd.DateOffset(second=1)]))

    assert not com.is_offsetlike(pd.Timedelta(1))
    assert not com.is_offsetlike(np.array([1 + 1j, 5]))

    # mixed case
    assert not com.is_offsetlike(np.array([pd.DateOffset(), pd.Timestamp(0)]))


@pytest.mark.parametrize('input_param,result', [
    (int, np.dtype(int)),
    ('int32', np.dtype('int32')),
    (float, np.dtype(float)),
    ('float64', np.dtype('float64')),
    (np.dtype('float64'), np.dtype('float64')),
    (str, np.dtype(str)),
    (pd.Series([1, 2], dtype=np.dtype('int16')), np.dtype('int16')),
    (pd.Series(['a', 'b']), np.dtype(object)),
    (pd.Index([1, 2]), np.dtype('int64')),
    (pd.Index(['a', 'b']), np.dtype(object)),
    ('category', 'category'),
    (pd.Categorical(['a', 'b']).dtype, CategoricalDtype(['a', 'b'])),
    (pd.Categorical(['a', 'b']), CategoricalDtype(['a', 'b'])),
    (pd.CategoricalIndex(['a', 'b']).dtype, CategoricalDtype(['a', 'b'])),
    (pd.CategoricalIndex(['a', 'b']), CategoricalDtype(['a', 'b'])),
    (CategoricalDtype(), CategoricalDtype()),
    (CategoricalDtype(['a', 'b']), CategoricalDtype()),
    (pd.DatetimeIndex([1, 2]), np.dtype('=M8[ns]')),
    (pd.DatetimeIndex([1, 2]).dtype, np.dtype('=M8[ns]')),
    ('<M8[ns]', np.dtype('<M8[ns]')),
    ('datetime64[ns, Europe/London]', DatetimeTZDtype('ns', 'Europe/London')),
    (pd.SparseSeries([1, 2], dtype='int32'), SparseDtype('int32')),
    (pd.SparseSeries([1, 2], dtype='int32').dtype, SparseDtype('int32')),
    (PeriodDtype(freq='D'), PeriodDtype(freq='D')),
    ('period[D]', PeriodDtype(freq='D')),
    (IntervalDtype(), IntervalDtype()),
])
def test__get_dtype(input_param, result):
    assert com._get_dtype(input_param) == result


@pytest.mark.parametrize('input_param', [None,
                                         1, 1.2,
                                         'random string',
                                         pd.DataFrame([1, 2])])
def test__get_dtype_fails(input_param):
    # python objects
    pytest.raises(TypeError, com._get_dtype, input_param)


@pytest.mark.parametrize('input_param,result', [
    (int, np.dtype(int).type),
    ('int32', np.int32),
    (float, np.dtype(float).type),
    ('float64', np.float64),
    (np.dtype('float64'), np.float64),
    (str, np.dtype(str).type),
    (pd.Series([1, 2], dtype=np.dtype('int16')), np.int16),
    (pd.Series(['a', 'b']), np.object_),
    (pd.Index([1, 2], dtype='int64'), np.int64),
    (pd.Index(['a', 'b']), np.object_),
    ('category', CategoricalDtypeType),
    (pd.Categorical(['a', 'b']).dtype, CategoricalDtypeType),
    (pd.Categorical(['a', 'b']), CategoricalDtypeType),
    (pd.CategoricalIndex(['a', 'b']).dtype, CategoricalDtypeType),
    (pd.CategoricalIndex(['a', 'b']), CategoricalDtypeType),
    (pd.DatetimeIndex([1, 2]), np.datetime64),
    (pd.DatetimeIndex([1, 2]).dtype, np.datetime64),
    ('<M8[ns]', np.datetime64),
    (pd.DatetimeIndex(['2000'], tz='Europe/London'), pd.Timestamp),
    (pd.DatetimeIndex(['2000'], tz='Europe/London').dtype,
     pd.Timestamp),
    ('datetime64[ns, Europe/London]', pd.Timestamp),
    (pd.SparseSeries([1, 2], dtype='int32'), np.int32),
    (pd.SparseSeries([1, 2], dtype='int32').dtype, np.int32),
    (PeriodDtype(freq='D'), pd.Period),
    ('period[D]', pd.Period),
    (IntervalDtype(), pd.Interval),
    (None, type(None)),
    (1, type(None)),
    (1.2, type(None)),
    (pd.DataFrame([1, 2]), type(None)),  # composite dtype
])
def test__is_dtype_type(input_param, result):
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)
