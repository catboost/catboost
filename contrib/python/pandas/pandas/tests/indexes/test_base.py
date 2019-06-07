# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import datetime, timedelta
import math
import operator
import sys

import numpy as np
import pytest

from pandas._libs.tslib import Timestamp
from pandas.compat import (
    PY3, PY35, PY36, StringIO, lrange, lzip, range, text_type, u, zip)
from pandas.compat.numpy import np_datetime64_compat

from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.generic import ABCIndex

import pandas as pd
from pandas import (
    CategoricalIndex, DataFrame, DatetimeIndex, Float64Index, Int64Index,
    PeriodIndex, RangeIndex, Series, TimedeltaIndex, UInt64Index, date_range,
    isna, period_range)
import pandas.core.config as cf
from pandas.core.index import _get_combined_index, ensure_index_from_sequences
from pandas.core.indexes.api import Index, MultiIndex
from pandas.core.sorting import safe_sort
from pandas.tests.indexes.common import Base
import pandas.util.testing as tm
from pandas.util.testing import assert_almost_equal


class TestIndex(Base):
    _holder = Index

    def setup_method(self, method):
        self.indices = dict(unicodeIndex=tm.makeUnicodeIndex(100),
                            strIndex=tm.makeStringIndex(100),
                            dateIndex=tm.makeDateIndex(100),
                            periodIndex=tm.makePeriodIndex(100),
                            tdIndex=tm.makeTimedeltaIndex(100),
                            intIndex=tm.makeIntIndex(100),
                            uintIndex=tm.makeUIntIndex(100),
                            rangeIndex=tm.makeRangeIndex(100),
                            floatIndex=tm.makeFloatIndex(100),
                            boolIndex=Index([True, False]),
                            catIndex=tm.makeCategoricalIndex(100),
                            empty=Index([]),
                            tuples=MultiIndex.from_tuples(lzip(
                                ['foo', 'bar', 'baz'], [1, 2, 3])),
                            repeats=Index([0, 0, 1, 1, 2, 2]))
        self.setup_indices()

    def create_index(self):
        return Index(list('abcde'))

    def generate_index_types(self, skip_index_keys=[]):
        """
        Return a generator of the various index types, leaving
        out the ones with a key in skip_index_keys
        """
        for key, index in self.indices.items():
            if key not in skip_index_keys:
                yield key, index

    def test_can_hold_identifiers(self):
        index = self.create_index()
        key = index[0]
        assert index._can_hold_identifiers_and_holds_name(key) is True

    def test_new_axis(self):
        new_index = self.dateIndex[None, :]
        assert new_index.ndim == 2
        assert isinstance(new_index, np.ndarray)

    def test_copy_and_deepcopy(self):
        new_copy2 = self.intIndex.copy(dtype=int)
        assert new_copy2.dtype.kind == 'i'

    @pytest.mark.parametrize("attr", ['strIndex', 'dateIndex'])
    def test_constructor_regular(self, attr):
        # regular instance creation
        index = getattr(self, attr)
        tm.assert_contains_all(index, index)

    def test_constructor_casting(self):
        # casting
        arr = np.array(self.strIndex)
        index = Index(arr)
        tm.assert_contains_all(arr, index)
        tm.assert_index_equal(self.strIndex, index)

    def test_constructor_copy(self):
        # copy
        arr = np.array(self.strIndex)
        index = Index(arr, copy=True, name='name')
        assert isinstance(index, Index)
        assert index.name == 'name'
        tm.assert_numpy_array_equal(arr, index.values)
        arr[0] = "SOMEBIGLONGSTRING"
        assert index[0] != "SOMEBIGLONGSTRING"

        # what to do here?
        # arr = np.array(5.)
        # pytest.raises(Exception, arr.view, Index)

    def test_constructor_corner(self):
        # corner case
        pytest.raises(TypeError, Index, 0)

    @pytest.mark.parametrize("index_vals", [
        [('A', 1), 'B'], ['B', ('A', 1)]])
    def test_construction_list_mixed_tuples(self, index_vals):
        # see gh-10697: if we are constructing from a mixed list of tuples,
        # make sure that we are independent of the sorting order.
        index = Index(index_vals)
        assert isinstance(index, Index)
        assert not isinstance(index, MultiIndex)

    @pytest.mark.parametrize('na_value', [None, np.nan])
    @pytest.mark.parametrize('vtype', [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value, vtype):
        # GH 18505 : valid tuples containing NaN
        values = [(1, 'two'), (3., na_value)]
        result = Index(vtype(values))
        expected = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cast_as_obj", [True, False])
    @pytest.mark.parametrize("index", [
        pd.date_range('2015-01-01 10:00', freq='D', periods=3,
                      tz='US/Eastern', name='Green Eggs & Ham'),  # DTI with tz
        pd.date_range('2015-01-01 10:00', freq='D', periods=3),  # DTI no tz
        pd.timedelta_range('1 days', freq='D', periods=3),  # td
        pd.period_range('2015-01-01', freq='D', periods=3)  # period
    ])
    def test_constructor_from_index_dtlike(self, cast_as_obj, index):
        if cast_as_obj:
            result = pd.Index(index.astype(object))
        else:
            result = pd.Index(index)

        tm.assert_index_equal(result, index)

        if isinstance(index, pd.DatetimeIndex):
            assert result.tz == index.tz
            if cast_as_obj:
                # GH#23524 check that Index(dti, dtype=object) does not
                #  incorrectly raise ValueError, and that nanoseconds are not
                #  dropped
                index += pd.Timedelta(nanoseconds=50)
                result = pd.Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)

    @pytest.mark.parametrize("index,has_tz", [
        (pd.date_range('2015-01-01 10:00', freq='D', periods=3,
                       tz='US/Eastern'), True),  # datetimetz
        (pd.timedelta_range('1 days', freq='D', periods=3), False),  # td
        (pd.period_range('2015-01-01', freq='D', periods=3), False)  # period
    ])
    def test_constructor_from_series_dtlike(self, index, has_tz):
        result = pd.Index(pd.Series(index))
        tm.assert_index_equal(result, index)

        if has_tz:
            assert result.tz == index.tz

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_from_series(self, klass):
        expected = DatetimeIndex([Timestamp('20110101'), Timestamp('20120101'),
                                  Timestamp('20130101')])
        s = Series([Timestamp('20110101'), Timestamp('20120101'),
                    Timestamp('20130101')])
        result = klass(s)
        tm.assert_index_equal(result, expected)

    def test_constructor_from_series_freq(self):
        # GH 6273
        # create from a series, passing a freq
        dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
        expected = DatetimeIndex(dts, freq='MS')

        s = Series(pd.to_datetime(dts))
        result = DatetimeIndex(s, freq='MS')

        tm.assert_index_equal(result, expected)

    def test_constructor_from_frame_series_freq(self):
        # GH 6273
        # create from a series, passing a freq
        dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
        expected = DatetimeIndex(dts, freq='MS')

        df = pd.DataFrame(np.random.rand(5, 3))
        df['date'] = dts
        result = DatetimeIndex(df['date'], freq='MS')

        assert df['date'].dtype == object
        expected.name = 'date'
        tm.assert_index_equal(result, expected)

        expected = pd.Series(dts, name='date')
        tm.assert_series_equal(df['date'], expected)

        # GH 6274
        # infer freq of same
        freq = pd.infer_freq(df['date'])
        assert freq == 'MS'

    @pytest.mark.parametrize("array", [
        np.arange(5), np.array(['a', 'b', 'c']), date_range(
            '2000-01-01', periods=3).values
    ])
    def test_constructor_ndarray_like(self, array):
        # GH 5460#issuecomment-44474502
        # it should be possible to convert any object that satisfies the numpy
        # ndarray interface directly into an Index
        class ArrayLike(object):
            def __init__(self, array):
                self.array = array

            def __array__(self, dtype=None):
                return self.array

        expected = pd.Index(array)
        result = pd.Index(ArrayLike(array))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', [
        int, 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32',
        'uint16', 'uint8'])
    def test_constructor_int_dtype_float(self, dtype):
        # GH 18400
        if is_unsigned_integer_dtype(dtype):
            index_type = UInt64Index
        else:
            index_type = Int64Index

        expected = index_type([0, 1, 2, 3])
        result = Index([0., 1., 2., 3.], dtype=dtype)
        tm.assert_index_equal(result, expected)

    def test_constructor_int_dtype_nan(self):
        # see gh-15187
        data = [np.nan]
        expected = Float64Index(data)
        result = Index(data, dtype='float')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ['int64', 'uint64'])
    def test_constructor_int_dtype_nan_raises(self, dtype):
        # see gh-15187
        data = [np.nan]
        msg = "cannot convert"
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)

    def test_constructor_no_pandas_array(self):
        ser = pd.Series([1, 2, 3])
        result = pd.Index(ser.array)
        expected = pd.Index([1, 2, 3])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass,dtype,na_val", [
        (pd.Float64Index, np.float64, np.nan),
        (pd.DatetimeIndex, 'datetime64[ns]', pd.NaT)
    ])
    def test_index_ctor_infer_nan_nat(self, klass, dtype, na_val):
        # GH 13467
        na_list = [na_val, na_val]
        expected = klass(na_list)
        assert expected.dtype == dtype

        result = Index(na_list)
        tm.assert_index_equal(result, expected)

        result = Index(np.array(na_list))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("pos", [0, 1])
    @pytest.mark.parametrize("klass,dtype,ctor", [
        (pd.DatetimeIndex, 'datetime64[ns]', np.datetime64('nat')),
        (pd.TimedeltaIndex, 'timedelta64[ns]', np.timedelta64('nat'))
    ])
    def test_index_ctor_infer_nat_dt_like(self, pos, klass, dtype, ctor,
                                          nulls_fixture):
        expected = klass([pd.NaT, pd.NaT])
        assert expected.dtype == dtype
        data = [ctor]
        data.insert(pos, nulls_fixture)

        result = Index(data)
        tm.assert_index_equal(result, expected)

        result = Index(np.array(data, dtype=object))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_index_ctor_nat_result(self, swap_objs):
        # mixed np.datetime64/timedelta64 nat results in object
        data = [np.datetime64('nat'), np.timedelta64('nat')]
        if swap_objs:
            data = data[::-1]

        expected = pd.Index(data, dtype=object)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    def test_index_ctor_infer_periodindex(self):
        xp = period_range('2012-1-1', freq='M', periods=3)
        rs = Index(xp)
        tm.assert_index_equal(rs, xp)
        assert isinstance(rs, PeriodIndex)

    @pytest.mark.parametrize("vals,dtype", [
        ([1, 2, 3, 4, 5], 'int'), ([1.1, np.nan, 2.2, 3.0], 'float'),
        (['A', 'B', 'C', np.nan], 'obj')
    ])
    def test_constructor_simple_new(self, vals, dtype):
        index = Index(vals, name=dtype)
        result = index._simple_new(index.values, dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("vals", [
        [1, 2, 3], np.array([1, 2, 3]), np.array([1, 2, 3], dtype=int),
        # below should coerce
        [1., 2., 3.], np.array([1., 2., 3.], dtype=float)
    ])
    def test_constructor_dtypes_to_int64(self, vals):
        index = Index(vals, dtype=int)
        assert isinstance(index, Int64Index)

    @pytest.mark.parametrize("vals", [
        [1, 2, 3], [1., 2., 3.], np.array([1., 2., 3.]),
        np.array([1, 2, 3], dtype=int), np.array([1., 2., 3.], dtype=float)
    ])
    def test_constructor_dtypes_to_float64(self, vals):
        index = Index(vals, dtype=float)
        assert isinstance(index, Float64Index)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize("vals", [
        [True, False, True], np.array([True, False, True], dtype=bool)
    ])
    def test_constructor_dtypes_to_object(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=bool)
        else:
            index = Index(vals)

        assert isinstance(index, Index)
        assert index.dtype == object

    @pytest.mark.parametrize("vals", [
        [1, 2, 3], np.array([1, 2, 3], dtype=int),
        np.array([np_datetime64_compat('2011-01-01'),
                  np_datetime64_compat('2011-01-02')]),
        [datetime(2011, 1, 1), datetime(2011, 1, 2)]
    ])
    def test_constructor_dtypes_to_categorical(self, vals):
        index = Index(vals, dtype='category')
        assert isinstance(index, CategoricalIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize("vals", [
        Index(np.array([np_datetime64_compat('2011-01-01'),
                        np_datetime64_compat('2011-01-02')])),
        Index([datetime(2011, 1, 1), datetime(2011, 1, 2)])

    ])
    def test_constructor_dtypes_to_datetime(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, DatetimeIndex)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize("vals", [
        np.array([np.timedelta64(1, 'D'), np.timedelta64(1, 'D')]),
        [timedelta(1), timedelta(1)]
    ])
    def test_constructor_dtypes_to_timedelta(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=object)
            assert isinstance(index, Index)
            assert index.dtype == object
        else:
            index = Index(vals)
            assert isinstance(index, TimedeltaIndex)

    @pytest.mark.parametrize("attr, utc", [
        ['values', False],
        ['asi8', True]])
    @pytest.mark.parametrize("klass", [pd.Index, pd.DatetimeIndex])
    def test_constructor_dtypes_datetime(self, tz_naive_fixture, attr, utc,
                                         klass):
        # Test constructing with a datetimetz dtype
        # .values produces numpy datetimes, so these are considered naive
        # .asi8 produces integers, so these are considered epoch timestamps
        # ^the above will be true in a later version. Right now we `.view`
        # the i8 values as NS_DTYPE, effectively treating them as wall times.
        index = pd.date_range('2011-01-01', periods=5)
        arg = getattr(index, attr)
        index = index.tz_localize(tz_naive_fixture)
        dtype = index.dtype

        # TODO(GH-24559): Remove the sys.modules and warnings
        # not sure what this is from. It's Py2 only.
        modules = [sys.modules['pandas.core.indexes.base']]

        if (tz_naive_fixture and attr == "asi8" and
                str(tz_naive_fixture) not in ('UTC', 'tzutc()')):
            ex_warn = FutureWarning
        else:
            ex_warn = None

        # stacklevel is checked elsewhere. We don't do it here since
        # Index will have an frame, throwing off the expected.
        with tm.assert_produces_warning(ex_warn, check_stacklevel=False,
                                        clear=modules):
            result = klass(arg, tz=tz_naive_fixture)
        tm.assert_index_equal(result, index)

        with tm.assert_produces_warning(ex_warn, check_stacklevel=False):
            result = klass(arg, dtype=dtype)
        tm.assert_index_equal(result, index)

        with tm.assert_produces_warning(ex_warn, check_stacklevel=False):
            result = klass(list(arg), tz=tz_naive_fixture)
        tm.assert_index_equal(result, index)

        with tm.assert_produces_warning(ex_warn, check_stacklevel=False):
            result = klass(list(arg), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("attr", ['values', 'asi8'])
    @pytest.mark.parametrize("klass", [pd.Index, pd.TimedeltaIndex])
    def test_constructor_dtypes_timedelta(self, attr, klass):
        index = pd.timedelta_range('1 days', periods=5)
        dtype = index.dtype

        values = getattr(index, attr)

        result = klass(values, dtype=dtype)
        tm.assert_index_equal(result, index)

        result = klass(list(values), dtype=dtype)
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("value", [[], iter([]), (x for x in [])])
    @pytest.mark.parametrize("klass",
                             [Index, Float64Index, Int64Index, UInt64Index,
                              CategoricalIndex, DatetimeIndex, TimedeltaIndex])
    def test_constructor_empty(self, value, klass):
        empty = klass(value)
        assert isinstance(empty, klass)
        assert not len(empty)

    @pytest.mark.parametrize("empty,klass", [
        (PeriodIndex([], freq='B'), PeriodIndex),
        (PeriodIndex(iter([]), freq='B'), PeriodIndex),
        (PeriodIndex((x for x in []), freq='B'), PeriodIndex),
        (RangeIndex(step=1), pd.RangeIndex),
        (MultiIndex(levels=[[1, 2], ['blue', 'red']],
                    codes=[[], []]), MultiIndex)
    ])
    def test_constructor_empty_special(self, empty, klass):
        assert isinstance(empty, klass)
        assert not len(empty)

    def test_constructor_overflow_int64(self):
        # see gh-15832
        msg = ("The elements provided in the data cannot "
               "all be casted to the dtype int64")
        with pytest.raises(OverflowError, match=msg):
            Index([np.iinfo(np.uint64).max - 1], dtype="int64")

    @pytest.mark.xfail(reason="see GH#21311: Index "
                              "doesn't enforce dtype argument")
    def test_constructor_cast(self):
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Index(["a", "b", "c"], dtype=float)

    def test_view_with_args(self):

        restricted = ['unicodeIndex', 'strIndex', 'catIndex', 'boolIndex',
                      'empty']

        for i in restricted:
            ind = self.indices[i]

            # with arguments
            pytest.raises(TypeError, lambda: ind.view('i8'))

        # these are ok
        for i in list(set(self.indices.keys()) - set(restricted)):
            ind = self.indices[i]

            # with arguments
            ind.view('i8')

    def test_astype(self):
        casted = self.intIndex.astype('i8')

        # it works!
        casted.get_loc(5)

        # pass on name
        self.intIndex.name = 'foobar'
        casted = self.intIndex.astype('i8')
        assert casted.name == 'foobar'

    def test_equals_object(self):
        # same
        assert Index(['a', 'b', 'c']).equals(Index(['a', 'b', 'c']))

    @pytest.mark.parametrize("comp", [
        Index(['a', 'b']), Index(['a', 'b', 'd']), ['a', 'b', 'c']])
    def test_not_equals_object(self, comp):
        assert not Index(['a', 'b', 'c']).equals(comp)

    def test_insert(self):

        # GH 7256
        # validate neg/pos inserts
        result = Index(['b', 'c', 'd'])

        # test 0th element
        tm.assert_index_equal(Index(['a', 'b', 'c', 'd']),
                              result.insert(0, 'a'))

        # test Nth element that follows Python list behavior
        tm.assert_index_equal(Index(['b', 'c', 'e', 'd']),
                              result.insert(-1, 'e'))

        # test loc +/- neq (0, -1)
        tm.assert_index_equal(result.insert(1, 'z'), result.insert(-2, 'z'))

        # test empty
        null_index = Index([])
        tm.assert_index_equal(Index(['a']), null_index.insert(0, 'a'))

    def test_insert_missing(self, nulls_fixture):
        # GH 22295
        # test there is no mangling of NA values
        expected = Index(['a', nulls_fixture, 'b', 'c'])
        result = Index(list('abc')).insert(1, nulls_fixture)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("pos,expected", [
        (0, Index(['b', 'c', 'd'], name='index')),
        (-1, Index(['a', 'b', 'c'], name='index'))
    ])
    def test_delete(self, pos, expected):
        index = Index(['a', 'b', 'c', 'd'], name='index')
        result = index.delete(pos)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

    def test_delete_raises(self):
        index = Index(['a', 'b', 'c', 'd'], name='index')
        with pytest.raises((IndexError, ValueError)):
            # either depending on numpy version
            index.delete(5)

    def test_identical(self):

        # index
        i1 = Index(['a', 'b', 'c'])
        i2 = Index(['a', 'b', 'c'])

        assert i1.identical(i2)

        i1 = i1.rename('foo')
        assert i1.equals(i2)
        assert not i1.identical(i2)

        i2 = i2.rename('foo')
        assert i1.identical(i2)

        i3 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')])
        i4 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')], tupleize_cols=False)
        assert not i3.identical(i4)

    def test_is_(self):
        ind = Index(range(10))
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))

        # quasi-implementation dependent
        assert ind.is_(ind.view())
        ind2 = ind.view()
        ind2.name = 'bob'
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        # doesn't matter if Indices are *actually* views of underlying data,
        assert not ind.is_(Index(ind.values))
        arr = np.array(range(1, 11))
        ind1 = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    def test_asof(self):
        d = self.dateIndex[0]
        assert self.dateIndex.asof(d) == d
        assert isna(self.dateIndex.asof(d - timedelta(1)))

        d = self.dateIndex[-1]
        assert self.dateIndex.asof(d + timedelta(1)) == d

        d = self.dateIndex[0].to_pydatetime()
        assert isinstance(self.dateIndex.asof(d), Timestamp)

    def test_asof_datetime_partial(self):
        index = pd.date_range('2010-01-01', periods=2, freq='m')
        expected = Timestamp('2010-02-28')
        result = index.asof('2010-02')
        assert result == expected
        assert not isinstance(result, Index)

    def test_nanosecond_index_access(self):
        s = Series([Timestamp('20130101')]).values.view('i8')[0]
        r = DatetimeIndex([s + 50 + i for i in range(100)])
        x = Series(np.random.randn(100), index=r)

        first_value = x.asof(x.index[0])

        # this does not yet work, as parsing strings is done via dateutil
        # assert first_value == x['2013-01-01 00:00:00.000000050+0000']

        expected_ts = np_datetime64_compat('2013-01-01 00:00:00.000000050+'
                                           '0000', 'ns')
        assert first_value == x[Timestamp(expected_ts)]

    def test_booleanindex(self):
        boolIndex = np.repeat(True, len(self.strIndex)).astype(bool)
        boolIndex[5:30:2] = False

        subIndex = self.strIndex[boolIndex]

        for i, val in enumerate(subIndex):
            assert subIndex.get_loc(val) == i

        subIndex = self.strIndex[list(boolIndex)]
        for i, val in enumerate(subIndex):
            assert subIndex.get_loc(val) == i

    def test_fancy(self):
        sl = self.strIndex[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    @pytest.mark.parametrize("attr", [
        'strIndex', 'intIndex', 'floatIndex'])
    @pytest.mark.parametrize("dtype", [np.int_, np.bool_])
    def test_empty_fancy(self, attr, dtype):
        empty_arr = np.array([], dtype=dtype)
        index = getattr(self, attr)
        empty_index = index.__class__([])

        assert index[[]].identical(empty_index)
        assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize("attr", [
        'strIndex', 'intIndex', 'floatIndex'])
    def test_empty_fancy_raises(self, attr):
        # pd.DatetimeIndex is excluded, because it overrides getitem and should
        # be tested separately.
        empty_farr = np.array([], dtype=np.float_)
        index = getattr(self, attr)
        empty_index = index.__class__([])

        assert index[[]].identical(empty_index)
        # np.ndarray only accepts ndarray of int & bool dtypes, so should Index
        pytest.raises(IndexError, index.__getitem__, empty_farr)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection(self, sort):
        first = self.strIndex[:20]
        second = self.strIndex[:10]
        intersect = first.intersection(second, sort=sort)
        if sort is None:
            tm.assert_index_equal(intersect, second.sort_values())
        assert tm.equalContents(intersect, second)

        # Corner cases
        inter = first.intersection(first, sort=sort)
        assert inter is first

    @pytest.mark.parametrize("index2,keeps_name", [
        (Index([3, 4, 5, 6, 7], name="index"), True),  # preserve same name
        (Index([3, 4, 5, 6, 7], name="other"), False),  # drop diff names
        (Index([3, 4, 5, 6, 7]), False)])
    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_name_preservation(self, index2, keeps_name, sort):
        index1 = Index([1, 2, 3, 4, 5], name='index')
        expected = Index([3, 4, 5])
        result = index1.intersection(index2, sort)

        if keeps_name:
            expected.name = 'index'

        assert result.name == expected.name
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("first_name,second_name,expected_name", [
        ('A', 'A', 'A'), ('A', 'B', None), (None, 'B', None)])
    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_name_preservation2(self, first_name, second_name,
                                             expected_name, sort):
        first = self.strIndex[5:20]
        second = self.strIndex[:10]
        first.name = first_name
        second.name = second_name
        intersect = first.intersection(second, sort=sort)
        assert intersect.name == expected_name

    @pytest.mark.parametrize("index2,keeps_name", [
        (Index([4, 7, 6, 5, 3], name='index'), True),
        (Index([4, 7, 6, 5, 3], name='other'), False)])
    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_monotonic(self, index2, keeps_name, sort):
        index1 = Index([5, 3, 2, 4, 1], name='index')
        expected = Index([5, 3, 4])

        if keeps_name:
            expected.name = "index"

        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index2,expected_arr", [
        (Index(['B', 'D']), ['B']),
        (Index(['B', 'D', 'A']), ['A', 'B', 'A'])])
    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_non_monotonic_non_unique(self, index2, expected_arr,
                                                   sort):
        # non-monotonic non-unique
        index1 = Index(['A', 'B', 'A', 'C'])
        expected = Index(expected_arr, dtype='object')
        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersect_str_dates(self, sort):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        i1 = Index(dt_dates, dtype=object)
        i2 = Index(['aa'], dtype=object)
        result = i2.intersection(i1, sort=sort)

        assert len(result) == 0

    def test_intersect_nosort(self):
        result = pd.Index(['c', 'b', 'a']).intersection(['b', 'a'])
        expected = pd.Index(['b', 'a'])
        tm.assert_index_equal(result, expected)

    def test_intersection_equal_sort(self):
        idx = pd.Index(['c', 'a', 'b'])
        tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
        tm.assert_index_equal(idx.intersection(idx, sort=None), idx)

    @pytest.mark.xfail(reason="Not implemented")
    def test_intersection_equal_sort_true(self):
        # TODO decide on True behaviour
        idx = pd.Index(['c', 'a', 'b'])
        sorted_ = pd.Index(['a', 'b', 'c'])
        tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)

    @pytest.mark.parametrize("sort", [None, False])
    def test_chained_union(self, sort):
        # Chained unions handles names correctly
        i1 = Index([1, 2], name='i1')
        i2 = Index([5, 6], name='i2')
        i3 = Index([3, 4], name='i3')
        union = i1.union(i2.union(i3, sort=sort), sort=sort)
        expected = i1.union(i2, sort=sort).union(i3, sort=sort)
        tm.assert_index_equal(union, expected)

        j1 = Index([1, 2], name='j1')
        j2 = Index([], name='j2')
        j3 = Index([], name='j3')
        union = j1.union(j2.union(j3, sort=sort), sort=sort)
        expected = j1.union(j2, sort=sort).union(j3, sort=sort)
        tm.assert_index_equal(union, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_union(self, sort):
        # TODO: Replace with fixturesult
        first = self.strIndex[5:20]
        second = self.strIndex[:10]
        everything = self.strIndex[:20]

        union = first.union(second, sort=sort)
        if sort is None:
            tm.assert_index_equal(union, everything.sort_values())
        assert tm.equalContents(union, everything)

    @pytest.mark.parametrize('slice_', [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        # https://github.com/pandas-dev/pandas/issues/24959

        idx = pd.Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]
        tm.assert_index_equal(idx.union(other), idx)
        tm.assert_index_equal(other.union(idx), idx)

        # sort=False
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize('slice_', [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        # TODO decide on True behaviour
        # sort=True
        idx = pd.Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]

        result = idx.union(other, sort=True)
        expected = pd.Index([0, 1, 2])
        tm.assert_index_equal(result, expected)

    def test_union_sort_other_incomparable(self):
        # https://github.com/pandas-dev/pandas/issues/24959
        idx = pd.Index([1, pd.Timestamp('2000')])
        # default (sort=None)
        with tm.assert_produces_warning(RuntimeWarning):
            result = idx.union(idx[:1])

        tm.assert_index_equal(result, idx)

        # sort=None
        with tm.assert_produces_warning(RuntimeWarning):
            result = idx.union(idx[:1], sort=None)
        tm.assert_index_equal(result, idx)

        # sort=False
        result = idx.union(idx[:1], sort=False)
        tm.assert_index_equal(result, idx)

    @pytest.mark.xfail(reason="Not implemented")
    def test_union_sort_other_incomparable_true(self):
        # TODO decide on True behaviour
        # sort=True
        idx = pd.Index([1, pd.Timestamp('2000')])
        with pytest.raises(TypeError, match='.*'):
            idx.union(idx[:1], sort=True)

    @pytest.mark.parametrize("klass", [
        np.array, Series, list])
    @pytest.mark.parametrize("sort", [None, False])
    def test_union_from_iterables(self, klass, sort):
        # GH 10149
        # TODO: Replace with fixturesult
        first = self.strIndex[5:20]
        second = self.strIndex[:10]
        everything = self.strIndex[:20]

        case = klass(second.values)
        result = first.union(case, sort=sort)
        if sort is None:
            tm.assert_index_equal(result, everything.sort_values())
        assert tm.equalContents(result, everything)

    @pytest.mark.parametrize("sort", [None, False])
    def test_union_identity(self, sort):
        # TODO: replace with fixturesult
        first = self.strIndex[5:20]

        union = first.union(first, sort=sort)
        # i.e. identity is not preserved when sort is True
        assert (union is first) is (not sort)

        union = first.union([], sort=sort)
        assert (union is first) is (not sort)

        union = Index([]).union(first, sort=sort)
        assert (union is first) is (not sort)

    @pytest.mark.parametrize("first_list", [list('ba'), list()])
    @pytest.mark.parametrize("second_list", [list('ab'), list()])
    @pytest.mark.parametrize("first_name, second_name, expected_name", [
        ('A', 'B', None), (None, 'B', None), ('A', None, None)])
    @pytest.mark.parametrize("sort", [None, False])
    def test_union_name_preservation(self, first_list, second_list, first_name,
                                     second_name, expected_name, sort):
        first = Index(first_list, name=first_name)
        second = Index(second_list, name=second_name)
        union = first.union(second, sort=sort)

        vals = set(first_list).union(second_list)

        if sort is None and len(first_list) > 0 and len(second_list) > 0:
            expected = Index(sorted(vals), name=expected_name)
            tm.assert_index_equal(union, expected)
        else:
            expected = Index(vals, name=expected_name)
            assert tm.equalContents(union, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_union_dt_as_obj(self, sort):
        # TODO: Replace with fixturesult
        firstCat = self.strIndex.union(self.dateIndex)
        secondCat = self.strIndex.union(self.strIndex)

        if self.dateIndex.dtype == np.object_:
            appended = np.append(self.strIndex, self.dateIndex)
        else:
            appended = np.append(self.strIndex, self.dateIndex.astype('O'))

        assert tm.equalContents(firstCat, appended)
        assert tm.equalContents(secondCat, self.strIndex)
        tm.assert_contains_all(self.strIndex, firstCat)
        tm.assert_contains_all(self.strIndex, secondCat)
        tm.assert_contains_all(self.dateIndex, firstCat)

    @pytest.mark.parametrize("method", ['union', 'intersection', 'difference',
                                        'symmetric_difference'])
    def test_setops_disallow_true(self, method):
        idx1 = pd.Index(['a', 'b'])
        idx2 = pd.Index(['b', 'c'])

        with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
            getattr(idx1, method)(idx2, sort=True)

    def test_map_identity_mapping(self):
        # GH 12766
        # TODO: replace with fixture
        for name, cur_index in self.indices.items():
            tm.assert_index_equal(cur_index, cur_index.map(lambda x: x))

    def test_map_with_tuples(self):
        # GH 12766

        # Test that returning a single tuple from an Index
        #   returns an Index.
        index = tm.makeIntIndex(3)
        result = tm.makeIntIndex(3).map(lambda x: (x,))
        expected = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)

        # Test that returning a tuple from a map of a single index
        #   returns a MultiIndex object.
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    def test_map_with_tuples_mi(self):
        # Test that returning a single object from a MultiIndex
        #   returns an Index.
        first_level = ['foo', 'bar', 'baz']
        multi_index = MultiIndex.from_tuples(lzip(first_level, [1, 2, 3]))
        reduced_index = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))

    @pytest.mark.parametrize("attr", [
        'makeDateIndex', 'makePeriodIndex', 'makeTimedeltaIndex'])
    def test_map_tseries_indices_return_index(self, attr):
        index = getattr(tm, attr)(10)
        expected = Index([1] * 10)
        result = index.map(lambda x: 1)
        tm.assert_index_equal(expected, result)

    def test_map_tseries_indices_accsr_return_index(self):
        date_index = tm.makeDateIndex(24, freq='h', name='hourly')
        expected = Index(range(24), name='hourly')
        tm.assert_index_equal(expected, date_index.map(lambda x: x.hour))

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: pd.Series(values, index)])
    def test_map_dictlike(self, mapper):
        # GH 12756
        expected = Index(['foo', 'bar', 'baz'])
        index = tm.makeIntIndex(3)
        result = index.map(mapper(expected.values, index))
        tm.assert_index_equal(result, expected)

        # TODO: replace with fixture
        for name in self.indices.keys():
            if name == 'catIndex':
                # Tested in test_categorical
                continue
            elif name == 'repeats':
                # Cannot map duplicated index
                continue

            index = self.indices[name]
            expected = Index(np.arange(len(index), 0, -1))

            # to match proper result coercion for uints
            if name == 'empty':
                expected = Index([])

            result = index.map(mapper(expected, index))
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("mapper", [
        Series(['foo', 2., 'baz'], index=[0, 2, -1]),
        {0: 'foo', 2: 2.0, -1: 'baz'}])
    def test_map_with_non_function_missing_values(self, mapper):
        # GH 12756
        expected = Index([2., np.nan, 'foo'])
        result = Index([2, 1, 0]).map(mapper)

        tm.assert_index_equal(expected, result)

    def test_map_na_exclusion(self):
        index = Index([1.5, np.nan, 3, np.nan, 5])

        result = index.map(lambda x: x * 2, na_action='ignore')
        expected = index * 2
        tm.assert_index_equal(result, expected)

    def test_map_defaultdict(self):
        index = Index([1, 2, 3])
        default_dict = defaultdict(lambda: 'blank')
        default_dict[1] = 'stuff'
        result = index.map(default_dict)
        expected = Index(['stuff', 'blank', 'blank'])
        tm.assert_index_equal(result, expected)

    def test_append_multiple(self):
        index = Index(['a', 'b', 'c', 'd', 'e', 'f'])

        foos = [index[:2], index[2:4], index[4:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, index)

        # empty
        result = index.append([])
        tm.assert_index_equal(result, index)

    @pytest.mark.parametrize("name,expected", [
        ('foo', 'foo'), ('bar', None)])
    def test_append_empty_preserve_name(self, name, expected):
        left = Index([], name='foo')
        right = Index([1, 2, 3], name=name)

        result = left.append(right)
        assert result.name == expected

    @pytest.mark.parametrize("second_name,expected", [
        (None, None), ('name', 'name')])
    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_name_preservation(self, second_name, expected, sort):
        # TODO: replace with fixturesult
        first = self.strIndex[5:20]
        second = self.strIndex[:10]
        answer = self.strIndex[10:20]

        first.name = 'name'
        second.name = second_name
        result = first.difference(second, sort=sort)

        assert tm.equalContents(result, answer)

        if expected is None:
            assert result.name is None
        else:
            assert result.name == expected

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_empty_arg(self, sort):
        first = self.strIndex[5:20]
        first.name == 'name'
        result = first.difference([], sort)

        assert tm.equalContents(result, first)
        assert result.name == first.name

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_identity(self, sort):
        first = self.strIndex[5:20]
        first.name == 'name'
        result = first.difference(first, sort)

        assert len(result) == 0
        assert result.name == first.name

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_sort(self, sort):
        first = self.strIndex[5:20]
        second = self.strIndex[:10]

        result = first.difference(second, sort)
        expected = self.strIndex[10:20]

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_symmetric_difference(self, sort):
        # smoke
        index1 = Index([5, 2, 3, 4], name='index1')
        index2 = Index([2, 3, 4, 1])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = Index([5, 1])
        assert tm.equalContents(result, expected)
        assert result.name is None
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

        # __xor__ syntax
        expected = index1 ^ index2
        assert tm.equalContents(result, expected)
        assert result.name is None

    @pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
    def test_difference_incomparable(self, opname):
        a = pd.Index([3, pd.Timestamp('2000'), 1])
        b = pd.Index([2, pd.Timestamp('1999'), 1])
        op = operator.methodcaller(opname, b)

        # sort=None, the default
        result = op(a)
        expected = pd.Index([3, pd.Timestamp('2000'), 2, pd.Timestamp('1999')])
        if opname == 'difference':
            expected = expected[:2]
        tm.assert_index_equal(result, expected)

        # sort=False
        op = operator.methodcaller(opname, b, sort=False)
        result = op(a)
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
    def test_difference_incomparable_true(self, opname):
        # TODO decide on True behaviour
        # # sort=True, raises
        a = pd.Index([3, pd.Timestamp('2000'), 1])
        b = pd.Index([2, pd.Timestamp('1999'), 1])
        op = operator.methodcaller(opname, b, sort=True)

        with pytest.raises(TypeError, match='Cannot compare'):
            op(a)

    @pytest.mark.parametrize("sort", [None, False])
    def test_symmetric_difference_mi(self, sort):
        index1 = MultiIndex.from_tuples(self.tuples)
        index2 = MultiIndex.from_tuples([('foo', 1), ('bar', 3)])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = MultiIndex.from_tuples([('bar', 2), ('baz', 3), ('bar', 3)])
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
        assert tm.equalContents(result, expected)

    @pytest.mark.parametrize("index2,expected", [
        (Index([0, 1, np.nan]), Index([2.0, 3.0, 0.0])),
        (Index([0, 1]), Index([np.nan, 2.0, 3.0, 0.0]))])
    @pytest.mark.parametrize("sort", [None, False])
    def test_symmetric_difference_missing(self, index2, expected, sort):
        # GH 13514 change: {nan} - {nan} == {}
        # (GH 6444, sorting of nans, is no longer an issue)
        index1 = Index([1, np.nan, 2, 3])

        result = index1.symmetric_difference(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_symmetric_difference_non_index(self, sort):
        index1 = Index([1, 2, 3, 4], name='index1')
        index2 = np.array([2, 3, 4, 5])
        expected = Index([1, 5])
        result = index1.symmetric_difference(index2, sort=sort)
        assert tm.equalContents(result, expected)
        assert result.name == 'index1'

        result = index1.symmetric_difference(index2, result_name='new_name',
                                             sort=sort)
        assert tm.equalContents(result, expected)
        assert result.name == 'new_name'

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_type(self, sort):
        # GH 20040
        # If taking difference of a set and itself, it
        # needs to preserve the type of the index
        skip_index_keys = ['repeats']
        for key, index in self.generate_index_types(skip_index_keys):
            result = index.difference(index, sort=sort)
            expected = index.drop(index)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_difference(self, sort):
        # GH 20040
        # Test that the intersection of an index with an
        # empty index produces the same index as the difference
        # of an index with itself.  Test for all types
        skip_index_keys = ['repeats']
        for key, index in self.generate_index_types(skip_index_keys):
            inter = index.intersection(index.drop(index))
            diff = index.difference(index, sort=sort)
            tm.assert_index_equal(inter, diff)

    @pytest.mark.parametrize("attr,expected", [
        ('strIndex', False), ('boolIndex', False), ('catIndex', False),
        ('intIndex', True), ('dateIndex', False), ('floatIndex', True)])
    def test_is_numeric(self, attr, expected):
        assert getattr(self, attr).is_numeric() == expected

    @pytest.mark.parametrize("attr,expected", [
        ('strIndex', True), ('boolIndex', True), ('catIndex', False),
        ('intIndex', False), ('dateIndex', False), ('floatIndex', False)])
    def test_is_object(self, attr, expected):
        assert getattr(self, attr).is_object() == expected

    @pytest.mark.parametrize("attr,expected", [
        ('strIndex', False), ('boolIndex', False), ('catIndex', False),
        ('intIndex', False), ('dateIndex', True), ('floatIndex', False)])
    def test_is_all_dates(self, attr, expected):
        assert getattr(self, attr).is_all_dates == expected

    def test_summary(self):
        self._check_method_works(Index._summary)
        # GH3869
        ind = Index(['{other}%s', "~:{range}:0"], name='A')
        result = ind._summary()
        # shouldn't be formatted accidentally.
        assert '~:{range}:0' in result
        assert '{other}%s' in result

    # GH18217
    def test_summary_deprecated(self):
        ind = Index(['{other}%s', "~:{range}:0"], name='A')

        with tm.assert_produces_warning(FutureWarning):
            ind.summary()

    def test_format(self):
        self._check_method_works(Index.format)

        # GH 14626
        # windows has different precision on datetime.datetime.now (it doesn't
        # include us since the default for Timestamp shows these but Index
        # formatting does not we are skipping)
        now = datetime.now()
        if not str(now).endswith("000"):
            index = Index([now])
            formatted = index.format()
            expected = [str(index[0])]
            assert formatted == expected

        self.strIndex[:0].format()

    @pytest.mark.parametrize("vals", [
        [1, 2.0 + 3.0j, 4.], ['a', 'b', 'c']])
    def test_format_missing(self, vals, nulls_fixture):
        # 2845
        vals = list(vals)  # Copy for each iteration
        vals.append(nulls_fixture)
        index = Index(vals)

        formatted = index.format()
        expected = [str(index[0]), str(index[1]), str(index[2]), u('NaN')]

        assert formatted == expected
        assert index[3] is nulls_fixture

    def test_format_with_name_time_info(self):
        # bug I fixed 12/20/2011
        inc = timedelta(hours=4)
        dates = Index([dt + inc for dt in self.dateIndex], name='something')

        formatted = dates.format(name=True)
        assert formatted[0] == 'something'

    def test_format_datetime_with_time(self):
        t = Index([datetime(2012, 2, 7), datetime(2012, 2, 7, 23)])

        result = t.format()
        expected = ['2012-02-07 00:00:00', '2012-02-07 23:00:00']
        assert len(result) == 2
        assert result == expected

    @pytest.mark.parametrize("op", ['any', 'all'])
    def test_logical_compat(self, op):
        index = self.create_index()
        assert getattr(index, op)() == getattr(index.values, op)()

    def _check_method_works(self, method):
        # TODO: make this a dedicated test with parametrized methods
        method(self.empty)
        method(self.dateIndex)
        method(self.unicodeIndex)
        method(self.strIndex)
        method(self.intIndex)
        method(self.tuples)
        method(self.catIndex)

    def test_get_indexer(self):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        r1 = index1.get_indexer(index2)
        e1 = np.array([1, 3, -1], dtype=np.intp)
        assert_almost_equal(r1, e1)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("expected,method", [
        (np.array([-1, 0, 0, 1, 1], dtype=np.intp), 'pad'),
        (np.array([-1, 0, 0, 1, 1], dtype=np.intp), 'ffill'),
        (np.array([0, 0, 1, 1, 2], dtype=np.intp), 'backfill'),
        (np.array([0, 0, 1, 1, 2], dtype=np.intp), 'bfill')])
    def test_get_indexer_methods(self, reverse, expected, method):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        if reverse:
            index1 = index1[::-1]
            expected = expected[::-1]

        result = index2.get_indexer(index1, method=method)
        assert_almost_equal(result, expected)

    def test_get_indexer_invalid(self):
        # GH10411
        index = Index(np.arange(10))

        with pytest.raises(ValueError, match='tolerance argument'):
            index.get_indexer([1, 0], tolerance=1)

        with pytest.raises(ValueError, match='limit argument'):
            index.get_indexer([1, 0], limit=1)

    @pytest.mark.parametrize(
        'method, tolerance, indexer, expected',
        [
            ('pad', None, [0, 5, 9], [0, 5, 9]),
            ('backfill', None, [0, 5, 9], [0, 5, 9]),
            ('nearest', None, [0, 5, 9], [0, 5, 9]),
            ('pad', 0, [0, 5, 9], [0, 5, 9]),
            ('backfill', 0, [0, 5, 9], [0, 5, 9]),
            ('nearest', 0, [0, 5, 9], [0, 5, 9]),

            ('pad', None, [0.2, 1.8, 8.5], [0, 1, 8]),
            ('backfill', None, [0.2, 1.8, 8.5], [1, 2, 9]),
            ('nearest', None, [0.2, 1.8, 8.5], [0, 2, 9]),
            ('pad', 1, [0.2, 1.8, 8.5], [0, 1, 8]),
            ('backfill', 1, [0.2, 1.8, 8.5], [1, 2, 9]),
            ('nearest', 1, [0.2, 1.8, 8.5], [0, 2, 9]),

            ('pad', 0.2, [0.2, 1.8, 8.5], [0, -1, -1]),
            ('backfill', 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]),
            ('nearest', 0.2, [0.2, 1.8, 8.5], [0, 2, -1])])
    def test_get_indexer_nearest(self, method, tolerance, indexer, expected):
        index = Index(np.arange(10))

        actual = index.get_indexer(indexer, method=method, tolerance=tolerance)
        tm.assert_numpy_array_equal(actual, np.array(expected,
                                                     dtype=np.intp))

    @pytest.mark.parametrize('listtype', [list, tuple, Series, np.array])
    @pytest.mark.parametrize(
        'tolerance, expected',
        list(zip([[0.3, 0.3, 0.1], [0.2, 0.1, 0.1],
                  [0.1, 0.5, 0.5]],
                 [[0, 2, -1], [0, -1, -1],
                  [-1, 2, 9]])))
    def test_get_indexer_nearest_listlike_tolerance(self, tolerance,
                                                    expected, listtype):
        index = Index(np.arange(10))

        actual = index.get_indexer([0.2, 1.8, 8.5], method='nearest',
                                   tolerance=listtype(tolerance))
        tm.assert_numpy_array_equal(actual, np.array(expected,
                                                     dtype=np.intp))

    def test_get_indexer_nearest_error(self):
        index = Index(np.arange(10))
        with pytest.raises(ValueError, match='limit argument'):
            index.get_indexer([1, 0], method='nearest', limit=1)

        with pytest.raises(ValueError, match='tolerance size must match'):
            index.get_indexer([1, 0], method='nearest',
                              tolerance=[1, 2, 3])

    @pytest.mark.parametrize("method,expected", [
        ('pad', [8, 7, 0]), ('backfill', [9, 8, 1]), ('nearest', [9, 7, 0])])
    def test_get_indexer_nearest_decreasing(self, method, expected):
        index = Index(np.arange(10))[::-1]

        actual = index.get_indexer([0, 5, 9], method=method)
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))

        actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("method,expected", [
        ('pad', np.array([-1, 0, 1, 1], dtype=np.intp)),
        ('backfill', np.array([0, 0, 1, -1], dtype=np.intp))])
    def test_get_indexer_strings(self, method, expected):
        index = pd.Index(['b', 'c'])
        actual = index.get_indexer(['a', 'b', 'c', 'd'], method=method)

        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self):
        index = pd.Index(['b', 'c'])

        with pytest.raises(TypeError):
            index.get_indexer(['a', 'b', 'c', 'd'], method='nearest')

        with pytest.raises(TypeError):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=2)

        with pytest.raises(TypeError):
            index.get_indexer(['a', 'b', 'c', 'd'], method='pad',
                              tolerance=[2, 2, 2, 2])

    def test_get_indexer_numeric_index_boolean_target(self):
        # GH 16877
        numeric_index = pd.Index(range(4))
        result = numeric_index.get_indexer([True, False, True])
        expected = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_with_NA_values(self, unique_nulls_fixture,
                                        unique_nulls_fixture2):
        # GH 22332
        # check pairwise, that no pair of na values
        # is mangled
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values are not unique
        arr = np.array([unique_nulls_fixture,
                        unique_nulls_fixture2], dtype=np.object)
        index = pd.Index(arr, dtype=np.object)
        result = index.get_indexer([unique_nulls_fixture,
                                    unique_nulls_fixture2, 'Unknown'])
        expected = np.array([0, 1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("method", [None, 'pad', 'backfill', 'nearest'])
    def test_get_loc(self, method):
        index = pd.Index([0, 1, 2])
        assert index.get_loc(1, method=method) == 1

        if method:
            assert index.get_loc(1, method=method, tolerance=0) == 1

    @pytest.mark.parametrize("method", [None, 'pad', 'backfill', 'nearest'])
    def test_get_loc_raises_bad_label(self, method):
        index = pd.Index([0, 1, 2])
        if method:
            # Messages vary across versions
            if PY36:
                msg = 'not supported between'
            elif PY35:
                msg = 'unorderable types'
            else:
                if method == 'nearest':
                    msg = 'unsupported operand'
                else:
                    msg = 'requires scalar valued input'
        else:
            msg = 'invalid key'

        with pytest.raises(TypeError, match=msg):
            index.get_loc([1, 2], method=method)

    @pytest.mark.parametrize("method,loc", [
        ('pad', 1), ('backfill', 2), ('nearest', 1)])
    def test_get_loc_tolerance(self, method, loc):
        index = pd.Index([0, 1, 2])
        assert index.get_loc(1.1, method) == loc
        assert index.get_loc(1.1, method, tolerance=1) == loc

    @pytest.mark.parametrize("method", ['pad', 'backfill', 'nearest'])
    def test_get_loc_outside_tolerance_raises(self, method):
        index = pd.Index([0, 1, 2])
        with pytest.raises(KeyError, match='1.1'):
            index.get_loc(1.1, method, tolerance=0.05)

    def test_get_loc_bad_tolerance_raises(self):
        index = pd.Index([0, 1, 2])
        with pytest.raises(ValueError, match='must be numeric'):
            index.get_loc(1.1, 'nearest', tolerance='invalid')

    def test_get_loc_tolerance_no_method_raises(self):
        index = pd.Index([0, 1, 2])
        with pytest.raises(ValueError, match='tolerance .* valid if'):
            index.get_loc(1.1, tolerance=1)

    def test_get_loc_raises_missized_tolerance(self):
        index = pd.Index([0, 1, 2])
        with pytest.raises(ValueError, match='tolerance size must match'):
            index.get_loc(1.1, 'nearest', tolerance=[1, 1])

    def test_get_loc_raises_object_nearest(self):
        index = pd.Index(['a', 'c'])
        with pytest.raises(TypeError, match='unsupported operand type'):
            index.get_loc('a', method='nearest')

    def test_get_loc_raises_object_tolerance(self):
        index = pd.Index(['a', 'c'])
        with pytest.raises(TypeError, match='unsupported operand type'):
            index.get_loc('a', method='pad', tolerance='invalid')

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs(self, dtype):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)

        assert index.slice_locs(start=2) == (2, n)
        assert index.slice_locs(start=3) == (3, n)
        assert index.slice_locs(3, 8) == (3, 6)
        assert index.slice_locs(5, 10) == (3, n)
        assert index.slice_locs(end=8) == (0, 6)
        assert index.slice_locs(end=9) == (0, 7)

        # reversed
        index2 = index[::-1]
        assert index2.slice_locs(8, 2) == (2, 6)
        assert index2.slice_locs(7, 3) == (2, 5)

    def test_slice_float_locs(self):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=float))
        n = len(index)
        assert index.slice_locs(5.0, 10.0) == (3, n)
        assert index.slice_locs(4.5, 10.5) == (3, 8)

        index2 = index[::-1]
        assert index2.slice_locs(8.5, 1.5) == (2, 6)
        assert index2.slice_locs(10.5, -1) == (0, n)

    @pytest.mark.xfail(reason="Assertions were not correct - see GH#20915")
    def test_slice_ints_with_floats_raises(self):
        # int slicing with floats
        # GH 4892, these are all TypeErrors
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=int))
        n = len(index)

        pytest.raises(TypeError,
                      lambda: index.slice_locs(5.0, 10.0))
        pytest.raises(TypeError,
                      lambda: index.slice_locs(4.5, 10.5))

        index2 = index[::-1]
        pytest.raises(TypeError,
                      lambda: index2.slice_locs(8.5, 1.5), (2, 6))
        pytest.raises(TypeError,
                      lambda: index2.slice_locs(10.5, -1), (0, n))

    def test_slice_locs_dup(self):
        index = Index(['a', 'a', 'b', 'c', 'd', 'd'])
        assert index.slice_locs('a', 'd') == (0, 6)
        assert index.slice_locs(end='d') == (0, 6)
        assert index.slice_locs('a', 'c') == (0, 4)
        assert index.slice_locs('b', 'd') == (2, 6)

        index2 = index[::-1]
        assert index2.slice_locs('d', 'a') == (0, 6)
        assert index2.slice_locs(end='a') == (0, 6)
        assert index2.slice_locs('d', 'b') == (0, 4)
        assert index2.slice_locs('c', 'a') == (2, 6)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_dup_numeric(self, dtype):
        index = Index(np.array([10, 12, 12, 14], dtype=dtype))
        assert index.slice_locs(12, 12) == (1, 3)
        assert index.slice_locs(11, 13) == (1, 3)

        index2 = index[::-1]
        assert index2.slice_locs(12, 12) == (1, 3)
        assert index2.slice_locs(13, 11) == (1, 3)

    def test_slice_locs_na(self):
        index = Index([np.nan, 1, 2])
        assert index.slice_locs(1) == (1, 3)
        assert index.slice_locs(np.nan) == (0, 3)

        index = Index([0, np.nan, np.nan, 1, 2])
        assert index.slice_locs(np.nan) == (1, 5)

    def test_slice_locs_na_raises(self):
        index = Index([np.nan, 1, 2])
        with pytest.raises(KeyError, match=''):
            index.slice_locs(start=1.5)

        with pytest.raises(KeyError, match=''):
            index.slice_locs(end=1.5)

    @pytest.mark.parametrize("in_slice,expected", [
        (pd.IndexSlice[::-1], 'yxdcb'), (pd.IndexSlice['b':'y':-1], ''),
        (pd.IndexSlice['b'::-1], 'b'), (pd.IndexSlice[:'b':-1], 'yxdcb'),
        (pd.IndexSlice[:'y':-1], 'y'), (pd.IndexSlice['y'::-1], 'yxdcb'),
        (pd.IndexSlice['y'::-4], 'yb'),
        # absent labels
        (pd.IndexSlice[:'a':-1], 'yxdcb'), (pd.IndexSlice[:'a':-2], 'ydb'),
        (pd.IndexSlice['z'::-1], 'yxdcb'), (pd.IndexSlice['z'::-3], 'yc'),
        (pd.IndexSlice['m'::-1], 'dcb'), (pd.IndexSlice[:'m':-1], 'yx'),
        (pd.IndexSlice['a':'a':-1], ''), (pd.IndexSlice['z':'z':-1], ''),
        (pd.IndexSlice['m':'m':-1], '')
    ])
    def test_slice_locs_negative_step(self, in_slice, expected):
        index = Index(list('bcdxy'))

        s_start, s_stop = index.slice_locs(in_slice.start, in_slice.stop,
                                           in_slice.step)
        result = index[s_start:s_stop:in_slice.step]
        expected = pd.Index(list(expected))
        tm.assert_index_equal(result, expected)

    def test_drop_by_str_label(self):
        # TODO: Parametrize these after replacing self.strIndex with fixture
        n = len(self.strIndex)
        drop = self.strIndex[lrange(5, 10)]
        dropped = self.strIndex.drop(drop)

        expected = self.strIndex[lrange(5) + lrange(10, n)]
        tm.assert_index_equal(dropped, expected)

        dropped = self.strIndex.drop(self.strIndex[0])
        expected = self.strIndex[1:]
        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize("keys", [['foo', 'bar'], ['1', 'bar']])
    def test_drop_by_str_label_raises_missing_keys(self, keys):
        with pytest.raises(KeyError, match=''):
            self.strIndex.drop(keys)

    def test_drop_by_str_label_errors_ignore(self):
        # TODO: Parametrize these after replacing self.strIndex with fixture

        # errors='ignore'
        n = len(self.strIndex)
        drop = self.strIndex[lrange(5, 10)]
        mixed = drop.tolist() + ['foo']
        dropped = self.strIndex.drop(mixed, errors='ignore')

        expected = self.strIndex[lrange(5) + lrange(10, n)]
        tm.assert_index_equal(dropped, expected)

        dropped = self.strIndex.drop(['foo', 'bar'], errors='ignore')
        expected = self.strIndex[lrange(n)]
        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_loc(self):
        # TODO: Parametrize numeric and str tests after self.strIndex fixture
        index = Index([1, 2, 3])
        dropped = index.drop(1)
        expected = Index([2, 3])

        tm.assert_index_equal(dropped, expected)

    def test_drop_by_numeric_label_raises_missing_keys(self):
        index = Index([1, 2, 3])
        with pytest.raises(KeyError, match=''):
            index.drop([3, 4])

    @pytest.mark.parametrize("key,expected", [
        (4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))])
    def test_drop_by_numeric_label_errors_ignore(self, key, expected):
        index = Index([1, 2, 3])
        dropped = index.drop(key, errors='ignore')

        tm.assert_index_equal(dropped, expected)

    @pytest.mark.parametrize("values", [['a', 'b', ('c', 'd')],
                                        ['a', ('c', 'd'), 'b'],
                                        [('c', 'd'), 'a', 'b']])
    @pytest.mark.parametrize("to_drop", [[('c', 'd'), 'a'], ['a', ('c', 'd')]])
    def test_drop_tuple(self, values, to_drop):
        # GH 18304
        index = pd.Index(values)
        expected = pd.Index(['b'])

        result = index.drop(to_drop)
        tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[0])
        for drop_me in to_drop[1], [to_drop[1]]:
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)

        removed = index.drop(to_drop[1])
        for drop_me in to_drop[1], [to_drop[1]]:
            pytest.raises(KeyError, removed.drop, drop_me)

    @pytest.mark.parametrize("method,expected,sort", [
        ('intersection', np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')],
                                  dtype=[('num', int), ('let', 'a1')]),
         False),

        ('intersection', np.array([(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')],
                                  dtype=[('num', int), ('let', 'a1')]),
         None),

        ('union', np.array([(1, 'A'), (1, 'B'), (1, 'C'), (2, 'A'), (2, 'B'),
                            (2, 'C')], dtype=[('num', int), ('let', 'a1')]),
         None)
    ])
    def test_tuple_union_bug(self, method, expected, sort):
        index1 = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')],
                                dtype=[('num', int), ('let', 'a1')]))
        index2 = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'),
                                 (2, 'B'), (1, 'C'), (2, 'C')],
                                dtype=[('num', int), ('let', 'a1')]))

        result = getattr(index1, method)(index2, sort=sort)
        assert result.ndim == 1

        expected = Index(expected)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("attr", [
        'is_monotonic_increasing', 'is_monotonic_decreasing',
        '_is_strictly_monotonic_increasing',
        '_is_strictly_monotonic_decreasing'])
    def test_is_monotonic_incomparable(self, attr):
        index = Index([5, datetime.now(), 7])
        assert not getattr(index, attr)

    def test_get_set_value(self):
        # TODO: Remove function? GH 19728
        values = np.random.randn(100)
        date = self.dateIndex[67]

        assert_almost_equal(self.dateIndex.get_value(values, date), values[67])

        self.dateIndex.set_value(values, date, 10)
        assert values[67] == 10

    @pytest.mark.parametrize("values", [
        ['foo', 'bar', 'quux'], {'foo', 'bar', 'quux'}])
    @pytest.mark.parametrize("index,expected", [
        (Index(['qux', 'baz', 'foo', 'bar']),
         np.array([False, False, True, True])),
        (Index([]), np.array([], dtype=bool))  # empty
    ])
    def test_isin(self, values, index, expected):
        result = index.isin(values)
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_nan_common_object(self, nulls_fixture, nulls_fixture2):
        # Test cartesian product of null fixtures and ensure that we don't
        # mangle the various types (save a corner case with PyPy)

        # all nans are the same
        if (isinstance(nulls_fixture, float) and
                isinstance(nulls_fixture2, float) and
                math.isnan(nulls_fixture) and
                math.isnan(nulls_fixture2)):
            tm.assert_numpy_array_equal(Index(['a', nulls_fixture]).isin(
                [nulls_fixture2]), np.array([False, True]))

        elif nulls_fixture is nulls_fixture2:  # should preserve NA type
            tm.assert_numpy_array_equal(Index(['a', nulls_fixture]).isin(
                [nulls_fixture2]), np.array([False, True]))

        else:
            tm.assert_numpy_array_equal(Index(['a', nulls_fixture]).isin(
                [nulls_fixture2]), np.array([False, False]))

    def test_isin_nan_common_float64(self, nulls_fixture):
        if nulls_fixture is pd.NaT:
            pytest.skip("pd.NaT not compatible with Float64Index")

        # Float64Index overrides isin, so must be checked separately
        tm.assert_numpy_array_equal(Float64Index([1.0, nulls_fixture]).isin(
            [np.nan]), np.array([False, True]))

        # we cannot compare NaT with NaN
        tm.assert_numpy_array_equal(Float64Index([1.0, nulls_fixture]).isin(
            [pd.NaT]), np.array([False, False]))

    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize("index", [
        Index(['qux', 'baz', 'foo', 'bar']),
        # Float64Index overrides isin, so must be checked separately
        Float64Index([1.0, 2.0, 3.0, 4.0])])
    def test_isin_level_kwarg(self, level, index):
        values = index.tolist()[-2:] + ['nonexisting']

        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(expected, index.isin(values, level=level))

        index.name = 'foobar'
        tm.assert_numpy_array_equal(expected,
                                    index.isin(values, level='foobar'))

    @pytest.mark.parametrize("level", [1, 10, -2])
    @pytest.mark.parametrize("index", [
        Index(['qux', 'baz', 'foo', 'bar']),
        # Float64Index overrides isin, so must be checked separately
        Float64Index([1.0, 2.0, 3.0, 4.0])])
    def test_isin_level_kwarg_raises_bad_index(self, level, index):
        with pytest.raises(IndexError, match='Too many levels'):
            index.isin([], level=level)

    @pytest.mark.parametrize("level", [1.0, 'foobar', 'xyzzy', np.nan])
    @pytest.mark.parametrize("index", [
        Index(['qux', 'baz', 'foo', 'bar']),
        Float64Index([1.0, 2.0, 3.0, 4.0])])
    def test_isin_level_kwarg_raises_key(self, level, index):
        with pytest.raises(KeyError, match='must be same as name'):
            index.isin([], level=level)

    @pytest.mark.parametrize("empty", [[], Series(), np.array([])])
    def test_isin_empty(self, empty):
        # see gh-16991
        index = Index(["a", "b"])
        expected = np.array([False, False])

        result = index.isin(empty)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("values", [
        [1, 2, 3, 4],
        [1., 2., 3., 4.],
        [True, True, True, True],
        ["foo", "bar", "baz", "qux"],
        pd.date_range('2018-01-01', freq='D', periods=4)])
    def test_boolean_cmp(self, values):
        index = Index(values)
        result = (index == values)
        expected = np.array([True, True, True, True], dtype=bool)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("name,level", [
        (None, 0), ('a', 'a')])
    def test_get_level_values(self, name, level):
        expected = self.strIndex.copy()
        if name:
            expected.name = name

        result = expected.get_level_values(level)
        tm.assert_index_equal(result, expected)

    def test_slice_keep_name(self):
        index = Index(['a', 'b'], name='asdf')
        assert index.name == index[1:].name

    # instance attributes of the form self.<name>Index
    @pytest.mark.parametrize('index_kind',
                             ['unicode', 'str', 'date', 'int', 'float'])
    def test_join_self(self, join_type, index_kind):

        res = getattr(self, '{0}Index'.format(index_kind))

        joined = res.join(res, how=join_type)
        assert res is joined

    @pytest.mark.parametrize("method", ['strip', 'rstrip', 'lstrip'])
    def test_str_attribute(self, method):
        # GH9068
        index = Index([' jack', 'jill ', ' jesse ', 'frank'])
        expected = Index([getattr(str, method)(x) for x in index.values])

        result = getattr(index.str, method)()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", [
        Index(range(5)), tm.makeDateIndex(10),
        MultiIndex.from_tuples([('foo', '1'), ('bar', '3')]),
        period_range(start='2000', end='2010', freq='A')])
    def test_str_attribute_raises(self, index):
        with pytest.raises(AttributeError, match='only use .str accessor'):
            index.str.repeat(2)

    @pytest.mark.parametrize("expand,expected", [
        (None, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])),
        (False, Index([['a', 'b', 'c'], ['d', 'e'], ['f']])),
        (True, MultiIndex.from_tuples([('a', 'b', 'c'), ('d', 'e', np.nan),
                                       ('f', np.nan, np.nan)]))])
    def test_str_split(self, expand, expected):
        index = Index(['a b c', 'd e', 'f'])
        if expand is not None:
            result = index.str.split(expand=expand)
        else:
            result = index.str.split()

        tm.assert_index_equal(result, expected)

    def test_str_bool_return(self):
        # test boolean case, should return np.array instead of boolean Index
        index = Index(['a1', 'a2', 'b1', 'b2'])
        result = index.str.startswith('a')
        expected = np.array([True, True, False, False])

        tm.assert_numpy_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_str_bool_series_indexing(self):
        index = Index(['a1', 'a2', 'b1', 'b2'])
        s = Series(range(4), index=index)

        result = s[s.index.str.startswith('a')]
        expected = Series(range(2), index=['a1', 'a2'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("index,expected", [
        (Index(list('abcd')), True), (Index(range(4)), False)])
    def test_tab_completion(self, index, expected):
        # GH 9910
        result = 'str' in dir(index)
        assert result == expected

    def test_indexing_doesnt_change_class(self):
        index = Index([1, 2, 3, 'a', 'b', 'c'])

        assert index[1:3].identical(pd.Index([2, 3], dtype=np.object_))
        assert index[[0, 1]].identical(pd.Index([1, 2], dtype=np.object_))

    def test_outer_join_sort(self):
        left_index = Index(np.random.permutation(15))
        right_index = tm.makeDateIndex(10)

        with tm.assert_produces_warning(RuntimeWarning):
            result = left_index.join(right_index, how='outer')

        # right_index in this case because DatetimeIndex has join precedence
        # over Int64Index
        with tm.assert_produces_warning(RuntimeWarning):
            expected = right_index.astype(object).union(
                left_index.astype(object))

        tm.assert_index_equal(result, expected)

    def test_nan_first_take_datetime(self):
        index = Index([pd.NaT, Timestamp('20130101'), Timestamp('20130102')])
        result = index.take([-1, 0, 1])
        expected = Index([index[-1], index[0], index[1]])
        tm.assert_index_equal(result, expected)

    def test_take_fill_value(self):
        # GH 12631
        index = pd.Index(list('ABC'), name='xxx')
        result = index.take(np.array([1, 0, -1]))
        expected = pd.Index(list('BAC'), name='xxx')
        tm.assert_index_equal(result, expected)

        # fill_value
        result = index.take(np.array([1, 0, -1]), fill_value=True)
        expected = pd.Index(['B', 'A', np.nan], name='xxx')
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = index.take(np.array([1, 0, -1]), allow_fill=False,
                            fill_value=True)
        expected = pd.Index(['B', 'A', 'C'], name='xxx')
        tm.assert_index_equal(result, expected)

    def test_take_fill_value_none_raises(self):
        index = pd.Index(list('ABC'), name='xxx')
        msg = ('When allow_fill=True and fill_value is not None, '
               'all indices must be >= -1')

        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_bad_bounds_raises(self):
        index = pd.Index(list('ABC'), name='xxx')
        with pytest.raises(IndexError, match='out of bounds'):
            index.take(np.array([1, -5]))

    @pytest.mark.parametrize("name", [None, 'foobar'])
    @pytest.mark.parametrize("labels", [
        [], np.array([]), ['A', 'B', 'C'], ['C', 'B', 'A'],
        np.array(['A', 'B', 'C']), np.array(['C', 'B', 'A']),
        # Must preserve name even if dtype changes
        pd.date_range('20130101', periods=3).values,
        pd.date_range('20130101', periods=3).tolist()])
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name,
                                                                 labels):
        # GH6552
        index = pd.Index([0, 1, 2])
        index.name = name
        assert index.reindex(labels)[0].name == name

    @pytest.mark.parametrize("labels", [
        [], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self,
                                                                     labels):
        # GH7774
        index = pd.Index(list('abc'))
        assert index.reindex(labels)[0].dtype.type == np.object_

    @pytest.mark.parametrize("labels,dtype", [
        (pd.Int64Index([]), np.int64),
        (pd.Float64Index([]), np.float64),
        (pd.DatetimeIndex([]), np.datetime64)])
    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self,
                                                                   labels,
                                                                   dtype):
        # GH7774
        index = pd.Index(list('abc'))
        assert index.reindex(labels)[0].dtype.type == dtype

    def test_reindex_no_type_preserve_target_empty_mi(self):
        index = pd.Index(list('abc'))
        result = index.reindex(pd.MultiIndex(
            [pd.Int64Index([]), pd.Float64Index([])], [[], []]))[0]
        assert result.levels[0].dtype.type == np.int64
        assert result.levels[1].dtype.type == np.float64

    def test_groupby(self):
        index = Index(range(5))
        result = index.groupby(np.array([1, 1, 2, 2, 2]))
        expected = {1: pd.Index([0, 1]), 2: pd.Index([2, 3, 4])}

        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize("mi,expected", [
        (MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])),
        (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False]))])
    def test_equals_op_multiindex(self, mi, expected):
        # GH9785
        # test comparisons of multiindex
        df = pd.read_csv(StringIO('a,b,c\n1,2,3\n4,5,6'), index_col=[0, 1])

        result = df.index == mi
        tm.assert_numpy_array_equal(result, expected)

    def test_equals_op_multiindex_identify(self):
        df = pd.read_csv(StringIO('a,b,c\n1,2,3\n4,5,6'), index_col=[0, 1])

        result = df.index == df.index
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", [
        MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]),
        Index(['foo', 'bar', 'baz'])])
    def test_equals_op_mismatched_multiindex_raises(self, index):
        df = pd.read_csv(StringIO('a,b,c\n1,2,3\n4,5,6'), index_col=[0, 1])

        with pytest.raises(ValueError, match="Lengths must match"):
            df.index == index

    def test_equals_op_index_vs_mi_same_length(self):
        mi = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index = Index(['foo', 'bar', 'baz'])

        result = mi == index
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dt_conv", [
        pd.to_datetime, pd.to_timedelta])
    def test_dt_conversion_preserves_name(self, dt_conv):
        # GH 10875
        index = pd.Index(['01:02:03', '01:02:04'], name='label')
        assert index.name == dt_conv(index).name

    @pytest.mark.skipif(not PY3, reason="compat test")
    @pytest.mark.parametrize("index,expected", [
        # ASCII
        # short
        (pd.Index(['a', 'bb', 'ccc']),
         u"""Index(['a', 'bb', 'ccc'], dtype='object')"""),
        # multiple lines
        (pd.Index(['a', 'bb', 'ccc'] * 10),
         u"""\
Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc',
       'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc',
       'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
      dtype='object')"""),
        # truncated
        (pd.Index(['a', 'bb', 'ccc'] * 100),
         u"""\
Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',
       ...
       'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
      dtype='object', length=300)"""),

        # Non-ASCII
        # short
        (pd.Index([u'あ', u'いい', u'ううう']),
         u"""Index(['あ', 'いい', 'ううう'], dtype='object')"""),
        # multiple lines
        (pd.Index([u'あ', u'いい', u'ううう'] * 10),
         (u"Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
          u"'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
          u"'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう'],\n"
          u"      dtype='object')")),
        # truncated
        (pd.Index([u'あ', u'いい', u'ううう'] * 100),
         (u"Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
          u"'あ', 'いい', 'ううう', 'あ',\n"
          u"       ...\n"
          u"       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう', 'あ', 'いい', 'ううう'],\n"
          u"      dtype='object', length=300)"))])
    def test_string_index_repr(self, index, expected):
        result = repr(index)
        assert result == expected

    @pytest.mark.skipif(PY3, reason="compat test")
    @pytest.mark.parametrize("index,expected", [
        # ASCII
        # short
        (pd.Index(['a', 'bb', 'ccc']),
         u"""Index([u'a', u'bb', u'ccc'], dtype='object')"""),
        # multiple lines
        (pd.Index(['a', 'bb', 'ccc'] * 10),
         u"""\
Index([u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a',
       u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb',
       u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc'],
      dtype='object')"""),
        # truncated
        (pd.Index(['a', 'bb', 'ccc'] * 100),
         u"""\
Index([u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a',
       ...
       u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc', u'a', u'bb', u'ccc'],
      dtype='object', length=300)"""),

        # Non-ASCII
        # short
        (pd.Index([u'あ', u'いい', u'ううう']),
         u"""Index([u'あ', u'いい', u'ううう'], dtype='object')"""),
        # multiple lines
        (pd.Index([u'あ', u'いい', u'ううう'] * 10),
         (u"Index([u'あ', u'いい', u'ううう', u'あ', u'いい', "
          u"u'ううう', u'あ', u'いい', u'ううう', u'あ',\n"
          u"       u'いい', u'ううう', u'あ', u'いい', u'ううう', "
          u"u'あ', u'いい', u'ううう', u'あ', u'いい',\n"
          u"       u'ううう', u'あ', u'いい', u'ううう', u'あ', "
          u"u'いい', u'ううう', u'あ', u'いい', u'ううう'],\n"
          u"      dtype='object')")),
        # truncated
        (pd.Index([u'あ', u'いい', u'ううう'] * 100),
         (u"Index([u'あ', u'いい', u'ううう', u'あ', u'いい', "
          u"u'ううう', u'あ', u'いい', u'ううう', u'あ',\n"
          u"       ...\n"
          u"       u'ううう', u'あ', u'いい', u'ううう', u'あ', "
          u"u'いい', u'ううう', u'あ', u'いい', u'ううう'],\n"
          u"      dtype='object', length=300)"))])
    def test_string_index_repr_compat(self, index, expected):
        result = unicode(index)  # noqa
        assert result == expected

    @pytest.mark.skipif(not PY3, reason="compat test")
    @pytest.mark.parametrize("index,expected", [
        # short
        (pd.Index([u'あ', u'いい', u'ううう']),
         (u"Index(['あ', 'いい', 'ううう'], "
          u"dtype='object')")),
        # multiple lines
        (pd.Index([u'あ', u'いい', u'ううう'] * 10),
         (u"Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ', 'いい', 'ううう'],\n"
          u"      dtype='object')""")),
        # truncated
        (pd.Index([u'あ', u'いい', u'ううう'] * 100),
         (u"Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
          u"'ううう', 'あ', 'いい', 'ううう',\n"
          u"       'あ',\n"
          u"       ...\n"
          u"       'ううう', 'あ', 'いい', 'ううう', 'あ', "
          u"'いい', 'ううう', 'あ', 'いい',\n"
          u"       'ううう'],\n"
          u"      dtype='object', length=300)"))])
    def test_string_index_repr_with_unicode_option(self, index, expected):
        # Enable Unicode option -----------------------------------------
        with cf.option_context('display.unicode.east_asian_width', True):
            result = repr(index)
            assert result == expected

    @pytest.mark.skipif(PY3, reason="compat test")
    @pytest.mark.parametrize("index,expected", [
        # short
        (pd.Index([u'あ', u'いい', u'ううう']),
         (u"Index([u'あ', u'いい', u'ううう'], "
          u"dtype='object')")),
        # multiple lines
        (pd.Index([u'あ', u'いい', u'ううう'] * 10),
         (u"Index([u'あ', u'いい', u'ううう', u'あ', u'いい', "
          u"u'ううう', u'あ', u'いい',\n"
          u"       u'ううう', u'あ', u'いい', u'ううう', "
          u"u'あ', u'いい', u'ううう', u'あ',\n"
          u"       u'いい', u'ううう', u'あ', u'いい', "
          u"u'ううう', u'あ', u'いい',\n"
          u"       u'ううう', u'あ', u'いい', u'ううう', "
          u"u'あ', u'いい', u'ううう'],\n"
          u"      dtype='object')")),
        # truncated
        (pd.Index([u'あ', u'いい', u'ううう'] * 100),
         (u"Index([u'あ', u'いい', u'ううう', u'あ', u'いい', "
          u"u'ううう', u'あ', u'いい',\n"
          u"       u'ううう', u'あ',\n"
          u"       ...\n"
          u"       u'ううう', u'あ', u'いい', u'ううう', "
          u"u'あ', u'いい', u'ううう', u'あ',\n"
          u"       u'いい', u'ううう'],\n"
          u"      dtype='object', length=300)"))])
    def test_string_index_repr_with_unicode_option_compat(self, index,
                                                          expected):
        # Enable Unicode option -----------------------------------------
        with cf.option_context('display.unicode.east_asian_width', True):
            result = unicode(index)  # noqa
            assert result == expected

    def test_cached_properties_not_settable(self):
        index = pd.Index([1, 2, 3])
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False

    def test_get_duplicates_deprecated(self):
        index = pd.Index([1, 2, 3])
        with tm.assert_produces_warning(FutureWarning):
            index.get_duplicates()

    def test_tab_complete_warning(self, ip):
        # https://github.com/pandas-dev/pandas/issues/16409
        pytest.importorskip('IPython', minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; idx = pd.Index([1, 2])"
        ip.run_code(code)
        with tm.assert_produces_warning(None):
            with provisionalcompleter('ignore'):
                list(ip.Completer.completions('idx.', 4))


class TestMixedIntIndex(Base):
    # Mostly the tests from common.py for which the results differ
    # in py2 and py3 because ints and strings are uncomparable in py3
    # (GH 13514)

    _holder = Index

    def setup_method(self, method):
        self.indices = dict(mixedIndex=Index([0, 'a', 1, 'b', 2, 'c']))
        self.setup_indices()

    def create_index(self):
        return self.mixedIndex

    def test_argsort(self):
        index = self.create_index()
        if PY36:
            with pytest.raises(TypeError, match="'>|<' not supported"):
                result = index.argsort()
        elif PY3:
            with pytest.raises(TypeError, match="unorderable types"):
                result = index.argsort()
        else:
            result = index.argsort()
            expected = np.array(index).argsort()
            tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self):
        index = self.create_index()
        if PY36:
            with pytest.raises(TypeError, match="'>|<' not supported"):
                result = np.argsort(index)
        elif PY3:
            with pytest.raises(TypeError, match="unorderable types"):
                result = np.argsort(index)
        else:
            result = np.argsort(index)
            expected = index.argsort()
            tm.assert_numpy_array_equal(result, expected)

    def test_copy_name(self):
        # Check that "name" argument passed at initialization is honoured
        # GH12309
        index = self.create_index()

        first = index.__class__(index, copy=True, name='mario')
        second = first.__class__(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second
        tm.assert_index_equal(first, second)

        assert first.name == 'mario'
        assert second.name == 'mario'

        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])

        s3 = s1 * s2

        assert s3.index.name == 'mario'

    def test_copy_name2(self):
        # Check that adding a "name" parameter to the copy is honored
        # GH14302
        index = pd.Index([1, 2], name='MyName')
        index1 = index.copy()

        tm.assert_index_equal(index, index1)

        index2 = index.copy(name='NewName')
        tm.assert_index_equal(index, index2, check_names=False)
        assert index.name == 'MyName'
        assert index2.name == 'NewName'

        index3 = index.copy(names=['NewName'])
        tm.assert_index_equal(index, index3, check_names=False)
        assert index.name == 'MyName'
        assert index.names == ['MyName']
        assert index3.name == 'NewName'
        assert index3.names == ['NewName']

    def test_union_base(self):
        index = self.create_index()
        first = index[3:]
        second = index[:5]

        result = first.union(second)

        expected = Index([0, 1, 2, 'a', 'b', 'c'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [
        np.array, Series, list])
    def test_union_different_type_base(self, klass):
        # GH 10149
        index = self.create_index()
        first = index[3:]
        second = index[:5]

        result = first.union(klass(second.values))

        assert tm.equalContents(result, index)

    def test_unique_na(self):
        idx = pd.Index([2, np.nan, 2, 1], name='my_index')
        expected = pd.Index([2, np.nan, 1], name='my_index')
        result = idx.unique()
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_base(self, sort):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = self.create_index()
        first = index[:5]
        second = index[:3]

        expected = Index([0, 1, 'a']) if sort is None else Index([0, 'a', 1])
        result = first.intersection(second, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [
        np.array, Series, list])
    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection_different_type_base(self, klass, sort):
        # GH 10149
        index = self.create_index()
        first = index[:5]
        second = index[:3]

        result = first.intersection(klass(second.values), sort=sort)
        assert tm.equalContents(result, second)

    @pytest.mark.parametrize("sort", [None, False])
    def test_difference_base(self, sort):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = self.create_index()
        first = index[:4]
        second = index[3:]

        result = first.difference(second, sort)
        expected = Index([0, 'a', 1])
        if sort is None:
            expected = Index(safe_sort(expected))
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = self.create_index()
        first = index[:4]
        second = index[3:]

        result = first.symmetric_difference(second)
        expected = Index([0, 1, 2, 'a', 'c'])
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self):
        index = self.create_index()
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize("how", ['any', 'all'])
    @pytest.mark.parametrize("dtype", [
        None, object, 'category'])
    @pytest.mark.parametrize("vals,expected", [
        ([1, 2, 3], [1, 2, 3]), ([1., 2., 3.], [1., 2., 3.]),
        ([1., 2., np.nan, 3.], [1., 2., 3.]),
        (['A', 'B', 'C'], ['A', 'B', 'C']),
        (['A', np.nan, 'B', 'C'], ['A', 'B', 'C'])])
    def test_dropna(self, how, dtype, vals, expected):
        # GH 6194
        index = pd.Index(vals, dtype=dtype)
        result = index.dropna(how=how)
        expected = pd.Index(expected, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("how", ['any', 'all'])
    @pytest.mark.parametrize("index,expected", [
        (pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03']),
         pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])),
        (pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', pd.NaT]),
         pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'])),
        (pd.TimedeltaIndex(['1 days', '2 days', '3 days']),
         pd.TimedeltaIndex(['1 days', '2 days', '3 days'])),
        (pd.TimedeltaIndex([pd.NaT, '1 days', '2 days', '3 days', pd.NaT]),
         pd.TimedeltaIndex(['1 days', '2 days', '3 days'])),
        (pd.PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'),
         pd.PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M')),
        (pd.PeriodIndex(['2012-02', '2012-04', 'NaT', '2012-05'], freq='M'),
         pd.PeriodIndex(['2012-02', '2012-04', '2012-05'], freq='M'))])
    def test_dropna_dt_like(self, how, index, expected):
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self):
        msg = "invalid how option: xxx"
        with pytest.raises(ValueError, match=msg):
            pd.Index([1, 2, 3]).dropna(how='xxx')

    def test_get_combined_index(self):
        result = _get_combined_index([])
        expected = Index([])
        tm.assert_index_equal(result, expected)

    def test_repeat(self):
        repeats = 2
        index = pd.Index([1, 2, 3])
        expected = pd.Index([1, 1, 2, 2, 3, 3])

        result = index.repeat(repeats)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", [
        pd.Index([np.nan]), pd.Index([np.nan, 1]),
        pd.Index([1, 2, np.nan]), pd.Index(['a', 'b', np.nan]),
        pd.to_datetime(['NaT']), pd.to_datetime(['NaT', '2000-01-01']),
        pd.to_datetime(['2000-01-01', 'NaT', '2000-01-02']),
        pd.to_timedelta(['1 day', 'NaT'])])
    def test_is_monotonic_na(self, index):
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    def test_repr_summary(self):
        with cf.option_context('display.max_seq_items', 10):
            result = repr(pd.Index(np.arange(1000)))
            assert len(result) < 200
            assert "..." in result

    @pytest.mark.parametrize("klass", [Series, DataFrame])
    def test_int_name_format(self, klass):
        index = Index(['a', 'b', 'c'], name=0)
        result = klass(lrange(3), index=index)
        assert '0' in repr(result)

    def test_print_unicode_columns(self):
        df = pd.DataFrame({u("\u05d0"): [1, 2, 3],
                           "\u05d1": [4, 5, 6],
                           "c": [7, 8, 9]})
        repr(df.columns)  # should not raise UnicodeDecodeError

    @pytest.mark.parametrize("func,compat_func", [
        (str, text_type),  # unicode string
        (bytes, str)  # byte string
    ])
    def test_with_unicode(self, func, compat_func):
        index = Index(lrange(1000))

        if PY3:
            func(index)
        else:
            compat_func(index)

    def test_intersect_str_dates(self):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        index1 = Index(dt_dates, dtype=object)
        index2 = Index(['aa'], dtype=object)
        result = index2.intersection(index1)

        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)


class TestIndexUtils(object):

    @pytest.mark.parametrize('data, names, expected', [
        ([[1, 2, 3]], None, Index([1, 2, 3])),
        ([[1, 2, 3]], ['name'], Index([1, 2, 3], name='name')),
        ([['a', 'a'], ['c', 'd']], None,
         MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]])),
        ([['a', 'a'], ['c', 'd']], ['L1', 'L2'],
         MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]],
                    names=['L1', 'L2'])),
    ])
    def test_ensure_index_from_sequences(self, data, names, expected):
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize('opname', ['eq', 'ne', 'le', 'lt', 'ge', 'gt',
                                    'add', 'radd', 'sub', 'rsub',
                                    'mul', 'rmul', 'truediv', 'rtruediv',
                                    'floordiv', 'rfloordiv',
                                    'pow', 'rpow', 'mod', 'divmod'])
def test_generated_op_names(opname, indices):
    index = indices
    if isinstance(index, ABCIndex) and opname == 'rsub':
        # pd.Index.__rsub__ does not exist; though the method does exist
        # for subclasses.  see GH#19723
        return
    opname = '__{name}__'.format(name=opname)
    method = getattr(index, opname)
    assert method.__name__ == opname


@pytest.mark.parametrize('index_maker', tm.index_subclass_makers_generator())
def test_index_subclass_constructor_wrong_kwargs(index_maker):
    # GH #19348
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        index_maker(foo='bar')


def test_deprecated_fastpath():

    with tm.assert_produces_warning(FutureWarning):
        idx = pd.Index(
            np.array(['a', 'b'], dtype=object), name='test', fastpath=True)

    expected = pd.Index(['a', 'b'], name='test')
    tm.assert_index_equal(idx, expected)

    with tm.assert_produces_warning(FutureWarning):
        idx = pd.Int64Index(
            np.array([1, 2, 3], dtype='int64'), name='test', fastpath=True)

    expected = pd.Index([1, 2, 3], name='test', dtype='int64')
    tm.assert_index_equal(idx, expected)

    with tm.assert_produces_warning(FutureWarning):
        idx = pd.RangeIndex(0, 5, 2, name='test', fastpath=True)

    expected = pd.RangeIndex(0, 5, 2, name='test')
    tm.assert_index_equal(idx, expected)

    with tm.assert_produces_warning(FutureWarning):
        idx = pd.CategoricalIndex(['a', 'b', 'c'], name='test', fastpath=True)

    expected = pd.CategoricalIndex(['a', 'b', 'c'], name='test')
    tm.assert_index_equal(idx, expected)
