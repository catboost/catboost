# -*- coding: utf-8 -*-
import re

import numpy as np
import pytest

from pandas.core.dtypes.common import (
    is_bool_dtype, is_categorical, is_categorical_dtype,
    is_datetime64_any_dtype, is_datetime64_dtype, is_datetime64_ns_dtype,
    is_datetime64tz_dtype, is_datetimetz, is_dtype_equal, is_interval_dtype,
    is_period, is_period_dtype, is_string_dtype)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype, DatetimeTZDtype, IntervalDtype, PeriodDtype, registry)

import pandas as pd
from pandas import (
    Categorical, CategoricalIndex, IntervalIndex, Series, date_range)
from pandas.core.sparse.api import SparseDtype
import pandas.util.testing as tm


@pytest.fixture(params=[True, False, None])
def ordered(request):
    return request.param


class Base(object):

    def setup_method(self, method):
        self.dtype = self.create()

    def test_hash(self):
        hash(self.dtype)

    def test_equality_invalid(self):
        assert not self.dtype == 'foo'
        assert not is_dtype_equal(self.dtype, np.int64)

    def test_numpy_informed(self):
        pytest.raises(TypeError, np.dtype, self.dtype)

        assert not self.dtype == np.str_
        assert not np.str_ == self.dtype

    def test_pickle(self):
        # make sure our cache is NOT pickled

        # clear the cache
        type(self.dtype).reset_cache()
        assert not len(self.dtype._cache)

        # force back to the cache
        result = tm.round_trip_pickle(self.dtype)
        assert not len(self.dtype._cache)
        assert result == self.dtype


class TestCategoricalDtype(Base):

    def create(self):
        return CategoricalDtype()

    def test_pickle(self):
        # make sure our cache is NOT pickled

        # clear the cache
        type(self.dtype).reset_cache()
        assert not len(self.dtype._cache)

        # force back to the cache
        result = tm.round_trip_pickle(self.dtype)
        assert result == self.dtype

    def test_hash_vs_equality(self):
        dtype = self.dtype
        dtype2 = CategoricalDtype()
        assert dtype == dtype2
        assert dtype2 == dtype
        assert hash(dtype) == hash(dtype2)

    def test_equality(self):
        assert is_dtype_equal(self.dtype, 'category')
        assert is_dtype_equal(self.dtype, CategoricalDtype())
        assert not is_dtype_equal(self.dtype, 'foo')

    def test_construction_from_string(self):
        result = CategoricalDtype.construct_from_string('category')
        assert is_dtype_equal(self.dtype, result)
        pytest.raises(
            TypeError, lambda: CategoricalDtype.construct_from_string('foo'))

    def test_constructor_invalid(self):
        msg = "Parameter 'categories' must be list-like"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype("category")

    dtype1 = CategoricalDtype(['a', 'b'], ordered=True)
    dtype2 = CategoricalDtype(['x', 'y'], ordered=False)
    c = Categorical([0, 1], dtype=dtype1, fastpath=True)

    @pytest.mark.parametrize('values, categories, ordered, dtype, expected',
                             [
                                 [None, None, None, None,
                                  CategoricalDtype()],
                                 [None, ['a', 'b'], True, None, dtype1],
                                 [c, None, None, dtype2, dtype2],
                                 [c, ['x', 'y'], False, None, dtype2],
                             ])
    def test_from_values_or_dtype(
            self, values, categories, ordered, dtype, expected):
        result = CategoricalDtype._from_values_or_dtype(values, categories,
                                                        ordered, dtype)
        assert result == expected

    @pytest.mark.parametrize('values, categories, ordered, dtype', [
        [None, ['a', 'b'], True, dtype2],
        [None, ['a', 'b'], None, dtype2],
        [None, None, True, dtype2],
    ])
    def test_from_values_or_dtype_raises(self, values, categories,
                                         ordered, dtype):
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(values, categories,
                                                   ordered, dtype)

    def test_is_dtype(self):
        assert CategoricalDtype.is_dtype(self.dtype)
        assert CategoricalDtype.is_dtype('category')
        assert CategoricalDtype.is_dtype(CategoricalDtype())
        assert not CategoricalDtype.is_dtype('foo')
        assert not CategoricalDtype.is_dtype(np.float64)

    def test_basic(self):

        assert is_categorical_dtype(self.dtype)

        factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])

        s = Series(factor, name='A')

        # dtypes
        assert is_categorical_dtype(s.dtype)
        assert is_categorical_dtype(s)
        assert not is_categorical_dtype(np.dtype('float64'))

        assert is_categorical(s.dtype)
        assert is_categorical(s)
        assert not is_categorical(np.dtype('float64'))
        assert not is_categorical(1.0)

    def test_tuple_categories(self):
        categories = [(1, 'a'), (2, 'b'), (3, 'c')]
        result = CategoricalDtype(categories)
        assert all(result.categories == categories)

    @pytest.mark.parametrize("categories, expected", [
        ([True, False], True),
        ([True, False, None], True),
        ([True, False, "a", "b'"], False),
        ([0, 1], False),
    ])
    def test_is_boolean(self, categories, expected):
        cat = Categorical(categories)
        assert cat.dtype._is_boolean is expected
        assert is_bool_dtype(cat) is expected
        assert is_bool_dtype(cat.dtype) is expected


class TestDatetimeTZDtype(Base):

    def create(self):
        return DatetimeTZDtype('ns', 'US/Eastern')

    def test_alias_to_unit_raises(self):
        # 23990
        with tm.assert_produces_warning(FutureWarning):
            DatetimeTZDtype('datetime64[ns, US/Central]')

    def test_alias_to_unit_bad_alias_raises(self):
        # 23990
        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('this is a bad string')

        with pytest.raises(TypeError, match=''):
            DatetimeTZDtype('datetime64[ns, US/NotATZ]')

    def test_hash_vs_equality(self):
        # make sure that we satisfy is semantics
        dtype = self.dtype
        dtype2 = DatetimeTZDtype('ns', 'US/Eastern')
        dtype3 = DatetimeTZDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

        dtype4 = DatetimeTZDtype("ns", "US/Central")
        assert dtype2 != dtype4
        assert hash(dtype2) != hash(dtype4)

    def test_construction(self):
        pytest.raises(ValueError,
                      lambda: DatetimeTZDtype('ms', 'US/Eastern'))

    def test_subclass(self):
        a = DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]')
        b = DatetimeTZDtype.construct_from_string('datetime64[ns, CET]')

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_compat(self):
        assert is_datetime64tz_dtype(self.dtype)
        assert is_datetime64tz_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_any_dtype(self.dtype)
        assert is_datetime64_any_dtype('datetime64[ns, US/Eastern]')
        assert is_datetime64_ns_dtype(self.dtype)
        assert is_datetime64_ns_dtype('datetime64[ns, US/Eastern]')
        assert not is_datetime64_dtype(self.dtype)
        assert not is_datetime64_dtype('datetime64[ns, US/Eastern]')

    def test_construction_from_string(self):
        result = DatetimeTZDtype.construct_from_string(
            'datetime64[ns, US/Eastern]')
        assert is_dtype_equal(self.dtype, result)
        pytest.raises(TypeError,
                      lambda: DatetimeTZDtype.construct_from_string('foo'))

    def test_construct_from_string_raises(self):
        with pytest.raises(TypeError, match="notatz"):
            DatetimeTZDtype.construct_from_string('datetime64[ns, notatz]')

        with pytest.raises(TypeError,
                           match="^Could not construct DatetimeTZDtype$"):
            DatetimeTZDtype.construct_from_string(['datetime64[ns, notatz]'])

    def test_is_dtype(self):
        assert not DatetimeTZDtype.is_dtype(None)
        assert DatetimeTZDtype.is_dtype(self.dtype)
        assert DatetimeTZDtype.is_dtype('datetime64[ns, US/Eastern]')
        assert not DatetimeTZDtype.is_dtype('foo')
        assert DatetimeTZDtype.is_dtype(DatetimeTZDtype('ns', 'US/Pacific'))
        assert not DatetimeTZDtype.is_dtype(np.float64)

    def test_equality(self):
        assert is_dtype_equal(self.dtype, 'datetime64[ns, US/Eastern]')
        assert is_dtype_equal(self.dtype, DatetimeTZDtype('ns', 'US/Eastern'))
        assert not is_dtype_equal(self.dtype, 'foo')
        assert not is_dtype_equal(self.dtype, DatetimeTZDtype('ns', 'CET'))
        assert not is_dtype_equal(DatetimeTZDtype('ns', 'US/Eastern'),
                                  DatetimeTZDtype('ns', 'US/Pacific'))

        # numpy compat
        assert is_dtype_equal(np.dtype("M8[ns]"), "datetime64[ns]")

    def test_basic(self):

        assert is_datetime64tz_dtype(self.dtype)

        dr = date_range('20130101', periods=3, tz='US/Eastern')
        s = Series(dr, name='A')

        # dtypes
        assert is_datetime64tz_dtype(s.dtype)
        assert is_datetime64tz_dtype(s)
        assert not is_datetime64tz_dtype(np.dtype('float64'))
        assert not is_datetime64tz_dtype(1.0)

        with tm.assert_produces_warning(FutureWarning):
            assert is_datetimetz(s)
            assert is_datetimetz(s.dtype)
            assert not is_datetimetz(np.dtype('float64'))
            assert not is_datetimetz(1.0)

    def test_dst(self):

        dr1 = date_range('2013-01-01', periods=3, tz='US/Eastern')
        s1 = Series(dr1, name='A')
        assert is_datetime64tz_dtype(s1)
        with tm.assert_produces_warning(FutureWarning):
            assert is_datetimetz(s1)

        dr2 = date_range('2013-08-01', periods=3, tz='US/Eastern')
        s2 = Series(dr2, name='A')
        assert is_datetime64tz_dtype(s2)
        with tm.assert_produces_warning(FutureWarning):
            assert is_datetimetz(s2)
        assert s1.dtype == s2.dtype

    @pytest.mark.parametrize('tz', ['UTC', 'US/Eastern'])
    @pytest.mark.parametrize('constructor', ['M8', 'datetime64'])
    def test_parser(self, tz, constructor):
        # pr #11245
        dtz_str = '{con}[ns, {tz}]'.format(con=constructor, tz=tz)
        result = DatetimeTZDtype.construct_from_string(dtz_str)
        expected = DatetimeTZDtype('ns', tz)
        assert result == expected

    def test_empty(self):
        with pytest.raises(TypeError, match="A 'tz' is required."):
            DatetimeTZDtype()


class TestPeriodDtype(Base):

    def create(self):
        return PeriodDtype('D')

    def test_hash_vs_equality(self):
        # make sure that we satisfy is semantics
        dtype = self.dtype
        dtype2 = PeriodDtype('D')
        dtype3 = PeriodDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is dtype2
        assert dtype2 is dtype
        assert dtype3 is dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

    def test_construction(self):
        with pytest.raises(ValueError):
            PeriodDtype('xx')

        for s in ['period[D]', 'Period[D]', 'D']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day()
            assert is_period_dtype(dt)

        for s in ['period[3D]', 'Period[3D]', '3D']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day(3)
            assert is_period_dtype(dt)

        for s in ['period[26H]', 'Period[26H]', '26H',
                  'period[1D2H]', 'Period[1D2H]', '1D2H']:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Hour(26)
            assert is_period_dtype(dt)

    def test_subclass(self):
        a = PeriodDtype('period[D]')
        b = PeriodDtype('period[3D]')

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_identity(self):
        assert PeriodDtype('period[D]') == PeriodDtype('period[D]')
        assert PeriodDtype('period[D]') is PeriodDtype('period[D]')

        assert PeriodDtype('period[3D]') == PeriodDtype('period[3D]')
        assert PeriodDtype('period[3D]') is PeriodDtype('period[3D]')

        assert PeriodDtype('period[1S1U]') == PeriodDtype('period[1000001U]')
        assert PeriodDtype('period[1S1U]') is PeriodDtype('period[1000001U]')

    def test_compat(self):
        assert not is_datetime64_ns_dtype(self.dtype)
        assert not is_datetime64_ns_dtype('period[D]')
        assert not is_datetime64_dtype(self.dtype)
        assert not is_datetime64_dtype('period[D]')

    def test_construction_from_string(self):
        result = PeriodDtype('period[D]')
        assert is_dtype_equal(self.dtype, result)
        result = PeriodDtype.construct_from_string('period[D]')
        assert is_dtype_equal(self.dtype, result)
        with pytest.raises(TypeError):
            PeriodDtype.construct_from_string('foo')
        with pytest.raises(TypeError):
            PeriodDtype.construct_from_string('period[foo]')
        with pytest.raises(TypeError):
            PeriodDtype.construct_from_string('foo[D]')

        with pytest.raises(TypeError):
            PeriodDtype.construct_from_string('datetime64[ns]')
        with pytest.raises(TypeError):
            PeriodDtype.construct_from_string('datetime64[ns, US/Eastern]')

    def test_is_dtype(self):
        assert PeriodDtype.is_dtype(self.dtype)
        assert PeriodDtype.is_dtype('period[D]')
        assert PeriodDtype.is_dtype('period[3D]')
        assert PeriodDtype.is_dtype(PeriodDtype('3D'))
        assert PeriodDtype.is_dtype('period[U]')
        assert PeriodDtype.is_dtype('period[S]')
        assert PeriodDtype.is_dtype(PeriodDtype('U'))
        assert PeriodDtype.is_dtype(PeriodDtype('S'))

        assert not PeriodDtype.is_dtype('D')
        assert not PeriodDtype.is_dtype('3D')
        assert not PeriodDtype.is_dtype('U')
        assert not PeriodDtype.is_dtype('S')
        assert not PeriodDtype.is_dtype('foo')
        assert not PeriodDtype.is_dtype(np.object_)
        assert not PeriodDtype.is_dtype(np.int64)
        assert not PeriodDtype.is_dtype(np.float64)

    def test_equality(self):
        assert is_dtype_equal(self.dtype, 'period[D]')
        assert is_dtype_equal(self.dtype, PeriodDtype('D'))
        assert is_dtype_equal(self.dtype, PeriodDtype('D'))
        assert is_dtype_equal(PeriodDtype('D'), PeriodDtype('D'))

        assert not is_dtype_equal(self.dtype, 'D')
        assert not is_dtype_equal(PeriodDtype('D'), PeriodDtype('2D'))

    def test_basic(self):
        assert is_period_dtype(self.dtype)

        pidx = pd.period_range('2013-01-01 09:00', periods=5, freq='H')

        assert is_period_dtype(pidx.dtype)
        assert is_period_dtype(pidx)
        with tm.assert_produces_warning(FutureWarning):
            assert is_period(pidx)

        s = Series(pidx, name='A')

        assert is_period_dtype(s.dtype)
        assert is_period_dtype(s)
        with tm.assert_produces_warning(FutureWarning):
            assert is_period(s)

        assert not is_period_dtype(np.dtype('float64'))
        assert not is_period_dtype(1.0)
        with tm.assert_produces_warning(FutureWarning):
            assert not is_period(np.dtype('float64'))
        with tm.assert_produces_warning(FutureWarning):
            assert not is_period(1.0)

    def test_empty(self):
        dt = PeriodDtype()
        with pytest.raises(AttributeError):
            str(dt)

    def test_not_string(self):
        # though PeriodDtype has object kind, it cannot be string
        assert not is_string_dtype(PeriodDtype('D'))


class TestIntervalDtype(Base):

    def create(self):
        return IntervalDtype('int64')

    def test_hash_vs_equality(self):
        # make sure that we satisfy is semantics
        dtype = self.dtype
        dtype2 = IntervalDtype('int64')
        dtype3 = IntervalDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert dtype is dtype2
        assert dtype2 is dtype3
        assert dtype3 is dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

        dtype1 = IntervalDtype('interval')
        dtype2 = IntervalDtype(dtype1)
        dtype3 = IntervalDtype('interval')
        assert dtype2 == dtype1
        assert dtype2 == dtype2
        assert dtype2 == dtype3
        assert dtype2 is dtype1
        assert dtype2 is dtype2
        assert dtype2 is dtype3
        assert hash(dtype2) == hash(dtype1)
        assert hash(dtype2) == hash(dtype2)
        assert hash(dtype2) == hash(dtype3)

    @pytest.mark.parametrize('subtype', [
        'interval[int64]', 'Interval[int64]', 'int64', np.dtype('int64')])
    def test_construction(self, subtype):
        i = IntervalDtype(subtype)
        assert i.subtype == np.dtype('int64')
        assert is_interval_dtype(i)

    @pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
    def test_construction_generic(self, subtype):
        # generic
        i = IntervalDtype(subtype)
        assert i.subtype is None
        assert is_interval_dtype(i)

    @pytest.mark.parametrize('subtype', [
        CategoricalDtype(list('abc'), False),
        CategoricalDtype(list('wxyz'), True),
        object, str, '<U10', 'interval[category]', 'interval[object]'])
    def test_construction_not_supported(self, subtype):
        # GH 19016
        msg = ('category, object, and string subtypes are not supported '
               'for IntervalDtype')
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    @pytest.mark.parametrize('subtype', ['xx', 'IntervalA', 'Interval[foo]'])
    def test_construction_errors(self, subtype):
        msg = 'could not construct IntervalDtype'
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    def test_construction_from_string(self):
        result = IntervalDtype('interval[int64]')
        assert is_dtype_equal(self.dtype, result)
        result = IntervalDtype.construct_from_string('interval[int64]')
        assert is_dtype_equal(self.dtype, result)

    @pytest.mark.parametrize('string', [
        0, 3.14, ('a', 'b'), None])
    def test_construction_from_string_errors(self, string):
        # these are invalid entirely
        msg = 'a string needs to be passed, got type'

        with pytest.raises(TypeError, match=msg):
            IntervalDtype.construct_from_string(string)

    @pytest.mark.parametrize('string', [
        'foo', 'foo[int64]', 'IntervalA'])
    def test_construction_from_string_error_subtype(self, string):
        # this is an invalid subtype
        msg = ("Incorrectly formatted string passed to constructor. "
               r"Valid formats include Interval or Interval\[dtype\] "
               "where dtype is numeric, datetime, or timedelta")

        with pytest.raises(TypeError, match=msg):
            IntervalDtype.construct_from_string(string)

    def test_subclass(self):
        a = IntervalDtype('interval[int64]')
        b = IntervalDtype('interval[int64]')

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_is_dtype(self):
        assert IntervalDtype.is_dtype(self.dtype)
        assert IntervalDtype.is_dtype('interval')
        assert IntervalDtype.is_dtype(IntervalDtype('float64'))
        assert IntervalDtype.is_dtype(IntervalDtype('int64'))
        assert IntervalDtype.is_dtype(IntervalDtype(np.int64))

        assert not IntervalDtype.is_dtype('D')
        assert not IntervalDtype.is_dtype('3D')
        assert not IntervalDtype.is_dtype('U')
        assert not IntervalDtype.is_dtype('S')
        assert not IntervalDtype.is_dtype('foo')
        assert not IntervalDtype.is_dtype('IntervalA')
        assert not IntervalDtype.is_dtype(np.object_)
        assert not IntervalDtype.is_dtype(np.int64)
        assert not IntervalDtype.is_dtype(np.float64)

    def test_equality(self):
        assert is_dtype_equal(self.dtype, 'interval[int64]')
        assert is_dtype_equal(self.dtype, IntervalDtype('int64'))
        assert is_dtype_equal(IntervalDtype('int64'), IntervalDtype('int64'))

        assert not is_dtype_equal(self.dtype, 'int64')
        assert not is_dtype_equal(IntervalDtype('int64'),
                                  IntervalDtype('float64'))

        # invalid subtype comparisons do not raise when directly compared
        dtype1 = IntervalDtype('float64')
        dtype2 = IntervalDtype('datetime64[ns, US/Eastern]')
        assert dtype1 != dtype2
        assert dtype2 != dtype1

    @pytest.mark.parametrize('subtype', [
        None, 'interval', 'Interval', 'int64', 'uint64', 'float64',
        'complex128', 'datetime64', 'timedelta64', PeriodDtype('Q')])
    def test_equality_generic(self, subtype):
        # GH 18980
        dtype = IntervalDtype(subtype)
        assert is_dtype_equal(dtype, 'interval')
        assert is_dtype_equal(dtype, IntervalDtype())

    @pytest.mark.parametrize('subtype', [
        'int64', 'uint64', 'float64', 'complex128', 'datetime64',
        'timedelta64', PeriodDtype('Q')])
    def test_name_repr(self, subtype):
        # GH 18980
        dtype = IntervalDtype(subtype)
        expected = 'interval[{subtype}]'.format(subtype=subtype)
        assert str(dtype) == expected
        assert dtype.name == 'interval'

    @pytest.mark.parametrize('subtype', [None, 'interval', 'Interval'])
    def test_name_repr_generic(self, subtype):
        # GH 18980
        dtype = IntervalDtype(subtype)
        assert str(dtype) == 'interval'
        assert dtype.name == 'interval'

    def test_basic(self):
        assert is_interval_dtype(self.dtype)

        ii = IntervalIndex.from_breaks(range(3))

        assert is_interval_dtype(ii.dtype)
        assert is_interval_dtype(ii)

        s = Series(ii, name='A')

        assert is_interval_dtype(s.dtype)
        assert is_interval_dtype(s)

    def test_basic_dtype(self):
        assert is_interval_dtype('interval[int64]')
        assert is_interval_dtype(IntervalIndex.from_tuples([(0, 1)]))
        assert is_interval_dtype(IntervalIndex.from_breaks(np.arange(4)))
        assert is_interval_dtype(IntervalIndex.from_breaks(
            date_range('20130101', periods=3)))
        assert not is_interval_dtype('U')
        assert not is_interval_dtype('S')
        assert not is_interval_dtype('foo')
        assert not is_interval_dtype(np.object_)
        assert not is_interval_dtype(np.int64)
        assert not is_interval_dtype(np.float64)

    def test_caching(self):
        IntervalDtype.reset_cache()
        dtype = IntervalDtype("int64")
        assert len(IntervalDtype._cache) == 1

        IntervalDtype("interval")
        assert len(IntervalDtype._cache) == 2

        IntervalDtype.reset_cache()
        tm.round_trip_pickle(dtype)
        assert len(IntervalDtype._cache) == 0


class TestCategoricalDtypeParametrized(object):

    @pytest.mark.parametrize('categories', [
        list('abcd'),
        np.arange(1000),
        ['a', 'b', 10, 2, 1.3, True],
        [True, False],
        pd.date_range('2017', periods=4)])
    def test_basic(self, categories, ordered):
        c1 = CategoricalDtype(categories, ordered=ordered)
        tm.assert_index_equal(c1.categories, pd.Index(categories))
        assert c1.ordered is ordered

    def test_order_matters(self):
        categories = ['a', 'b']
        c1 = CategoricalDtype(categories, ordered=True)
        c2 = CategoricalDtype(categories, ordered=False)
        c3 = CategoricalDtype(categories, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    @pytest.mark.parametrize('ordered', [False, None])
    def test_unordered_same(self, ordered):
        c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
        c2 = CategoricalDtype(['b', 'a'], ordered=ordered)
        assert hash(c1) == hash(c2)

    def test_categories(self):
        result = CategoricalDtype(['a', 'b', 'c'])
        tm.assert_index_equal(result.categories, pd.Index(['a', 'b', 'c']))
        assert result.ordered is None

    def test_equal_but_different(self, ordered):
        c1 = CategoricalDtype([1, 2, 3])
        c2 = CategoricalDtype([1., 2., 3.])
        assert c1 is not c2
        assert c1 != c2

    @pytest.mark.parametrize('v1, v2', [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [3, 2, 1]),
    ])
    def test_order_hashes_different(self, v1, v2):
        c1 = CategoricalDtype(v1, ordered=False)
        c2 = CategoricalDtype(v2, ordered=True)
        c3 = CategoricalDtype(v1, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    def test_nan_invalid(self):
        with pytest.raises(ValueError):
            CategoricalDtype([1, 2, np.nan])

    def test_non_unique_invalid(self):
        with pytest.raises(ValueError):
            CategoricalDtype([1, 2, 1])

    def test_same_categories_different_order(self):
        c1 = CategoricalDtype(['a', 'b'], ordered=True)
        c2 = CategoricalDtype(['b', 'a'], ordered=True)
        assert c1 is not c2

    @pytest.mark.parametrize('ordered1', [True, False, None])
    @pytest.mark.parametrize('ordered2', [True, False, None])
    def test_categorical_equality(self, ordered1, ordered2):
        # same categories, same order
        # any combination of None/False are equal
        # True/True is the only combination with True that are equal
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(list('abc'), ordered2)
        result = c1 == c2
        expected = bool(ordered1) is bool(ordered2)
        assert result is expected

        # same categories, different order
        # any combination of None/False are equal (order doesn't matter)
        # any combination with True are not equal (different order of cats)
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(list('cab'), ordered2)
        result = c1 == c2
        expected = (bool(ordered1) is False) and (bool(ordered2) is False)
        assert result is expected

        # different categories
        c2 = CategoricalDtype([1, 2, 3], ordered2)
        assert c1 != c2

        # none categories
        c1 = CategoricalDtype(list('abc'), ordered1)
        c2 = CategoricalDtype(None, ordered2)
        c3 = CategoricalDtype(None, ordered1)
        assert c1 == c2
        assert c2 == c1
        assert c2 == c3

    @pytest.mark.parametrize('categories', [list('abc'), None])
    @pytest.mark.parametrize('other', ['category', 'not a category'])
    def test_categorical_equality_strings(self, categories, ordered, other):
        c1 = CategoricalDtype(categories, ordered)
        result = c1 == other
        expected = other == 'category'
        assert result is expected

    def test_invalid_raises(self):
        with pytest.raises(TypeError, match='ordered'):
            CategoricalDtype(['a', 'b'], ordered='foo')

        with pytest.raises(TypeError, match="'categories' must be list-like"):
            CategoricalDtype('category')

    def test_mixed(self):
        a = CategoricalDtype(['a', 'b', 1, 2])
        b = CategoricalDtype(['a', 'b', '1', '2'])
        assert hash(a) != hash(b)

    def test_from_categorical_dtype_identity(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # Identity test for no changes
        c2 = CategoricalDtype._from_categorical_dtype(c1)
        assert c2 is c1

    def test_from_categorical_dtype_categories(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override categories
        result = CategoricalDtype._from_categorical_dtype(
            c1, categories=[2, 3])
        assert result == CategoricalDtype([2, 3], ordered=True)

    def test_from_categorical_dtype_ordered(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override ordered
        result = CategoricalDtype._from_categorical_dtype(
            c1, ordered=False)
        assert result == CategoricalDtype([1, 2, 3], ordered=False)

    def test_from_categorical_dtype_both(self):
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # override ordered
        result = CategoricalDtype._from_categorical_dtype(
            c1, categories=[1, 2], ordered=False)
        assert result == CategoricalDtype([1, 2], ordered=False)

    def test_str_vs_repr(self, ordered):
        c1 = CategoricalDtype(['a', 'b'], ordered=ordered)
        assert str(c1) == 'category'
        # Py2 will have unicode prefixes
        pat = r"CategoricalDtype\(categories=\[.*\], ordered={ordered}\)"
        assert re.match(pat.format(ordered=ordered), repr(c1))

    def test_categorical_categories(self):
        # GH17884
        c1 = CategoricalDtype(Categorical(['a', 'b']))
        tm.assert_index_equal(c1.categories, pd.Index(['a', 'b']))
        c1 = CategoricalDtype(CategoricalIndex(['a', 'b']))
        tm.assert_index_equal(c1.categories, pd.Index(['a', 'b']))

    @pytest.mark.parametrize('new_categories', [
        list('abc'), list('cba'), list('wxyz'), None])
    @pytest.mark.parametrize('new_ordered', [True, False, None])
    def test_update_dtype(self, ordered, new_categories, new_ordered):
        dtype = CategoricalDtype(list('abc'), ordered)
        new_dtype = CategoricalDtype(new_categories, new_ordered)

        expected_categories = new_dtype.categories
        if expected_categories is None:
            expected_categories = dtype.categories

        expected_ordered = new_dtype.ordered
        if expected_ordered is None:
            expected_ordered = dtype.ordered

        result = dtype.update_dtype(new_dtype)
        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    def test_update_dtype_string(self, ordered):
        dtype = CategoricalDtype(list('abc'), ordered)
        expected_categories = dtype.categories
        expected_ordered = dtype.ordered
        result = dtype.update_dtype('category')
        tm.assert_index_equal(result.categories, expected_categories)
        assert result.ordered is expected_ordered

    @pytest.mark.parametrize('bad_dtype', [
        'foo', object, np.int64, PeriodDtype('Q')])
    def test_update_dtype_errors(self, bad_dtype):
        dtype = CategoricalDtype(list('abc'), False)
        msg = 'a CategoricalDtype must be passed to perform an update, '
        with pytest.raises(ValueError, match=msg):
            dtype.update_dtype(bad_dtype)


@pytest.mark.parametrize('dtype', [
    CategoricalDtype,
    IntervalDtype,
    DatetimeTZDtype,
    PeriodDtype,
])
def test_registry(dtype):
    assert dtype in registry.dtypes


@pytest.mark.parametrize('dtype, expected', [
    ('int64', None),
    ('interval', IntervalDtype()),
    ('interval[int64]', IntervalDtype()),
    ('interval[datetime64[ns]]', IntervalDtype('datetime64[ns]')),
    ('period[D]', PeriodDtype('D')),
    ('category', CategoricalDtype()),
    ('datetime64[ns, US/Eastern]', DatetimeTZDtype('ns', 'US/Eastern')),
])
def test_registry_find(dtype, expected):
    assert registry.find(dtype) == expected


@pytest.mark.parametrize('dtype, expected', [
    (str, False),
    (int, False),
    (bool, True),
    (np.bool, True),
    (np.array(['a', 'b']), False),
    (pd.Series([1, 2]), False),
    (np.array([True, False]), True),
    (pd.Series([True, False]), True),
    (pd.SparseSeries([True, False]), True),
    (pd.SparseArray([True, False]), True),
    (SparseDtype(bool), True)
])
def test_is_bool_dtype(dtype, expected):
    result = is_bool_dtype(dtype)
    assert result is expected


@pytest.mark.parametrize("check", [
    is_categorical_dtype,
    is_datetime64tz_dtype,
    is_period_dtype,
    is_datetime64_ns_dtype,
    is_datetime64_dtype,
    is_interval_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
    is_bool_dtype,
])
def test_is_dtype_no_warning(check):
    data = pd.DataFrame({"A": [1, 2]})
    with tm.assert_produces_warning(None):
        check(data)

    with tm.assert_produces_warning(None):
        check(data["A"])
