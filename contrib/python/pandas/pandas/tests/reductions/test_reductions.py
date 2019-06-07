# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical, DataFrame, DatetimeIndex, Index, NaT, Period, PeriodIndex,
    RangeIndex, Series, Timedelta, TimedeltaIndex, Timestamp, compat, isna,
    timedelta_range, to_timedelta)
from pandas.core import nanops
import pandas.util.testing as tm


def get_objs():
    indexes = [
        tm.makeBoolIndex(10, name='a'),
        tm.makeIntIndex(10, name='a'),
        tm.makeFloatIndex(10, name='a'),
        tm.makeDateIndex(10, name='a'),
        tm.makeDateIndex(10, name='a').tz_localize(tz='US/Eastern'),
        tm.makePeriodIndex(10, name='a'),
        tm.makeStringIndex(10, name='a'),
        tm.makeUnicodeIndex(10, name='a')
    ]

    arr = np.random.randn(10)
    series = [Series(arr, index=idx, name='a') for idx in indexes]

    objs = indexes + series
    return objs


objs = get_objs()


class TestReductions(object):

    @pytest.mark.parametrize('opname', ['max', 'min'])
    @pytest.mark.parametrize('obj', objs)
    def test_ops(self, opname, obj):
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            expected = getattr(obj.values, opname)()
        else:
            expected = pd.Period(
                ordinal=getattr(obj._ndarray_values, opname)(),
                freq=obj.freq)
        try:
            assert result == expected
        except TypeError:
            # comparing tz-aware series with np.array results in
            # TypeError
            expected = expected.astype('M8[ns]').astype('int64')
            assert result.value == expected

    def test_nanops(self):
        # GH#7261
        for opname in ['max', 'min']:
            for klass in [Index, Series]:
                arg_op = 'arg' + opname if klass is Index else 'idx' + opname

                obj = klass([np.nan, 2.0])
                assert getattr(obj, opname)() == 2.0

                obj = klass([np.nan])
                assert pd.isna(getattr(obj, opname)())
                assert pd.isna(getattr(obj, opname)(skipna=False))

                obj = klass([])
                assert pd.isna(getattr(obj, opname)())
                assert pd.isna(getattr(obj, opname)(skipna=False))

                obj = klass([pd.NaT, datetime(2011, 11, 1)])
                # check DatetimeIndex monotonic path
                assert getattr(obj, opname)() == datetime(2011, 11, 1)
                assert getattr(obj, opname)(skipna=False) is pd.NaT

                assert getattr(obj, arg_op)() == 1
                result = getattr(obj, arg_op)(skipna=False)
                if klass is Series:
                    assert np.isnan(result)
                else:
                    assert result == -1

                obj = klass([pd.NaT, datetime(2011, 11, 1), pd.NaT])
                # check DatetimeIndex non-monotonic path
                assert getattr(obj, opname)(), datetime(2011, 11, 1)
                assert getattr(obj, opname)(skipna=False) is pd.NaT

                assert getattr(obj, arg_op)() == 1
                result = getattr(obj, arg_op)(skipna=False)
                if klass is Series:
                    assert np.isnan(result)
                else:
                    assert result == -1

                for dtype in ["M8[ns]", "datetime64[ns, UTC]"]:
                    # cases with empty Series/DatetimeIndex
                    obj = klass([], dtype=dtype)

                    assert getattr(obj, opname)() is pd.NaT
                    assert getattr(obj, opname)(skipna=False) is pd.NaT

                    with pytest.raises(ValueError, match="empty sequence"):
                        getattr(obj, arg_op)()
                    with pytest.raises(ValueError, match="empty sequence"):
                        getattr(obj, arg_op)(skipna=False)

        # argmin/max
        obj = Index(np.arange(5, dtype='int64'))
        assert obj.argmin() == 0
        assert obj.argmax() == 4

        obj = Index([np.nan, 1, np.nan, 2])
        assert obj.argmin() == 1
        assert obj.argmax() == 3
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([np.nan])
        assert obj.argmin() == -1
        assert obj.argmax() == -1
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([pd.NaT, datetime(2011, 11, 1), datetime(2011, 11, 2),
                     pd.NaT])
        assert obj.argmin() == 1
        assert obj.argmax() == 2
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

        obj = Index([pd.NaT])
        assert obj.argmin() == -1
        assert obj.argmax() == -1
        assert obj.argmin(skipna=False) == -1
        assert obj.argmax(skipna=False) == -1

    @pytest.mark.parametrize('op, expected_col', [
        ['max', 'a'], ['min', 'b']
    ])
    def test_same_tz_min_max_axis_1(self, op, expected_col):
        # GH 10390
        df = DataFrame(pd.date_range('2016-01-01 00:00:00', periods=3,
                                     tz='UTC'),
                       columns=['a'])
        df['b'] = df.a.subtract(pd.Timedelta(seconds=3600))
        result = getattr(df, op)(axis=1)
        expected = df[expected_col]
        tm.assert_series_equal(result, expected)


class TestIndexReductions(object):
    # Note: the name TestIndexReductions indicates these tests
    #  were moved from a Index-specific test file, _not_ that these tests are
    #  intended long-term to be Index-specific

    @pytest.mark.parametrize('start,stop,step',
                             [(0, 400, 3), (500, 0, -6), (-10**6, 10**6, 4),
                              (10**6, -10**6, -4), (0, 10, 20)])
    def test_max_min_range(self, start, stop, step):
        # GH#17607
        idx = RangeIndex(start, stop, step)
        expected = idx._int64index.max()
        result = idx.max()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.max(skipna=False)
        assert result2 == expected

        expected = idx._int64index.min()
        result = idx.min()
        assert result == expected

        # skipna should be irrelevant since RangeIndex should never have NAs
        result2 = idx.min(skipna=False)
        assert result2 == expected

        # empty
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self):

        # monotonic
        idx1 = TimedeltaIndex(['1 days', '2 days', '3 days'])
        assert idx1.is_monotonic

        # non-monotonic
        idx2 = TimedeltaIndex(['1 days', np.nan, '3 days', 'NaT'])
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta('1 days')
            assert idx.max() == Timedelta('3 days')
            assert idx.argmin() == 0
            assert idx.argmax() == 2

        for op in ['min', 'max']:
            # Return NaT
            obj = TimedeltaIndex([])
            assert pd.isna(getattr(obj, op)())

            obj = TimedeltaIndex([pd.NaT])
            assert pd.isna(getattr(obj, op)())

            obj = TimedeltaIndex([pd.NaT, pd.NaT, pd.NaT])
            assert pd.isna(getattr(obj, op)())

    def test_numpy_minmax_timedelta64(self):
        td = timedelta_range('16815 days', '16820 days', freq='D')

        assert np.min(td) == Timedelta('16815 days')
        assert np.max(td) == Timedelta('16820 days')

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)

        assert np.argmin(td) == 0
        assert np.argmax(td) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self):
        # GH#4984
        # make sure ops return Timedelta
        s = Series([Timestamp('20130101') + timedelta(seconds=i * i)
                    for i in range(10)])
        td = s.diff()

        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected

        result = td.to_frame().mean()
        assert result[0] == expected

        result = td.quantile(.1)
        expected = Timedelta(np.timedelta64(2600, 'ms'))
        assert result == expected

        result = td.median()
        expected = to_timedelta('00:00:09')
        assert result == expected

        result = td.to_frame().median()
        assert result[0] == expected

        # GH#6462
        # consistency in returned values for sum
        result = td.sum()
        expected = to_timedelta('00:01:21')
        assert result == expected

        result = td.to_frame().sum()
        assert result[0] == expected

        # std
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected

        result = td.to_frame().std()
        assert result[0] == expected

        # invalid ops
        for op in ['skew', 'kurt', 'sem', 'prod']:
            pytest.raises(TypeError, getattr(td, op))

        # GH#10040
        # make sure NaT is properly handled by median()
        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07')])
        assert s.diff().median() == timedelta(days=4)

        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07'),
                    Timestamp('2015-02-15')])
        assert s.diff().median() == timedelta(days=6)

    def test_minmax_tz(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # monotonic
        idx1 = pd.DatetimeIndex(['2011-01-01', '2011-01-02',
                                 '2011-01-03'], tz=tz)
        assert idx1.is_monotonic

        # non-monotonic
        idx2 = pd.DatetimeIndex(['2011-01-01', pd.NaT, '2011-01-03',
                                 '2011-01-02', pd.NaT], tz=tz)
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp('2011-01-01', tz=tz)
            assert idx.max() == Timestamp('2011-01-03', tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize('op', ['min', 'max'])
    def test_minmax_nat_datetime64(self, op):
        # Return NaT
        obj = DatetimeIndex([])
        assert pd.isna(getattr(obj, op)())

        obj = DatetimeIndex([pd.NaT])
        assert pd.isna(getattr(obj, op)())

        obj = DatetimeIndex([pd.NaT, pd.NaT, pd.NaT])
        assert pd.isna(getattr(obj, op)())

    def test_numpy_minmax_datetime64(self):
        dr = pd.date_range(start='2016-01-15', end='2016-01-20')

        assert np.min(dr) == Timestamp('2016-01-15 00:00:00', freq='D')
        assert np.max(dr) == Timestamp('2016-01-20 00:00:00', freq='D')

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)

        assert np.argmin(dr) == 0
        assert np.argmax(dr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)

        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)

    def test_minmax_period(self):

        # monotonic
        idx1 = pd.PeriodIndex([NaT, '2011-01-01', '2011-01-02',
                               '2011-01-03'], freq='D')
        assert idx1.is_monotonic

        # non-monotonic
        idx2 = pd.PeriodIndex(['2011-01-01', NaT, '2011-01-03',
                               '2011-01-02', NaT], freq='D')
        assert not idx2.is_monotonic

        for idx in [idx1, idx2]:
            assert idx.min() == pd.Period('2011-01-01', freq='D')
            assert idx.max() == pd.Period('2011-01-03', freq='D')
        assert idx1.argmin() == 1
        assert idx2.argmin() == 0
        assert idx1.argmax() == 3
        assert idx2.argmax() == 2

        for op in ['min', 'max']:
            # Return NaT
            obj = PeriodIndex([], freq='M')
            result = getattr(obj, op)()
            assert result is NaT

            obj = PeriodIndex([NaT], freq='M')
            result = getattr(obj, op)()
            assert result is NaT

            obj = PeriodIndex([NaT, NaT, NaT], freq='M')
            result = getattr(obj, op)()
            assert result is NaT

    def test_numpy_minmax_period(self):
        pr = pd.period_range(start='2016-01-15', end='2016-01-20')

        assert np.min(pr) == Period('2016-01-15', freq='D')
        assert np.max(pr) == Period('2016-01-20', freq='D')

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)

        assert np.argmin(pr) == 0
        assert np.argmax(pr) == 5

        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self):

        ci = pd.CategoricalIndex(list('aabbca'),
                                 categories=list('cab'),
                                 ordered=False)
        with pytest.raises(TypeError):
            ci.min()
        with pytest.raises(TypeError):
            ci.max()

        ci = pd.CategoricalIndex(list('aabbca'),
                                 categories=list('cab'),
                                 ordered=True)
        assert ci.min() == 'c'
        assert ci.max() == 'b'


class TestSeriesReductions(object):
    # Note: the name TestSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def test_sum_inf(self):
        s = Series(np.random.randn(10))
        s2 = s.copy()

        s[5:8] = np.inf
        s2[5:8] = np.nan

        assert np.isinf(s.sum())

        arr = np.random.randn(100, 100).astype('f4')
        arr[:, 2] = np.inf

        with pd.option_context("mode.use_inf_as_na", True):
            tm.assert_almost_equal(s.sum(), s2.sum())

        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    @pytest.mark.parametrize("method, unit", [
        ("sum", 0.0),
        ("prod", 1.0)
    ])
    def test_empty(self, method, unit, use_bottleneck):
        with pd.option_context("use_bottleneck", use_bottleneck):
            # GH#9422 / GH#18921
            # Entirely empty
            s = Series([])
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert pd.isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # Skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert pd.isna(result)

            # All-NA
            s = Series([np.nan])
            # NA by default
            result = getattr(s, method)()
            assert result == unit

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == unit

            result = getattr(s, method)(min_count=1)
            assert pd.isna(result)

            # Skipna, default
            result = getattr(s, method)(skipna=True)
            result == unit

            # skipna, explicit
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == unit

            result = getattr(s, method)(skipna=True, min_count=1)
            assert pd.isna(result)

            # Mix of valid, empty
            s = Series([np.nan, 1])
            # Default
            result = getattr(s, method)()
            assert result == 1.0

            # Explicit
            result = getattr(s, method)(min_count=0)
            assert result == 1.0

            result = getattr(s, method)(min_count=1)
            assert result == 1.0

            # Skipna
            result = getattr(s, method)(skipna=True)
            assert result == 1.0

            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 1.0

            result = getattr(s, method)(skipna=True, min_count=1)
            assert result == 1.0

            # GH#844 (changed in GH#9422)
            df = DataFrame(np.empty((10, 0)))
            assert (getattr(df, method)(1) == unit).all()

            s = pd.Series([1])
            result = getattr(s, method)(min_count=2)
            assert pd.isna(result)

            s = pd.Series([np.nan])
            result = getattr(s, method)(min_count=2)
            assert pd.isna(result)

            s = pd.Series([np.nan, 1])
            result = getattr(s, method)(min_count=2)
            assert pd.isna(result)

    @pytest.mark.parametrize('method, unit', [
        ('sum', 0.0),
        ('prod', 1.0),
    ])
    def test_empty_multi(self, method, unit):
        s = pd.Series([1, np.nan, np.nan, np.nan],
                      index=pd.MultiIndex.from_product([('a', 'b'), (0, 1)]))
        # 1 / 0 by default
        result = getattr(s, method)(level=0)
        expected = pd.Series([1, unit], index=['a', 'b'])
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = getattr(s, method)(level=0, min_count=0)
        expected = pd.Series([1, unit], index=['a', 'b'])
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = getattr(s, method)(level=0, min_count=1)
        expected = pd.Series([1, np.nan], index=['a', 'b'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method", ['mean', 'median', 'std', 'var'])
    def test_ops_consistency_on_empty(self, method):

        # GH#7869
        # consistency on empty

        # float
        result = getattr(Series(dtype=float), method)()
        assert pd.isna(result)

        # timedelta64[ns]
        result = getattr(Series(dtype='m8[ns]'), method)()
        assert result is pd.NaT

    def test_nansum_buglet(self):
        ser = Series([1.0, np.nan], index=[0, 1])
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    @pytest.mark.parametrize("use_bottleneck", [True, False])
    def test_sum_overflow(self, use_bottleneck):

        with pd.option_context('use_bottleneck', use_bottleneck):
            # GH#6915
            # overflowing on the smaller int dtypes
            for dtype in ['int32', 'int64']:
                v = np.arange(5000000, dtype=dtype)
                s = Series(v)

                result = s.sum(skipna=False)
                assert int(result) == v.sum(dtype='int64')
                result = s.min(skipna=False)
                assert int(result) == 0
                result = s.max(skipna=False)
                assert int(result) == v[-1]

            for dtype in ['float32', 'float64']:
                v = np.arange(5000000, dtype=dtype)
                s = Series(v)

                result = s.sum(skipna=False)
                assert result == v.sum(dtype=dtype)
                result = s.min(skipna=False)
                assert np.allclose(float(result), 0.0)
                result = s.max(skipna=False)
                assert np.allclose(float(result), v[-1])

    def test_empty_timeseries_reductions_return_nat(self):
        # covers GH#11245
        for dtype in ('m8[ns]', 'm8[ns]', 'M8[ns]', 'M8[ns, UTC]'):
            assert Series([], dtype=dtype).min() is pd.NaT
            assert Series([], dtype=dtype).max() is pd.NaT
            assert Series([], dtype=dtype).min(skipna=False) is pd.NaT
            assert Series([], dtype=dtype).max(skipna=False) is pd.NaT

    def test_numpy_argmin_deprecated(self):
        # See GH#16830
        data = np.arange(1, 11)

        s = Series(data, index=data)
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            # The deprecation of Series.argmin also causes a deprecation
            # warning when calling np.argmin. This behavior is temporary
            # until the implementation of Series.argmin is corrected.
            result = np.argmin(s)

        assert result == 1

        with tm.assert_produces_warning(FutureWarning):
            # argmin is aliased to idxmin
            result = s.argmin()

        assert result == 1

        with tm.assert_produces_warning(FutureWarning,
                                        check_stacklevel=False):
            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argmin(s, out=data)

    def test_numpy_argmax_deprecated(self):
        # See GH#16830
        data = np.arange(1, 11)

        s = Series(data, index=data)
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            # The deprecation of Series.argmax also causes a deprecation
            # warning when calling np.argmax. This behavior is temporary
            # until the implementation of Series.argmax is corrected.
            result = np.argmax(s)
        assert result == 10

        with tm.assert_produces_warning(FutureWarning):
            # argmax is aliased to idxmax
            result = s.argmax()

        assert result == 10

        with tm.assert_produces_warning(FutureWarning,
                                        check_stacklevel=False):
            msg = "the 'out' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argmax(s, out=data)

    def test_idxmin(self):
        # test idxmin
        # _check_stat_op approach can not be used here because of isna check.
        string_series = tm.makeStringSeries().rename('series')

        # add some NaNs
        string_series[5:15] = np.NaN

        # skipna or no
        assert string_series[string_series.idxmin()] == string_series.min()
        assert pd.isna(string_series.idxmin(skipna=False))

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmin()] == nona.min()
        assert (nona.index.values.tolist().index(nona.idxmin()) ==
                nona.values.argmin())

        # all NaNs
        allna = string_series * np.nan
        assert pd.isna(allna.idxmin())

        # datetime64[ns]
        s = Series(pd.date_range('20130102', periods=6))
        result = s.idxmin()
        assert result == 0

        s[0] = np.nan
        result = s.idxmin()
        assert result == 1

    def test_idxmax(self):
        # test idxmax
        # _check_stat_op approach can not be used here because of isna check.
        string_series = tm.makeStringSeries().rename('series')

        # add some NaNs
        string_series[5:15] = np.NaN

        # skipna or no
        assert string_series[string_series.idxmax()] == string_series.max()
        assert pd.isna(string_series.idxmax(skipna=False))

        # no NaNs
        nona = string_series.dropna()
        assert nona[nona.idxmax()] == nona.max()
        assert (nona.index.values.tolist().index(nona.idxmax()) ==
                nona.values.argmax())

        # all NaNs
        allna = string_series * np.nan
        assert pd.isna(allna.idxmax())

        from pandas import date_range
        s = Series(date_range('20130102', periods=6))
        result = s.idxmax()
        assert result == 5

        s[5] = np.nan
        result = s.idxmax()
        assert result == 4

        # Float64Index
        # GH#5914
        s = pd.Series([1, 2, 3], [1.1, 2.1, 3.1])
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

        s = pd.Series(s.index, s.index)
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self):
        ts = tm.makeTimeSeries()
        bool_series = ts > 0
        assert not bool_series.all()
        assert bool_series.any()

        # Alternative types, with implicit 'object' dtype.
        s = Series(['abc', True])
        assert 'abc' == s.any()  # 'abc' || True => 'abc'

    def test_all_any_params(self):
        # Check skipna, with implicit 'object' dtype.
        s1 = Series([np.nan, True])
        s2 = Series([np.nan, False])
        assert s1.all(skipna=False)  # nan && True => True
        assert s1.all(skipna=True)
        assert np.isnan(s2.any(skipna=False))  # nan || False => nan
        assert not s2.any(skipna=True)

        # Check level.
        s = pd.Series([False, False, True, True, False, True],
                      index=[0, 0, 1, 1, 2, 2])
        tm.assert_series_equal(s.all(level=0), Series([False, True, False]))
        tm.assert_series_equal(s.any(level=0), Series([False, True, True]))

        # bool_only is not implemented with level option.
        with pytest.raises(NotImplementedError):
            s.any(bool_only=True, level=0)
        with pytest.raises(NotImplementedError):
            s.all(bool_only=True, level=0)

        # bool_only is not implemented alone.
        with pytest.raises(NotImplementedError):
            s.any(bool_only=True,)
        with pytest.raises(NotImplementedError):
            s.all(bool_only=True)

    def test_timedelta64_analytics(self):

        # index min/max
        dti = pd.date_range('2012-1-1', periods=3, freq='D')
        td = Series(dti) - pd.Timestamp('20120101')

        result = td.idxmin()
        assert result == 0

        result = td.idxmax()
        assert result == 2

        # GH#2982
        # with NaT
        td[0] = np.nan

        result = td.idxmin()
        assert result == 1

        result = td.idxmax()
        assert result == 2

        # abs
        s1 = Series(pd.date_range('20120101', periods=3))
        s2 = Series(pd.date_range('20120102', periods=3))
        expected = Series(s2 - s1)

        # FIXME: don't leave commented-out code
        # this fails as numpy returns timedelta64[us]
        # result = np.abs(s1-s2)
        # assert_frame_equal(result,expected)

        result = (s1 - s2).abs()
        tm.assert_series_equal(result, expected)

        # max/min
        result = td.max()
        expected = pd.Timedelta('2 days')
        assert result == expected

        result = td.min()
        expected = pd.Timedelta('1 days')
        assert result == expected

    @pytest.mark.parametrize(
        "test_input,error_type",
        [
            (pd.Series([]), ValueError),

            # For strings, or any Series with dtype 'O'
            (pd.Series(['foo', 'bar', 'baz']), TypeError),
            (pd.Series([(1,), (2,)]), TypeError),

            # For mixed data types
            (
                pd.Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz']),
                TypeError
            ),
        ]
    )
    def test_assert_idxminmax_raises(self, test_input, error_type):
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        with pytest.raises(error_type):
            test_input.idxmin()
        with pytest.raises(error_type):
            test_input.idxmin(skipna=False)
        with pytest.raises(error_type):
            test_input.idxmax()
        with pytest.raises(error_type):
            test_input.idxmax(skipna=False)

    def test_idxminmax_with_inf(self):
        # For numeric data with NA and Inf (GH #13595)
        s = pd.Series([0, -np.inf, np.inf, np.nan])

        assert s.idxmin() == 1
        assert np.isnan(s.idxmin(skipna=False))

        assert s.idxmax() == 2
        assert np.isnan(s.idxmax(skipna=False))

        # Using old-style behavior that treats floating point nan, -inf, and
        # +inf as missing
        with pd.option_context('mode.use_inf_as_na', True):
            assert s.idxmin() == 0
            assert np.isnan(s.idxmin(skipna=False))
            assert s.idxmax() == 0
            np.isnan(s.idxmax(skipna=False))


class TestDatetime64SeriesReductions(object):
    # Note: the name TestDatetime64SeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize('nat_ser', [
        Series([pd.NaT, pd.NaT]),
        Series([pd.NaT, pd.Timedelta('nat')]),
        Series([pd.Timedelta('nat'), pd.Timedelta('nat')])])
    def test_minmax_nat_series(self, nat_ser):
        # GH#23282
        assert nat_ser.min() is pd.NaT
        assert nat_ser.max() is pd.NaT
        assert nat_ser.min(skipna=False) is pd.NaT
        assert nat_ser.max(skipna=False) is pd.NaT

    @pytest.mark.parametrize('nat_df', [
        pd.DataFrame([pd.NaT, pd.NaT]),
        pd.DataFrame([pd.NaT, pd.Timedelta('nat')]),
        pd.DataFrame([pd.Timedelta('nat'), pd.Timedelta('nat')])])
    def test_minmax_nat_dataframe(self, nat_df):
        # GH#23282
        assert nat_df.min()[0] is pd.NaT
        assert nat_df.max()[0] is pd.NaT
        assert nat_df.min(skipna=False)[0] is pd.NaT
        assert nat_df.max(skipna=False)[0] is pd.NaT

    def test_min_max(self):
        rng = pd.date_range('1/1/2000', '12/31/2000')
        rng2 = rng.take(np.random.permutation(len(rng)))

        the_min = rng2.min()
        the_max = rng2.max()
        assert isinstance(the_min, pd.Timestamp)
        assert isinstance(the_max, pd.Timestamp)
        assert the_min == rng[0]
        assert the_max == rng[-1]

        assert rng.min() == rng[0]
        assert rng.max() == rng[-1]

    def test_min_max_series(self):
        rng = pd.date_range('1/1/2000', periods=10, freq='4h')
        lvls = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
        df = DataFrame({'TS': rng, 'V': np.random.randn(len(rng)), 'L': lvls})

        result = df.TS.max()
        exp = pd.Timestamp(df.TS.iat[-1])
        assert isinstance(result, pd.Timestamp)
        assert result == exp

        result = df.TS.min()
        exp = pd.Timestamp(df.TS.iat[0])
        assert isinstance(result, pd.Timestamp)
        assert result == exp


class TestCategoricalSeriesReductions(object):
    # Note: the name TestCategoricalSeriesReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def test_min_max(self):
        # unordered cats have no min/max
        cat = Series(Categorical(["a", "b", "c", "d"], ordered=False))
        with pytest.raises(TypeError):
            cat.min()
        with pytest.raises(TypeError):
            cat.max()

        cat = Series(Categorical(["a", "b", "c", "d"], ordered=True))
        _min = cat.min()
        _max = cat.max()
        assert _min == "a"
        assert _max == "d"

        cat = Series(Categorical(["a", "b", "c", "d"], categories=[
                     'd', 'c', 'b', 'a'], ordered=True))
        _min = cat.min()
        _max = cat.max()
        assert _min == "d"
        assert _max == "a"

        cat = Series(Categorical(
            [np.nan, "b", "c", np.nan], categories=['d', 'c', 'b', 'a'
                                                    ], ordered=True))
        _min = cat.min()
        _max = cat.max()
        assert np.isnan(_min)
        assert _max == "b"

        cat = Series(Categorical(
            [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True))
        _min = cat.min()
        _max = cat.max()
        assert np.isnan(_min)
        assert _max == 1

    def test_min_max_numeric_only(self):
        # TODO deprecate numeric_only argument for Categorical and use
        # skipna as well, see GH25303
        cat = Series(Categorical(
            ["a", "b", np.nan, "a"], categories=['b', 'a'], ordered=True))

        _min = cat.min()
        _max = cat.max()
        assert np.isnan(_min)
        assert _max == "a"

        _min = cat.min(numeric_only=True)
        _max = cat.max(numeric_only=True)
        assert _min == "b"
        assert _max == "a"

        _min = cat.min(numeric_only=False)
        _max = cat.max(numeric_only=False)
        assert np.isnan(_min)
        assert _max == "a"


class TestSeriesMode(object):
    # Note: the name TestSeriesMode indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    @pytest.mark.parametrize('dropna, expected', [
        (True, Series([], dtype=np.float64)),
        (False, Series([], dtype=np.float64))
    ])
    def test_mode_empty(self, dropna, expected):
        s = Series([], dtype=np.float64)
        result = s.mode(dropna)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, data, expected', [
        (True, [1, 1, 1, 2], [1]),
        (True, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
        (False, [1, 1, 1, 2], [1]),
        (False, [1, 1, 1, 2, 3, 3, 3], [1, 3]),
    ])
    @pytest.mark.parametrize(
        'dt',
        list(np.typecodes['AllInteger'] + np.typecodes['Float'])
    )
    def test_mode_numerical(self, dropna, data, expected, dt):
        s = Series(data, dtype=dt)
        result = s.mode(dropna)
        expected = Series(expected, dtype=dt)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected', [
        (True, [1.0]),
        (False, [1, np.nan]),
    ])
    def test_mode_numerical_nan(self, dropna, expected):
        s = Series([1, 1, 2, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected1, expected2, expected3', [
        (True, ['b'], ['bar'], ['nan']),
        (False, ['b'], [np.nan], ['nan'])
    ])
    def test_mode_str_obj(self, dropna, expected1, expected2, expected3):
        # Test string and object types.
        data = ['a'] * 2 + ['b'] * 3

        s = Series(data, dtype='c')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='c')
        tm.assert_series_equal(result, expected1)

        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]

        s = Series(data, dtype=object)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected2)

        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]

        s = Series(data, dtype=object).astype(str)
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype=str)
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize('dropna, expected1, expected2', [
        (True, ['foo'], ['foo']),
        (False, ['foo'], [np.nan])
    ])
    def test_mode_mixeddtype(self, dropna, expected1, expected2):
        s = Series([1, 'foo', 'foo'])
        result = s.mode(dropna)
        expected = Series(expected1)
        tm.assert_series_equal(result, expected)

        s = Series([1, 'foo', 'foo', np.nan, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(expected2, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna, expected1, expected2', [
        (True, ['1900-05-03', '2011-01-03', '2013-01-02'],
               ['2011-01-03', '2013-01-02']),
        (False, [np.nan], [np.nan, '2011-01-03', '2013-01-02']),
    ])
    def test_mode_datetime(self, dropna, expected1, expected2):
        s = Series(['2011-01-03', '2013-01-02',
                    '1900-05-03', 'nan', 'nan'], dtype='M8[ns]')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='M8[ns]')
        tm.assert_series_equal(result, expected1)

        s = Series(['2011-01-03', '2013-01-02', '1900-05-03',
                    '2011-01-03', '2013-01-02', 'nan', 'nan'],
                   dtype='M8[ns]')
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='M8[ns]')
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('dropna, expected1, expected2', [
        (True, ['-1 days', '0 days', '1 days'], ['2 min', '1 day']),
        (False, [np.nan], [np.nan, '2 min', '1 day']),
    ])
    def test_mode_timedelta(self, dropna, expected1, expected2):
        # gh-5986: Test timedelta types.

        s = Series(['1 days', '-1 days', '0 days', 'nan', 'nan'],
                   dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected1)

        s = Series(['1 day', '1 day', '-1 day', '-1 day 2 min',
                    '2 min', '2 min', 'nan', 'nan'],
                   dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('dropna, expected1, expected2, expected3', [
        (True, Categorical([1, 2], categories=[1, 2]),
         Categorical(['a'], categories=[1, 'a']),
         Categorical([3, 1], categories=[3, 2, 1], ordered=True)),
        (False, Categorical([np.nan], categories=[1, 2]),
         Categorical([np.nan, 'a'], categories=[1, 'a']),
         Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True)),
    ])
    def test_mode_category(self, dropna, expected1, expected2, expected3):
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype='category')
        tm.assert_series_equal(result, expected1)

        s = Series(Categorical([1, 'a', 'a', np.nan, np.nan]))
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype='category')
        tm.assert_series_equal(result, expected2)

        s = Series(Categorical([1, 1, 2, 3, 3, np.nan, np.nan],
                               categories=[3, 2, 1], ordered=True))
        result = s.mode(dropna)
        expected3 = Series(expected3, dtype='category')
        tm.assert_series_equal(result, expected3)

    @pytest.mark.parametrize('dropna, expected1, expected2', [
        (True, [2**63], [1, 2**63]),
        (False, [2**63], [1, 2**63])
    ])
    def test_mode_intoverflow(self, dropna, expected1, expected2):
        # Test for uint64 overflow.
        s = Series([1, 2**63, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected1 = Series(expected1, dtype=np.uint64)
        tm.assert_series_equal(result, expected1)

        s = Series([1, 2**63], dtype=np.uint64)
        result = s.mode(dropna)
        expected2 = Series(expected2, dtype=np.uint64)
        tm.assert_series_equal(result, expected2)

    @pytest.mark.skipif(not compat.PY3, reason="only PY3")
    def test_mode_sortwarning(self):
        # Check for the warning that is raised when the mode
        # results cannot be sorted

        expected = Series(['foo', np.nan])
        s = Series([1, 'foo', 'foo', np.nan, np.nan])

        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            result = s.mode(dropna=False)
            result = result.sort_values().reset_index(drop=True)

        tm.assert_series_equal(result, expected)
