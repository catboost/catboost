# coding=utf-8
# pylint: disable-msg=E1101,W0612

from datetime import datetime, timedelta
from distutils.version import LooseVersion

import numpy as np
from numpy import nan
import pytest
import pytz

from pandas._libs.tslib import iNaT
from pandas.compat import range
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical, DataFrame, Index, IntervalIndex, MultiIndex, NaT, Series,
    Timestamp, date_range, isna)
from pandas.core.series import remove_na
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal, assert_series_equal

try:
    import scipy
    _is_scipy_ge_0190 = (LooseVersion(scipy.__version__) >=
                         LooseVersion('0.19.0'))
except ImportError:
    _is_scipy_ge_0190 = False


def _skip_if_no_pchip():
    try:
        from scipy.interpolate import pchip_interpolate  # noqa
    except ImportError:
        import pytest
        pytest.skip('scipy.interpolate.pchip missing')


def _skip_if_no_akima():
    try:
        from scipy.interpolate import Akima1DInterpolator  # noqa
    except ImportError:
        import pytest
        pytest.skip('scipy.interpolate.Akima1DInterpolator missing')


def _simple_ts(start, end, freq='D'):
    rng = date_range(start, end, freq=freq)
    return Series(np.random.randn(len(rng)), index=rng)


class TestSeriesMissingData():

    def test_remove_na_deprecation(self):
        # see gh-16971
        with tm.assert_produces_warning(FutureWarning):
            remove_na(Series([]))

    def test_timedelta_fillna(self):
        # GH 3371
        s = Series([Timestamp('20130101'), Timestamp('20130101'),
                    Timestamp('20130102'), Timestamp('20130103 9:01:01')])
        td = s.diff()

        # reg fillna
        with tm.assert_produces_warning(FutureWarning):
            result = td.fillna(0)
        expected = Series([timedelta(0), timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9 * 3600 + 60 + 1)])
        assert_series_equal(result, expected)

        # interpreted as seconds, deprecated
        with tm.assert_produces_warning(FutureWarning):
            result = td.fillna(1)
        expected = Series([timedelta(seconds=1),
                           timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9 * 3600 + 60 + 1)])
        assert_series_equal(result, expected)

        result = td.fillna(timedelta(days=1, seconds=1))
        expected = Series([timedelta(days=1, seconds=1), timedelta(0),
                           timedelta(1),
                           timedelta(days=1, seconds=9 * 3600 + 60 + 1)])
        assert_series_equal(result, expected)

        result = td.fillna(np.timedelta64(int(1e9)))
        expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9 * 3600 + 60 + 1)])
        assert_series_equal(result, expected)

        result = td.fillna(NaT)
        expected = Series([NaT, timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9 * 3600 + 60 + 1)],
                          dtype='m8[ns]')
        assert_series_equal(result, expected)

        # ffill
        td[2] = np.nan
        result = td.ffill()
        with tm.assert_produces_warning(FutureWarning):
            expected = td.fillna(0)
        expected[0] = np.nan
        assert_series_equal(result, expected)

        # bfill
        td[2] = np.nan
        result = td.bfill()
        with tm.assert_produces_warning(FutureWarning):
            expected = td.fillna(0)
        expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
        assert_series_equal(result, expected)

    def test_datetime64_fillna(self):

        s = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp(
            '20130102'), Timestamp('20130103 9:01:01')])
        s[2] = np.nan

        # reg fillna
        result = s.fillna(Timestamp('20130104'))
        expected = Series([Timestamp('20130101'), Timestamp(
            '20130101'), Timestamp('20130104'), Timestamp('20130103 9:01:01')])
        assert_series_equal(result, expected)

        result = s.fillna(NaT)
        expected = s
        assert_series_equal(result, expected)

        # ffill
        result = s.ffill()
        expected = Series([Timestamp('20130101'), Timestamp(
            '20130101'), Timestamp('20130101'), Timestamp('20130103 9:01:01')])
        assert_series_equal(result, expected)

        # bfill
        result = s.bfill()
        expected = Series([Timestamp('20130101'), Timestamp('20130101'),
                           Timestamp('20130103 9:01:01'), Timestamp(
                               '20130103 9:01:01')])
        assert_series_equal(result, expected)

        # GH 6587
        # make sure that we are treating as integer when filling
        # this also tests inference of a datetime-like with NaT's
        s = Series([pd.NaT, pd.NaT, '2013-08-05 15:30:00.000001'])
        expected = Series(
            ['2013-08-05 15:30:00.000001', '2013-08-05 15:30:00.000001',
             '2013-08-05 15:30:00.000001'], dtype='M8[ns]')
        result = s.fillna(method='backfill')
        assert_series_equal(result, expected)

    def test_datetime64_tz_fillna(self):

        for tz in ['US/Eastern', 'Asia/Tokyo']:
            # DatetimeBlock
            s = Series([Timestamp('2011-01-01 10:00'), pd.NaT,
                        Timestamp('2011-01-03 10:00'), pd.NaT])
            null_loc = pd.Series([False, True, False, True])

            result = s.fillna(pd.Timestamp('2011-01-02 10:00'))
            expected = Series([Timestamp('2011-01-01 10:00'),
                               Timestamp('2011-01-02 10:00'),
                               Timestamp('2011-01-03 10:00'),
                               Timestamp('2011-01-02 10:00')])
            tm.assert_series_equal(expected, result)
            # check s is not changed
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna(pd.Timestamp('2011-01-02 10:00', tz=tz))
            expected = Series([Timestamp('2011-01-01 10:00'),
                               Timestamp('2011-01-02 10:00', tz=tz),
                               Timestamp('2011-01-03 10:00'),
                               Timestamp('2011-01-02 10:00', tz=tz)])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna('AAA')
            expected = Series([Timestamp('2011-01-01 10:00'), 'AAA',
                               Timestamp('2011-01-03 10:00'), 'AAA'],
                              dtype=object)
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna({1: pd.Timestamp('2011-01-02 10:00', tz=tz),
                               3: pd.Timestamp('2011-01-04 10:00')})
            expected = Series([Timestamp('2011-01-01 10:00'),
                               Timestamp('2011-01-02 10:00', tz=tz),
                               Timestamp('2011-01-03 10:00'),
                               Timestamp('2011-01-04 10:00')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna({1: pd.Timestamp('2011-01-02 10:00'),
                               3: pd.Timestamp('2011-01-04 10:00')})
            expected = Series([Timestamp('2011-01-01 10:00'),
                               Timestamp('2011-01-02 10:00'),
                               Timestamp('2011-01-03 10:00'),
                               Timestamp('2011-01-04 10:00')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            # DatetimeBlockTZ
            idx = pd.DatetimeIndex(['2011-01-01 10:00', pd.NaT,
                                    '2011-01-03 10:00', pd.NaT], tz=tz)
            s = pd.Series(idx)
            assert s.dtype == 'datetime64[ns, {0}]'.format(tz)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna(pd.Timestamp('2011-01-02 10:00'))
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz),
                               Timestamp('2011-01-02 10:00'),
                               Timestamp('2011-01-03 10:00', tz=tz),
                               Timestamp('2011-01-02 10:00')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna(pd.Timestamp('2011-01-02 10:00', tz=tz))
            idx = pd.DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00',
                                    '2011-01-03 10:00', '2011-01-02 10:00'],
                                   tz=tz)
            expected = Series(idx)
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna(pd.Timestamp('2011-01-02 10:00',
                                           tz=tz).to_pydatetime())
            idx = pd.DatetimeIndex(['2011-01-01 10:00', '2011-01-02 10:00',
                                    '2011-01-03 10:00', '2011-01-02 10:00'],
                                   tz=tz)
            expected = Series(idx)
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna('AAA')
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz), 'AAA',
                               Timestamp('2011-01-03 10:00', tz=tz), 'AAA'],
                              dtype=object)
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna({1: pd.Timestamp('2011-01-02 10:00', tz=tz),
                               3: pd.Timestamp('2011-01-04 10:00')})
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz),
                               Timestamp('2011-01-02 10:00', tz=tz),
                               Timestamp('2011-01-03 10:00', tz=tz),
                               Timestamp('2011-01-04 10:00')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna({1: pd.Timestamp('2011-01-02 10:00', tz=tz),
                               3: pd.Timestamp('2011-01-04 10:00', tz=tz)})
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz),
                               Timestamp('2011-01-02 10:00', tz=tz),
                               Timestamp('2011-01-03 10:00', tz=tz),
                               Timestamp('2011-01-04 10:00', tz=tz)])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            # filling with a naive/other zone, coerce to object
            result = s.fillna(Timestamp('20130101'))
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz),
                               Timestamp('2013-01-01'),
                               Timestamp('2011-01-03 10:00', tz=tz),
                               Timestamp('2013-01-01')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

            result = s.fillna(Timestamp('20130101', tz='US/Pacific'))
            expected = Series([Timestamp('2011-01-01 10:00', tz=tz),
                               Timestamp('2013-01-01', tz='US/Pacific'),
                               Timestamp('2011-01-03 10:00', tz=tz),
                               Timestamp('2013-01-01', tz='US/Pacific')])
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(pd.isna(s), null_loc)

        # with timezone
        # GH 15855
        df = pd.Series([pd.Timestamp('2012-11-11 00:00:00+01:00'), pd.NaT])
        exp = pd.Series([pd.Timestamp('2012-11-11 00:00:00+01:00'),
                         pd.Timestamp('2012-11-11 00:00:00+01:00')])
        assert_series_equal(df.fillna(method='pad'), exp)

        df = pd.Series([pd.NaT, pd.Timestamp('2012-11-11 00:00:00+01:00')])
        exp = pd.Series([pd.Timestamp('2012-11-11 00:00:00+01:00'),
                         pd.Timestamp('2012-11-11 00:00:00+01:00')])
        assert_series_equal(df.fillna(method='bfill'), exp)

    def test_fillna_consistency(self):
        # GH 16402
        # fillna with a tz aware to a tz-naive, should result in object

        s = Series([Timestamp('20130101'), pd.NaT])

        result = s.fillna(Timestamp('20130101', tz='US/Eastern'))
        expected = Series([Timestamp('20130101'),
                           Timestamp('2013-01-01', tz='US/Eastern')],
                          dtype='object')
        assert_series_equal(result, expected)

        # where (we ignore the errors=)
        result = s.where([True, False],
                         Timestamp('20130101', tz='US/Eastern'),
                         errors='ignore')
        assert_series_equal(result, expected)

        result = s.where([True, False],
                         Timestamp('20130101', tz='US/Eastern'),
                         errors='ignore')
        assert_series_equal(result, expected)

        # with a non-datetime
        result = s.fillna('foo')
        expected = Series([Timestamp('20130101'),
                           'foo'])
        assert_series_equal(result, expected)

        # assignment
        s2 = s.copy()
        s2[1] = 'foo'
        assert_series_equal(s2, expected)

    def test_datetime64tz_fillna_round_issue(self):
        # GH 14872

        data = pd.Series([pd.NaT, pd.NaT,
                          datetime(2016, 12, 12, 22, 24, 6, 100001,
                                   tzinfo=pytz.utc)])

        filled = data.fillna(method='bfill')

        expected = pd.Series([datetime(2016, 12, 12, 22, 24, 6,
                                       100001, tzinfo=pytz.utc),
                              datetime(2016, 12, 12, 22, 24, 6,
                                       100001, tzinfo=pytz.utc),
                              datetime(2016, 12, 12, 22, 24, 6,
                                       100001, tzinfo=pytz.utc)])

        assert_series_equal(filled, expected)

    def test_fillna_downcast(self):
        # GH 15277
        # infer int64 from float64
        s = pd.Series([1., np.nan])
        result = s.fillna(0, downcast='infer')
        expected = pd.Series([1, 0])
        assert_series_equal(result, expected)

        # infer int64 from float64 when fillna value is a dict
        s = pd.Series([1., np.nan])
        result = s.fillna({1: 0}, downcast='infer')
        expected = pd.Series([1, 0])
        assert_series_equal(result, expected)

    def test_fillna_int(self):
        s = Series(np.random.randint(-100, 100, 50))
        s.fillna(method='ffill', inplace=True)
        assert_series_equal(s.fillna(method='ffill', inplace=False), s)

    def test_fillna_raise(self):
        s = Series(np.random.randint(-100, 100, 50))
        msg = ('"value" parameter must be a scalar or dict, but you passed a'
               ' "list"')
        with pytest.raises(TypeError, match=msg):
            s.fillna([1, 2])

        msg = ('"value" parameter must be a scalar or dict, but you passed a'
               ' "tuple"')
        with pytest.raises(TypeError, match=msg):
            s.fillna((1, 2))

        # related GH 9217, make sure limit is an int and greater than 0
        s = Series([1, 2, 3, None])
        msg = (r"Cannot specify both 'value' and 'method'\.|"
               r"Limit must be greater than 0|"
               "Limit must be an integer")
        for limit in [-1, 0, 1., 2.]:
            for method in ['backfill', 'bfill', 'pad', 'ffill', None]:
                with pytest.raises(ValueError, match=msg):
                    s.fillna(1, limit=limit, method=method)

    def test_categorical_nan_equality(self):
        cat = Series(Categorical(["a", "b", "c", np.nan]))
        exp = Series([True, True, True, False])
        res = (cat == cat)
        tm.assert_series_equal(res, exp)

    def test_categorical_nan_handling(self):

        # NaNs are represented as -1 in labels
        s = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(s.cat.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(s.values.codes,
                                    np.array([0, 1, -1, 0], dtype=np.int8))

    @pytest.mark.parametrize('fill_value, expected_output', [
        ('a', ['a', 'a', 'b', 'a', 'a']),
        ({1: 'a', 3: 'b', 4: 'b'}, ['a', 'a', 'b', 'b', 'b']),
        ({1: 'a'}, ['a', 'a', 'b', np.nan, np.nan]),
        ({1: 'a', 3: 'b'}, ['a', 'a', 'b', 'b', np.nan]),
        (Series('a'), ['a', np.nan, 'b', np.nan, np.nan]),
        (Series('a', index=[1]), ['a', 'a', 'b', np.nan, np.nan]),
        (Series({1: 'a', 3: 'b'}), ['a', 'a', 'b', 'b', np.nan]),
        (Series(['a', 'b'], index=[3, 4]), ['a', np.nan, 'b', 'a', 'b'])
    ])
    def test_fillna_categorical(self, fill_value, expected_output):
        # GH 17033
        # Test fillna for a Categorical series
        data = ['a', np.nan, 'b', np.nan, np.nan]
        s = Series(Categorical(data, categories=['a', 'b']))
        exp = Series(Categorical(expected_output, categories=['a', 'b']))
        tm.assert_series_equal(s.fillna(fill_value), exp)

    def test_fillna_categorical_raise(self):
        data = ['a', np.nan, 'b', np.nan, np.nan]
        s = Series(Categorical(data, categories=['a', 'b']))

        with pytest.raises(ValueError,
                           match="fill value must be in categories"):
            s.fillna('d')

        with pytest.raises(ValueError,
                           match="fill value must be in categories"):
            s.fillna(Series('d'))

        with pytest.raises(ValueError,
                           match="fill value must be in categories"):
            s.fillna({1: 'd', 3: 'a'})

        msg = ('"value" parameter must be a scalar or '
               'dict, but you passed a "list"')
        with pytest.raises(TypeError, match=msg):
            s.fillna(['a', 'b'])

        msg = ('"value" parameter must be a scalar or '
               'dict, but you passed a "tuple"')
        with pytest.raises(TypeError, match=msg):
            s.fillna(('a', 'b'))

        msg = ('"value" parameter must be a scalar, dict '
               'or Series, but you passed a "DataFrame"')
        with pytest.raises(TypeError, match=msg):
            s.fillna(DataFrame({1: ['a'], 3: ['b']}))

    def test_fillna_nat(self):
        series = Series([0, 1, 2, iNaT], dtype='M8[ns]')

        filled = series.fillna(method='pad')
        filled2 = series.fillna(value=series.values[2])

        expected = series.copy()
        expected.values[3] = expected.values[2]

        assert_series_equal(filled, expected)
        assert_series_equal(filled2, expected)

        df = DataFrame({'A': series})
        filled = df.fillna(method='pad')
        filled2 = df.fillna(value=series.values[2])
        expected = DataFrame({'A': expected})
        assert_frame_equal(filled, expected)
        assert_frame_equal(filled2, expected)

        series = Series([iNaT, 0, 1, 2], dtype='M8[ns]')

        filled = series.fillna(method='bfill')
        filled2 = series.fillna(value=series[1])

        expected = series.copy()
        expected[0] = expected[1]

        assert_series_equal(filled, expected)
        assert_series_equal(filled2, expected)

        df = DataFrame({'A': series})
        filled = df.fillna(method='bfill')
        filled2 = df.fillna(value=series[1])
        expected = DataFrame({'A': expected})
        assert_frame_equal(filled, expected)
        assert_frame_equal(filled2, expected)

    def test_isna_for_inf(self):
        s = Series(['a', np.inf, np.nan, 1.0])
        with pd.option_context('mode.use_inf_as_na', True):
            r = s.isna()
            dr = s.dropna()
        e = Series([False, True, True, False])
        de = Series(['a', 1.0], index=[0, 3])
        tm.assert_series_equal(r, e)
        tm.assert_series_equal(dr, de)

    def test_isnull_for_inf_deprecated(self):
        # gh-17115
        s = Series(['a', np.inf, np.nan, 1.0])
        with pd.option_context('mode.use_inf_as_null', True):
            r = s.isna()
            dr = s.dropna()

        e = Series([False, True, True, False])
        de = Series(['a', 1.0], index=[0, 3])
        tm.assert_series_equal(r, e)
        tm.assert_series_equal(dr, de)

    def test_fillna(self, datetime_series):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))

        tm.assert_series_equal(ts, ts.fillna(method='ffill'))

        ts[2] = np.NaN

        exp = Series([0., 1., 1., 3., 4.], index=ts.index)
        tm.assert_series_equal(ts.fillna(method='ffill'), exp)

        exp = Series([0., 1., 3., 3., 4.], index=ts.index)
        tm.assert_series_equal(ts.fillna(method='backfill'), exp)

        exp = Series([0., 1., 5., 3., 4.], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)

        msg = "Must specify a fill 'value' or 'method'"
        with pytest.raises(ValueError, match=msg):
            ts.fillna()

        msg = "Cannot specify both 'value' and 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_series.fillna(value=0, method='ffill')

        # GH 5703
        s1 = Series([np.nan])
        s2 = Series([1])
        result = s1.fillna(s2)
        expected = Series([1.])
        assert_series_equal(result, expected)
        result = s1.fillna({})
        assert_series_equal(result, s1)
        result = s1.fillna(Series(()))
        assert_series_equal(result, s1)
        result = s2.fillna(s1)
        assert_series_equal(result, s2)
        result = s1.fillna({0: 1})
        assert_series_equal(result, expected)
        result = s1.fillna({1: 1})
        assert_series_equal(result, Series([np.nan]))
        result = s1.fillna({0: 1, 1: 1})
        assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}))
        assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        assert_series_equal(result, s1)

        s1 = Series([0, 1, 2], list('abc'))
        s2 = Series([0, np.nan, 2], list('bac'))
        result = s2.fillna(s1)
        expected = Series([0, 0, 2.], list('bac'))
        assert_series_equal(result, expected)

        # limit
        s = Series(np.nan, index=[0, 1, 2])
        result = s.fillna(999, limit=1)
        expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
        assert_series_equal(result, expected)

        result = s.fillna(999, limit=2)
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        assert_series_equal(result, expected)

        # GH 9043
        # make sure a string representation of int/float values can be filled
        # correctly without raising errors or being converted
        vals = ['0', '1.5', '-0.3']
        for val in vals:
            s = Series([0, 1, np.nan, np.nan, 4], dtype='float64')
            result = s.fillna(val)
            expected = Series([0, 1, val, val, 4], dtype='object')
            assert_series_equal(result, expected)

    def test_fillna_bug(self):
        x = Series([nan, 1., nan, 3., nan], ['z', 'a', 'b', 'c', 'd'])
        filled = x.fillna(method='ffill')
        expected = Series([nan, 1., 1., 3., 3.], x.index)
        assert_series_equal(filled, expected)

        filled = x.fillna(method='bfill')
        expected = Series([1., 1., 3., 3., nan], x.index)
        assert_series_equal(filled, expected)

    def test_fillna_inplace(self):
        x = Series([nan, 1., nan, 3., nan], ['z', 'a', 'b', 'c', 'd'])
        y = x.copy()

        y.fillna(value=0, inplace=True)

        expected = x.fillna(value=0)
        assert_series_equal(y, expected)

    def test_fillna_invalid_method(self, datetime_series):
        try:
            datetime_series.fillna(method='ffil')
        except ValueError as inst:
            assert 'ffil' in str(inst)

    def test_ffill(self):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        assert_series_equal(ts.ffill(), ts.fillna(method='ffill'))

    def test_ffill_mixed_dtypes_without_missing_data(self):
        # GH14956
        series = pd.Series([datetime(2015, 1, 1, tzinfo=pytz.utc), 1])
        result = series.ffill()
        assert_series_equal(series, result)

    def test_bfill(self):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        assert_series_equal(ts.bfill(), ts.fillna(method='bfill'))

    def test_timedelta64_nan(self):

        td = Series([timedelta(days=i) for i in range(10)])

        # nan ops on timedeltas
        td1 = td.copy()
        td1[0] = np.nan
        assert isna(td1[0])
        assert td1[0].value == iNaT
        td1[0] = td[0]
        assert not isna(td1[0])

        td1[1] = iNaT
        assert isna(td1[1])
        assert td1[1].value == iNaT
        td1[1] = td[1]
        assert not isna(td1[1])

        td1[2] = NaT
        assert isna(td1[2])
        assert td1[2].value == iNaT
        td1[2] = td[2]
        assert not isna(td1[2])

        # boolean setting
        # this doesn't work, not sure numpy even supports it
        # result = td[(td>np.timedelta64(timedelta(days=3))) &
        # td<np.timedelta64(timedelta(days=7)))] = np.nan
        # assert isna(result).sum() == 7

        # NumPy limitiation =(

        # def test_logical_range_select(self):
        #     np.random.seed(12345)
        #     selector = -0.5 <= datetime_series <= 0.5
        #     expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)
        #     assert_series_equal(selector, expected)

    def test_dropna_empty(self):
        s = Series([])
        assert len(s.dropna()) == 0
        s.dropna(inplace=True)
        assert len(s) == 0

        # invalid axis
        msg = r"No axis named 1 for object type <(class|type) 'type'>"
        with pytest.raises(ValueError, match=msg):
            s.dropna(axis=1)

    def test_datetime64_tz_dropna(self):
        # DatetimeBlock
        s = Series([Timestamp('2011-01-01 10:00'), pd.NaT, Timestamp(
            '2011-01-03 10:00'), pd.NaT])
        result = s.dropna()
        expected = Series([Timestamp('2011-01-01 10:00'),
                           Timestamp('2011-01-03 10:00')], index=[0, 2])
        tm.assert_series_equal(result, expected)

        # DatetimeBlockTZ
        idx = pd.DatetimeIndex(['2011-01-01 10:00', pd.NaT,
                                '2011-01-03 10:00', pd.NaT],
                               tz='Asia/Tokyo')
        s = pd.Series(idx)
        assert s.dtype == 'datetime64[ns, Asia/Tokyo]'
        result = s.dropna()
        expected = Series([Timestamp('2011-01-01 10:00', tz='Asia/Tokyo'),
                           Timestamp('2011-01-03 10:00', tz='Asia/Tokyo')],
                          index=[0, 2])
        assert result.dtype == 'datetime64[ns, Asia/Tokyo]'
        tm.assert_series_equal(result, expected)

    def test_dropna_no_nan(self):
        for s in [Series([1, 2, 3], name='x'), Series(
                [False, True, False], name='x')]:

            result = s.dropna()
            tm.assert_series_equal(result, s)
            assert result is not s

            s2 = s.copy()
            s2.dropna(inplace=True)
            tm.assert_series_equal(s2, s)

    def test_dropna_intervals(self):
        s = Series([np.nan, 1, 2, 3], IntervalIndex.from_arrays(
            [np.nan, 0, 1, 2],
            [np.nan, 1, 2, 3]))

        result = s.dropna()
        expected = s.iloc[1:]
        assert_series_equal(result, expected)

    def test_valid(self, datetime_series):
        ts = datetime_series.copy()
        ts[::2] = np.NaN

        result = ts.dropna()
        assert len(result) == ts.count()
        tm.assert_series_equal(result, ts[1::2])
        tm.assert_series_equal(result, ts[pd.notna(ts)])

    def test_isna(self):
        ser = Series([0, 5.4, 3, nan, -0.001])
        expected = Series([False, False, False, True, False])
        tm.assert_series_equal(ser.isna(), expected)

        ser = Series(["hi", "", nan])
        expected = Series([False, False, True])
        tm.assert_series_equal(ser.isna(), expected)

    def test_notna(self):
        ser = Series([0, 5.4, 3, nan, -0.001])
        expected = Series([True, True, True, False, True])
        tm.assert_series_equal(ser.notna(), expected)

        ser = Series(["hi", "", nan])
        expected = Series([True, True, False])
        tm.assert_series_equal(ser.notna(), expected)

    def test_pad_nan(self):
        x = Series([np.nan, 1., np.nan, 3., np.nan], ['z', 'a', 'b', 'c', 'd'],
                   dtype=float)

        x.fillna(method='pad', inplace=True)

        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0],
                          ['z', 'a', 'b', 'c', 'd'], dtype=float)
        assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x[0]), np.isnan(expected[0])

    def test_pad_require_monotonicity(self):
        rng = date_range('1/1/2000', '3/1/2000', freq='B')

        # neither monotonic increasing or decreasing
        rng2 = rng[[1, 0, 2]]

        msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(ValueError, match=msg):
            rng2.get_indexer(rng, method='pad')

    def test_dropna_preserve_name(self, datetime_series):
        datetime_series[:5] = np.nan
        result = datetime_series.dropna()
        assert result.name == datetime_series.name
        name = datetime_series.name
        ts = datetime_series.copy()
        ts.dropna(inplace=True)
        assert ts.name == name

    def test_fill_value_when_combine_const(self):
        # GH12723
        s = Series([0, 1, np.nan, 3, 4, 5])

        exp = s.fillna(0).add(2)
        res = s.add(2, fill_value=0)
        assert_series_equal(res, exp)

    def test_series_fillna_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)

        result = s[:2].reindex(index)
        result = result.fillna(method='pad', limit=5)

        expected = s[:2].reindex(index).fillna(method='pad')
        expected[-3:] = np.nan
        assert_series_equal(result, expected)

        result = s[-2:].reindex(index)
        result = result.fillna(method='bfill', limit=5)

        expected = s[-2:].reindex(index).fillna(method='backfill')
        expected[:3] = np.nan
        assert_series_equal(result, expected)

    def test_sparse_series_fillna_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)

        ss = s[:2].reindex(index).to_sparse()
        # TODO: what is this test doing? why are result an expected
        # the same call to fillna?
        with tm.assert_produces_warning(PerformanceWarning):
            # TODO: release-note fillna performance warning
            result = ss.fillna(method='pad', limit=5)
            expected = ss.fillna(method='pad', limit=5)
        expected = expected.to_dense()
        expected[-3:] = np.nan
        expected = expected.to_sparse()
        assert_series_equal(result, expected)

        ss = s[-2:].reindex(index).to_sparse()
        with tm.assert_produces_warning(PerformanceWarning):
            result = ss.fillna(method='backfill', limit=5)
            expected = ss.fillna(method='backfill')
        expected = expected.to_dense()
        expected[:3] = np.nan
        expected = expected.to_sparse()
        assert_series_equal(result, expected)

    def test_sparse_series_pad_backfill_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)
        s = s.to_sparse()

        result = s[:2].reindex(index, method='pad', limit=5)
        with tm.assert_produces_warning(PerformanceWarning):
            expected = s[:2].reindex(index).fillna(method='pad')
        expected = expected.to_dense()
        expected[-3:] = np.nan
        expected = expected.to_sparse()
        assert_series_equal(result, expected)

        result = s[-2:].reindex(index, method='backfill', limit=5)
        with tm.assert_produces_warning(PerformanceWarning):
            expected = s[-2:].reindex(index).fillna(method='backfill')
        expected = expected.to_dense()
        expected[:3] = np.nan
        expected = expected.to_sparse()
        assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)

        result = s[:2].reindex(index, method='pad', limit=5)

        expected = s[:2].reindex(index).fillna(method='pad')
        expected[-3:] = np.nan
        assert_series_equal(result, expected)

        result = s[-2:].reindex(index, method='backfill', limit=5)

        expected = s[-2:].reindex(index).fillna(method='backfill')
        expected[:3] = np.nan
        assert_series_equal(result, expected)


class TestSeriesInterpolateData():

    def test_interpolate(self, datetime_series, string_series):
        ts = Series(np.arange(len(datetime_series), dtype=float),
                    datetime_series.index)

        ts_copy = ts.copy()
        ts_copy[5:10] = np.NaN

        linear_interp = ts_copy.interpolate(method='linear')
        tm.assert_series_equal(linear_interp, ts)

        ord_ts = Series([d.toordinal() for d in datetime_series.index],
                        index=datetime_series.index).astype(float)

        ord_ts_copy = ord_ts.copy()
        ord_ts_copy[5:10] = np.NaN

        time_interp = ord_ts_copy.interpolate(method='time')
        tm.assert_series_equal(time_interp, ord_ts)

        # try time interpolation on a non-TimeSeries
        # Only raises ValueError if there are NaNs.
        non_ts = string_series.copy()
        non_ts[0] = np.NaN
        msg = ("time-weighted interpolation only works on Series or DataFrames"
               " with a DatetimeIndex")
        with pytest.raises(ValueError, match=msg):
            non_ts.interpolate(method='time')

    @td.skip_if_no_scipy
    def test_interpolate_pchip(self):
        _skip_if_no_pchip()

        ser = Series(np.sort(np.random.uniform(size=100)))

        # interpolate at new_index
        new_index = ser.index.union(Index([49.25, 49.5, 49.75, 50.25, 50.5,
                                           50.75]))
        interp_s = ser.reindex(new_index).interpolate(method='pchip')
        # does not blow up, GH5977
        interp_s[49:51]

    @td.skip_if_no_scipy
    def test_interpolate_akima(self):
        _skip_if_no_akima()

        ser = Series([10, 11, 12, 13])

        expected = Series([11.00, 11.25, 11.50, 11.75,
                           12.00, 12.25, 12.50, 12.75, 13.00],
                          index=Index([1.0, 1.25, 1.5, 1.75,
                                       2.0, 2.25, 2.5, 2.75, 3.0]))
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75]))
        interp_s = ser.reindex(new_index).interpolate(method='akima')
        assert_series_equal(interp_s[1:3], expected)

    @td.skip_if_no_scipy
    def test_interpolate_piecewise_polynomial(self):
        ser = Series([10, 11, 12, 13])

        expected = Series([11.00, 11.25, 11.50, 11.75,
                           12.00, 12.25, 12.50, 12.75, 13.00],
                          index=Index([1.0, 1.25, 1.5, 1.75,
                                       2.0, 2.25, 2.5, 2.75, 3.0]))
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75]))
        interp_s = ser.reindex(new_index).interpolate(
            method='piecewise_polynomial')
        assert_series_equal(interp_s[1:3], expected)

    @td.skip_if_no_scipy
    def test_interpolate_from_derivatives(self):
        ser = Series([10, 11, 12, 13])

        expected = Series([11.00, 11.25, 11.50, 11.75,
                           12.00, 12.25, 12.50, 12.75, 13.00],
                          index=Index([1.0, 1.25, 1.5, 1.75,
                                       2.0, 2.25, 2.5, 2.75, 3.0]))
        # interpolate at new_index
        new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75]))
        interp_s = ser.reindex(new_index).interpolate(
            method='from_derivatives')
        assert_series_equal(interp_s[1:3], expected)

    @pytest.mark.parametrize("kwargs", [
        {},
        pytest.param({'method': 'polynomial', 'order': 1},
                     marks=td.skip_if_no_scipy)
    ])
    def test_interpolate_corners(self, kwargs):
        s = Series([np.nan, np.nan])
        assert_series_equal(s.interpolate(**kwargs), s)

        s = Series([]).interpolate()
        assert_series_equal(s.interpolate(**kwargs), s)

    def test_interpolate_index_values(self):
        s = Series(np.nan, index=np.sort(np.random.rand(30)))
        s[::3] = np.random.randn(10)

        vals = s.index.values.astype(float)

        result = s.interpolate(method='index')

        expected = s.copy()
        bad = isna(expected.values)
        good = ~bad
        expected = Series(np.interp(vals[bad], vals[good],
                                    s.values[good]),
                          index=s.index[bad])

        assert_series_equal(result[bad], expected)

        # 'values' is synonymous with 'index' for the method kwarg
        other_result = s.interpolate(method='values')

        assert_series_equal(other_result, result)
        assert_series_equal(other_result[bad], expected)

    def test_interpolate_non_ts(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        msg = ("time-weighted interpolation only works on Series or DataFrames"
               " with a DatetimeIndex")
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='time')

    @pytest.mark.parametrize("kwargs", [
        {},
        pytest.param({'method': 'polynomial', 'order': 1},
                     marks=td.skip_if_no_scipy)
    ])
    def test_nan_interpolate(self, kwargs):
        s = Series([0, 1, np.nan, 3])
        result = s.interpolate(**kwargs)
        expected = Series([0., 1., 2., 3.])
        assert_series_equal(result, expected)

    def test_nan_irregular_index(self):
        s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
        result = s.interpolate()
        expected = Series([1., 2., 3., 4.], index=[1, 3, 5, 9])
        assert_series_equal(result, expected)

    def test_nan_str_index(self):
        s = Series([0, 1, 2, np.nan], index=list('abcd'))
        result = s.interpolate()
        expected = Series([0., 1., 2., 2.], index=list('abcd'))
        assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_interp_quad(self):
        sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
        result = sq.interpolate(method='quadratic')
        expected = Series([1., 4., 9., 16.], index=[1, 2, 3, 4])
        assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_interp_scipy_basic(self):
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        # slinear
        expected = Series([1., 3., 7.5, 12., 18.5, 25.])
        result = s.interpolate(method='slinear')
        assert_series_equal(result, expected)

        result = s.interpolate(method='slinear', downcast='infer')
        assert_series_equal(result, expected)
        # nearest
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method='nearest')
        assert_series_equal(result, expected.astype('float'))

        result = s.interpolate(method='nearest', downcast='infer')
        assert_series_equal(result, expected)
        # zero
        expected = Series([1, 3, 3, 12, 12, 25])
        result = s.interpolate(method='zero')
        assert_series_equal(result, expected.astype('float'))

        result = s.interpolate(method='zero', downcast='infer')
        assert_series_equal(result, expected)
        # quadratic
        # GH #15662.
        # new cubic and quadratic interpolation algorithms from scipy 0.19.0.
        # previously `splmake` was used. See scipy/scipy#6710
        if _is_scipy_ge_0190:
            expected = Series([1, 3., 6.823529, 12., 18.058824, 25.])
        else:
            expected = Series([1, 3., 6.769231, 12., 18.230769, 25.])
        result = s.interpolate(method='quadratic')
        assert_series_equal(result, expected)

        result = s.interpolate(method='quadratic', downcast='infer')
        assert_series_equal(result, expected)
        # cubic
        expected = Series([1., 3., 6.8, 12., 18.2, 25.])
        result = s.interpolate(method='cubic')
        assert_series_equal(result, expected)

    def test_interp_limit(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1., 3., 5., 7., np.nan, 11.])
        result = s.interpolate(method='linear', limit=2)
        assert_series_equal(result, expected)

        # GH 9217, make sure limit is an int and greater than 0
        methods = ['linear', 'time', 'index', 'values', 'nearest', 'zero',
                   'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh',
                   'polynomial', 'spline', 'piecewise_polynomial', None,
                   'from_derivatives', 'pchip', 'akima']
        s = pd.Series([1, 2, np.nan, np.nan, 5])
        msg = (r"Limit must be greater than 0|"
               "time-weighted interpolation only works on Series or"
               r" DataFrames with a DatetimeIndex|"
               r"invalid method '(polynomial|spline|None)' to interpolate|"
               "Limit must be an integer")
        for limit in [-1, 0, 1., 2.]:
            for method in methods:
                with pytest.raises(ValueError, match=msg):
                    s.interpolate(limit=limit, method=method)

    def test_interp_limit_forward(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        # Provide 'forward' (the default) explicitly here.
        expected = Series([1., 3., 5., 7., np.nan, 11.])

        result = s.interpolate(method='linear', limit=2,
                               limit_direction='forward')
        assert_series_equal(result, expected)

        result = s.interpolate(method='linear', limit=2,
                               limit_direction='FORWARD')
        assert_series_equal(result, expected)

    def test_interp_unlimited(self):
        # these test are for issue #16282 default Limit=None is unlimited
        s = Series([np.nan, 1., 3., np.nan, np.nan, np.nan, 11., np.nan])
        expected = Series([1., 1., 3., 5., 7., 9., 11., 11.])
        result = s.interpolate(method='linear',
                               limit_direction='both')
        assert_series_equal(result, expected)

        expected = Series([np.nan, 1., 3., 5., 7., 9., 11., 11.])
        result = s.interpolate(method='linear',
                               limit_direction='forward')
        assert_series_equal(result, expected)

        expected = Series([1., 1., 3., 5., 7., 9., 11., np.nan])
        result = s.interpolate(method='linear',
                               limit_direction='backward')
        assert_series_equal(result, expected)

    def test_interp_limit_bad_direction(self):
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        msg = (r"Invalid limit_direction: expecting one of \['forward',"
               r" 'backward', 'both'\], got 'abc'")
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit=2, limit_direction='abc')

        # raises an error even if no limit is specified.
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit_direction='abc')

    # limit_area introduced GH #16284
    def test_interp_limit_area(self):
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([nan, nan, 3, nan, nan, nan, 7, nan, nan])

        expected = Series([nan, nan, 3., 4., 5., 6., 7., nan, nan])
        result = s.interpolate(method='linear', limit_area='inside')
        assert_series_equal(result, expected)

        expected = Series([nan, nan, 3., 4., nan, nan, 7., nan, nan])
        result = s.interpolate(method='linear', limit_area='inside',
                               limit=1)

        expected = Series([nan, nan, 3., 4., nan, 6., 7., nan, nan])
        result = s.interpolate(method='linear', limit_area='inside',
                               limit_direction='both', limit=1)
        assert_series_equal(result, expected)

        expected = Series([nan, nan, 3., nan, nan, nan, 7., 7., 7.])
        result = s.interpolate(method='linear', limit_area='outside')
        assert_series_equal(result, expected)

        expected = Series([nan, nan, 3., nan, nan, nan, 7., 7., nan])
        result = s.interpolate(method='linear', limit_area='outside',
                               limit=1)

        expected = Series([nan, 3., 3., nan, nan, nan, 7., 7., nan])
        result = s.interpolate(method='linear', limit_area='outside',
                               limit_direction='both', limit=1)
        assert_series_equal(result, expected)

        expected = Series([3., 3., 3., nan, nan, nan, 7., nan, nan])
        result = s.interpolate(method='linear', limit_area='outside',
                               direction='backward')

        # raises an error even if limit type is wrong.
        msg = (r"Invalid limit_area: expecting one of \['inside', 'outside'\],"
               " got abc")
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='linear', limit_area='abc')

    def test_interp_limit_direction(self):
        # These tests are for issue #9218 -- fill NaNs in both directions.
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])

        expected = Series([1., 3., np.nan, 7., 9., 11.])
        result = s.interpolate(method='linear', limit=2,
                               limit_direction='backward')
        assert_series_equal(result, expected)

        expected = Series([1., 3., 5., np.nan, 9., 11.])
        result = s.interpolate(method='linear', limit=1,
                               limit_direction='both')
        assert_series_equal(result, expected)

        # Check that this works on a longer series of nans.
        s = Series([1, 3, np.nan, np.nan, np.nan, 7, 9, np.nan, np.nan, 12,
                    np.nan])

        expected = Series([1., 3., 4., 5., 6., 7., 9., 10., 11., 12., 12.])
        result = s.interpolate(method='linear', limit=2,
                               limit_direction='both')
        assert_series_equal(result, expected)

        expected = Series([1., 3., 4., np.nan, 6., 7., 9., 10., 11., 12., 12.])
        result = s.interpolate(method='linear', limit=1,
                               limit_direction='both')
        assert_series_equal(result, expected)

    def test_interp_limit_to_ends(self):
        # These test are for issue #10420 -- flow back to beginning.
        s = Series([np.nan, np.nan, 5, 7, 9, np.nan])

        expected = Series([5., 5., 5., 7., 9., np.nan])
        result = s.interpolate(method='linear', limit=2,
                               limit_direction='backward')
        assert_series_equal(result, expected)

        expected = Series([5., 5., 5., 7., 9., 9.])
        result = s.interpolate(method='linear', limit=2,
                               limit_direction='both')
        assert_series_equal(result, expected)

    def test_interp_limit_before_ends(self):
        # These test are for issue #11115 -- limit ends properly.
        s = Series([np.nan, np.nan, 5, 7, np.nan, np.nan])

        expected = Series([np.nan, np.nan, 5., 7., 7., np.nan])
        result = s.interpolate(method='linear', limit=1,
                               limit_direction='forward')
        assert_series_equal(result, expected)

        expected = Series([np.nan, 5., 5., 7., np.nan, np.nan])
        result = s.interpolate(method='linear', limit=1,
                               limit_direction='backward')
        assert_series_equal(result, expected)

        expected = Series([np.nan, 5., 5., 7., 7., np.nan])
        result = s.interpolate(method='linear', limit=1,
                               limit_direction='both')
        assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_interp_all_good(self):
        s = Series([1, 2, 3])
        result = s.interpolate(method='polynomial', order=1)
        assert_series_equal(result, s)

        # non-scipy
        result = s.interpolate()
        assert_series_equal(result, s)

    @pytest.mark.parametrize("check_scipy", [
        False,
        pytest.param(True, marks=td.skip_if_no_scipy)
    ])
    def test_interp_multiIndex(self, check_scipy):
        idx = MultiIndex.from_tuples([(0, 'a'), (1, 'b'), (2, 'c')])
        s = Series([1, 2, np.nan], index=idx)

        expected = s.copy()
        expected.loc[2] = 2
        result = s.interpolate()
        assert_series_equal(result, expected)

        msg = "Only `method=linear` interpolation is supported on MultiIndexes"
        if check_scipy:
            with pytest.raises(ValueError, match=msg):
                s.interpolate(method='polynomial', order=1)

    @td.skip_if_no_scipy
    def test_interp_nonmono_raise(self):
        s = Series([1, np.nan, 3], index=[0, 2, 1])
        msg = "krogh interpolation requires that the index be monotonic"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='krogh')

    @td.skip_if_no_scipy
    def test_interp_datetime64(self):
        df = Series([1, np.nan, 3], index=date_range('1/1/2000', periods=3))
        result = df.interpolate(method='nearest')
        expected = Series([1., 1., 3.],
                          index=date_range('1/1/2000', periods=3))
        assert_series_equal(result, expected)

    def test_interp_limit_no_nans(self):
        # GH 7173
        s = pd.Series([1., 2., 3.])
        result = s.interpolate(limit=1)
        expected = s
        assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    @pytest.mark.parametrize("method", ['polynomial', 'spline'])
    def test_no_order(self, method):
        s = Series([0, 1, np.nan, 3])
        msg = "invalid method '{}' to interpolate".format(method)
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method=method)

    @td.skip_if_no_scipy
    def test_spline(self):
        s = Series([1, 2, np.nan, 4, 5, np.nan, 7])
        result = s.interpolate(method='spline', order=1)
        expected = Series([1., 2., 3., 4., 5., 6., 7.])
        assert_series_equal(result, expected)

    @td.skip_if_no('scipy', min_version='0.15')
    def test_spline_extrapolate(self):
        s = Series([1, 2, 3, 4, np.nan, 6, np.nan])
        result3 = s.interpolate(method='spline', order=1, ext=3)
        expected3 = Series([1., 2., 3., 4., 5., 6., 6.])
        assert_series_equal(result3, expected3)

        result1 = s.interpolate(method='spline', order=1, ext=0)
        expected1 = Series([1., 2., 3., 4., 5., 6., 7.])
        assert_series_equal(result1, expected1)

    @td.skip_if_no_scipy
    def test_spline_smooth(self):
        s = Series([1, 2, np.nan, 4, 5.1, np.nan, 7])
        assert (s.interpolate(method='spline', order=3, s=0)[5] !=
                s.interpolate(method='spline', order=3)[5])

    @td.skip_if_no_scipy
    def test_spline_interpolation(self):
        s = Series(np.arange(10) ** 2)
        s[np.random.randint(0, 9, 3)] = np.nan
        result1 = s.interpolate(method='spline', order=1)
        expected1 = s.interpolate(method='spline', order=1)
        assert_series_equal(result1, expected1)

    @td.skip_if_no_scipy
    def test_spline_error(self):
        # see gh-10633
        s = pd.Series(np.arange(10) ** 2)
        s[np.random.randint(0, 9, 3)] = np.nan
        msg = "invalid method 'spline' to interpolate"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='spline')

        msg = "order needs to be specified and greater than 0"
        with pytest.raises(ValueError, match=msg):
            s.interpolate(method='spline', order=0)

    def test_interp_timedelta64(self):
        # GH 6424
        df = Series([1, np.nan, 3],
                    index=pd.to_timedelta([1, 2, 3]))
        result = df.interpolate(method='time')
        expected = Series([1., 2., 3.],
                          index=pd.to_timedelta([1, 2, 3]))
        assert_series_equal(result, expected)

        # test for non uniform spacing
        df = Series([1, np.nan, 3],
                    index=pd.to_timedelta([1, 2, 4]))
        result = df.interpolate(method='time')
        expected = Series([1., 1.666667, 3.],
                          index=pd.to_timedelta([1, 2, 4]))
        assert_series_equal(result, expected)

    def test_series_interpolate_method_values(self):
        # #1646
        ts = _simple_ts('1/1/2000', '1/20/2000')
        ts[::2] = np.nan

        result = ts.interpolate(method='values')
        exp = ts.interpolate()
        assert_series_equal(result, exp)

    def test_series_interpolate_intraday(self):
        # #1698
        index = pd.date_range('1/1/2012', periods=4, freq='12D')
        ts = pd.Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(days=1)).sort_values()

        exp = ts.reindex(new_index).interpolate(method='time')

        index = pd.date_range('1/1/2012', periods=4, freq='12H')
        ts = pd.Series([0, 12, 24, 36], index)
        new_index = index.append(index + pd.DateOffset(hours=1)).sort_values()
        result = ts.reindex(new_index).interpolate(method='time')

        tm.assert_numpy_array_equal(result.values, exp.values)

    def test_nonzero_warning(self):
        # GH 24048
        ser = pd.Series([1, 0, 3, 4])
        with tm.assert_produces_warning(FutureWarning):
            ser.nonzero()
