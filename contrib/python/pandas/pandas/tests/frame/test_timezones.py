# -*- coding: utf-8 -*-
"""
Tests for DataFrame timezone-related methods
"""
from datetime import datetime

import numpy as np
import pytest
import pytz

from pandas.compat import lrange

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import DataFrame, Series
from pandas.core.indexes.datetimes import date_range
import pandas.util.testing as tm


class TestDataFrameTimezones(object):

    def test_frame_values_with_tz(self):
        tz = "US/Central"
        df = DataFrame({"A": date_range('2000', periods=4, tz=tz)})
        result = df.values
        expected = np.array([
            [pd.Timestamp('2000-01-01', tz=tz)],
            [pd.Timestamp('2000-01-02', tz=tz)],
            [pd.Timestamp('2000-01-03', tz=tz)],
            [pd.Timestamp('2000-01-04', tz=tz)],
        ])
        tm.assert_numpy_array_equal(result, expected)

        # two columns, homogenous

        df = df.assign(B=df.A)
        result = df.values
        expected = np.concatenate([expected, expected], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        # three columns, heterogenous
        est = "US/Eastern"
        df = df.assign(C=df.A.dt.tz_convert(est))

        new = np.array([
            [pd.Timestamp('2000-01-01T01:00:00', tz=est)],
            [pd.Timestamp('2000-01-02T01:00:00', tz=est)],
            [pd.Timestamp('2000-01-03T01:00:00', tz=est)],
            [pd.Timestamp('2000-01-04T01:00:00', tz=est)],
        ])
        expected = np.concatenate([expected, new], axis=1)
        result = df.values
        tm.assert_numpy_array_equal(result, expected)

    def test_frame_from_records_utc(self):
        rec = {'datum': 1.5,
               'begin_time': datetime(2006, 4, 27, tzinfo=pytz.utc)}

        # it works
        DataFrame.from_records([rec], index='begin_time')

    def test_frame_tz_localize(self):
        rng = date_range('1/1/2011', periods=100, freq='H')

        df = DataFrame({'a': 1}, index=rng)
        result = df.tz_localize('utc')
        expected = DataFrame({'a': 1}, rng.tz_localize('UTC'))
        assert result.index.tz.zone == 'UTC'
        tm.assert_frame_equal(result, expected)

        df = df.T
        result = df.tz_localize('utc', axis=1)
        assert result.columns.tz.zone == 'UTC'
        tm.assert_frame_equal(result, expected.T)

    def test_frame_tz_convert(self):
        rng = date_range('1/1/2011', periods=200, freq='D', tz='US/Eastern')

        df = DataFrame({'a': 1}, index=rng)
        result = df.tz_convert('Europe/Berlin')
        expected = DataFrame({'a': 1}, rng.tz_convert('Europe/Berlin'))
        assert result.index.tz.zone == 'Europe/Berlin'
        tm.assert_frame_equal(result, expected)

        df = df.T
        result = df.tz_convert('Europe/Berlin', axis=1)
        assert result.columns.tz.zone == 'Europe/Berlin'
        tm.assert_frame_equal(result, expected.T)

    def test_frame_join_tzaware(self):
        test1 = DataFrame(np.zeros((6, 3)),
                          index=date_range("2012-11-15 00:00:00", periods=6,
                                           freq="100L", tz="US/Central"))
        test2 = DataFrame(np.zeros((3, 3)),
                          index=date_range("2012-11-15 00:00:00", periods=3,
                                           freq="250L", tz="US/Central"),
                          columns=lrange(3, 6))

        result = test1.join(test2, how='outer')
        ex_index = test1.index.union(test2.index)

        tm.assert_index_equal(result.index, ex_index)
        assert result.index.tz.zone == 'US/Central'

    def test_frame_add_tz_mismatch_converts_to_utc(self):
        rng = date_range('1/1/2011', periods=10, freq='H', tz='US/Eastern')
        df = DataFrame(np.random.randn(len(rng)), index=rng, columns=['a'])

        df_moscow = df.tz_convert('Europe/Moscow')
        result = df + df_moscow
        assert result.index.tz is pytz.utc

        result = df_moscow + df
        assert result.index.tz is pytz.utc

    def test_frame_align_aware(self):
        idx1 = date_range('2001', periods=5, freq='H', tz='US/Eastern')
        idx2 = date_range('2001', periods=5, freq='2H', tz='US/Eastern')
        df1 = DataFrame(np.random.randn(len(idx1), 3), idx1)
        df2 = DataFrame(np.random.randn(len(idx2), 3), idx2)
        new1, new2 = df1.align(df2)
        assert df1.index.tz == new1.index.tz
        assert df2.index.tz == new2.index.tz

        # different timezones convert to UTC

        # frame with frame
        df1_central = df1.tz_convert('US/Central')
        new1, new2 = df1.align(df1_central)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

        # frame with Series
        new1, new2 = df1.align(df1_central[0], axis=0)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

        df1[0].align(df1_central, axis=0)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_frame_no_datetime64_dtype(self, tz):
        # after GH#7822
        # these retain the timezones on dict construction
        dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({'A': 'foo', 'B': dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype('ns', dr_tz.tzinfo)
        assert df['B'].dtype == tz_expected

        # GH#2810 (with timezones)
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({'dr': dr,
                        'dr_tz': dr_tz,
                        'datetimes_naive': datetimes_naive,
                        'datetimes_with_tz': datetimes_with_tz})
        result = df.get_dtype_counts().sort_index()
        expected = Series({'datetime64[ns]': 2,
                           str(tz_expected): 2}).sort_index()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_frame_reset_index(self, tz):
        dr = date_range('2012-06-02', periods=10, tz=tz)
        df = DataFrame(np.random.randn(len(dr)), dr)
        roundtripped = df.reset_index().set_index('index')
        xp = df.index.tz
        rs = roundtripped.index.tz
        assert xp == rs

    @pytest.mark.parametrize('tz', [None, 'America/New_York'])
    def test_boolean_compare_transpose_tzindex_with_dst(self, tz):
        # GH 19970
        idx = date_range('20161101', '20161130', freq='4H', tz=tz)
        df = DataFrame({'a': range(len(idx)), 'b': range(len(idx))},
                       index=idx)
        result = df.T == df.T
        expected = DataFrame(True, index=list('ab'), columns=idx)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('copy', [True, False])
    @pytest.mark.parametrize('method, tz', [
        ['tz_localize', None],
        ['tz_convert', 'Europe/Berlin']
    ])
    def test_tz_localize_convert_copy_inplace_mutate(self, copy, method, tz):
        # GH 6326
        result = DataFrame(np.arange(0, 5),
                           index=date_range('20131027', periods=5,
                                            freq='1H', tz=tz))
        getattr(result, method)('UTC', copy=copy)
        expected = DataFrame(np.arange(0, 5),
                             index=date_range('20131027', periods=5,
                                              freq='1H', tz=tz))
        tm.assert_frame_equal(result, expected)
