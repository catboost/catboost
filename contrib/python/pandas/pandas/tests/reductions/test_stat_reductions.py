# -*- coding: utf-8 -*-
"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""

import numpy as np
import pytest

from pandas.compat import lrange
import pandas.util._test_decorators as td

import pandas as pd
from pandas import DataFrame, Series, compat
import pandas.util.testing as tm


class TestSeriesStatReductions(object):
    # Note: the name TestSeriesStatReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def _check_stat_op(self, name, alternate, string_series_,
                       check_objects=False, check_allna=False):

        with pd.option_context('use_bottleneck', False):
            f = getattr(Series, name)

            # add some NaNs
            string_series_[5:15] = np.NaN

            # mean, idxmax, idxmin, min, and max are valid for dates
            if name not in ['max', 'min', 'mean']:
                ds = Series(pd.date_range('1/1/2001', periods=10))
                with pytest.raises(TypeError):
                    f(ds)

            # skipna or no
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # check the result is correct
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            allna = string_series_ * np.nan

            if check_allna:
                assert np.isnan(f(allna))

            # dtype=object with None, it works!
            s = Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888
            items = [0]
            items.extend(lrange(2 ** 40, 2 ** 40 + 1000))
            s = Series(items, dtype='int64')
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # check date range
            if check_objects:
                s = Series(pd.bdate_range('1/1/2000', periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # check on string data
            if name not in ['sum', 'min', 'max']:
                with pytest.raises(TypeError):
                    f(Series(list('abc')))

            # Invalid axis.
            with pytest.raises(ValueError):
                f(string_series_, axis=1)

            # Unimplemented numeric_only parameter.
            if 'numeric_only' in compat.signature(f).args:
                with pytest.raises(NotImplementedError, match=name):
                    f(string_series_, numeric_only=True)

    def test_sum(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('sum', np.sum, string_series, check_allna=False)

    def test_mean(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('mean', np.mean, string_series)

    def test_median(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('median', np.median, string_series)

        # test with integers, test failure
        int_ts = Series(np.ones(10, dtype=int), index=lrange(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('prod', np.prod, string_series)

    def test_min(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('min', np.min, string_series, check_objects=True)

    def test_max(self):
        string_series = tm.makeStringSeries().rename('series')
        self._check_stat_op('max', np.max, string_series, check_objects=True)

    def test_var_std(self):
        string_series = tm.makeStringSeries().rename('series')
        datetime_series = tm.makeTimeSeries().rename('ts')

        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op('std', alt, string_series)

        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op('var', alt, string_series)

        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)

        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        string_series = tm.makeStringSeries().rename('series')
        datetime_series = tm.makeTimeSeries().rename('ts')

        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op('sem', alt, string_series)

        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values,
                          ddof=4) / np.sqrt(len(datetime_series.values))
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    @td.skip_if_no_scipy
    def test_skew(self):
        from scipy.stats import skew

        string_series = tm.makeStringSeries().rename('series')

        alt = lambda x: skew(x, bias=False)
        self._check_stat_op('skew', alt, string_series)

        # test corner cases, skew() returns NaN unless there's at least 3
        # values
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                assert (df.skew() == 0).all()

    @td.skip_if_no_scipy
    def test_kurt(self):
        from scipy.stats import kurtosis

        string_series = tm.makeStringSeries().rename('series')

        alt = lambda x: kurtosis(x, bias=False)
        self._check_stat_op('kurt', alt, string_series)

        index = pd.MultiIndex(
            levels=[['bar'], ['one', 'two', 'three'], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]]
        )
        s = Series(np.random.randn(6), index=index)
        tm.assert_almost_equal(s.kurt(), s.kurt(level=0)['bar'])

        # test corner cases, kurt() returns NaN unless there's at least 4
        # values
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                assert (df.kurt() == 0).all()
