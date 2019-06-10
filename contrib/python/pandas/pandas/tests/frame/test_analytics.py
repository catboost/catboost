# -*- coding: utf-8 -*-

from datetime import timedelta
import operator
from string import ascii_lowercase
import warnings

import numpy as np
import pytest

from pandas.compat import PY35, lrange
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical, DataFrame, MultiIndex, Series, Timestamp, compat, date_range,
    isna, notna, to_datetime, to_timedelta)
import pandas.core.algorithms as algorithms
import pandas.core.nanops as nanops
import pandas.util.testing as tm


def assert_stat_op_calc(opname, alternative, frame, has_skipna=True,
                        check_dtype=True, check_dates=False,
                        check_less_precise=False, skipna_alternative=None):
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : string
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    check_less_precise : bool, default False
        Whether results should only be compared approximately;
        passed on to tm.assert_series_equal
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """

    f = getattr(frame, opname)

    if check_dates:
        df = DataFrame({'b': date_range('1/1/2001', periods=2)})
        result = getattr(df, opname)()
        assert isinstance(result, Series)

        df['a'] = lrange(len(df))
        result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)

    if has_skipna:
        def wrapper(x):
            return alternative(x.values)

        skipna_wrapper = tm._make_skipna_wrapper(alternative,
                                                 skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(result0, frame.apply(wrapper),
                               check_dtype=check_dtype,
                               check_less_precise=check_less_precise)
        # HACK: win32
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1),
                               check_dtype=False,
                               check_less_precise=check_less_precise)
    else:
        skipna_wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(result0, frame.apply(skipna_wrapper),
                           check_dtype=check_dtype,
                           check_less_precise=check_less_precise)

    if opname in ['sum', 'prod']:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(result1, expected, check_dtype=False,
                               check_less_precise=check_less_precise)

    # check dtypes
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype

    # bad axis
    with pytest.raises(ValueError, match='No axis named 2'):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.NaN
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ['sum', 'prod']:
            unit = 1 if opname == 'prod' else 0  # result for empty sum/prod
            expected = pd.Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = pd.Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


def assert_stat_op_api(opname, float_frame, float_string_frame,
                       has_numeric_only=False):
    """
    Check that API for operator opname works as advertised on frame

    Parameters
    ----------
    opname : string
        Name of the operator to test on frame
    float_frame : DataFrame
        DataFrame with columns of type float
    float_string_frame : DataFrame
        DataFrame with both float and string columns
    has_numeric_only : bool, default False
        Whether the method "opname" has the kwarg "numeric_only"
    """

    # make sure works on mixed-type frame
    getattr(float_string_frame, opname)(axis=0)
    getattr(float_string_frame, opname)(axis=1)

    if has_numeric_only:
        getattr(float_string_frame, opname)(axis=0, numeric_only=True)
        getattr(float_string_frame, opname)(axis=1, numeric_only=True)
        getattr(float_frame, opname)(axis=0, numeric_only=False)
        getattr(float_frame, opname)(axis=1, numeric_only=False)


def assert_bool_op_calc(opname, alternative, frame, has_skipna=True):
    """
    Check that bool operator opname works as advertised on frame

    Parameters
    ----------
    opname : string
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    """

    f = getattr(frame, opname)

    if has_skipna:
        def skipna_wrapper(x):
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x):
            return alternative(x.values)

        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)

        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1),
                               check_dtype=False)  # HACK: win32
    else:
        skipna_wrapper = alternative
        wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)

    tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
    tm.assert_series_equal(result1, frame.apply(skipna_wrapper, axis=1),
                           check_dtype=False)

    # bad axis
    with pytest.raises(ValueError, match='No axis named 2'):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.NaN
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname == 'any':
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()


def assert_bool_op_api(opname, bool_frame_with_na, float_string_frame,
                       has_bool_only=False):
    """
    Check that API for boolean operator opname works as advertised on frame

    Parameters
    ----------
    opname : string
        Name of the operator to test on frame
    float_frame : DataFrame
        DataFrame with columns of type float
    float_string_frame : DataFrame
        DataFrame with both float and string columns
    has_bool_only : bool, default False
        Whether the method "opname" has the kwarg "bool_only"
    """
    # make sure op works on mixed-type frame
    mixed = float_string_frame
    mixed['_bool_'] = np.random.randn(len(mixed)) > 0.5
    getattr(mixed, opname)(axis=0)
    getattr(mixed, opname)(axis=1)

    if has_bool_only:
        getattr(mixed, opname)(axis=0, bool_only=True)
        getattr(mixed, opname)(axis=1, bool_only=True)
        getattr(bool_frame_with_na, opname)(axis=0, bool_only=False)
        getattr(bool_frame_with_na, opname)(axis=1, bool_only=False)


class TestDataFrameAnalytics():

    # ---------------------------------------------------------------------=
    # Correlation and covariance

    @td.skip_if_no_scipy
    def test_corr_pearson(self, float_frame):
        float_frame['A'][:5] = np.nan
        float_frame['B'][5:10] = np.nan

        self._check_method(float_frame, 'pearson')

    @td.skip_if_no_scipy
    def test_corr_kendall(self, float_frame):
        float_frame['A'][:5] = np.nan
        float_frame['B'][5:10] = np.nan

        self._check_method(float_frame, 'kendall')

    @td.skip_if_no_scipy
    def test_corr_spearman(self, float_frame):
        float_frame['A'][:5] = np.nan
        float_frame['B'][5:10] = np.nan

        self._check_method(float_frame, 'spearman')

    def _check_method(self, frame, method='pearson'):
        correls = frame.corr(method=method)
        expected = frame['A'].corr(frame['C'], method=method)
        tm.assert_almost_equal(correls['A']['C'], expected)

    @td.skip_if_no_scipy
    def test_corr_non_numeric(self, float_frame, float_string_frame):
        float_frame['A'][:5] = np.nan
        float_frame['B'][5:10] = np.nan

        # exclude non-numeric types
        result = float_string_frame.corr()
        expected = float_string_frame.loc[:, ['A', 'B', 'C', 'D']].corr()
        tm.assert_frame_equal(result, expected)

    @td.skip_if_no_scipy
    @pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
    def test_corr_nooverlap(self, meth):
        # nothing in common
        df = DataFrame({'A': [1, 1.5, 1, np.nan, np.nan, np.nan],
                        'B': [np.nan, np.nan, np.nan, 1, 1.5, 1],
                        'C': [np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan]})
        rs = df.corr(meth)
        assert isna(rs.loc['A', 'B'])
        assert isna(rs.loc['B', 'A'])
        assert rs.loc['A', 'A'] == 1
        assert rs.loc['B', 'B'] == 1
        assert isna(rs.loc['C', 'C'])

    @td.skip_if_no_scipy
    @pytest.mark.parametrize('meth', ['pearson', 'spearman'])
    def test_corr_constant(self, meth):
        # constant --> all NA

        df = DataFrame({'A': [1, 1, 1, np.nan, np.nan, np.nan],
                        'B': [np.nan, np.nan, np.nan, 1, 1, 1]})
        rs = df.corr(meth)
        assert isna(rs.values).all()

    def test_corr_int(self):
        # dtypes other than float64 #1761
        df3 = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        df3.cov()
        df3.corr()

    @td.skip_if_no_scipy
    def test_corr_int_and_boolean(self):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        df = DataFrame({"a": [True, False], "b": [1, 0]})

        expected = DataFrame(np.ones((2, 2)), index=[
                             'a', 'b'], columns=['a', 'b'])
        for meth in ['pearson', 'kendall', 'spearman']:

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", RuntimeWarning)
                result = df.corr(meth)
            tm.assert_frame_equal(result, expected)

    def test_corr_cov_independent_index_column(self):
        # GH 14617
        df = pd.DataFrame(np.random.randn(4 * 10).reshape(10, 4),
                          columns=list("abcd"))
        for method in ['cov', 'corr']:
            result = getattr(df, method)()
            assert result.index is not result.columns
            assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        # GH 22298
        df = pd.DataFrame(np.random.normal(size=(10, 2)))
        msg = ("method must be either 'pearson', 'spearman', "
               "or 'kendall'")
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    def test_cov(self, float_frame, float_string_frame):
        # min_periods no NAs (corner case)
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))

        tm.assert_frame_equal(expected, result)

        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()

        # with NAs
        frame = float_frame.copy()
        frame['A'][:5] = np.nan
        frame['B'][5:10] = np.nan
        result = float_frame.cov(min_periods=len(float_frame) - 8)
        expected = float_frame.cov()
        expected.loc['A', 'B'] = np.nan
        expected.loc['B', 'A'] = np.nan

        # regular
        float_frame['A'][:5] = np.nan
        float_frame['B'][:10] = np.nan
        cov = float_frame.cov()

        tm.assert_almost_equal(cov['A']['C'],
                               float_frame['A'].cov(float_frame['C']))

        # exclude non-numeric types
        result = float_string_frame.cov()
        expected = float_string_frame.loc[:, ['A', 'B', 'C', 'D']].cov()
        tm.assert_frame_equal(result, expected)

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(np.cov(df.values.T).reshape((1, 1)),
                             index=df.columns, columns=df.columns)
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(np.cov(df.values[1:].T).reshape((1, 1)),
                             index=df.columns, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_corrwith(self, datetime_frame):
        a = datetime_frame
        noise = Series(np.random.randn(len(a)), index=a.index)

        b = datetime_frame.add(noise, axis=0)

        # make sure order does not matter
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][10:])
        del b['B']

        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr['A'], a['A'].corr(b['A']))

        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped['A'], a['A'].corr(b['A']))
        assert 'B' not in dropped

        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

        # non time-series data
        index = ['a', 'b', 'c', 'd', 'e']
        columns = ['one', 'two', 'three', 'four']
        df1 = DataFrame(np.random.randn(5, 4), index=index, columns=columns)
        df2 = DataFrame(np.random.randn(4, 4),
                        index=index[:4], columns=columns)
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row],
                                   df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self):
        df1 = tm.makeTimeDataFrame()
        df2 = tm.makeTimeDataFrame()
        cols = ['A', 'B', 'C', 'D']

        df1['obj'] = 'foo'
        df2['obj'] = 'bar'

        result = df1.corrwith(df2)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        result = df1.corrwith(df2, axis=1)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        result = datetime_frame.corrwith(datetime_frame['A'])
        expected = datetime_frame.apply(datetime_frame['A'].corr)

        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        df1 = DataFrame(np.arange(10000), columns=['a'])
        df2 = DataFrame(np.arange(10000) ** 2, columns=['a'])
        c1 = df1.corrwith(df2)['a']
        c2 = np.corrcoef(df1['a'], df2['a'])[0][1]

        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    def test_corrwith_mixed_dtypes(self):
        # GH 18570
        df = pd.DataFrame({'a': [1, 4, 3, 2], 'b': [4, 6, 7, 3],
                           'c': ['a', 'b', 'c', 'd']})
        s = pd.Series([0, 6, 7, 3])
        result = df.corrwith(s)
        corrs = [df['a'].corr(s), df['b'].corr(s)]
        expected = pd.Series(data=corrs, index=['a', 'b'])
        tm.assert_series_equal(result, expected)

    def test_corrwith_index_intersection(self):
        df1 = pd.DataFrame(np.random.random(size=(10, 2)),
                           columns=["a", "b"])
        df2 = pd.DataFrame(np.random.random(size=(10, 3)),
                           columns=["a", "b", "c"])

        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        df1 = pd.DataFrame(np.random.random(size=(10, 2)),
                           columns=["a", "b"])
        df2 = pd.DataFrame(np.random.random(size=(10, 3)),
                           columns=["a", "b", "c"])

        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        # GH 21925
        df1 = pd.DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        result = df1.corrwith(df2)
        expected = pd.Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_corrwith_spearman(self):
        # GH 21925
        df = pd.DataFrame(np.random.random(size=(100, 3)))
        result = df.corrwith(df**2, method="spearman")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_corrwith_kendall(self):
        # GH 21925
        df = pd.DataFrame(np.random.random(size=(100, 3)))
        result = df.corrwith(df**2, method="kendall")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    def test_bool_describe_in_mixed_frame(self):
        df = DataFrame({
            'string_data': ['a', 'b', 'c', 'd', 'e'],
            'bool_data': [True, True, False, False, False],
            'int_data': [10, 20, 30, 40, 50],
        })

        # Integer data are included in .describe() output,
        # Boolean and string data are not.
        result = df.describe()
        expected = DataFrame({'int_data': [5, 30, df.int_data.std(),
                                           10, 20, 30, 40, 50]},
                             index=['count', 'mean', 'std', 'min', '25%',
                                    '50%', '75%', 'max'])
        tm.assert_frame_equal(result, expected)

        # Top value is a boolean value that is False
        result = df.describe(include=['bool'])

        expected = DataFrame({'bool_data': [5, 2, False, 3]},
                             index=['count', 'unique', 'top', 'freq'])
        tm.assert_frame_equal(result, expected)

    def test_describe_bool_frame(self):
        # GH 13891
        df = pd.DataFrame({
            'bool_data_1': [False, False, True, True],
            'bool_data_2': [False, True, True, True]
        })
        result = df.describe()
        expected = DataFrame({'bool_data_1': [4, 2, True, 2],
                              'bool_data_2': [4, 2, True, 3]},
                             index=['count', 'unique', 'top', 'freq'])
        tm.assert_frame_equal(result, expected)

        df = pd.DataFrame({
            'bool_data': [False, False, True, True, False],
            'int_data': [0, 1, 2, 3, 4]
        })
        result = df.describe()
        expected = DataFrame({'int_data': [5, 2, df.int_data.std(), 0, 1,
                                           2, 3, 4]},
                             index=['count', 'mean', 'std', 'min', '25%',
                                    '50%', '75%', 'max'])
        tm.assert_frame_equal(result, expected)

        df = pd.DataFrame({
            'bool_data': [False, False, True, True],
            'str_data': ['a', 'b', 'c', 'a']
        })
        result = df.describe()
        expected = DataFrame({'bool_data': [4, 2, True, 2],
                              'str_data': [4, 3, 'a', 2]},
                             index=['count', 'unique', 'top', 'freq'])
        tm.assert_frame_equal(result, expected)

    def test_describe_categorical(self):
        df = DataFrame({'value': np.random.randint(0, 10000, 100)})
        labels = ["{0} - {1}".format(i, i + 499) for i in range(0, 10000, 500)]
        cat_labels = Categorical(labels, labels)

        df = df.sort_values(by=['value'], ascending=True)
        df['value_group'] = pd.cut(df.value, range(0, 10500, 500),
                                   right=False, labels=cat_labels)
        cat = df

        # Categoricals should not show up together with numerical columns
        result = cat.describe()
        assert len(result.columns) == 1

        # In a frame, describe() for the cat should be the same as for string
        # arrays (count, unique, top, freq)

        cat = Categorical(["a", "b", "b", "b"], categories=['a', 'b', 'c'],
                          ordered=True)
        s = Series(cat)
        result = s.describe()
        expected = Series([4, 2, "b", 3],
                          index=['count', 'unique', 'top', 'freq'])
        tm.assert_series_equal(result, expected)

        cat = Series(Categorical(["a", "b", "c", "c"]))
        df3 = DataFrame({"cat": cat, "s": ["a", "b", "c", "c"]})
        result = df3.describe()
        tm.assert_numpy_array_equal(result["cat"].values, result["s"].values)

    def test_describe_categorical_columns(self):
        # GH 11558
        columns = pd.CategoricalIndex(['int1', 'int2', 'obj'],
                                      ordered=True, name='XXX')
        df = DataFrame({'int1': [10, 20, 30, 40, 50],
                        'int2': [10, 20, 30, 40, 50],
                        'obj': ['A', 0, None, 'X', 1]},
                       columns=columns)
        result = df.describe()

        exp_columns = pd.CategoricalIndex(['int1', 'int2'],
                                          categories=['int1', 'int2', 'obj'],
                                          ordered=True, name='XXX')
        expected = DataFrame({'int1': [5, 30, df.int1.std(),
                                       10, 20, 30, 40, 50],
                              'int2': [5, 30, df.int2.std(),
                                       10, 20, 30, 40, 50]},
                             index=['count', 'mean', 'std', 'min', '25%',
                                    '50%', '75%', 'max'],
                             columns=exp_columns)
        tm.assert_frame_equal(result, expected)
        tm.assert_categorical_equal(result.columns.values,
                                    expected.columns.values)

    def test_describe_datetime_columns(self):
        columns = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'],
                                   freq='MS', tz='US/Eastern', name='XXX')
        df = DataFrame({0: [10, 20, 30, 40, 50],
                        1: [10, 20, 30, 40, 50],
                        2: ['A', 0, None, 'X', 1]})
        df.columns = columns
        result = df.describe()

        exp_columns = pd.DatetimeIndex(['2011-01-01', '2011-02-01'],
                                       freq='MS', tz='US/Eastern', name='XXX')
        expected = DataFrame({0: [5, 30, df.iloc[:, 0].std(),
                                  10, 20, 30, 40, 50],
                              1: [5, 30, df.iloc[:, 1].std(),
                                  10, 20, 30, 40, 50]},
                             index=['count', 'mean', 'std', 'min', '25%',
                                    '50%', '75%', 'max'])
        expected.columns = exp_columns
        tm.assert_frame_equal(result, expected)
        assert result.columns.freq == 'MS'
        assert result.columns.tz == expected.columns.tz

    def test_describe_timedelta_values(self):
        # GH 6145
        t1 = pd.timedelta_range('1 days', freq='D', periods=5)
        t2 = pd.timedelta_range('1 hours', freq='H', periods=5)
        df = pd.DataFrame({'t1': t1, 't2': t2})

        expected = DataFrame({'t1': [5, pd.Timedelta('3 days'),
                                     df.iloc[:, 0].std(),
                                     pd.Timedelta('1 days'),
                                     pd.Timedelta('2 days'),
                                     pd.Timedelta('3 days'),
                                     pd.Timedelta('4 days'),
                                     pd.Timedelta('5 days')],
                              't2': [5, pd.Timedelta('3 hours'),
                                     df.iloc[:, 1].std(),
                                     pd.Timedelta('1 hours'),
                                     pd.Timedelta('2 hours'),
                                     pd.Timedelta('3 hours'),
                                     pd.Timedelta('4 hours'),
                                     pd.Timedelta('5 hours')]},
                             index=['count', 'mean', 'std', 'min', '25%',
                                    '50%', '75%', 'max'])

        result = df.describe()
        tm.assert_frame_equal(result, expected)

        exp_repr = ("                           t1                      t2\n"
                    "count                       5                       5\n"
                    "mean          3 days 00:00:00         0 days 03:00:00\n"
                    "std    1 days 13:56:50.394919  0 days 01:34:52.099788\n"
                    "min           1 days 00:00:00         0 days 01:00:00\n"
                    "25%           2 days 00:00:00         0 days 02:00:00\n"
                    "50%           3 days 00:00:00         0 days 03:00:00\n"
                    "75%           4 days 00:00:00         0 days 04:00:00\n"
                    "max           5 days 00:00:00         0 days 05:00:00")
        assert repr(result) == exp_repr

    def test_describe_tz_values(self, tz_naive_fixture):
        # GH 21332
        tz = tz_naive_fixture
        s1 = Series(range(5))
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s2 = Series(date_range(start, end, tz=tz))
        df = pd.DataFrame({'s1': s1, 's2': s2})

        expected = DataFrame({'s1': [5, np.nan, np.nan, np.nan, np.nan, np.nan,
                                     2, 1.581139, 0, 1, 2, 3, 4],
                              's2': [5, 5, s2.value_counts().index[0], 1,
                                     start.tz_localize(tz),
                                     end.tz_localize(tz), np.nan, np.nan,
                                     np.nan, np.nan, np.nan, np.nan, np.nan]},
                             index=['count', 'unique', 'top', 'freq', 'first',
                                    'last', 'mean', 'std', 'min', '25%', '50%',
                                    '75%', 'max']
                             )
        result = df.describe(include='all')
        tm.assert_frame_equal(result, expected)

    def test_reduce_mixed_frame(self):
        # GH 6806
        df = DataFrame({
            'bool_data': [True, True, False, False, False],
            'int_data': [10, 20, 30, 40, 50],
            'string_data': ['a', 'b', 'c', 'd', 'e'],
        })
        df.reindex(columns=['bool_data', 'int_data', 'string_data'])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(test.values,
                                    np.array([2, 150, 'abcde'], dtype=object))
        tm.assert_series_equal(test, df.T.sum(axis=1))

    def test_count(self, float_frame_with_na, float_frame, float_string_frame):
        f = lambda s: notna(s).sum()
        assert_stat_op_calc('count', f, float_frame_with_na, has_skipna=False,
                            check_dtype=False, check_dates=True)
        assert_stat_op_api('count', float_frame, float_string_frame,
                           has_numeric_only=True)

        # corner case
        frame = DataFrame()
        ct1 = frame.count(1)
        assert isinstance(ct1, Series)

        ct2 = frame.count(0)
        assert isinstance(ct2, Series)

        # GH 423
        df = DataFrame(index=lrange(10))
        result = df.count(1)
        expected = Series(0, index=df.index)
        tm.assert_series_equal(result, expected)

        df = DataFrame(columns=lrange(10))
        result = df.count(0)
        expected = Series(0, index=df.columns)
        tm.assert_series_equal(result, expected)

        df = DataFrame()
        result = df.count()
        expected = Series(0, index=[])
        tm.assert_series_equal(result, expected)

    def test_nunique(self, float_frame_with_na, float_frame,
                     float_string_frame):
        f = lambda s: len(algorithms.unique1d(s.dropna()))
        assert_stat_op_calc('nunique', f, float_frame_with_na,
                            has_skipna=False, check_dtype=False,
                            check_dates=True)
        assert_stat_op_api('nunique', float_frame, float_string_frame)

        df = DataFrame({'A': [1, 1, 1],
                        'B': [1, 2, 3],
                        'C': [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({'A': 1, 'B': 3, 'C': 2}))
        tm.assert_series_equal(df.nunique(dropna=False),
                               Series({'A': 1, 'B': 3, 'C': 3}))
        tm.assert_series_equal(df.nunique(axis=1), Series({0: 1, 1: 2, 2: 2}))
        tm.assert_series_equal(df.nunique(axis=1, dropna=False),
                               Series({0: 1, 1: 3, 2: 2}))

    def test_sum(self, float_frame_with_na, mixed_float_frame,
                 float_frame, float_string_frame):
        assert_stat_op_api('sum', float_frame, float_string_frame,
                           has_numeric_only=True)
        assert_stat_op_calc('sum', np.sum, float_frame_with_na,
                            skipna_alternative=np.nansum)
        # mixed types (with upcasting happening)
        assert_stat_op_calc('sum', np.sum, mixed_float_frame.astype('float32'),
                            check_dtype=False, check_less_precise=True)

    @pytest.mark.parametrize('method', ['sum', 'mean', 'prod', 'var',
                                        'std', 'skew', 'min', 'max'])
    def test_stat_operators_attempt_obj_array(self, method):
        # GH 676
        data = {
            'a': [-0.00049987540199591344, -0.0016467257772919831,
                  0.00067695870775883013],
            'b': [-0, -0, 0.0],
            'c': [0.00031111847529610595, 0.0014902627951905339,
                  -0.00094099200035979691]
        }
        df1 = DataFrame(data, index=['foo', 'bar', 'baz'], dtype='O')

        df2 = DataFrame({0: [np.nan, 2], 1: [np.nan, 3],
                         2: [np.nan, 4]}, dtype=object)

        for df in [df1, df2]:
            assert df.values.dtype == np.object_
            result = getattr(df, method)(1)
            expected = getattr(df.astype('f8'), method)(1)

            if method in ['sum', 'prod']:
                tm.assert_series_equal(result, expected)

    def test_mean(self, float_frame_with_na, float_frame, float_string_frame):
        assert_stat_op_calc('mean', np.mean, float_frame_with_na,
                            check_dates=True)
        assert_stat_op_api('mean', float_frame, float_string_frame)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_mixed_datetime_numeric(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        df = pd.DataFrame({"A": [1, 1],
                           "B": [pd.Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = pd.Series([1.0], index=['A'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_excludeds_datetimes(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        # Our long-term desired behavior is unclear, but the behavior in
        # 0.24.0rc1 was buggy.
        df = pd.DataFrame({"A": [pd.Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = pd.Series()
        tm.assert_series_equal(result, expected)

    def test_product(self, float_frame_with_na, float_frame,
                     float_string_frame):
        assert_stat_op_calc('product', np.prod, float_frame_with_na)
        assert_stat_op_api('product', float_frame, float_string_frame)

    # TODO: Ensure warning isn't emitted in the first place
    @pytest.mark.filterwarnings("ignore:All-NaN:RuntimeWarning")
    def test_median(self, float_frame_with_na, float_frame,
                    float_string_frame):
        def wrapper(x):
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc('median', wrapper, float_frame_with_na,
                            check_dates=True)
        assert_stat_op_api('median', float_frame, float_string_frame)

    def test_min(self, float_frame_with_na, int_frame,
                 float_frame, float_string_frame):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_stat_op_calc('min', np.min, float_frame_with_na,
                                check_dates=True)
        assert_stat_op_calc('min', np.min, int_frame)
        assert_stat_op_api('min', float_frame, float_string_frame)

    def test_cummin(self, datetime_frame):
        datetime_frame.loc[5:10, 0] = np.nan
        datetime_frame.loc[10:15, 1] = np.nan
        datetime_frame.loc[15:, 2] = np.nan

        # axis = 0
        cummin = datetime_frame.cummin()
        expected = datetime_frame.apply(Series.cummin)
        tm.assert_frame_equal(cummin, expected)

        # axis = 1
        cummin = datetime_frame.cummin(axis=1)
        expected = datetime_frame.apply(Series.cummin, axis=1)
        tm.assert_frame_equal(cummin, expected)

        # it works
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        result = df.cummin()  # noqa

        # fix issue
        cummin_xs = datetime_frame.cummin(axis=1)
        assert np.shape(cummin_xs) == np.shape(datetime_frame)

    def test_cummax(self, datetime_frame):
        datetime_frame.loc[5:10, 0] = np.nan
        datetime_frame.loc[10:15, 1] = np.nan
        datetime_frame.loc[15:, 2] = np.nan

        # axis = 0
        cummax = datetime_frame.cummax()
        expected = datetime_frame.apply(Series.cummax)
        tm.assert_frame_equal(cummax, expected)

        # axis = 1
        cummax = datetime_frame.cummax(axis=1)
        expected = datetime_frame.apply(Series.cummax, axis=1)
        tm.assert_frame_equal(cummax, expected)

        # it works
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        result = df.cummax()  # noqa

        # fix issue
        cummax_xs = datetime_frame.cummax(axis=1)
        assert np.shape(cummax_xs) == np.shape(datetime_frame)

    def test_max(self, float_frame_with_na, int_frame,
                 float_frame, float_string_frame):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_stat_op_calc('max', np.max, float_frame_with_na,
                                check_dates=True)
        assert_stat_op_calc('max', np.max, int_frame)
        assert_stat_op_api('max', float_frame, float_string_frame)

    def test_mad(self, float_frame_with_na, float_frame, float_string_frame):
        f = lambda x: np.abs(x - x.mean()).mean()
        assert_stat_op_calc('mad', f, float_frame_with_na)
        assert_stat_op_api('mad', float_frame, float_string_frame)

    def test_var_std(self, float_frame_with_na, datetime_frame, float_frame,
                     float_string_frame):
        alt = lambda x: np.var(x, ddof=1)
        assert_stat_op_calc('var', alt, float_frame_with_na)
        assert_stat_op_api('var', float_frame, float_string_frame)

        alt = lambda x: np.std(x, ddof=1)
        assert_stat_op_calc('std', alt, float_frame_with_na)
        assert_stat_op_api('std', float_frame, float_string_frame)

        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)

        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context('use_bottleneck', False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize(
        "meth", ['sem', 'var', 'std'])
    def test_numeric_only_flag(self, meth):
        # GH 9201
        df1 = DataFrame(np.random.randn(5, 3), columns=['foo', 'bar', 'baz'])
        # set one entry to a number in str format
        df1.loc[0, 'foo'] = '100'

        df2 = DataFrame(np.random.randn(5, 3), columns=['foo', 'bar', 'baz'])
        # set one entry to a non-number str
        df2.loc[0, 'foo'] = 'a'

        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        # df1 has all numbers, df2 has a letter inside
        pytest.raises(TypeError, lambda: getattr(df1, meth)(
            axis=1, numeric_only=False))
        pytest.raises(TypeError, lambda: getattr(df2, meth)(
            axis=1, numeric_only=False))

    @pytest.mark.parametrize('op', ['mean', 'std', 'var',
                                    'skew', 'kurt', 'sem'])
    def test_mixed_ops(self, op):
        # GH 16116
        df = DataFrame({'int': [1, 2, 3, 4],
                        'float': [1., 2., 3., 4.],
                        'str': ['a', 'b', 'c', 'd']})

        result = getattr(df, op)()
        assert len(result) == 2

        with pd.option_context('use_bottleneck', False):
            result = getattr(df, op)()
            assert len(result) == 2

    def test_cumsum(self, datetime_frame):
        datetime_frame.loc[5:10, 0] = np.nan
        datetime_frame.loc[10:15, 1] = np.nan
        datetime_frame.loc[15:, 2] = np.nan

        # axis = 0
        cumsum = datetime_frame.cumsum()
        expected = datetime_frame.apply(Series.cumsum)
        tm.assert_frame_equal(cumsum, expected)

        # axis = 1
        cumsum = datetime_frame.cumsum(axis=1)
        expected = datetime_frame.apply(Series.cumsum, axis=1)
        tm.assert_frame_equal(cumsum, expected)

        # works
        df = DataFrame({'A': np.arange(20)}, index=np.arange(20))
        result = df.cumsum()  # noqa

        # fix issue
        cumsum_xs = datetime_frame.cumsum(axis=1)
        assert np.shape(cumsum_xs) == np.shape(datetime_frame)

    def test_cumprod(self, datetime_frame):
        datetime_frame.loc[5:10, 0] = np.nan
        datetime_frame.loc[10:15, 1] = np.nan
        datetime_frame.loc[15:, 2] = np.nan

        # axis = 0
        cumprod = datetime_frame.cumprod()
        expected = datetime_frame.apply(Series.cumprod)
        tm.assert_frame_equal(cumprod, expected)

        # axis = 1
        cumprod = datetime_frame.cumprod(axis=1)
        expected = datetime_frame.apply(Series.cumprod, axis=1)
        tm.assert_frame_equal(cumprod, expected)

        # fix issue
        cumprod_xs = datetime_frame.cumprod(axis=1)
        assert np.shape(cumprod_xs) == np.shape(datetime_frame)

        # ints
        df = datetime_frame.fillna(0).astype(int)
        df.cumprod(0)
        df.cumprod(1)

        # ints32
        df = datetime_frame.fillna(0).astype(np.int32)
        df.cumprod(0)
        df.cumprod(1)

    def test_sem(self, float_frame_with_na, datetime_frame,
                 float_frame, float_string_frame):
        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        assert_stat_op_calc('sem', alt, float_frame_with_na)
        assert_stat_op_api('sem', float_frame, float_string_frame)

        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(
            lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context('use_bottleneck', False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @td.skip_if_no_scipy
    def test_skew(self, float_frame_with_na, float_frame, float_string_frame):
        from scipy.stats import skew

        def alt(x):
            if len(x) < 3:
                return np.nan
            return skew(x, bias=False)

        assert_stat_op_calc('skew', alt, float_frame_with_na)
        assert_stat_op_api('skew', float_frame, float_string_frame)

    @td.skip_if_no_scipy
    def test_kurt(self, float_frame_with_na, float_frame, float_string_frame):
        from scipy.stats import kurtosis

        def alt(x):
            if len(x) < 4:
                return np.nan
            return kurtosis(x, bias=False)

        assert_stat_op_calc('kurt', alt, float_frame_with_na)
        assert_stat_op_api('kurt', float_frame, float_string_frame)

        index = MultiIndex(levels=[['bar'], ['one', 'two', 'three'], [0, 1]],
                           codes=[[0, 0, 0, 0, 0, 0],
                                  [0, 1, 2, 0, 1, 2],
                                  [0, 1, 0, 1, 0, 1]])
        df = DataFrame(np.random.randn(6, 3), index=index)

        kurt = df.kurt()
        kurt2 = df.kurt(level=0).xs('bar')
        tm.assert_series_equal(kurt, kurt2, check_names=False)
        assert kurt.name is None
        assert kurt2.name == 'bar'

    @pytest.mark.parametrize("dropna, expected", [
        (True, {'A': [12],
                'B': [10.0],
                'C': [1.0],
                'D': ['a'],
                'E': Categorical(['a'], categories=['a']),
                'F': to_datetime(['2000-1-2']),
                'G': to_timedelta(['1 days'])}),
        (False, {'A': [12],
                 'B': [10.0],
                 'C': [np.nan],
                 'D': np.array([np.nan], dtype=object),
                 'E': Categorical([np.nan], categories=['a']),
                 'F': [pd.NaT],
                 'G': to_timedelta([pd.NaT])}),
        (True, {'H': [8, 9, np.nan, np.nan],
                'I': [8, 9, np.nan, np.nan],
                'J': [1, np.nan, np.nan, np.nan],
                'K': Categorical(['a', np.nan, np.nan, np.nan],
                                 categories=['a']),
                'L': to_datetime(['2000-1-2', 'NaT', 'NaT', 'NaT']),
                'M': to_timedelta(['1 days', 'nan', 'nan', 'nan']),
                'N': [0, 1, 2, 3]}),
        (False, {'H': [8, 9, np.nan, np.nan],
                 'I': [8, 9, np.nan, np.nan],
                 'J': [1, np.nan, np.nan, np.nan],
                 'K': Categorical([np.nan, 'a', np.nan, np.nan],
                                  categories=['a']),
                 'L': to_datetime(['NaT', '2000-1-2', 'NaT', 'NaT']),
                 'M': to_timedelta(['nan', '1 days', 'nan', 'nan']),
                 'N': [0, 1, 2, 3]})
    ])
    def test_mode_dropna(self, dropna, expected):

        df = DataFrame({"A": [12, 12, 19, 11],
                        "B": [10, 10, np.nan, 3],
                        "C": [1, np.nan, np.nan, np.nan],
                        "D": [np.nan, np.nan, 'a', np.nan],
                        "E": Categorical([np.nan, np.nan, 'a', np.nan]),
                        "F": to_datetime(['NaT', '2000-1-2', 'NaT', 'NaT']),
                        "G": to_timedelta(['1 days', 'nan', 'nan', 'nan']),
                        "H": [8, 8, 9, 9],
                        "I": [9, 9, 8, 8],
                        "J": [1, 1, np.nan, np.nan],
                        "K": Categorical(['a', np.nan, 'a', np.nan]),
                        "L": to_datetime(['2000-1-2', '2000-1-2',
                                          'NaT', 'NaT']),
                        "M": to_timedelta(['1 days', 'nan',
                                           '1 days', 'nan']),
                        "N": np.arange(4, dtype='int64')})

        result = df[sorted(list(expected.keys()))].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(not compat.PY3, reason="only PY3")
    def test_mode_sortwarning(self):
        # Check for the warning that is raised when the mode
        # results cannot be sorted

        df = DataFrame({"A": [np.nan, np.nan, 'a', 'a']})
        expected = DataFrame({'A': ['a', np.nan]})

        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            result = df.mode(dropna=False)
            result = result.sort_values(by='A').reset_index(drop=True)

        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self):
        df = DataFrame(dict(A=date_range('2012-1-1', periods=3, freq='D'),
                            B=date_range('2012-1-2', periods=3, freq='D'),
                            C=Timestamp('20120101') -
                            timedelta(minutes=5, seconds=5)))

        diffs = DataFrame(dict(A=df['A'] - df['C'],
                               B=df['A'] - df['B']))

        # min
        result = diffs.min()
        assert result[0] == diffs.loc[0, 'A']
        assert result[1] == diffs.loc[0, 'B']

        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, 'B']).all()

        # max
        result = diffs.max()
        assert result[0] == diffs.loc[2, 'A']
        assert result[1] == diffs.loc[2, 'B']

        result = diffs.max(axis=1)
        assert (result == diffs['A']).all()

        # abs
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame(dict(A=df['A'] - df['C'],
                                  B=df['B'] - df['A']))
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # mixed frame
        mixed = diffs.copy()
        mixed['C'] = 'foo'
        mixed['D'] = 1
        mixed['E'] = 1.
        mixed['F'] = Timestamp('20130101')

        # results in an object array
        result = mixed.min()
        expected = Series([pd.Timedelta(timedelta(seconds=5 * 60 + 5)),
                           pd.Timedelta(timedelta(days=-1)),
                           'foo', 1, 1.0,
                           Timestamp('20130101')],
                          index=mixed.columns)
        tm.assert_series_equal(result, expected)

        # excludes numeric
        result = mixed.min(axis=1)
        expected = Series([1, 1, 1.], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        # works when only those columns are selected
        result = mixed[['A', 'B']].min(1)
        expected = Series([timedelta(days=-1)] * 3)
        tm.assert_series_equal(result, expected)

        result = mixed[['A', 'B']].min()
        expected = Series([timedelta(seconds=5 * 60 + 5),
                           timedelta(days=-1)], index=['A', 'B'])
        tm.assert_series_equal(result, expected)

        # GH 3106
        df = DataFrame({'time': date_range('20130102', periods=5),
                        'time2': date_range('20130105', periods=5)})
        df['off1'] = df['time2'] - df['time']
        assert df['off1'].dtype == 'timedelta64[ns]'

        df['off2'] = df['time'] - df['time2']
        df._consolidate_inplace()
        assert df['off1'].dtype == 'timedelta64[ns]'
        assert df['off2'].dtype == 'timedelta64[ns]'

    def test_sum_corner(self, empty_frame):
        axis0 = empty_frame.sum(0)
        axis1 = empty_frame.sum(1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize('method, unit', [
        ('sum', 0),
        ('prod', 1),
    ])
    def test_sum_prod_nanops(self, method, unit):
        idx = ['a', 'b', 'c']
        df = pd.DataFrame({"a": [unit, unit],
                           "b": [unit, np.nan],
                           "c": [np.nan, np.nan]})
        # The default
        result = getattr(df, method)
        expected = pd.Series([unit, unit, unit], index=idx, dtype='float64')

        # min_count=1
        result = getattr(df, method)(min_count=1)
        expected = pd.Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = getattr(df, method)(min_count=0)
        expected = pd.Series([unit, unit, unit], index=idx, dtype='float64')
        tm.assert_series_equal(result, expected)

        result = getattr(df.iloc[1:], method)(min_count=1)
        expected = pd.Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count > 1
        df = pd.DataFrame({"A": [unit] * 10, "B": [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(min_count=5)
        expected = pd.Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)

        result = getattr(df, method)(min_count=6)
        expected = pd.Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self):
        # prod isn't defined on timedeltas
        idx = ['a', 'b', 'c']
        df = pd.DataFrame({"a": [0, 0],
                           "b": [0, np.nan],
                           "c": [np.nan, np.nan]})

        df2 = df.apply(pd.to_timedelta)

        # 0 by default
        result = df2.sum()
        expected = pd.Series([0, 0, 0], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = df2.sum(min_count=1)
        expected = pd.Series([0, 0, np.nan], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index,
                          columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame):
        # ensure this works, bug report
        bools = np.isnan(float_frame)
        bools.sum(1)
        bools.sum(0)

    def test_mean_corner(self, float_frame, float_string_frame):
        # unit test when have object data
        the_mean = float_string_frame.mean(axis=0)
        the_sum = float_string_frame.sum(axis=0, numeric_only=True)
        tm.assert_index_equal(the_sum.index, the_mean.index)
        assert len(the_mean.index) < len(float_string_frame.columns)

        # xs sum mixed type, just want to know it works...
        the_mean = float_string_frame.mean(axis=1)
        the_sum = float_string_frame.sum(axis=1, numeric_only=True)
        tm.assert_index_equal(the_sum.index, the_mean.index)

        # take mean of boolean column
        float_frame['bool'] = float_frame['A'] > 0
        means = float_frame.mean(0)
        assert means['bool'] == float_frame['bool'].values.mean()

    def test_stats_mixed_type(self, float_string_frame):
        # don't blow up
        float_string_frame.std(1)
        float_string_frame.var(1)
        float_string_frame.mean(1)
        float_string_frame.skew(1)

    # TODO: Ensure warning isn't emitted in the first place
    @pytest.mark.filterwarnings("ignore:All-NaN:RuntimeWarning")
    def test_median_corner(self, int_frame, float_frame, float_string_frame):
        def wrapper(x):
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc('median', wrapper, int_frame, check_dtype=False,
                            check_dates=True)
        assert_stat_op_api('median', float_frame, float_string_frame)

    # Miscellanea

    def test_count_objects(self, float_string_frame):
        dm = DataFrame(float_string_frame._series)
        df = DataFrame(float_string_frame._series)

        tm.assert_series_equal(dm.count(), df.count())
        tm.assert_series_equal(dm.count(1), df.count(1))

    def test_cumsum_corner(self):
        dm = DataFrame(np.arange(20).reshape(4, 5),
                       index=lrange(4), columns=lrange(5))
        # ?(wesm)
        result = dm.cumsum()  # noqa

    def test_sum_bools(self):
        df = DataFrame(index=lrange(1), columns=lrange(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    # Index of max / min

    def test_idxmin(self, float_frame, int_frame):
        frame = float_frame
        frame.loc[5:10] = np.nan
        frame.loc[15:20, -2:] = np.nan
        for skipna in [True, False]:
            for axis in [0, 1]:
                for df in [frame, int_frame]:
                    result = df.idxmin(axis=axis, skipna=skipna)
                    expected = df.apply(Series.idxmin, axis=axis,
                                        skipna=skipna)
                    tm.assert_series_equal(result, expected)

        pytest.raises(ValueError, frame.idxmin, axis=2)

    def test_idxmax(self, float_frame, int_frame):
        frame = float_frame
        frame.loc[5:10] = np.nan
        frame.loc[15:20, -2:] = np.nan
        for skipna in [True, False]:
            for axis in [0, 1]:
                for df in [frame, int_frame]:
                    result = df.idxmax(axis=axis, skipna=skipna)
                    expected = df.apply(Series.idxmax, axis=axis,
                                        skipna=skipna)
                    tm.assert_series_equal(result, expected)

        pytest.raises(ValueError, frame.idxmax, axis=2)

    # ----------------------------------------------------------------------
    # Logical reductions

    @pytest.mark.parametrize('opname', ['any', 'all'])
    def test_any_all(self, opname, bool_frame_with_na, float_string_frame):
        assert_bool_op_calc(opname, getattr(np, opname), bool_frame_with_na,
                            has_skipna=True)
        assert_bool_op_api(opname, bool_frame_with_na, float_string_frame,
                           has_bool_only=True)

    def test_any_all_extra(self):
        df = DataFrame({
            'A': [True, False, False],
            'B': [True, True, False],
            'C': [True, True, True],
        }, index=['a', 'b', 'c'])
        result = df[['A', 'B']].any(1)
        expected = Series([True, True, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

        result = df[['A', 'B']].any(1, bool_only=True)
        tm.assert_series_equal(result, expected)

        result = df.all(1)
        expected = Series([True, False, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

        result = df.all(1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # Axis is None
        result = df.all(axis=None).item()
        assert result is False

        result = df.any(axis=None).item()
        assert result is True

        result = df[['C']].all(axis=None).item()
        assert result is True

    def test_any_datetime(self):

        # GH 23070
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [pd.Timestamp('1960-02-15'),
                         pd.Timestamp('1960-02-16'),
                         pd.NaT,
                         pd.NaT]
        df = DataFrame({
            "A": float_data,
            "B": datetime_data
        })

        result = df.any(1)
        expected = Series([True, True, True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_bool_only(self):

        # GH 25101
        df = DataFrame({"col1": [1, 2, 3],
                        "col2": [4, 5, 6],
                        "col3": [None, None, None]})

        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool)
        tm.assert_series_equal(result, expected)

        df = DataFrame({"col1": [1, 2, 3],
                        "col2": [4, 5, 6],
                        "col3": [None, None, None],
                        "col4": [False, False, True]})

        result = df.all(bool_only=True)
        expected = Series({"col4": False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func, data, expected', [
        (np.any, {}, False),
        (np.all, {}, True),
        (np.any, {'A': []}, False),
        (np.all, {'A': []}, True),
        (np.any, {'A': [False, False]}, False),
        (np.all, {'A': [False, False]}, False),
        (np.any, {'A': [True, False]}, True),
        (np.all, {'A': [True, False]}, False),
        (np.any, {'A': [True, True]}, True),
        (np.all, {'A': [True, True]}, True),

        (np.any, {'A': [False], 'B': [False]}, False),
        (np.all, {'A': [False], 'B': [False]}, False),

        (np.any, {'A': [False, False], 'B': [False, True]}, True),
        (np.all, {'A': [False, False], 'B': [False, True]}, False),

        # other types
        (np.all, {'A': pd.Series([0.0, 1.0], dtype='float')}, False),
        (np.any, {'A': pd.Series([0.0, 1.0], dtype='float')}, True),
        (np.all, {'A': pd.Series([0, 1], dtype=int)}, False),
        (np.any, {'A': pd.Series([0, 1], dtype=int)}, True),
        pytest.param(np.all, {'A': pd.Series([0, 1], dtype='M8[ns]')}, False,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.any, {'A': pd.Series([0, 1], dtype='M8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.all, {'A': pd.Series([1, 2], dtype='M8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.any, {'A': pd.Series([1, 2], dtype='M8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.all, {'A': pd.Series([0, 1], dtype='m8[ns]')}, False,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.any, {'A': pd.Series([0, 1], dtype='m8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.all, {'A': pd.Series([1, 2], dtype='m8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        pytest.param(np.any, {'A': pd.Series([1, 2], dtype='m8[ns]')}, True,
                     marks=[td.skip_if_np_lt_115]),
        (np.all, {'A': pd.Series([0, 1], dtype='category')}, False),
        (np.any, {'A': pd.Series([0, 1], dtype='category')}, True),
        (np.all, {'A': pd.Series([1, 2], dtype='category')}, True),
        (np.any, {'A': pd.Series([1, 2], dtype='category')}, True),

        # # Mix
        # GH 21484
        # (np.all, {'A': pd.Series([10, 20], dtype='M8[ns]'),
        #           'B': pd.Series([10, 20], dtype='m8[ns]')}, True),
    ])
    def test_any_all_np_func(self, func, data, expected):
        # GH 19976
        data = DataFrame(data)
        result = func(data)
        assert isinstance(result, np.bool_)
        assert result.item() is expected

        # method version
        result = getattr(DataFrame(data), func.__name__)(axis=None)
        assert isinstance(result, np.bool_)
        assert result.item() is expected

    def test_any_all_object(self):
        # GH 19976
        result = np.all(DataFrame(columns=['a', 'b'])).item()
        assert result is True

        result = np.any(DataFrame(columns=['a', 'b'])).item()
        assert result is False

    @pytest.mark.parametrize('method', ['any', 'all'])
    def test_any_all_level_axis_none_raises(self, method):
        df = DataFrame(
            {"A": 1},
            index=MultiIndex.from_product([['A', 'B'], ['a', 'b']],
                                          names=['out', 'in'])
        )
        xpr = "Must specify 'axis' when aggregating by level."
        with pytest.raises(ValueError, match=xpr):
            getattr(df, method)(axis=None, level='out')

    # ----------------------------------------------------------------------
    # Isin

    def test_isin(self):
        # GH 4211
        df = DataFrame({'vals': [1, 2, 3, 4], 'ids': ['a', 'b', 'f', 'n'],
                        'ids2': ['a', 'n', 'c', 'n']},
                       index=['foo', 'bar', 'baz', 'qux'])
        other = ['a', 'b', 'c']

        result = df.isin(other)
        expected = DataFrame([df.loc[s].isin(other) for s in df.index])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(), np.array([])])
    def test_isin_empty(self, empty):
        # GH 16991
        df = DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'e', 'f']})
        expected = DataFrame(False, df.index, df.columns)

        result = df.isin(empty)
        tm.assert_frame_equal(result, expected)

    def test_isin_dict(self):
        df = DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'e', 'f']})
        d = {'A': ['a']}

        expected = DataFrame(False, df.index, df.columns)
        expected.loc[0, 'A'] = True

        result = df.isin(d)
        tm.assert_frame_equal(result, expected)

        # non unique columns
        df = DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'e', 'f']})
        df.columns = ['A', 'A']
        expected = DataFrame(False, df.index, df.columns)
        expected.loc[0, 'A'] = True
        result = df.isin(d)
        tm.assert_frame_equal(result, expected)

    def test_isin_with_string_scalar(self):
        # GH 4763
        df = DataFrame({'vals': [1, 2, 3, 4], 'ids': ['a', 'b', 'f', 'n'],
                        'ids2': ['a', 'n', 'c', 'n']},
                       index=['foo', 'bar', 'baz', 'qux'])
        with pytest.raises(TypeError):
            df.isin('a')

        with pytest.raises(TypeError):
            df.isin('aaa')

    def test_isin_df(self):
        df1 = DataFrame({'A': [1, 2, 3, 4], 'B': [2, np.nan, 4, 4]})
        df2 = DataFrame({'A': [0, 2, 12, 4], 'B': [2, np.nan, 4, 5]})
        expected = DataFrame(False, df1.index, df1.columns)
        result = df1.isin(df2)
        expected['A'].loc[[1, 3]] = True
        expected['B'].loc[[0, 2]] = True
        tm.assert_frame_equal(result, expected)

        # partial overlapping columns
        df2.columns = ['A', 'C']
        result = df1.isin(df2)
        expected['B'] = False
        tm.assert_frame_equal(result, expected)

    def test_isin_tuples(self):
        # GH 16394
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'f']})
        df['C'] = list(zip(df['A'], df['B']))
        result = df['C'].isin([(1, 'a')])
        tm.assert_series_equal(result,
                               Series([True, False, False], name="C"))

    def test_isin_df_dupe_values(self):
        df1 = DataFrame({'A': [1, 2, 3, 4], 'B': [2, np.nan, 4, 4]})
        # just cols duped
        df2 = DataFrame([[0, 2], [12, 4], [2, np.nan], [4, 5]],
                        columns=['B', 'B'])
        with pytest.raises(ValueError):
            df1.isin(df2)

        # just index duped
        df2 = DataFrame([[0, 2], [12, 4], [2, np.nan], [4, 5]],
                        columns=['A', 'B'], index=[0, 0, 1, 1])
        with pytest.raises(ValueError):
            df1.isin(df2)

        # cols and index:
        df2.columns = ['B', 'B']
        with pytest.raises(ValueError):
            df1.isin(df2)

    def test_isin_dupe_self(self):
        other = DataFrame({'A': [1, 0, 1, 0], 'B': [1, 1, 0, 0]})
        df = DataFrame([[1, 1], [1, 0], [0, 0]], columns=['A', 'A'])
        result = df.isin(other)
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected.loc[0] = True
        expected.iloc[1, 1] = True
        tm.assert_frame_equal(result, expected)

    def test_isin_against_series(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, np.nan, 4, 4]},
                          index=['a', 'b', 'c', 'd'])
        s = pd.Series([1, 3, 11, 4], index=['a', 'b', 'c', 'd'])
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected['A'].loc['a'] = True
        expected.loc['d'] = True
        result = df.isin(s)
        tm.assert_frame_equal(result, expected)

    def test_isin_multiIndex(self):
        idx = MultiIndex.from_tuples([(0, 'a', 'foo'), (0, 'a', 'bar'),
                                      (0, 'b', 'bar'), (0, 'b', 'baz'),
                                      (2, 'a', 'foo'), (2, 'a', 'bar'),
                                      (2, 'c', 'bar'), (2, 'c', 'baz'),
                                      (1, 'b', 'foo'), (1, 'b', 'bar'),
                                      (1, 'c', 'bar'), (1, 'c', 'baz')])
        df1 = DataFrame({'A': np.ones(12),
                         'B': np.zeros(12)}, index=idx)
        df2 = DataFrame({'A': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                         'B': [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]})
        # against regular index
        expected = DataFrame(False, index=df1.index, columns=df1.columns)
        result = df1.isin(df2)
        tm.assert_frame_equal(result, expected)

        df2.index = idx
        expected = df2.values.astype(np.bool)
        expected[:, 1] = ~expected[:, 1]
        expected = DataFrame(expected, columns=['A', 'B'], index=idx)

        result = df1.isin(df2)
        tm.assert_frame_equal(result, expected)

    def test_isin_empty_datetimelike(self):
        # GH 15473
        df1_ts = DataFrame({'date':
                            pd.to_datetime(['2014-01-01', '2014-01-02'])})
        df1_td = DataFrame({'date':
                            [pd.Timedelta(1, 's'), pd.Timedelta(2, 's')]})
        df2 = DataFrame({'date': []})
        df3 = DataFrame()

        expected = DataFrame({'date': [False, False]})

        result = df1_ts.isin(df2)
        tm.assert_frame_equal(result, expected)
        result = df1_ts.isin(df3)
        tm.assert_frame_equal(result, expected)

        result = df1_td.isin(df2)
        tm.assert_frame_equal(result, expected)
        result = df1_td.isin(df3)
        tm.assert_frame_equal(result, expected)

    # Rounding
    def test_round(self):
        # GH 2665

        # Test that rounding an empty DataFrame does nothing
        df = DataFrame()
        tm.assert_frame_equal(df, df.round())

        # Here's the test frame we'll be working with
        df = DataFrame({'col1': [1.123, 2.123, 3.123],
                        'col2': [1.234, 2.234, 3.234]})

        # Default round to integer (i.e. decimals=0)
        expected_rounded = DataFrame(
            {'col1': [1., 2., 3.], 'col2': [1., 2., 3.]})
        tm.assert_frame_equal(df.round(), expected_rounded)

        # Round with an integer
        decimals = 2
        expected_rounded = DataFrame({'col1': [1.12, 2.12, 3.12],
                                      'col2': [1.23, 2.23, 3.23]})
        tm.assert_frame_equal(df.round(decimals), expected_rounded)

        # This should also work with np.round (since np.round dispatches to
        # df.round)
        tm.assert_frame_equal(np.round(df, decimals), expected_rounded)

        # Round with a list
        round_list = [1, 2]
        with pytest.raises(TypeError):
            df.round(round_list)

        # Round with a dictionary
        expected_rounded = DataFrame(
            {'col1': [1.1, 2.1, 3.1], 'col2': [1.23, 2.23, 3.23]})
        round_dict = {'col1': 1, 'col2': 2}
        tm.assert_frame_equal(df.round(round_dict), expected_rounded)

        # Incomplete dict
        expected_partially_rounded = DataFrame(
            {'col1': [1.123, 2.123, 3.123], 'col2': [1.2, 2.2, 3.2]})
        partial_round_dict = {'col2': 1}
        tm.assert_frame_equal(df.round(partial_round_dict),
                              expected_partially_rounded)

        # Dict with unknown elements
        wrong_round_dict = {'col3': 2, 'col2': 1}
        tm.assert_frame_equal(df.round(wrong_round_dict),
                              expected_partially_rounded)

        # float input to `decimals`
        non_int_round_dict = {'col1': 1, 'col2': 0.5}
        with pytest.raises(TypeError):
            df.round(non_int_round_dict)

        # String input
        non_int_round_dict = {'col1': 1, 'col2': 'foo'}
        with pytest.raises(TypeError):
            df.round(non_int_round_dict)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError):
            df.round(non_int_round_Series)

        # List input
        non_int_round_dict = {'col1': 1, 'col2': [1, 2]}
        with pytest.raises(TypeError):
            df.round(non_int_round_dict)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError):
            df.round(non_int_round_Series)

        # Non integer Series inputs
        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError):
            df.round(non_int_round_Series)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError):
            df.round(non_int_round_Series)

        # Negative numbers
        negative_round_dict = {'col1': -1, 'col2': -2}
        big_df = df * 100
        expected_neg_rounded = DataFrame(
            {'col1': [110., 210, 310], 'col2': [100., 200, 300]})
        tm.assert_frame_equal(big_df.round(negative_round_dict),
                              expected_neg_rounded)

        # nan in Series round
        nan_round_Series = Series({'col1': np.nan, 'col2': 1})

        # TODO(wesm): unused?
        expected_nan_round = DataFrame({  # noqa
            'col1': [1.123, 2.123, 3.123],
            'col2': [1.2, 2.2, 3.2]})

        with pytest.raises(TypeError):
            df.round(nan_round_Series)

        # Make sure this doesn't break existing Series.round
        tm.assert_series_equal(df['col1'].round(1), expected_rounded['col1'])

        # named columns
        # GH 11986
        decimals = 2
        expected_rounded = DataFrame(
            {'col1': [1.12, 2.12, 3.12], 'col2': [1.23, 2.23, 3.23]})
        df.columns.name = "cols"
        expected_rounded.columns.name = "cols"
        tm.assert_frame_equal(df.round(decimals), expected_rounded)

        # interaction of named columns & series
        tm.assert_series_equal(df['col1'].round(decimals),
                               expected_rounded['col1'])
        tm.assert_series_equal(df.round(decimals)['col1'],
                               expected_rounded['col1'])

    def test_numpy_round(self):
        # GH 12600
        df = DataFrame([[1.53, 1.36], [0.06, 7.01]])
        out = np.round(df, decimals=0)
        expected = DataFrame([[2., 1.], [0., 7.]])
        tm.assert_frame_equal(out, expected)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.round(df, decimals=0, out=df)

    def test_round_mixed_type(self):
        # GH 11885
        df = DataFrame({'col1': [1.1, 2.2, 3.3, 4.4],
                        'col2': ['1', 'a', 'c', 'f'],
                        'col3': date_range('20111111', periods=4)})
        round_0 = DataFrame({'col1': [1., 2., 3., 4.],
                             'col2': ['1', 'a', 'c', 'f'],
                             'col3': date_range('20111111', periods=4)})
        tm.assert_frame_equal(df.round(), round_0)
        tm.assert_frame_equal(df.round(1), df)
        tm.assert_frame_equal(df.round({'col1': 1}), df)
        tm.assert_frame_equal(df.round({'col1': 0}), round_0)
        tm.assert_frame_equal(df.round({'col1': 0, 'col2': 1}), round_0)
        tm.assert_frame_equal(df.round({'col3': 1}), df)

    def test_round_issue(self):
        # GH 11611

        df = pd.DataFrame(np.random.random([3, 3]), columns=['A', 'B', 'C'],
                          index=['first', 'second', 'third'])

        dfs = pd.concat((df, df), axis=1)
        rounded = dfs.round()
        tm.assert_index_equal(rounded.index, dfs.index)

        decimals = pd.Series([1, 0, 2], index=['A', 'B', 'A'])
        pytest.raises(ValueError, df.round, decimals)

    def test_built_in_round(self):
        if not compat.PY3:
            pytest.skip("build in round cannot be overridden "
                        "prior to Python 3")

        # GH 11763
        # Here's the test frame we'll be working with
        df = DataFrame(
            {'col1': [1.123, 2.123, 3.123], 'col2': [1.234, 2.234, 3.234]})

        # Default round to integer (i.e. decimals=0)
        expected_rounded = DataFrame(
            {'col1': [1., 2., 3.], 'col2': [1., 2., 3.]})
        tm.assert_frame_equal(round(df), expected_rounded)

    def test_round_nonunique_categorical(self):
        # See GH21809
        idx = pd.CategoricalIndex(['low'] * 3 + ['hi'] * 3)
        df = pd.DataFrame(np.random.rand(6, 3), columns=list('abc'))

        expected = df.round(3)
        expected.index = idx

        df_categorical = df.copy().set_index(idx)
        assert df_categorical.shape == (6, 3)
        result = df_categorical.round(3)
        assert result.shape == (6, 3)

        tm.assert_frame_equal(result, expected)

    def test_pct_change(self):
        # GH 11150
        pnl = DataFrame([np.arange(0, 40, 10), np.arange(0, 40, 10), np.arange(
            0, 40, 10)]).astype(np.float64)
        pnl.iat[1, 0] = np.nan
        pnl.iat[1, 1] = np.nan
        pnl.iat[2, 3] = 60

        for axis in range(2):
            expected = pnl.ffill(axis=axis) / pnl.ffill(axis=axis).shift(
                axis=axis) - 1
            result = pnl.pct_change(axis=axis, fill_method='pad')

            tm.assert_frame_equal(result, expected)

    # Clip
    def test_clip(self, float_frame):
        median = float_frame.median().median()
        original = float_frame.copy()

        with tm.assert_produces_warning(FutureWarning):
            capped = float_frame.clip_upper(median)
        assert not (capped.values > median).any()

        with tm.assert_produces_warning(FutureWarning):
            floored = float_frame.clip_lower(median)
        assert not (floored.values < median).any()

        double = float_frame.clip(upper=median, lower=median)
        assert not (double.values != median).any()

        # Verify that float_frame was not changed inplace
        assert (float_frame.values == original.values).all()

    def test_inplace_clip(self, float_frame):
        # GH 15388
        median = float_frame.median().median()
        frame_copy = float_frame.copy()

        with tm.assert_produces_warning(FutureWarning):
            frame_copy.clip_upper(median, inplace=True)
        assert not (frame_copy.values > median).any()
        frame_copy = float_frame.copy()

        with tm.assert_produces_warning(FutureWarning):
            frame_copy.clip_lower(median, inplace=True)
        assert not (frame_copy.values < median).any()
        frame_copy = float_frame.copy()

        frame_copy.clip(upper=median, lower=median, inplace=True)
        assert not (frame_copy.values != median).any()

    def test_dataframe_clip(self):
        # GH 2747
        df = DataFrame(np.random.randn(1000, 2))

        for lb, ub in [(-1, 1), (1, -1)]:
            clipped_df = df.clip(lb, ub)

            lb, ub = min(lb, ub), max(ub, lb)
            lb_mask = df.values <= lb
            ub_mask = df.values >= ub
            mask = ~lb_mask & ~ub_mask
            assert (clipped_df.values[lb_mask] == lb).all()
            assert (clipped_df.values[ub_mask] == ub).all()
            assert (clipped_df.values[mask] == df.values[mask]).all()

    def test_clip_mixed_numeric(self):
        # TODO(jreback)
        # clip on mixed integer or floats
        # with integer clippers coerces to float
        df = DataFrame({'A': [1, 2, 3],
                        'B': [1., np.nan, 3.]})
        result = df.clip(1, 2)
        expected = DataFrame({'A': [1, 2, 2],
                              'B': [1., np.nan, 2.]})
        tm.assert_frame_equal(result, expected, check_like=True)

        # GH 24162, clipping now preserves numeric types per column
        df = DataFrame([[1, 2, 3.4], [3, 4, 5.6]],
                       columns=['foo', 'bar', 'baz'])
        expected = df.dtypes
        result = df.clip(upper=3).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_clip_against_series(self, inplace):
        # GH 6966

        df = DataFrame(np.random.randn(1000, 2))
        lb = Series(np.random.randn(1000))
        ub = lb + 1

        original = df.copy()
        clipped_df = df.clip(lb, ub, axis=0, inplace=inplace)

        if inplace:
            clipped_df = df

        for i in range(2):
            lb_mask = original.iloc[:, i] <= lb
            ub_mask = original.iloc[:, i] >= ub
            mask = ~lb_mask & ~ub_mask

            result = clipped_df.loc[lb_mask, i]
            tm.assert_series_equal(result, lb[lb_mask], check_names=False)
            assert result.name == i

            result = clipped_df.loc[ub_mask, i]
            tm.assert_series_equal(result, ub[ub_mask], check_names=False)
            assert result.name == i

            tm.assert_series_equal(clipped_df.loc[mask, i], df.loc[mask, i])

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("lower", [[2, 3, 4], np.asarray([2, 3, 4])])
    @pytest.mark.parametrize("axis,res", [
        (0, [[2., 2., 3.], [4., 5., 6.], [7., 7., 7.]]),
        (1, [[2., 3., 4.], [4., 5., 6.], [5., 6., 7.]])
    ])
    def test_clip_against_list_like(self, simple_frame,
                                    inplace, lower, axis, res):
        # GH 15390
        original = simple_frame.copy(deep=True)

        result = original.clip(lower=lower, upper=[5, 6, 7],
                               axis=axis, inplace=inplace)

        expected = pd.DataFrame(res,
                                columns=original.columns,
                                index=original.index)
        if inplace:
            result = original
        tm.assert_frame_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize("axis", [0, 1, None])
    def test_clip_against_frame(self, axis):
        df = DataFrame(np.random.randn(1000, 2))
        lb = DataFrame(np.random.randn(1000, 2))
        ub = lb + 1

        clipped_df = df.clip(lb, ub, axis=axis)

        lb_mask = df <= lb
        ub_mask = df >= ub
        mask = ~lb_mask & ~ub_mask

        tm.assert_frame_equal(clipped_df[lb_mask], lb[lb_mask])
        tm.assert_frame_equal(clipped_df[ub_mask], ub[ub_mask])
        tm.assert_frame_equal(clipped_df[mask], df[mask])

    def test_clip_against_unordered_columns(self):
        # GH 20911
        df1 = DataFrame(np.random.randn(1000, 4), columns=['A', 'B', 'C', 'D'])
        df2 = DataFrame(np.random.randn(1000, 4), columns=['D', 'A', 'B', 'C'])
        df3 = DataFrame(df2.values - 1, columns=['B', 'D', 'C', 'A'])
        result_upper = df1.clip(lower=0, upper=df2)
        expected_upper = df1.clip(lower=0, upper=df2[df1.columns])
        result_lower = df1.clip(lower=df3, upper=3)
        expected_lower = df1.clip(lower=df3[df1.columns], upper=3)
        result_lower_upper = df1.clip(lower=df3, upper=df2)
        expected_lower_upper = df1.clip(lower=df3[df1.columns],
                                        upper=df2[df1.columns])
        tm.assert_frame_equal(result_upper, expected_upper)
        tm.assert_frame_equal(result_lower, expected_lower)
        tm.assert_frame_equal(result_lower_upper, expected_lower_upper)

    def test_clip_with_na_args(self, float_frame):
        """Should process np.nan argument as None """
        # GH 17276
        tm.assert_frame_equal(float_frame.clip(np.nan), float_frame)
        tm.assert_frame_equal(float_frame.clip(upper=np.nan, lower=np.nan),
                              float_frame)

        # GH 19992
        df = DataFrame({'col_0': [1, 2, 3], 'col_1': [4, 5, 6],
                        'col_2': [7, 8, 9]})

        result = df.clip(lower=[4, 5, np.nan], axis=0)
        expected = DataFrame({'col_0': [4, 5, np.nan], 'col_1': [4, 5, np.nan],
                              'col_2': [7, 8, np.nan]})
        tm.assert_frame_equal(result, expected)

        result = df.clip(lower=[4, 5, np.nan], axis=1)
        expected = DataFrame({'col_0': [4, 4, 4], 'col_1': [5, 5, 6],
                              'col_2': [np.nan, np.nan, np.nan]})
        tm.assert_frame_equal(result, expected)

    # Matrix-like
    def test_dot(self):
        a = DataFrame(np.random.randn(3, 4), index=['a', 'b', 'c'],
                      columns=['p', 'q', 'r', 's'])
        b = DataFrame(np.random.randn(4, 2), index=['p', 'q', 'r', 's'],
                      columns=['one', 'two'])

        result = a.dot(b)
        expected = DataFrame(np.dot(a.values, b.values),
                             index=['a', 'b', 'c'],
                             columns=['one', 'two'])
        # Check alignment
        b1 = b.reindex(index=reversed(b.index))
        result = a.dot(b)
        tm.assert_frame_equal(result, expected)

        # Check series argument
        result = a.dot(b['one'])
        tm.assert_series_equal(result, expected['one'], check_names=False)
        assert result.name is None

        result = a.dot(b1['one'])
        tm.assert_series_equal(result, expected['one'], check_names=False)
        assert result.name is None

        # can pass correct-length arrays
        row = a.iloc[0].values

        result = a.dot(row)
        expected = a.dot(a.iloc[0])
        tm.assert_series_equal(result, expected)

        with pytest.raises(ValueError, match='Dot product shape mismatch'):
            a.dot(row[:-1])

        a = np.random.rand(1, 5)
        b = np.random.rand(5, 1)
        A = DataFrame(a)

        # TODO(wesm): unused
        B = DataFrame(b)  # noqa

        # it works
        result = A.dot(b)

        # unaligned
        df = DataFrame(np.random.randn(3, 4),
                       index=[1, 2, 3], columns=lrange(4))
        df2 = DataFrame(np.random.randn(5, 3),
                        index=lrange(5), columns=[1, 2, 3])

        with pytest.raises(ValueError, match='aligned'):
            df.dot(df2)

    @pytest.mark.skipif(not PY35,
                        reason='matmul supported for Python>=3.5')
    def test_matmul(self):
        # matmul test is for GH 10259
        a = DataFrame(np.random.randn(3, 4), index=['a', 'b', 'c'],
                      columns=['p', 'q', 'r', 's'])
        b = DataFrame(np.random.randn(4, 2), index=['p', 'q', 'r', 's'],
                      columns=['one', 'two'])

        # DataFrame @ DataFrame
        result = operator.matmul(a, b)
        expected = DataFrame(np.dot(a.values, b.values),
                             index=['a', 'b', 'c'],
                             columns=['one', 'two'])
        tm.assert_frame_equal(result, expected)

        # DataFrame @ Series
        result = operator.matmul(a, b.one)
        expected = Series(np.dot(a.values, b.one.values),
                          index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

        # np.array @ DataFrame
        result = operator.matmul(a.values, b)
        assert isinstance(result, DataFrame)
        assert result.columns.equals(b.columns)
        assert result.index.equals(pd.Index(range(3)))
        expected = np.dot(a.values, b.values)
        tm.assert_almost_equal(result.values, expected)

        # nested list @ DataFrame (__rmatmul__)
        result = operator.matmul(a.values.tolist(), b)
        expected = DataFrame(np.dot(a.values, b.values),
                             index=['a', 'b', 'c'],
                             columns=['one', 'two'])
        tm.assert_almost_equal(result.values, expected.values)

        # mixed dtype DataFrame @ DataFrame
        a['q'] = a.q.round().astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(np.dot(a.values, b.values),
                             index=['a', 'b', 'c'],
                             columns=['one', 'two'])
        tm.assert_frame_equal(result, expected)

        # different dtypes DataFrame @ DataFrame
        a = a.astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(np.dot(a.values, b.values),
                             index=['a', 'b', 'c'],
                             columns=['one', 'two'])
        tm.assert_frame_equal(result, expected)

        # unaligned
        df = DataFrame(np.random.randn(3, 4),
                       index=[1, 2, 3], columns=lrange(4))
        df2 = DataFrame(np.random.randn(5, 3),
                        index=lrange(5), columns=[1, 2, 3])

        with pytest.raises(ValueError, match='aligned'):
            operator.matmul(df, df2)


@pytest.fixture
def df_duplicates():
    return pd.DataFrame({'a': [1, 2, 3, 4, 4],
                         'b': [1, 1, 1, 1, 1],
                         'c': [0, 1, 2, 5, 4]},
                        index=[0, 0, 1, 1, 1])


@pytest.fixture
def df_strings():
    return pd.DataFrame({'a': np.random.permutation(10),
                         'b': list(ascii_lowercase[:10]),
                         'c': np.random.permutation(10).astype('float64')})


@pytest.fixture
def df_main_dtypes():
    return pd.DataFrame(
        {'group': [1, 1, 2],
         'int': [1, 2, 3],
         'float': [4., 5., 6.],
         'string': list('abc'),
         'category_string': pd.Series(list('abc')).astype('category'),
         'category_int': [7, 8, 9],
         'datetime': pd.date_range('20130101', periods=3),
         'datetimetz': pd.date_range('20130101',
                                     periods=3,
                                     tz='US/Eastern'),
         'timedelta': pd.timedelta_range('1 s', periods=3, freq='s')},
        columns=['group', 'int', 'float', 'string',
                 'category_string', 'category_int',
                 'datetime', 'datetimetz',
                 'timedelta'])


class TestNLargestNSmallest(object):

    dtype_error_msg_template = ("Column {column!r} has dtype {dtype}, cannot "
                                "use method {method!r} with this dtype")

    # ----------------------------------------------------------------------
    # Top / bottom
    @pytest.mark.parametrize('order', [
        ['a'],
        ['c'],
        ['a', 'b'],
        ['a', 'c'],
        ['b', 'a'],
        ['b', 'c'],
        ['a', 'b', 'c'],
        ['c', 'a', 'b'],
        ['c', 'b', 'a'],
        ['b', 'c', 'a'],
        ['b', 'a', 'c'],

        # dups!
        ['b', 'c', 'c']])
    @pytest.mark.parametrize('n', range(1, 11))
    def test_n(self, df_strings, nselect_method, n, order):
        # GH 10393
        df = df_strings
        if 'b' in order:

            error_msg = self.dtype_error_msg_template.format(
                column='b', method=nselect_method, dtype='object')
            with pytest.raises(TypeError, match=error_msg):
                getattr(df, nselect_method)(n, order)
        else:
            ascending = nselect_method == 'nsmallest'
            result = getattr(df, nselect_method)(n, order)
            expected = df.sort_values(order, ascending=ascending).head(n)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('columns', [
        ['group', 'category_string'], ['group', 'string']])
    def test_n_error(self, df_main_dtypes, nselect_method, columns):
        df = df_main_dtypes
        col = columns[1]
        error_msg = self.dtype_error_msg_template.format(
            column=col, method=nselect_method, dtype=df[col].dtype)
        # escape some characters that may be in the repr
        error_msg = (error_msg.replace('(', '\\(').replace(")", "\\)")
                              .replace("[", "\\[").replace("]", "\\]"))
        with pytest.raises(TypeError, match=error_msg):
            getattr(df, nselect_method)(2, columns)

    def test_n_all_dtypes(self, df_main_dtypes):
        df = df_main_dtypes
        df.nsmallest(2, list(set(df) - {'category_string', 'string'}))
        df.nlargest(2, list(set(df) - {'category_string', 'string'}))

    @pytest.mark.parametrize('method,expected', [
        ('nlargest',
         pd.DataFrame({'a': [2, 2, 2, 1], 'b': [3, 2, 1, 3]},
                      index=[2, 1, 0, 3])),
        ('nsmallest',
         pd.DataFrame({'a': [1, 1, 1, 2], 'b': [1, 2, 3, 1]},
                      index=[5, 4, 3, 0]))])
    def test_duplicates_on_starter_columns(self, method, expected):
        # regression test for #22752

        df = pd.DataFrame({
            'a': [2, 2, 2, 1, 1, 1],
            'b': [1, 2, 3, 3, 2, 1]
        })

        result = getattr(df, method)(4, columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_n_identical_values(self):
        # GH 15297
        df = pd.DataFrame({'a': [1] * 5, 'b': [1, 2, 3, 4, 5]})

        result = df.nlargest(3, 'a')
        expected = pd.DataFrame(
            {'a': [1] * 3, 'b': [1, 2, 3]}, index=[0, 1, 2]
        )
        tm.assert_frame_equal(result, expected)

        result = df.nsmallest(3, 'a')
        expected = pd.DataFrame({'a': [1] * 3, 'b': [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('order', [
        ['a', 'b', 'c'],
        ['c', 'b', 'a'],
        ['a'],
        ['b'],
        ['a', 'b'],
        ['c', 'b']])
    @pytest.mark.parametrize('n', range(1, 6))
    def test_n_duplicate_index(self, df_duplicates, n, order):
        # GH 13412

        df = df_duplicates
        result = df.nsmallest(n, order)
        expected = df.sort_values(order).head(n)
        tm.assert_frame_equal(result, expected)

        result = df.nlargest(n, order)
        expected = df.sort_values(order, ascending=False).head(n)
        tm.assert_frame_equal(result, expected)

    def test_duplicate_keep_all_ties(self):
        # GH 16818
        df = pd.DataFrame({'a': [5, 4, 4, 2, 3, 3, 3, 3],
                           'b': [10, 9, 8, 7, 5, 50, 10, 20]})
        result = df.nlargest(4, 'a', keep='all')
        expected = pd.DataFrame({'a': {0: 5, 1: 4, 2: 4, 4: 3,
                                       5: 3, 6: 3, 7: 3},
                                 'b': {0: 10, 1: 9, 2: 8, 4: 5,
                                       5: 50, 6: 10, 7: 20}})
        tm.assert_frame_equal(result, expected)

        result = df.nsmallest(2, 'a', keep='all')
        expected = pd.DataFrame({'a': {3: 2, 4: 3, 5: 3, 6: 3, 7: 3},
                                 'b': {3: 7, 4: 5, 5: 50, 6: 10, 7: 20}})
        tm.assert_frame_equal(result, expected)

    def test_series_broadcasting(self):
        # smoke test for numpy warnings
        # GH 16378, GH 16306
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({'A': [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])

        with tm.assert_produces_warning(None):
            with tm.assert_produces_warning(FutureWarning):
                df_nan.clip_lower(s, axis=0)
            for op in ['lt', 'le', 'gt', 'ge', 'eq', 'ne']:
                getattr(df, op)(s_nan, axis=0)

    def test_series_nat_conversion(self):
        # GH 18521
        # Check rank does not mutate DataFrame
        df = DataFrame(np.random.randn(10, 3), dtype='float64')
        expected = df.copy()
        df.rank()
        result = df
        tm.assert_frame_equal(result, expected)

    def test_multiindex_column_lookup(self):
        # Check whether tuples are correctly treated as multi-level lookups.
        # GH 23033
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([['x'], ['a', 'b']]),
            data=[[0.33, 0.13], [0.86, 0.25], [0.25, 0.70], [0.85, 0.91]])

        # nsmallest
        result = df.nsmallest(3, ('x', 'a'))
        expected = df.iloc[[2, 0, 3]]
        tm.assert_frame_equal(result, expected)

        # nlargest
        result = df.nlargest(3, ('x', 'b'))
        expected = df.iloc[[3, 2, 1]]
        tm.assert_frame_equal(result, expected)
