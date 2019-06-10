# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from distutils.version import LooseVersion

import numpy as np
import pytest

from pandas import DataFrame, Series
from pandas.tests.frame.common import TestData
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal


class TestRank(TestData):
    s = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    df = DataFrame({'A': s, 'B': s})

    results = {
        'average': np.array([1.5, 5.5, 7.0, 3.5, np.nan,
                             3.5, 1.5, 8.0, np.nan, 5.5]),
        'min': np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        'max': np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        'first': np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        'dense': np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3]),
    }

    @pytest.fixture(params=['average', 'min', 'max', 'first', 'dense'])
    def method(self, request):
        """
        Fixture for trying all rank methods
        """
        return request.param

    def test_rank(self):
        rankdata = pytest.importorskip('scipy.stats.rankdata')

        self.frame['A'][::2] = np.nan
        self.frame['B'][::3] = np.nan
        self.frame['C'][::4] = np.nan
        self.frame['D'][::5] = np.nan

        ranks0 = self.frame.rank()
        ranks1 = self.frame.rank(1)
        mask = np.isnan(self.frame.values)

        fvals = self.frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(rankdata, 0, fvals)
        exp0[mask] = np.nan

        exp1 = np.apply_along_axis(rankdata, 1, fvals)
        exp1[mask] = np.nan

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # integers
        df = DataFrame(np.random.randint(0, 5, size=40).reshape((10, 4)))

        result = df.rank()
        exp = df.astype(float).rank()
        tm.assert_frame_equal(result, exp)

        result = df.rank(1)
        exp = df.astype(float).rank(1)
        tm.assert_frame_equal(result, exp)

    def test_rank2(self):
        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = DataFrame([[1.0, 3.0, 2.0], [1, 2, 3]]) / 3.0
        result = df.rank(1, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = df.rank(0) / 2.0
        result = df.rank(0, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([['b', 'c', 'a'], ['a', 'c', 'b']])
        expected = DataFrame([[2.0, 3.0, 1.0], [1, 3, 2]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, 1.5, 1.0], [1, 1.5, 2]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([['b', np.nan, 'a'], ['a', 'c', 'b']])
        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 3.0, 2.0]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 1.0, 2.0]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        # f7u12, this does not work without extensive workaround
        data = [[datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)],
                [datetime(2000, 1, 2), datetime(2000, 1, 3),
                 datetime(2000, 1, 1)]]
        df = DataFrame(data)

        # check the rank
        expected = DataFrame([[2., np.nan, 1.],
                              [2., 3., 1.]])
        result = df.rank(1, numeric_only=False, ascending=True)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[1., np.nan, 2.],
                              [2., 1., 3.]])
        result = df.rank(1, numeric_only=False, ascending=False)
        tm.assert_frame_equal(result, expected)

        # mixed-type frames
        self.mixed_frame['datetime'] = datetime.now()
        self.mixed_frame['timedelta'] = timedelta(days=1, seconds=1)

        result = self.mixed_frame.rank(1)
        expected = self.mixed_frame.rank(1, numeric_only=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": [1e-20, -5, 1e-20 + 1e-40, 10,
                              1e60, 1e80, 1e-30]})
        exp = DataFrame({"a": [3.5, 1., 3.5, 5., 6., 7., 2.]})
        tm.assert_frame_equal(df.rank(), exp)

    def test_rank_na_option(self):
        rankdata = pytest.importorskip('scipy.stats.rankdata')

        self.frame['A'][::2] = np.nan
        self.frame['B'][::3] = np.nan
        self.frame['C'][::4] = np.nan
        self.frame['D'][::5] = np.nan

        # bottom
        ranks0 = self.frame.rank(na_option='bottom')
        ranks1 = self.frame.rank(1, na_option='bottom')

        fvals = self.frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(rankdata, 0, fvals)
        exp1 = np.apply_along_axis(rankdata, 1, fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # top
        ranks0 = self.frame.rank(na_option='top')
        ranks1 = self.frame.rank(1, na_option='top')

        fval0 = self.frame.fillna((self.frame.min() - 1).to_dict()).values
        fval1 = self.frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(rankdata, 0, fval0)
        exp1 = np.apply_along_axis(rankdata, 1, fval1)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # bottom
        ranks0 = self.frame.rank(na_option='top', ascending=False)
        ranks1 = self.frame.rank(1, na_option='top', ascending=False)

        fvals = self.frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(rankdata, 0, -fvals)
        exp1 = np.apply_along_axis(rankdata, 1, -fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # top
        ranks0 = self.frame.rank(na_option='bottom', ascending=False)
        ranks1 = self.frame.rank(1, na_option='bottom', ascending=False)

        fval0 = self.frame.fillna((self.frame.min() - 1).to_dict()).values
        fval1 = self.frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(rankdata, 0, -fval0)
        exp1 = np.apply_along_axis(rankdata, 1, -fval1)

        tm.assert_numpy_array_equal(ranks0.values, exp0)
        tm.assert_numpy_array_equal(ranks1.values, exp1)

        # bad values throw error
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            self.frame.rank(na_option='bad', ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            self.frame.rank(na_option=True, ascending=False)

    def test_rank_axis(self):
        # check if using axes' names gives the same result
        df = DataFrame([[2, 1], [4, 3]])
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis='index'))
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis='columns'))

    def test_rank_methods_frame(self):
        pytest.importorskip('scipy.stats.special')
        rankdata = pytest.importorskip('scipy.stats.rankdata')
        import scipy

        xs = np.random.randint(0, 21, (100, 26))
        xs = (xs - 10.0) / 10.0
        cols = [chr(ord('z') - i) for i in range(xs.shape[1])]

        for vals in [xs, xs + 1e6, xs * 1e-6]:
            df = DataFrame(vals, columns=cols)

            for ax in [0, 1]:
                for m in ['average', 'min', 'max', 'first', 'dense']:
                    result = df.rank(axis=ax, method=m)
                    sprank = np.apply_along_axis(
                        rankdata, ax, vals,
                        m if m != 'first' else 'ordinal')
                    sprank = sprank.astype(np.float64)
                    expected = DataFrame(sprank, columns=cols)

                    if (LooseVersion(scipy.__version__) >=
                            LooseVersion('0.17.0')):
                        expected = expected.astype('float64')
                    tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['O', 'f8', 'i8'])
    def test_rank_descending(self, method, dtype):

        if 'i' in dtype:
            df = self.df.dropna()
        else:
            df = self.df.astype(dtype)

        res = df.rank(ascending=False)
        expected = (df.max() - df).rank()
        assert_frame_equal(res, expected)

        if method == 'first' and dtype == 'O':
            return

        expected = (df.max() - df).rank(method=method)

        if dtype != 'O':
            res2 = df.rank(method=method, ascending=False,
                           numeric_only=True)
            assert_frame_equal(res2, expected)

        res3 = df.rank(method=method, ascending=False,
                       numeric_only=False)
        assert_frame_equal(res3, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_rank_2d_tie_methods(self, method, axis, dtype):
        df = self.df

        def _check2d(df, expected, method='average', axis=0):
            exp_df = DataFrame({'A': expected, 'B': expected})

            if axis == 1:
                df = df.T
                exp_df = exp_df.T

            result = df.rank(method=method, axis=axis)
            assert_frame_equal(result, exp_df)

        disabled = {(object, 'first')}
        if (dtype, method) in disabled:
            return
        frame = df if dtype is None else df.astype(dtype)
        _check2d(frame, self.results[method], method=method, axis=axis)

    @pytest.mark.parametrize(
        "method,exp", [("dense",
                        [[1., 1., 1.],
                         [1., 0.5, 2. / 3],
                         [1., 0.5, 1. / 3]]),
                       ("min",
                        [[1. / 3, 1., 1.],
                         [1. / 3, 1. / 3, 2. / 3],
                         [1. / 3, 1. / 3, 1. / 3]]),
                       ("max",
                        [[1., 1., 1.],
                         [1., 2. / 3, 2. / 3],
                         [1., 2. / 3, 1. / 3]]),
                       ("average",
                        [[2. / 3, 1., 1.],
                         [2. / 3, 0.5, 2. / 3],
                         [2. / 3, 0.5, 1. / 3]]),
                       ("first",
                        [[1. / 3, 1., 1.],
                         [2. / 3, 1. / 3, 2. / 3],
                         [3. / 3, 2. / 3, 1. / 3]])])
    def test_rank_pct_true(self, method, exp):
        # see gh-15630.

        df = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])
        result = df.rank(method=method, pct=True)

        expected = DataFrame(exp)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single
    def test_pct_max_many_rows(self):
        # GH 18271
        df = DataFrame({'A': np.arange(2**24 + 1),
                        'B': np.arange(2**24 + 1, 0, -1)})
        result = df.rank(pct=True).max()
        assert (result == 1).all()
