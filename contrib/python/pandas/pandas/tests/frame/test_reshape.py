# -*- coding: utf-8 -*-

from __future__ import print_function

from datetime import datetime
import itertools
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest

from pandas.compat import u

import pandas as pd
from pandas import (
    DataFrame, Index, MultiIndex, Period, Series, Timedelta, date_range)
from pandas.tests.frame.common import TestData
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal, assert_series_equal


class TestDataFrameReshape(TestData):

    def test_pivot(self):
        data = {
            'index': ['A', 'B', 'C', 'C', 'B', 'A'],
            'columns': ['One', 'One', 'One', 'Two', 'Two', 'Two'],
            'values': [1., 2., 3., 3., 2., 1.]
        }

        frame = DataFrame(data)
        pivoted = frame.pivot(
            index='index', columns='columns', values='values')

        expected = DataFrame({
            'One': {'A': 1., 'B': 2., 'C': 3.},
            'Two': {'A': 1., 'B': 2., 'C': 3.}
        })

        expected.index.name, expected.columns.name = 'index', 'columns'
        tm.assert_frame_equal(pivoted, expected)

        # name tracking
        assert pivoted.index.name == 'index'
        assert pivoted.columns.name == 'columns'

        # don't specify values
        pivoted = frame.pivot(index='index', columns='columns')
        assert pivoted.index.name == 'index'
        assert pivoted.columns.names == (None, 'columns')

        with catch_warnings(record=True):
            # pivot multiple columns
            simplefilter("ignore", FutureWarning)
            wp = tm.makePanel()
            lp = wp.to_frame()
            df = lp.reset_index()
            tm.assert_frame_equal(df.pivot('major', 'minor'), lp.unstack())

    def test_pivot_duplicates(self):
        data = DataFrame({'a': ['bar', 'bar', 'foo', 'foo', 'foo'],
                          'b': ['one', 'two', 'one', 'one', 'two'],
                          'c': [1., 2., 3., 3., 4.]})
        with pytest.raises(ValueError, match='duplicate entries'):
            data.pivot('a', 'b', 'c')

    def test_pivot_empty(self):
        df = DataFrame({}, columns=['a', 'b', 'c'])
        result = df.pivot('a', 'b', 'c')
        expected = DataFrame({})
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_pivot_integer_bug(self):
        df = DataFrame(data=[("A", "1", "A1"), ("B", "2", "B2")])

        result = df.pivot(index=1, columns=0, values=2)
        repr(result)
        tm.assert_index_equal(result.columns, Index(['A', 'B'], name=0))

    def test_pivot_index_none(self):
        # gh-3962
        data = {
            'index': ['A', 'B', 'C', 'C', 'B', 'A'],
            'columns': ['One', 'One', 'One', 'Two', 'Two', 'Two'],
            'values': [1., 2., 3., 3., 2., 1.]
        }

        frame = DataFrame(data).set_index('index')
        result = frame.pivot(columns='columns', values='values')
        expected = DataFrame({
            'One': {'A': 1., 'B': 2., 'C': 3.},
            'Two': {'A': 1., 'B': 2., 'C': 3.}
        })

        expected.index.name, expected.columns.name = 'index', 'columns'
        assert_frame_equal(result, expected)

        # omit values
        result = frame.pivot(columns='columns')

        expected.columns = pd.MultiIndex.from_tuples([('values', 'One'),
                                                      ('values', 'Two')],
                                                     names=[None, 'columns'])
        expected.index.name = 'index'
        tm.assert_frame_equal(result, expected, check_names=False)
        assert result.index.name == 'index'
        assert result.columns.names == (None, 'columns')
        expected.columns = expected.columns.droplevel(0)
        result = frame.pivot(columns='columns', values='values')

        expected.columns.name = 'columns'
        tm.assert_frame_equal(result, expected)

    def test_stack_unstack(self):
        df = self.frame.copy()
        df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)

        stacked = df.stack()
        stacked_df = DataFrame({'foo': stacked, 'bar': stacked})

        unstacked = stacked.unstack()
        unstacked_df = stacked_df.unstack()

        assert_frame_equal(unstacked, df)
        assert_frame_equal(unstacked_df['bar'], df)

        unstacked_cols = stacked.unstack(0)
        unstacked_cols_df = stacked_df.unstack(0)
        assert_frame_equal(unstacked_cols.T, df)
        assert_frame_equal(unstacked_cols_df['bar'].T, df)

    def test_stack_mixed_level(self):
        # GH 18310
        levels = [range(3), [3, 'a', 'b'], [1, 2]]

        # flat columns:
        df = DataFrame(1, index=levels[0], columns=levels[1])
        result = df.stack()
        expected = Series(1, index=MultiIndex.from_product(levels[:2]))
        assert_series_equal(result, expected)

        # MultiIndex columns:
        df = DataFrame(1, index=levels[0],
                       columns=MultiIndex.from_product(levels[1:]))
        result = df.stack(1)
        expected = DataFrame(1, index=MultiIndex.from_product([levels[0],
                                                               levels[2]]),
                             columns=levels[1])
        assert_frame_equal(result, expected)

        # as above, but used labels in level are actually of homogeneous type
        result = df[['a', 'b']].stack(1)
        expected = expected[['a', 'b']]
        assert_frame_equal(result, expected)

    def test_unstack_fill(self):

        # GH #9746: fill_value keyword argument for Series
        # and DataFrame unstack

        # From a series
        data = Series([1, 2, 4, 5], dtype=np.int16)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

        result = data.unstack(fill_value=-1)
        expected = DataFrame({'a': [1, -1, 5], 'b': [2, 4, -1]},
                             index=['x', 'y', 'z'], dtype=np.int16)
        assert_frame_equal(result, expected)

        # From a series with incorrect data type for fill_value
        result = data.unstack(fill_value=0.5)
        expected = DataFrame({'a': [1, 0.5, 5], 'b': [2, 4, 0.5]},
                             index=['x', 'y', 'z'], dtype=np.float)
        assert_frame_equal(result, expected)

        # GH #13971: fill_value when unstacking multiple levels:
        df = DataFrame({'x': ['a', 'a', 'b'],
                        'y': ['j', 'k', 'j'],
                        'z': [0, 1, 2],
                        'w': [0, 1, 2]}).set_index(['x', 'y', 'z'])
        unstacked = df.unstack(['x', 'y'], fill_value=0)
        key = ('w', 'b', 'j')
        expected = unstacked[key]
        result = pd.Series([0, 0, 2], index=unstacked.index, name=key)
        assert_series_equal(result, expected)

        stacked = unstacked.stack(['x', 'y'])
        stacked.index = stacked.index.reorder_levels(df.index.names)
        # Workaround for GH #17886 (unnecessarily casts to float):
        stacked = stacked.astype(np.int64)
        result = stacked.loc[df.index]
        assert_frame_equal(result, df)

        # From a series
        s = df['w']
        result = s.unstack(['x', 'y'], fill_value=0)
        expected = unstacked['w']
        assert_frame_equal(result, expected)

    def test_unstack_fill_frame(self):

        # From a dataframe
        rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
        df = DataFrame(rows, columns=list('AB'), dtype=np.int32)
        df.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

        result = df.unstack(fill_value=-1)

        rows = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
        expected = DataFrame(rows, index=list('xyz'), dtype=np.int32)
        expected.columns = MultiIndex.from_tuples(
            [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        assert_frame_equal(result, expected)

        # From a mixed type dataframe
        df['A'] = df['A'].astype(np.int16)
        df['B'] = df['B'].astype(np.float64)

        result = df.unstack(fill_value=-1)
        expected['A'] = expected['A'].astype(np.int16)
        expected['B'] = expected['B'].astype(np.float64)
        assert_frame_equal(result, expected)

        # From a dataframe with incorrect data type for fill_value
        result = df.unstack(fill_value=0.5)

        rows = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
        expected = DataFrame(rows, index=list('xyz'), dtype=np.float)
        expected.columns = MultiIndex.from_tuples(
            [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')])
        assert_frame_equal(result, expected)

    def test_unstack_fill_frame_datetime(self):

        # Test unstacking with date times
        dv = pd.date_range('2012-01-01', periods=4).values
        data = Series(dv)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

        result = data.unstack()
        expected = DataFrame({'a': [dv[0], pd.NaT, dv[3]],
                              'b': [dv[1], dv[2], pd.NaT]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

        result = data.unstack(fill_value=dv[0])
        expected = DataFrame({'a': [dv[0], dv[0], dv[3]],
                              'b': [dv[1], dv[2], dv[0]]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

    def test_unstack_fill_frame_timedelta(self):

        # Test unstacking with time deltas
        td = [Timedelta(days=i) for i in range(4)]
        data = Series(td)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

        result = data.unstack()
        expected = DataFrame({'a': [td[0], pd.NaT, td[3]],
                              'b': [td[1], td[2], pd.NaT]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

        result = data.unstack(fill_value=td[1])
        expected = DataFrame({'a': [td[0], td[1], td[3]],
                              'b': [td[1], td[2], td[1]]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

    def test_unstack_fill_frame_period(self):

        # Test unstacking with period
        periods = [Period('2012-01'), Period('2012-02'), Period('2012-03'),
                   Period('2012-04')]
        data = Series(periods)
        data.index = MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

        result = data.unstack()
        expected = DataFrame({'a': [periods[0], None, periods[3]],
                              'b': [periods[1], periods[2], None]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

        result = data.unstack(fill_value=periods[1])
        expected = DataFrame({'a': [periods[0], periods[1], periods[3]],
                              'b': [periods[1], periods[2], periods[1]]},
                             index=['x', 'y', 'z'])
        assert_frame_equal(result, expected)

    def test_unstack_fill_frame_categorical(self):

        # Test unstacking with categorical
        data = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        data.index = pd.MultiIndex.from_tuples(
            [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')],
        )

        # By default missing values will be NaN
        result = data.unstack()
        expected = DataFrame({'a': pd.Categorical(list('axa'),
                                                  categories=list('abc')),
                              'b': pd.Categorical(list('bcx'),
                                                  categories=list('abc'))},
                             index=list('xyz'))
        assert_frame_equal(result, expected)

        # Fill with non-category results in a TypeError
        msg = r"'fill_value' \('d'\) is not in"
        with pytest.raises(TypeError, match=msg):
            data.unstack(fill_value='d')

        # Fill with category value replaces missing values as expected
        result = data.unstack(fill_value='c')
        expected = DataFrame({'a': pd.Categorical(list('aca'),
                                                  categories=list('abc')),
                              'b': pd.Categorical(list('bcc'),
                                                  categories=list('abc'))},
                             index=list('xyz'))
        assert_frame_equal(result, expected)

    def test_unstack_preserve_dtypes(self):
        # Checks fix for #11847
        df = pd.DataFrame(dict(state=['IL', 'MI', 'NC'],
                               index=['a', 'b', 'c'],
                               some_categories=pd.Series(['a', 'b', 'c']
                                                         ).astype('category'),
                               A=np.random.rand(3),
                               B=1,
                               C='foo',
                               D=pd.Timestamp('20010102'),
                               E=pd.Series([1.0, 50.0, 100.0]
                                           ).astype('float32'),
                               F=pd.Series([3.0, 4.0, 5.0]).astype('float64'),
                               G=False,
                               H=pd.Series([1, 200, 923442], dtype='int8')))

        def unstack_and_compare(df, column_name):
            unstacked1 = df.unstack([column_name])
            unstacked2 = df.unstack(column_name)
            assert_frame_equal(unstacked1, unstacked2)

        df1 = df.set_index(['state', 'index'])
        unstack_and_compare(df1, 'index')

        df1 = df.set_index(['state', 'some_categories'])
        unstack_and_compare(df1, 'some_categories')

        df1 = df.set_index(['F', 'C'])
        unstack_and_compare(df1, 'F')

        df1 = df.set_index(['G', 'B', 'state'])
        unstack_and_compare(df1, 'B')

        df1 = df.set_index(['E', 'A'])
        unstack_and_compare(df1, 'E')

        df1 = df.set_index(['state', 'index'])
        s = df1['A']
        unstack_and_compare(s, 'index')

    def test_stack_ints(self):
        columns = MultiIndex.from_tuples(list(itertools.product(range(3),
                                                                repeat=3)))
        df = DataFrame(np.random.randn(30, 27), columns=columns)

        assert_frame_equal(df.stack(level=[1, 2]),
                           df.stack(level=1).stack(level=1))
        assert_frame_equal(df.stack(level=[-2, -1]),
                           df.stack(level=1).stack(level=1))

        df_named = df.copy()
        df_named.columns.set_names(range(3), inplace=True)

        assert_frame_equal(df_named.stack(level=[1, 2]),
                           df_named.stack(level=1).stack(level=1))

    def test_stack_mixed_levels(self):
        columns = MultiIndex.from_tuples(
            [('A', 'cat', 'long'), ('B', 'cat', 'long'),
             ('A', 'dog', 'short'), ('B', 'dog', 'short')],
            names=['exp', 'animal', 'hair_length']
        )
        df = DataFrame(np.random.randn(4, 4), columns=columns)

        animal_hair_stacked = df.stack(level=['animal', 'hair_length'])
        exp_hair_stacked = df.stack(level=['exp', 'hair_length'])

        # GH #8584: Need to check that stacking works when a number
        # is passed that is both a level name and in the range of
        # the level numbers
        df2 = df.copy()
        df2.columns.names = ['exp', 'animal', 1]
        assert_frame_equal(df2.stack(level=['animal', 1]),
                           animal_hair_stacked, check_names=False)
        assert_frame_equal(df2.stack(level=['exp', 1]),
                           exp_hair_stacked, check_names=False)

        # When mixed types are passed and the ints are not level
        # names, raise
        pytest.raises(ValueError, df2.stack, level=['animal', 0])

        # GH #8584: Having 0 in the level names could raise a
        # strange error about lexsort depth
        df3 = df.copy()
        df3.columns.names = ['exp', 'animal', 0]
        assert_frame_equal(df3.stack(level=['animal', 0]),
                           animal_hair_stacked, check_names=False)

    def test_stack_int_level_names(self):
        columns = MultiIndex.from_tuples(
            [('A', 'cat', 'long'), ('B', 'cat', 'long'),
             ('A', 'dog', 'short'), ('B', 'dog', 'short')],
            names=['exp', 'animal', 'hair_length']
        )
        df = DataFrame(np.random.randn(4, 4), columns=columns)

        exp_animal_stacked = df.stack(level=['exp', 'animal'])
        animal_hair_stacked = df.stack(level=['animal', 'hair_length'])
        exp_hair_stacked = df.stack(level=['exp', 'hair_length'])

        df2 = df.copy()
        df2.columns.names = [0, 1, 2]
        assert_frame_equal(df2.stack(level=[1, 2]), animal_hair_stacked,
                           check_names=False)
        assert_frame_equal(df2.stack(level=[0, 1]), exp_animal_stacked,
                           check_names=False)
        assert_frame_equal(df2.stack(level=[0, 2]), exp_hair_stacked,
                           check_names=False)

        # Out-of-order int column names
        df3 = df.copy()
        df3.columns.names = [2, 0, 1]
        assert_frame_equal(df3.stack(level=[0, 1]), animal_hair_stacked,
                           check_names=False)
        assert_frame_equal(df3.stack(level=[2, 0]), exp_animal_stacked,
                           check_names=False)
        assert_frame_equal(df3.stack(level=[2, 1]), exp_hair_stacked,
                           check_names=False)

    def test_unstack_bool(self):
        df = DataFrame([False, False],
                       index=MultiIndex.from_arrays([['a', 'b'], ['c', 'l']]),
                       columns=['col'])
        rs = df.unstack()
        xp = DataFrame(np.array([[False, np.nan], [np.nan, False]],
                                dtype=object),
                       index=['a', 'b'],
                       columns=MultiIndex.from_arrays([['col', 'col'],
                                                       ['c', 'l']]))
        assert_frame_equal(rs, xp)

    def test_unstack_level_binding(self):
        # GH9856
        mi = pd.MultiIndex(
            levels=[[u('foo'), u('bar')], [u('one'), u('two')],
                    [u('a'), u('b')]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]],
            names=[u('first'), u('second'), u('third')])
        s = pd.Series(0, index=mi)
        result = s.unstack([1, 2]).stack(0)

        expected_mi = pd.MultiIndex(
            levels=[['foo', 'bar'], ['one', 'two']],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=['first', 'second'])

        expected = pd.DataFrame(np.array([[np.nan, 0],
                                          [0, np.nan],
                                          [np.nan, 0],
                                          [0, np.nan]],
                                         dtype=np.float64),
                                index=expected_mi,
                                columns=pd.Index(['a', 'b'], name='third'))

        assert_frame_equal(result, expected)

    def test_unstack_to_series(self):
        # check reversibility
        data = self.frame.unstack()

        assert isinstance(data, Series)
        undo = data.unstack().T
        assert_frame_equal(undo, self.frame)

        # check NA handling
        data = DataFrame({'x': [1, 2, np.NaN], 'y': [3.0, 4, np.NaN]})
        data.index = Index(['a', 'b', 'c'])
        result = data.unstack()

        midx = MultiIndex(levels=[['x', 'y'], ['a', 'b', 'c']],
                          codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        expected = Series([1, 2, np.NaN, 3, 4, np.NaN], index=midx)

        assert_series_equal(result, expected)

        # check composability of unstack
        old_data = data.copy()
        for _ in range(4):
            data = data.unstack()
        assert_frame_equal(old_data, data)

    def test_unstack_dtypes(self):

        # GH 2929
        rows = [[1, 1, 3, 4],
                [1, 2, 3, 4],
                [2, 1, 3, 4],
                [2, 2, 3, 4]]

        df = DataFrame(rows, columns=list('ABCD'))
        result = df.get_dtype_counts()
        expected = Series({'int64': 4})
        assert_series_equal(result, expected)

        # single dtype
        df2 = df.set_index(['A', 'B'])
        df3 = df2.unstack('B')
        result = df3.get_dtype_counts()
        expected = Series({'int64': 4})
        assert_series_equal(result, expected)

        # mixed
        df2 = df.set_index(['A', 'B'])
        df2['C'] = 3.
        df3 = df2.unstack('B')
        result = df3.get_dtype_counts()
        expected = Series({'int64': 2, 'float64': 2})
        assert_series_equal(result, expected)

        df2['D'] = 'foo'
        df3 = df2.unstack('B')
        result = df3.get_dtype_counts()
        expected = Series({'float64': 2, 'object': 2})
        assert_series_equal(result, expected)

        # GH7405
        for c, d in (np.zeros(5), np.zeros(5)), \
                    (np.arange(5, dtype='f8'), np.arange(5, 10, dtype='f8')):

            df = DataFrame({'A': ['a'] * 5, 'C': c, 'D': d,
                            'B': pd.date_range('2012-01-01', periods=5)})

            right = df.iloc[:3].copy(deep=True)

            df = df.set_index(['A', 'B'])
            df['D'] = df['D'].astype('int64')

            left = df.iloc[:3].unstack(0)
            right = right.set_index(['A', 'B']).unstack(0)
            right[('D', 'a')] = right[('D', 'a')].astype('int64')

            assert left.shape == (3, 2)
            tm.assert_frame_equal(left, right)

    def test_unstack_non_unique_index_names(self):
        idx = MultiIndex.from_tuples([('a', 'b'), ('c', 'd')],
                                     names=['c1', 'c1'])
        df = DataFrame([1, 2], index=idx)
        with pytest.raises(ValueError):
            df.unstack('c1')

        with pytest.raises(ValueError):
            df.T.stack('c1')

    def test_unstack_unused_levels(self):
        # GH 17845: unused codes in index make unstack() cast int to float
        idx = pd.MultiIndex.from_product([['a'], ['A', 'B', 'C', 'D']])[:-1]
        df = pd.DataFrame([[1, 0]] * 3, index=idx)

        result = df.unstack()
        exp_col = pd.MultiIndex.from_product([[0, 1], ['A', 'B', 'C']])
        expected = pd.DataFrame([[1, 1, 1, 0, 0, 0]], index=['a'],
                                columns=exp_col)
        tm.assert_frame_equal(result, expected)
        assert((result.columns.levels[1] == idx.levels[1]).all())

        # Unused items on both levels
        levels = [[0, 1, 7], [0, 1, 2, 3]]
        codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
        idx = pd.MultiIndex(levels, codes)
        block = np.arange(4).reshape(2, 2)
        df = pd.DataFrame(np.concatenate([block, block + 4]), index=idx)
        result = df.unstack()
        expected = pd.DataFrame(np.concatenate([block * 2, block * 2 + 1],
                                               axis=1),
                                columns=idx)
        tm.assert_frame_equal(result, expected)
        assert((result.columns.levels[1] == idx.levels[1]).all())

        # With mixed dtype and NaN
        levels = [['a', 2, 'c'], [1, 3, 5, 7]]
        codes = [[0, -1, 1, 1], [0, 2, -1, 2]]
        idx = pd.MultiIndex(levels, codes)
        data = np.arange(8)
        df = pd.DataFrame(data.reshape(4, 2), index=idx)

        cases = ((0, [13, 16, 6, 9, 2, 5, 8, 11],
                  [np.nan, 'a', 2], [np.nan, 5, 1]),
                 (1, [8, 11, 1, 4, 12, 15, 13, 16],
                  [np.nan, 5, 1], [np.nan, 'a', 2]))
        for level, idces, col_level, idx_level in cases:
            result = df.unstack(level=level)
            exp_data = np.zeros(18) * np.nan
            exp_data[idces] = data
            cols = pd.MultiIndex.from_product([[0, 1], col_level])
            expected = pd.DataFrame(exp_data.reshape(3, 6),
                                    index=idx_level, columns=cols)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("cols", [['A', 'C'], slice(None)])
    def test_unstack_unused_level(self, cols):
        # GH 18562 : unused codes on the unstacked level
        df = pd.DataFrame([[2010, 'a', 'I'],
                           [2011, 'b', 'II']],
                          columns=['A', 'B', 'C'])

        ind = df.set_index(['A', 'B', 'C'], drop=False)
        selection = ind.loc[(slice(None), slice(None), 'I'), cols]
        result = selection.unstack()

        expected = ind.iloc[[0]][cols]
        expected.columns = MultiIndex.from_product([expected.columns, ['I']],
                                                   names=[None, 'C'])
        expected.index = expected.index.droplevel('C')
        tm.assert_frame_equal(result, expected)

    def test_unstack_nan_index(self):  # GH7466
        cast = lambda val: '{0:1}'.format('' if val != val else val)

        def verify(df):
            mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]
            rows, cols = df.notna().values.nonzero()
            for i, j in zip(rows, cols):
                left = sorted(df.iloc[i, j].split('.'))
                right = mk_list(df.index[i]) + mk_list(df.columns[j])
                right = sorted(list(map(cast, right)))
                assert left == right

        df = DataFrame({'jim': ['a', 'b', np.nan, 'd'],
                        'joe': ['w', 'x', 'y', 'z'],
                        'jolie': ['a.w', 'b.x', ' .y', 'd.z']})

        left = df.set_index(['jim', 'joe']).unstack()['jolie']
        right = df.set_index(['joe', 'jim']).unstack()['jolie'].T
        assert_frame_equal(left, right)

        for idx in itertools.permutations(df.columns[:2]):
            mi = df.set_index(list(idx))
            for lev in range(2):
                udf = mi.unstack(level=lev)
                assert udf.notna().values.sum() == len(df)
                verify(udf['jolie'])

        df = DataFrame({'1st': ['d'] * 3 + [np.nan] * 5 + ['a'] * 2 +
                        ['c'] * 3 + ['e'] * 2 + ['b'] * 5,
                        '2nd': ['y'] * 2 + ['w'] * 3 + [np.nan] * 3 +
                        ['z'] * 4 + [np.nan] * 3 + ['x'] * 3 + [np.nan] * 2,
                        '3rd': [67, 39, 53, 72, 57, 80, 31, 18, 11, 30, 59,
                                50, 62, 59, 76, 52, 14, 53, 60, 51]})

        df['4th'], df['5th'] = \
            df.apply(lambda r: '.'.join(map(cast, r)), axis=1), \
            df.apply(lambda r: '.'.join(map(cast, r.iloc[::-1])), axis=1)

        for idx in itertools.permutations(['1st', '2nd', '3rd']):
            mi = df.set_index(list(idx))
            for lev in range(3):
                udf = mi.unstack(level=lev)
                assert udf.notna().values.sum() == 2 * len(df)
                for col in ['4th', '5th']:
                    verify(udf[col])

        # GH7403
        df = pd.DataFrame(
            {'A': list('aaaabbbb'), 'B': range(8), 'C': range(8)})
        df.iloc[3, 1] = np.NaN
        left = df.set_index(['A', 'B']).unstack(0)

        vals = [[3, 0, 1, 2, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7]]
        vals = list(map(list, zip(*vals)))
        idx = Index([np.nan, 0, 1, 2, 4, 5, 6, 7], name='B')
        cols = MultiIndex(levels=[['C'], ['a', 'b']],
                          codes=[[0, 0], [0, 1]],
                          names=[None, 'A'])

        right = DataFrame(vals, columns=cols, index=idx)
        assert_frame_equal(left, right)

        df = DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2,
                        'C': range(8)})
        df.iloc[2, 1] = np.NaN
        left = df.set_index(['A', 'B']).unstack(0)

        vals = [[2, np.nan], [0, 4], [1, 5], [np.nan, 6], [3, 7]]
        cols = MultiIndex(levels=[['C'], ['a', 'b']],
                          codes=[[0, 0], [0, 1]],
                          names=[None, 'A'])
        idx = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=idx)
        assert_frame_equal(left, right)

        df = pd.DataFrame({'A': list('aaaabbbb'), 'B': list(range(4)) * 2,
                           'C': range(8)})
        df.iloc[3, 1] = np.NaN
        left = df.set_index(['A', 'B']).unstack(0)

        vals = [[3, np.nan], [0, 4], [1, 5], [2, 6], [np.nan, 7]]
        cols = MultiIndex(levels=[['C'], ['a', 'b']],
                          codes=[[0, 0], [0, 1]],
                          names=[None, 'A'])
        idx = Index([np.nan, 0, 1, 2, 3], name='B')
        right = DataFrame(vals, columns=cols, index=idx)
        assert_frame_equal(left, right)

        # GH7401
        df = pd.DataFrame({'A': list('aaaaabbbbb'),
                           'B': (date_range('2012-01-01', periods=5)
                                 .tolist() * 2),
                           'C': np.arange(10)})

        df.iloc[3, 1] = np.NaN
        left = df.set_index(['A', 'B']).unstack()

        vals = np.array([[3, 0, 1, 2, np.nan, 4], [np.nan, 5, 6, 7, 8, 9]])
        idx = Index(['a', 'b'], name='A')
        cols = MultiIndex(levels=[['C'], date_range('2012-01-01', periods=5)],
                          codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
                          names=[None, 'B'])

        right = DataFrame(vals, columns=cols, index=idx)
        assert_frame_equal(left, right)

        # GH4862
        vals = [['Hg', np.nan, np.nan, 680585148],
                ['U', 0.0, np.nan, 680585148],
                ['Pb', 7.07e-06, np.nan, 680585148],
                ['Sn', 2.3614e-05, 0.0133, 680607017],
                ['Ag', 0.0, 0.0133, 680607017],
                ['Hg', -0.00015, 0.0133, 680607017]]
        df = DataFrame(vals, columns=['agent', 'change', 'dosage', 's_id'],
                       index=[17263, 17264, 17265, 17266, 17267, 17268])

        left = df.copy().set_index(['s_id', 'dosage', 'agent']).unstack()

        vals = [[np.nan, np.nan, 7.07e-06, np.nan, 0.0],
                [0.0, -0.00015, np.nan, 2.3614e-05, np.nan]]

        idx = MultiIndex(levels=[[680585148, 680607017], [0.0133]],
                         codes=[[0, 1], [-1, 0]],
                         names=['s_id', 'dosage'])

        cols = MultiIndex(levels=[['change'], ['Ag', 'Hg', 'Pb', 'Sn', 'U']],
                          codes=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],
                          names=[None, 'agent'])

        right = DataFrame(vals, columns=cols, index=idx)
        assert_frame_equal(left, right)

        left = df.loc[17264:].copy().set_index(['s_id', 'dosage', 'agent'])
        assert_frame_equal(left.unstack(), right)

        # GH9497 - multiple unstack with nulls
        df = DataFrame({'1st': [1, 2, 1, 2, 1, 2],
                        '2nd': pd.date_range('2014-02-01', periods=6,
                                             freq='D'),
                        'jim': 100 + np.arange(6),
                        'joe': (np.random.randn(6) * 10).round(2)})

        df['3rd'] = df['2nd'] - pd.Timestamp('2014-02-02')
        df.loc[1, '2nd'] = df.loc[3, '2nd'] = np.nan
        df.loc[1, '3rd'] = df.loc[4, '3rd'] = np.nan

        left = df.set_index(['1st', '2nd', '3rd']).unstack(['2nd', '3rd'])
        assert left.notna().values.sum() == 2 * len(df)

        for col in ['jim', 'joe']:
            for _, r in df.iterrows():
                key = r['1st'], (col, r['2nd'], r['3rd'])
                assert r[col] == left.loc[key]

    def test_stack_datetime_column_multiIndex(self):
        # GH 8039
        t = datetime(2014, 1, 1)
        df = DataFrame(
            [1, 2, 3, 4], columns=MultiIndex.from_tuples([(t, 'A', 'B')]))
        result = df.stack()

        eidx = MultiIndex.from_product([(0, 1, 2, 3), ('B',)])
        ecols = MultiIndex.from_tuples([(t, 'A')])
        expected = DataFrame([1, 2, 3, 4], index=eidx, columns=ecols)
        assert_frame_equal(result, expected)

    def test_stack_partial_multiIndex(self):
        # GH 8844
        def _test_stack_with_multiindex(multiindex):
            df = DataFrame(np.arange(3 * len(multiindex))
                           .reshape(3, len(multiindex)),
                           columns=multiindex)
            for level in (-1, 0, 1, [0, 1], [1, 0]):
                result = df.stack(level=level, dropna=False)

                if isinstance(level, int):
                    # Stacking a single level should not make any all-NaN rows,
                    # so df.stack(level=level, dropna=False) should be the same
                    # as df.stack(level=level, dropna=True).
                    expected = df.stack(level=level, dropna=True)
                    if isinstance(expected, Series):
                        assert_series_equal(result, expected)
                    else:
                        assert_frame_equal(result, expected)

                df.columns = MultiIndex.from_tuples(df.columns.get_values(),
                                                    names=df.columns.names)
                expected = df.stack(level=level, dropna=False)
                if isinstance(expected, Series):
                    assert_series_equal(result, expected)
                else:
                    assert_frame_equal(result, expected)

        full_multiindex = MultiIndex.from_tuples([('B', 'x'), ('B', 'z'),
                                                  ('A', 'y'),
                                                  ('C', 'x'), ('C', 'u')],
                                                 names=['Upper', 'Lower'])
        for multiindex_columns in ([0, 1, 2, 3, 4],
                                   [0, 1, 2, 3], [0, 1, 2, 4],
                                   [0, 1, 2], [1, 2, 3], [2, 3, 4],
                                   [0, 1], [0, 2], [0, 3],
                                   [0], [2], [4]):
            _test_stack_with_multiindex(full_multiindex[multiindex_columns])
            if len(multiindex_columns) > 1:
                multiindex_columns.reverse()
                _test_stack_with_multiindex(
                    full_multiindex[multiindex_columns])

        df = DataFrame(np.arange(6).reshape(2, 3),
                       columns=full_multiindex[[0, 1, 3]])
        result = df.stack(dropna=False)
        expected = DataFrame([[0, 2], [1, np.nan], [3, 5], [4, np.nan]],
                             index=MultiIndex(
                                 levels=[[0, 1], ['u', 'x', 'y', 'z']],
                                 codes=[[0, 0, 1, 1],
                                        [1, 3, 1, 3]],
                                 names=[None, 'Lower']),
                             columns=Index(['B', 'C'], name='Upper'),
                             dtype=df.dtypes[0])
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('ordered', [False, True])
    @pytest.mark.parametrize('labels', [list("yxz"), list("yxy")])
    def test_stack_preserve_categorical_dtype(self, ordered, labels):
        # GH13854
        cidx = pd.CategoricalIndex(labels, categories=list("xyz"),
                                   ordered=ordered)
        df = DataFrame([[10, 11, 12]], columns=cidx)
        result = df.stack()

        # `MutliIndex.from_product` preserves categorical dtype -
        # it's tested elsewhere.
        midx = pd.MultiIndex.from_product([df.index, cidx])
        expected = Series([10, 11, 12], index=midx)

        tm.assert_series_equal(result, expected)

    def test_stack_preserve_categorical_dtype_values(self):
        # GH-23077
        cat = pd.Categorical(['a', 'a', 'b', 'c'])
        df = pd.DataFrame({"A": cat, "B": cat})
        result = df.stack()
        index = pd.MultiIndex.from_product([[0, 1, 2, 3], ['A', 'B']])
        expected = pd.Series(pd.Categorical(['a', 'a', 'a', 'a',
                                             'b', 'b', 'c', 'c']),
                             index=index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('level', [0, 1])
    def test_unstack_mixed_extension_types(self, level):
        index = pd.MultiIndex.from_tuples([('A', 0), ('A', 1), ('B', 1)],
                                          names=['a', 'b'])
        df = pd.DataFrame({"A": pd.core.arrays.integer_array([0, 1, None]),
                           "B": pd.Categorical(['a', 'a', 'b'])}, index=index)

        result = df.unstack(level=level)
        expected = df.astype(object).unstack(level=level)

        expected_dtypes = pd.Series([df.A.dtype] * 2 + [df.B.dtype] * 2,
                                    index=result.columns)
        tm.assert_series_equal(result.dtypes, expected_dtypes)
        tm.assert_frame_equal(result.astype(object), expected)

    @pytest.mark.parametrize("level", [0, 'baz'])
    def test_unstack_swaplevel_sortlevel(self, level):
        # GH 20994
        mi = pd.MultiIndex.from_product([[0], ['d', 'c']],
                                        names=['bar', 'baz'])
        df = pd.DataFrame([[0, 2], [1, 3]], index=mi, columns=['B', 'A'])
        df.columns.name = 'foo'

        expected = pd.DataFrame([
            [3, 1, 2, 0]], columns=pd.MultiIndex.from_tuples([
                ('c', 'A'), ('c', 'B'), ('d', 'A'), ('d', 'B')], names=[
                    'baz', 'foo']))
        expected.index.name = 'bar'

        result = df.unstack().swaplevel(axis=1).sort_index(axis=1, level=level)
        tm.assert_frame_equal(result, expected)


def test_unstack_fill_frame_object():
    # GH12815 Test unstacking with object.
    data = pd.Series(['a', 'b', 'c', 'a'], dtype='object')
    data.index = pd.MultiIndex.from_tuples(
        [('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])

    # By default missing values will be NaN
    result = data.unstack()
    expected = pd.DataFrame(
        {'a': ['a', np.nan, 'a'], 'b': ['b', 'c', np.nan]},
        index=list('xyz')
    )
    assert_frame_equal(result, expected)

    # Fill with any value replaces missing values as expected
    result = data.unstack(fill_value='d')
    expected = pd.DataFrame(
        {'a': ['a', 'd', 'a'], 'b': ['b', 'c', 'd']},
        index=list('xyz')
    )
    assert_frame_equal(result, expected)


def test_unstack_timezone_aware_values():
    # GH 18338
    df = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2017-08-27 01:00:00.709949+0000', tz='UTC')],
        'a': ['a'],
        'b': ['b'],
        'c': ['c'],
    }, columns=['timestamp', 'a', 'b', 'c'])
    result = df.set_index(['a', 'b']).unstack()
    expected = pd.DataFrame([[pd.Timestamp('2017-08-27 01:00:00.709949+0000',
                                           tz='UTC'),
                              'c']],
                            index=pd.Index(['a'], name='a'),
                            columns=pd.MultiIndex(
                                levels=[['timestamp', 'c'], ['b']],
                                codes=[[0, 1], [0, 0]],
                                names=[None, 'b']))
    assert_frame_equal(result, expected)


def test_stack_timezone_aware_values():
    # GH 19420
    ts = pd.date_range(freq="D", start="20180101", end="20180103",
                       tz="America/New_York")
    df = pd.DataFrame({"A": ts}, index=["a", "b", "c"])
    result = df.stack()
    expected = pd.Series(ts,
                         index=pd.MultiIndex(levels=[['a', 'b', 'c'], ['A']],
                                             codes=[[0, 1, 2], [0, 0, 0]]))
    assert_series_equal(result, expected)
