# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import pytest

from pandas.compat import PY36, lrange, range

from pandas import DataFrame, Index, MultiIndex, Series
from pandas.tests.frame.common import TestData
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal

# Column add, remove, delete.


class TestDataFrameMutateColumns(TestData):

    def test_assign(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        original = df.copy()
        result = df.assign(C=df.B / df.A)
        expected = df.copy()
        expected['C'] = [4, 2.5, 2]
        assert_frame_equal(result, expected)

        # lambda syntax
        result = df.assign(C=lambda x: x.B / x.A)
        assert_frame_equal(result, expected)

        # original is unmodified
        assert_frame_equal(df, original)

        # Non-Series array-like
        result = df.assign(C=[4, 2.5, 2])
        assert_frame_equal(result, expected)
        # original is unmodified
        assert_frame_equal(df, original)

        result = df.assign(B=df.B / df.A)
        expected = expected.drop('B', axis=1).rename(columns={'C': 'B'})
        assert_frame_equal(result, expected)

        # overwrite
        result = df.assign(A=df.A + df.B)
        expected = df.copy()
        expected['A'] = [5, 7, 9]
        assert_frame_equal(result, expected)

        # lambda
        result = df.assign(A=lambda x: x.A + x.B)
        assert_frame_equal(result, expected)

    def test_assign_multiple(self):
        df = DataFrame([[1, 4], [2, 5], [3, 6]], columns=['A', 'B'])
        result = df.assign(C=[7, 8, 9], D=df.A, E=lambda x: x.B)
        expected = DataFrame([[1, 4, 7, 1, 4], [2, 5, 8, 2, 5],
                              [3, 6, 9, 3, 6]], columns=list('ABCDE'))
        assert_frame_equal(result, expected)

    def test_assign_order(self):
        # GH 9818
        df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        result = df.assign(D=df.A + df.B, C=df.A - df.B)

        if PY36:
            expected = DataFrame([[1, 2, 3, -1], [3, 4, 7, -1]],
                                 columns=list('ABDC'))
        else:
            expected = DataFrame([[1, 2, -1, 3], [3, 4, -1, 7]],
                                 columns=list('ABCD'))
        assert_frame_equal(result, expected)
        result = df.assign(C=df.A - df.B, D=df.A + df.B)

        expected = DataFrame([[1, 2, -1, 3], [3, 4, -1, 7]],
                             columns=list('ABCD'))

        assert_frame_equal(result, expected)

    def test_assign_bad(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        # non-keyword argument
        with pytest.raises(TypeError):
            df.assign(lambda x: x.A)
        with pytest.raises(AttributeError):
            df.assign(C=df.A, D=df.A + df.C)

    @pytest.mark.skipif(PY36, reason="""Issue #14207: valid for python
                        3.6 and above""")
    def test_assign_dependent_old_python(self):
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        # Key C does not exist at definition time of df
        with pytest.raises(KeyError):
            df.assign(C=lambda df: df.A,
                      D=lambda df: df['A'] + df['C'])
        with pytest.raises(KeyError):
            df.assign(C=df.A, D=lambda x: x['A'] + x['C'])

    @pytest.mark.skipif(not PY36, reason="""Issue #14207: not valid for
                        python 3.5 and below""")
    def test_assign_dependent(self):
        df = DataFrame({'A': [1, 2], 'B': [3, 4]})

        result = df.assign(C=df.A, D=lambda x: x['A'] + x['C'])
        expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]],
                             columns=list('ABCD'))
        assert_frame_equal(result, expected)

        result = df.assign(C=lambda df: df.A,
                           D=lambda df: df['A'] + df['C'])
        expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]],
                             columns=list('ABCD'))
        assert_frame_equal(result, expected)

    def test_insert_error_msmgs(self):

        # GH 7432
        df = DataFrame({'foo': ['a', 'b', 'c'], 'bar': [
                       1, 2, 3], 'baz': ['d', 'e', 'f']}).set_index('foo')
        s = DataFrame({'foo': ['a', 'b', 'c', 'a'], 'fiz': [
                      'g', 'h', 'i', 'j']}).set_index('foo')
        msg = 'cannot reindex from a duplicate axis'
        with pytest.raises(ValueError, match=msg):
            df['newcol'] = s

        # GH 4107, more descriptive error message
        df = DataFrame(np.random.randint(0, 2, (4, 4)),
                       columns=['a', 'b', 'c', 'd'])

        msg = 'incompatible index of inserted column with frame index'
        with pytest.raises(TypeError, match=msg):
            df['gr'] = df.groupby(['b', 'c']).count()

    def test_insert_benchmark(self):
        # from the vb_suite/frame_methods/frame_insert_columns
        N = 10
        K = 5
        df = DataFrame(index=lrange(N))
        new_col = np.random.randn(N)
        for i in range(K):
            df[i] = new_col
        expected = DataFrame(np.repeat(new_col, K).reshape(N, K),
                             index=lrange(N))
        assert_frame_equal(df, expected)

    def test_insert(self):
        df = DataFrame(np.random.randn(5, 3), index=np.arange(5),
                       columns=['c', 'b', 'a'])

        df.insert(0, 'foo', df['a'])
        tm.assert_index_equal(df.columns, Index(['foo', 'c', 'b', 'a']))
        tm.assert_series_equal(df['a'], df['foo'], check_names=False)

        df.insert(2, 'bar', df['c'])
        tm.assert_index_equal(df.columns,
                              Index(['foo', 'c', 'bar', 'b', 'a']))
        tm.assert_almost_equal(df['c'], df['bar'], check_names=False)

        # diff dtype

        # new item
        df['x'] = df['a'].astype('float32')
        result = Series(dict(float32=1, float64=5))
        assert (df.get_dtype_counts().sort_index() == result).all()

        # replacing current (in different block)
        df['a'] = df['a'].astype('float32')
        result = Series(dict(float32=2, float64=4))
        assert (df.get_dtype_counts().sort_index() == result).all()

        df['y'] = df['a'].astype('int32')
        result = Series(dict(float32=2, float64=4, int32=1))
        assert (df.get_dtype_counts().sort_index() == result).all()

        with pytest.raises(ValueError, match='already exists'):
            df.insert(1, 'a', df['b'])
        pytest.raises(ValueError, df.insert, 1, 'c', df['b'])

        df.columns.name = 'some_name'
        # preserve columns name field
        df.insert(0, 'baz', df['c'])
        assert df.columns.name == 'some_name'

        # GH 13522
        df = DataFrame(index=['A', 'B', 'C'])
        df['X'] = df.index
        df['X'] = ['x', 'y', 'z']
        exp = DataFrame(data={'X': ['x', 'y', 'z']}, index=['A', 'B', 'C'])
        assert_frame_equal(df, exp)

    def test_delitem(self):
        del self.frame['A']
        assert 'A' not in self.frame

    def test_delitem_multiindex(self):
        midx = MultiIndex.from_product([['A', 'B'], [1, 2]])
        df = DataFrame(np.random.randn(4, 4), columns=midx)
        assert len(df.columns) == 4
        assert ('A', ) in df.columns
        assert 'A' in df.columns

        result = df['A']
        assert isinstance(result, DataFrame)
        del df['A']

        assert len(df.columns) == 2

        # A still in the levels, BUT get a KeyError if trying
        # to delete
        assert ('A', ) not in df.columns
        with pytest.raises(KeyError):
            del df[('A',)]

        # behavior of dropped/deleted MultiIndex levels changed from
        # GH 2770 to GH 19027: MultiIndex no longer '.__contains__'
        # levels which are dropped/deleted
        assert 'A' not in df.columns
        with pytest.raises(KeyError):
            del df['A']

    def test_pop(self):
        self.frame.columns.name = 'baz'

        self.frame.pop('A')
        assert 'A' not in self.frame

        self.frame['foo'] = 'bar'
        self.frame.pop('foo')
        assert 'foo' not in self.frame
        assert self.frame.columns.name == 'baz'

        # gh-10912: inplace ops cause caching issue
        a = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[
                      'A', 'B', 'C'], index=['X', 'Y'])
        b = a.pop('B')
        b += 1

        # original frame
        expected = DataFrame([[1, 3], [4, 6]], columns=[
                             'A', 'C'], index=['X', 'Y'])
        tm.assert_frame_equal(a, expected)

        # result
        expected = Series([2, 5], index=['X', 'Y'], name='B') + 1
        tm.assert_series_equal(b, expected)

    def test_pop_non_unique_cols(self):
        df = DataFrame({0: [0, 1], 1: [0, 1], 2: [4, 5]})
        df.columns = ["a", "b", "a"]

        res = df.pop("a")
        assert type(res) == DataFrame
        assert len(res) == 2
        assert len(df.columns) == 1
        assert "b" in df.columns
        assert "a" not in df.columns
        assert len(df.index) == 2

    def test_insert_column_bug_4032(self):

        # GH4032, inserting a column and renaming causing errors
        df = DataFrame({'b': [1.1, 2.2]})
        df = df.rename(columns={})
        df.insert(0, 'a', [1, 2])

        result = df.rename(columns={})
        str(result)
        expected = DataFrame([[1, 1.1], [2, 2.2]], columns=['a', 'b'])
        assert_frame_equal(result, expected)
        df.insert(0, 'c', [1.3, 2.3])

        result = df.rename(columns={})
        str(result)

        expected = DataFrame([[1.3, 1, 1.1], [2.3, 2, 2.2]],
                             columns=['c', 'a', 'b'])
        assert_frame_equal(result, expected)
