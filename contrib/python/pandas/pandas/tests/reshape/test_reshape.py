# -*- coding: utf-8 -*-
# pylint: disable-msg=W0612,E1101

from collections import OrderedDict

import numpy as np
from numpy import nan
import pytest

from pandas.compat import u

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import Categorical, DataFrame, Index, Series, get_dummies
from pandas.core.sparse.api import SparseArray, SparseDtype
import pandas.util.testing as tm
from pandas.util.testing import assert_frame_equal


class TestGetDummies(object):

    @pytest.fixture
    def df(self):
        return DataFrame({'A': ['a', 'b', 'a'],
                          'B': ['b', 'b', 'c'],
                          'C': [1, 2, 3]})

    @pytest.fixture(params=['uint8', 'i8', np.float64, bool, None])
    def dtype(self, request):
        return np.dtype(request.param)

    @pytest.fixture(params=['dense', 'sparse'])
    def sparse(self, request):
        # params are strings to simplify reading test results,
        # e.g. TestGetDummies::test_basic[uint8-sparse] instead of [uint8-True]
        return request.param == 'sparse'

    def effective_dtype(self, dtype):
        if dtype is None:
            return np.uint8
        return dtype

    def test_raises_on_dtype_object(self, df):
        with pytest.raises(ValueError):
            get_dummies(df, dtype='object')

    def test_basic(self, sparse, dtype):
        s_list = list('abc')
        s_series = Series(s_list)
        s_series_index = Series(s_list, list('ABC'))

        expected = DataFrame({'a': [1, 0, 0],
                              'b': [0, 1, 0],
                              'c': [0, 0, 1]},
                             dtype=self.effective_dtype(dtype))
        if sparse:
            expected = expected.apply(pd.SparseArray, fill_value=0.0)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        assert_frame_equal(result, expected)

        expected.index = list('ABC')
        result = get_dummies(s_series_index, sparse=sparse, dtype=dtype)
        assert_frame_equal(result, expected)

    def test_basic_types(self, sparse, dtype):
        # GH 10531
        s_list = list('abc')
        s_series = Series(s_list)
        s_df = DataFrame({'a': [0, 1, 0, 1, 2],
                          'b': ['A', 'A', 'B', 'C', 'C'],
                          'c': [2, 3, 3, 3, 2]})

        expected = DataFrame({'a': [1, 0, 0],
                              'b': [0, 1, 0],
                              'c': [0, 0, 1]},
                             dtype=self.effective_dtype(dtype),
                             columns=list('abc'))
        if sparse:
            if is_integer_dtype(dtype):
                fill_value = 0
            elif dtype == bool:
                fill_value = False
            else:
                fill_value = 0.0

            expected = expected.apply(SparseArray, fill_value=fill_value)
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = get_dummies(s_df, columns=s_df.columns,
                             sparse=sparse, dtype=dtype)
        if sparse:
            dtype_name = 'Sparse[{}, {}]'.format(
                self.effective_dtype(dtype).name,
                fill_value
            )
        else:
            dtype_name = self.effective_dtype(dtype).name

        expected = Series({dtype_name: 8})
        tm.assert_series_equal(result.get_dtype_counts(), expected)

        result = get_dummies(s_df, columns=['a'], sparse=sparse, dtype=dtype)

        expected_counts = {'int64': 1, 'object': 1}
        expected_counts[dtype_name] = 3 + expected_counts.get(dtype_name, 0)

        expected = Series(expected_counts).sort_index()
        tm.assert_series_equal(result.get_dtype_counts().sort_index(),
                               expected)

    def test_just_na(self, sparse):
        just_na_list = [np.nan]
        just_na_series = Series(just_na_list)
        just_na_series_index = Series(just_na_list, index=['A'])

        res_list = get_dummies(just_na_list, sparse=sparse)
        res_series = get_dummies(just_na_series, sparse=sparse)
        res_series_index = get_dummies(just_na_series_index, sparse=sparse)

        assert res_list.empty
        assert res_series.empty
        assert res_series_index.empty

        assert res_list.index.tolist() == [0]
        assert res_series.index.tolist() == [0]
        assert res_series_index.index.tolist() == ['A']

    def test_include_na(self, sparse, dtype):
        s = ['a', 'b', np.nan]
        res = get_dummies(s, sparse=sparse, dtype=dtype)
        exp = DataFrame({'a': [1, 0, 0],
                         'b': [0, 1, 0]},
                        dtype=self.effective_dtype(dtype))
        if sparse:
            exp = exp.apply(pd.SparseArray, fill_value=0.0)
        assert_frame_equal(res, exp)

        # Sparse dataframes do not allow nan labelled columns, see #GH8822
        res_na = get_dummies(s, dummy_na=True, sparse=sparse, dtype=dtype)
        exp_na = DataFrame({nan: [0, 0, 1],
                            'a': [1, 0, 0],
                            'b': [0, 1, 0]},
                           dtype=self.effective_dtype(dtype))
        exp_na = exp_na.reindex(['a', 'b', nan], axis=1)
        # hack (NaN handling in assert_index_equal)
        exp_na.columns = res_na.columns
        if sparse:
            exp_na = exp_na.apply(pd.SparseArray, fill_value=0.0)
        assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies([nan], dummy_na=True,
                                  sparse=sparse, dtype=dtype)
        exp_just_na = DataFrame(Series(1, index=[0]), columns=[nan],
                                dtype=self.effective_dtype(dtype))
        tm.assert_numpy_array_equal(res_just_na.values, exp_just_na.values)

    def test_unicode(self, sparse):
        # See GH 6885 - get_dummies chokes on unicode values
        import unicodedata
        e = 'e'
        eacute = unicodedata.lookup('LATIN SMALL LETTER E WITH ACUTE')
        s = [e, eacute, eacute]
        res = get_dummies(s, prefix='letter', sparse=sparse)
        exp = DataFrame({'letter_e': [1, 0, 0],
                         u('letter_%s') % eacute: [0, 1, 1]},
                        dtype=np.uint8)
        if sparse:
            exp = exp.apply(pd.SparseArray, fill_value=0)
        assert_frame_equal(res, exp)

    def test_dataframe_dummies_all_obj(self, df, sparse):
        df = df[['A', 'B']]
        result = get_dummies(df, sparse=sparse)
        expected = DataFrame({'A_a': [1, 0, 1],
                              'A_b': [0, 1, 0],
                              'B_b': [1, 1, 0],
                              'B_c': [0, 0, 1]},
                             dtype=np.uint8)
        if sparse:
            expected = pd.DataFrame({
                "A_a": pd.SparseArray([1, 0, 1], dtype='uint8'),
                "A_b": pd.SparseArray([0, 1, 0], dtype='uint8'),
                "B_b": pd.SparseArray([1, 1, 0], dtype='uint8'),
                "B_c": pd.SparseArray([0, 0, 1], dtype='uint8'),
            })

        assert_frame_equal(result, expected)

    def test_dataframe_dummies_mix_default(self, df, sparse, dtype):
        result = get_dummies(df, sparse=sparse, dtype=dtype)
        if sparse:
            arr = SparseArray
            typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype
        expected = DataFrame({'C': [1, 2, 3],
                              'A_a': arr([1, 0, 1], dtype=typ),
                              'A_b': arr([0, 1, 0], dtype=typ),
                              'B_b': arr([1, 1, 0], dtype=typ),
                              'B_c': arr([0, 0, 1], dtype=typ)})
        expected = expected[['C', 'A_a', 'A_b', 'B_b', 'B_c']]
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_list(self, df, sparse):
        prefixes = ['from_A', 'from_B']
        result = get_dummies(df, prefix=prefixes, sparse=sparse)
        expected = DataFrame({'C': [1, 2, 3],
                              'from_A_a': [1, 0, 1],
                              'from_A_b': [0, 1, 0],
                              'from_B_b': [1, 1, 0],
                              'from_B_c': [0, 0, 1]},
                             dtype=np.uint8)
        expected[['C']] = df[['C']]
        cols = ['from_A_a', 'from_A_b', 'from_B_b', 'from_B_c']
        expected = expected[['C'] + cols]

        typ = pd.SparseArray if sparse else pd.Series
        expected[cols] = expected[cols].apply(lambda x: typ(x))
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_str(self, df, sparse):
        # not that you should do this...
        result = get_dummies(df, prefix='bad', sparse=sparse)
        bad_columns = ['bad_a', 'bad_b', 'bad_b', 'bad_c']
        expected = DataFrame([[1, 1, 0, 1, 0],
                              [2, 0, 1, 1, 0],
                              [3, 1, 0, 0, 1]],
                             columns=['C'] + bad_columns,
                             dtype=np.uint8)
        expected = expected.astype({"C": np.int64})
        if sparse:
            # work around astyping & assigning with duplicate columns
            # https://github.com/pandas-dev/pandas/issues/14427
            expected = pd.concat([
                pd.Series([1, 2, 3], name='C'),
                pd.Series([1, 0, 1], name='bad_a', dtype='Sparse[uint8]'),
                pd.Series([0, 1, 0], name='bad_b', dtype='Sparse[uint8]'),
                pd.Series([1, 1, 0], name='bad_b', dtype='Sparse[uint8]'),
                pd.Series([0, 0, 1], name='bad_c', dtype='Sparse[uint8]'),
            ], axis=1)

        assert_frame_equal(result, expected)

    def test_dataframe_dummies_subset(self, df, sparse):
        result = get_dummies(df, prefix=['from_A'], columns=['A'],
                             sparse=sparse)
        expected = DataFrame({'B': ['b', 'b', 'c'],
                              'C': [1, 2, 3],
                              'from_A_a': [1, 0, 1],
                              'from_A_b': [0, 1, 0]}, dtype=np.uint8)
        expected[['C']] = df[['C']]
        if sparse:
            cols = ['from_A_a', 'from_A_b']
            expected[cols] = expected[cols].apply(lambda x: pd.SparseSeries(x))
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_sep(self, df, sparse):
        result = get_dummies(df, prefix_sep='..', sparse=sparse)
        expected = DataFrame({'C': [1, 2, 3],
                              'A..a': [1, 0, 1],
                              'A..b': [0, 1, 0],
                              'B..b': [1, 1, 0],
                              'B..c': [0, 0, 1]},
                             dtype=np.uint8)
        expected[['C']] = df[['C']]
        expected = expected[['C', 'A..a', 'A..b', 'B..b', 'B..c']]
        if sparse:
            cols = ['A..a', 'A..b', 'B..b', 'B..c']
            expected[cols] = expected[cols].apply(lambda x: pd.SparseSeries(x))

        assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep=['..', '__'], sparse=sparse)
        expected = expected.rename(columns={'B..b': 'B__b', 'B..c': 'B__c'})
        assert_frame_equal(result, expected)

        result = get_dummies(df, prefix_sep={'A': '..', 'B': '__'},
                             sparse=sparse)
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_prefix_bad_length(self, df, sparse):
        with pytest.raises(ValueError):
            get_dummies(df, prefix=['too few'], sparse=sparse)

    def test_dataframe_dummies_prefix_sep_bad_length(self, df, sparse):
        with pytest.raises(ValueError):
            get_dummies(df, prefix_sep=['bad'], sparse=sparse)

    def test_dataframe_dummies_prefix_dict(self, sparse):
        prefixes = {'A': 'from_A', 'B': 'from_B'}
        df = DataFrame({'C': [1, 2, 3],
                        'A': ['a', 'b', 'a'],
                        'B': ['b', 'b', 'c']})
        result = get_dummies(df, prefix=prefixes, sparse=sparse)

        expected = DataFrame({'C': [1, 2, 3],
                              'from_A_a': [1, 0, 1],
                              'from_A_b': [0, 1, 0],
                              'from_B_b': [1, 1, 0],
                              'from_B_c': [0, 0, 1]})

        columns = ['from_A_a', 'from_A_b', 'from_B_b', 'from_B_c']
        expected[columns] = expected[columns].astype(np.uint8)
        if sparse:
            expected[columns] = expected[columns].apply(
                lambda x: pd.SparseSeries(x)
            )

        assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_na(self, df, sparse, dtype):
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(df, dummy_na=True,
                             sparse=sparse, dtype=dtype).sort_index(axis=1)

        if sparse:
            arr = SparseArray
            typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame({'C': [1, 2, 3, np.nan],
                              'A_a': arr([1, 0, 1, 0], dtype=typ),
                              'A_b': arr([0, 1, 0, 0], dtype=typ),
                              'A_nan': arr([0, 0, 0, 1], dtype=typ),
                              'B_b': arr([1, 1, 0, 0], dtype=typ),
                              'B_c': arr([0, 0, 1, 0], dtype=typ),
                              'B_nan': arr([0, 0, 0, 1], dtype=typ)
                              }).sort_index(axis=1)

        assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, sparse=sparse, dtype=dtype)
        expected = expected[['C', 'A_a', 'A_b', 'B_b', 'B_c']]
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_with_categorical(self, df, sparse, dtype):
        df['cat'] = pd.Categorical(['x', 'y', 'y'])
        result = get_dummies(df, sparse=sparse, dtype=dtype).sort_index(axis=1)
        if sparse:
            arr = SparseArray
            typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        expected = DataFrame({'C': [1, 2, 3],
                              'A_a': arr([1, 0, 1], dtype=typ),
                              'A_b': arr([0, 1, 0], dtype=typ),
                              'B_b': arr([1, 1, 0], dtype=typ),
                              'B_c': arr([0, 0, 1], dtype=typ),
                              'cat_x': arr([1, 0, 0], dtype=typ),
                              'cat_y': arr([0, 1, 1], dtype=typ)
                              }).sort_index(axis=1)

        assert_frame_equal(result, expected)

    @pytest.mark.parametrize('get_dummies_kwargs,expected', [
        ({'data': pd.DataFrame(({u'ä': ['a']}))},
         pd.DataFrame({u'ä_a': [1]}, dtype=np.uint8)),

        ({'data': pd.DataFrame({'x': [u'ä']})},
         pd.DataFrame({u'x_ä': [1]}, dtype=np.uint8)),

        ({'data': pd.DataFrame({'x': [u'a']}), 'prefix':u'ä'},
         pd.DataFrame({u'ä_a': [1]}, dtype=np.uint8)),

        ({'data': pd.DataFrame({'x': [u'a']}), 'prefix_sep':u'ä'},
         pd.DataFrame({u'xäa': [1]}, dtype=np.uint8))])
    def test_dataframe_dummies_unicode(self, get_dummies_kwargs, expected):
        # GH22084 pd.get_dummies incorrectly encodes unicode characters
        # in dataframe column names
        result = get_dummies(**get_dummies_kwargs)
        assert_frame_equal(result, expected)

    def test_basic_drop_first(self, sparse):
        # GH12402 Add a new parameter `drop_first` to avoid collinearity
        # Basic case
        s_list = list('abc')
        s_series = Series(s_list)
        s_series_index = Series(s_list, list('ABC'))

        expected = DataFrame({'b': [0, 1, 0],
                              'c': [0, 0, 1]},
                             dtype=np.uint8)

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        if sparse:
            expected = expected.apply(pd.SparseArray, fill_value=0)
        assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        assert_frame_equal(result, expected)

        expected.index = list('ABC')
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        assert_frame_equal(result, expected)

    def test_basic_drop_first_one_level(self, sparse):
        # Test the case that categorical variable only has one level.
        s_list = list('aaa')
        s_series = Series(s_list)
        s_series_index = Series(s_list, list('ABC'))

        expected = DataFrame(index=np.arange(3))

        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        assert_frame_equal(result, expected)

        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        assert_frame_equal(result, expected)

        expected = DataFrame(index=list('ABC'))
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        assert_frame_equal(result, expected)

    def test_basic_drop_first_NA(self, sparse):
        # Test NA handling together with drop_first
        s_NA = ['a', 'b', np.nan]
        res = get_dummies(s_NA, drop_first=True, sparse=sparse)
        exp = DataFrame({'b': [0, 1, 0]}, dtype=np.uint8)
        if sparse:
            exp = exp.apply(pd.SparseArray, fill_value=0)

        assert_frame_equal(res, exp)

        res_na = get_dummies(s_NA, dummy_na=True, drop_first=True,
                             sparse=sparse)
        exp_na = DataFrame(
            {'b': [0, 1, 0],
             nan: [0, 0, 1]},
            dtype=np.uint8).reindex(['b', nan], axis=1)
        if sparse:
            exp_na = exp_na.apply(pd.SparseArray, fill_value=0)
        assert_frame_equal(res_na, exp_na)

        res_just_na = get_dummies([nan], dummy_na=True, drop_first=True,
                                  sparse=sparse)
        exp_just_na = DataFrame(index=np.arange(1))
        assert_frame_equal(res_just_na, exp_just_na)

    def test_dataframe_dummies_drop_first(self, df, sparse):
        df = df[['A', 'B']]
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame({'A_b': [0, 1, 0],
                              'B_c': [0, 0, 1]},
                             dtype=np.uint8)
        if sparse:
            expected = expected.apply(pd.SparseArray, fill_value=0)
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_categorical(
            self, df, sparse, dtype):
        df['cat'] = pd.Categorical(['x', 'y', 'y'])
        result = get_dummies(df, drop_first=True, sparse=sparse)
        expected = DataFrame({'C': [1, 2, 3],
                              'A_b': [0, 1, 0],
                              'B_c': [0, 0, 1],
                              'cat_y': [0, 1, 1]})
        cols = ['A_b', 'B_c', 'cat_y']
        expected[cols] = expected[cols].astype(np.uint8)
        expected = expected[['C', 'A_b', 'B_c', 'cat_y']]
        if sparse:
            for col in cols:
                expected[col] = pd.SparseSeries(expected[col])
        assert_frame_equal(result, expected)

    def test_dataframe_dummies_drop_first_with_na(self, df, sparse):
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        result = get_dummies(df, dummy_na=True, drop_first=True,
                             sparse=sparse).sort_index(axis=1)
        expected = DataFrame({'C': [1, 2, 3, np.nan],
                              'A_b': [0, 1, 0, 0],
                              'A_nan': [0, 0, 0, 1],
                              'B_c': [0, 0, 1, 0],
                              'B_nan': [0, 0, 0, 1]})
        cols = ['A_b', 'A_nan', 'B_c', 'B_nan']
        expected[cols] = expected[cols].astype(np.uint8)
        expected = expected.sort_index(axis=1)
        if sparse:
            for col in cols:
                expected[col] = pd.SparseSeries(expected[col])

        assert_frame_equal(result, expected)

        result = get_dummies(df, dummy_na=False, drop_first=True,
                             sparse=sparse)
        expected = expected[['C', 'A_b', 'B_c']]
        assert_frame_equal(result, expected)

    def test_int_int(self):
        data = Series([1, 2, 1])
        result = pd.get_dummies(data)
        expected = DataFrame([[1, 0],
                              [0, 1],
                              [1, 0]],
                             columns=[1, 2],
                             dtype=np.uint8)
        tm.assert_frame_equal(result, expected)

        data = Series(pd.Categorical(['a', 'b', 'a']))
        result = pd.get_dummies(data)
        expected = DataFrame([[1, 0],
                              [0, 1],
                              [1, 0]],
                             columns=pd.Categorical(['a', 'b']),
                             dtype=np.uint8)
        tm.assert_frame_equal(result, expected)

    def test_int_df(self, dtype):
        data = DataFrame(
            {'A': [1, 2, 1],
             'B': pd.Categorical(['a', 'b', 'a']),
             'C': [1, 2, 1],
             'D': [1., 2., 1.]
             }
        )
        columns = ['C', 'D', 'A_1', 'A_2', 'B_a', 'B_b']
        expected = DataFrame([
            [1, 1., 1, 0, 1, 0],
            [2, 2., 0, 1, 0, 1],
            [1, 1., 1, 0, 1, 0]
        ], columns=columns)
        expected[columns[2:]] = expected[columns[2:]].astype(dtype)
        result = pd.get_dummies(data, columns=['A', 'B'], dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_dataframe_dummies_preserve_categorical_dtype(self, dtype):
        # GH13854
        for ordered in [False, True]:
            cat = pd.Categorical(list("xy"), categories=list("xyz"),
                                 ordered=ordered)
            result = get_dummies(cat, dtype=dtype)

            data = np.array([[1, 0, 0], [0, 1, 0]],
                            dtype=self.effective_dtype(dtype))
            cols = pd.CategoricalIndex(cat.categories,
                                       categories=cat.categories,
                                       ordered=ordered)
            expected = DataFrame(data, columns=cols,
                                 dtype=self.effective_dtype(dtype))

            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('sparse', [True, False])
    def test_get_dummies_dont_sparsify_all_columns(self, sparse):
        # GH18914
        df = DataFrame.from_dict(OrderedDict([('GDP', [1, 2]),
                                              ('Nation', ['AB', 'CD'])]))
        df = get_dummies(df, columns=['Nation'], sparse=sparse)
        df2 = df.reindex(columns=['GDP'])

        tm.assert_frame_equal(df[['GDP']], df2)

    def test_get_dummies_duplicate_columns(self, df):
        # GH20839
        df.columns = ["A", "A", "A"]
        result = get_dummies(df).sort_index(axis=1)

        expected = DataFrame([[1, 1, 0, 1, 0],
                              [2, 0, 1, 1, 0],
                              [3, 1, 0, 0, 1]],
                             columns=['A', 'A_a', 'A_b', 'A_b', 'A_c'],
                             dtype=np.uint8).sort_index(axis=1)

        expected = expected.astype({"A": np.int64})

        tm.assert_frame_equal(result, expected)


class TestCategoricalReshape(object):

    @pytest.mark.filterwarnings("ignore:\\nPanel:FutureWarning")
    def test_reshaping_panel_categorical(self):

        p = tm.makePanel()
        p['str'] = 'foo'
        df = p.to_frame()

        df['category'] = df['str'].astype('category')
        result = df['category'].unstack()

        c = Categorical(['foo'] * len(p.major_axis))
        expected = DataFrame({'A': c.copy(),
                              'B': c.copy(),
                              'C': c.copy(),
                              'D': c.copy()},
                             columns=Index(list('ABCD'), name='minor'),
                             index=p.major_axis.set_names('major'))
        tm.assert_frame_equal(result, expected)


class TestMakeAxisDummies(object):

    def test_preserve_categorical_dtype(self):
        # GH13854
        for ordered in [False, True]:
            cidx = pd.CategoricalIndex(list("xyz"), ordered=ordered)
            midx = pd.MultiIndex(levels=[['a'], cidx],
                                 codes=[[0, 0], [0, 1]])
            df = DataFrame([[10, 11]], index=midx)

            expected = DataFrame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                 index=midx, columns=cidx)

            from pandas.core.reshape.reshape import make_axis_dummies
            result = make_axis_dummies(df)
            tm.assert_frame_equal(result, expected)

            result = make_axis_dummies(df, transform=lambda x: x)
            tm.assert_frame_equal(result, expected)
