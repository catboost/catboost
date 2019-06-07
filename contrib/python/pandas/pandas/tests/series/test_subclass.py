# coding=utf-8
# pylint: disable-msg=E1101,W0612
import numpy as np

import pandas as pd
from pandas import SparseDtype
import pandas.util.testing as tm


class TestSeriesSubclassing(object):

    def test_indexing_sliced(self):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'))
        res = s.loc[['a', 'b']]
        exp = tm.SubclassedSeries([1, 2], index=list('ab'))
        tm.assert_series_equal(res, exp)

        res = s.iloc[[2, 3]]
        exp = tm.SubclassedSeries([3, 4], index=list('cd'))
        tm.assert_series_equal(res, exp)

        res = s.loc[['a', 'b']]
        exp = tm.SubclassedSeries([1, 2], index=list('ab'))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list('abcd'), name='xxx')
        res = s.to_frame()
        exp = tm.SubclassedDataFrame({'xxx': [1, 2, 3, 4]}, index=list('abcd'))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # GH 15564
        s = tm.SubclassedSeries(
            [1, 2, 3, 4], index=[list('aabb'), list('xyxy')])

        res = s.unstack()
        exp = tm.SubclassedDataFrame(
            {'x': [1, 3], 'y': [2, 4]}, index=['a', 'b'])

        tm.assert_frame_equal(res, exp)


class TestSparseSeriesSubclassing(object):

    def test_subclass_sparse_slice(self):
        # int64
        s = tm.SubclassedSparseSeries([1, 2, 3, 4, 5])
        exp = tm.SubclassedSparseSeries([2, 3, 4], index=[1, 2, 3])
        tm.assert_sp_series_equal(s.loc[1:3], exp)
        assert s.loc[1:3].dtype == SparseDtype(np.int64)

        exp = tm.SubclassedSparseSeries([2, 3], index=[1, 2])
        tm.assert_sp_series_equal(s.iloc[1:3], exp)
        assert s.iloc[1:3].dtype == SparseDtype(np.int64)

        exp = tm.SubclassedSparseSeries([2, 3], index=[1, 2])
        tm.assert_sp_series_equal(s[1:3], exp)
        assert s[1:3].dtype == SparseDtype(np.int64)

        # float64
        s = tm.SubclassedSparseSeries([1., 2., 3., 4., 5.])
        exp = tm.SubclassedSparseSeries([2., 3., 4.], index=[1, 2, 3])
        tm.assert_sp_series_equal(s.loc[1:3], exp)
        assert s.loc[1:3].dtype == SparseDtype(np.float64)

        exp = tm.SubclassedSparseSeries([2., 3.], index=[1, 2])
        tm.assert_sp_series_equal(s.iloc[1:3], exp)
        assert s.iloc[1:3].dtype == SparseDtype(np.float64)

        exp = tm.SubclassedSparseSeries([2., 3.], index=[1, 2])
        tm.assert_sp_series_equal(s[1:3], exp)
        assert s[1:3].dtype == SparseDtype(np.float64)

    def test_subclass_sparse_addition(self):
        s1 = tm.SubclassedSparseSeries([1, 3, 5])
        s2 = tm.SubclassedSparseSeries([-2, 5, 12])
        exp = tm.SubclassedSparseSeries([-1, 8, 17])
        tm.assert_sp_series_equal(s1 + s2, exp)

        s1 = tm.SubclassedSparseSeries([4.0, 5.0, 6.0])
        s2 = tm.SubclassedSparseSeries([1.0, 2.0, 3.0])
        exp = tm.SubclassedSparseSeries([5., 7., 9.])
        tm.assert_sp_series_equal(s1 + s2, exp)

    def test_subclass_sparse_to_frame(self):
        s = tm.SubclassedSparseSeries([1, 2], index=list('ab'), name='xxx')
        res = s.to_frame()

        exp_arr = pd.SparseArray([1, 2], dtype=np.int64, kind='block',
                                 fill_value=0)
        exp = tm.SubclassedSparseDataFrame({'xxx': exp_arr},
                                           index=list('ab'),
                                           default_fill_value=0)
        tm.assert_sp_frame_equal(res, exp)

        # create from int dict
        res = tm.SubclassedSparseDataFrame({'xxx': [1, 2]},
                                           index=list('ab'),
                                           default_fill_value=0)
        tm.assert_sp_frame_equal(res, exp)

        s = tm.SubclassedSparseSeries([1.1, 2.1], index=list('ab'),
                                      name='xxx')
        res = s.to_frame()
        exp = tm.SubclassedSparseDataFrame({'xxx': [1.1, 2.1]},
                                           index=list('ab'))
        tm.assert_sp_frame_equal(res, exp)
