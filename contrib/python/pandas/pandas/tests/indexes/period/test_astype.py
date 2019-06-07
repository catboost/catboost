# -*- coding: utf-8 -*-

import numpy as np
import pytest

import pandas as pd
from pandas import Index, Int64Index, NaT, Period, PeriodIndex, period_range
import pandas.util.testing as tm


class TestPeriodIndexAsType(object):
    @pytest.mark.parametrize('dtype', [
        float, 'timedelta64', 'timedelta64[ns]'])
    def test_astype_raises(self, dtype):
        # GH#13149, GH#13209
        idx = PeriodIndex(['2016-05-16', 'NaT', NaT, np.NaN], freq='D')
        msg = 'Cannot cast PeriodArray to dtype'
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_conversion(self):
        # GH#13149, GH#13209
        idx = PeriodIndex(['2016-05-16', 'NaT', NaT, np.NaN], freq='D')

        result = idx.astype(object)
        expected = Index([Period('2016-05-16', freq='D')] +
                         [Period(NaT, freq='D')] * 3, dtype='object')
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Int64Index([16937] + [-9223372036854775808] * 3,
                              dtype=np.int64)
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        expected = Index(str(x) for x in idx)
        tm.assert_index_equal(result, expected)

        idx = period_range('1990', '2009', freq='A')
        result = idx.astype('i8')
        tm.assert_index_equal(result, Index(idx.asi8))
        tm.assert_numpy_array_equal(result.values, idx.asi8)

    def test_astype_uint(self):
        arr = period_range('2000', periods=2)
        expected = pd.UInt64Index(np.array([10957, 10958], dtype='uint64'))
        tm.assert_index_equal(arr.astype("uint64"), expected)
        tm.assert_index_equal(arr.astype("uint32"), expected)

    def test_astype_object(self):
        idx = pd.PeriodIndex([], freq='M')

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        idx = pd.PeriodIndex(['2011-01', pd.NaT], freq='M')

        exp = np.array([pd.Period('2011-01', freq='M'), pd.NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        exp = np.array([pd.Period('2011-01-01', freq='D'), pd.NaT],
                       dtype=object)
        idx = pd.PeriodIndex(['2011-01-01', pd.NaT], freq='D')
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

    # TODO: de-duplicate this version (from test_ops) with the one above
    # (from test_period)
    def test_astype_object2(self):
        idx = pd.period_range(start='2013-01-01', periods=4, freq='M',
                              name='idx')
        expected_list = [pd.Period('2013-01-31', freq='M'),
                         pd.Period('2013-02-28', freq='M'),
                         pd.Period('2013-03-31', freq='M'),
                         pd.Period('2013-04-30', freq='M')]
        expected = pd.Index(expected_list, dtype=object, name='idx')
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert idx.tolist() == expected_list

        idx = PeriodIndex(['2013-01-01', '2013-01-02', 'NaT',
                           '2013-01-04'], freq='D', name='idx')
        expected_list = [pd.Period('2013-01-01', freq='D'),
                         pd.Period('2013-01-02', freq='D'),
                         pd.Period('NaT', freq='D'),
                         pd.Period('2013-01-04', freq='D')]
        expected = pd.Index(expected_list, dtype=object, name='idx')
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        for i in [0, 1, 3]:
            assert result[i] == expected[i]
        assert result[2] is pd.NaT
        assert result.name == expected.name

        result_list = idx.tolist()
        for i in [0, 1, 3]:
            assert result_list[i] == expected_list[i]
        assert result_list[2] is pd.NaT

    def test_astype_category(self):
        obj = pd.period_range("2000", periods=2)
        result = obj.astype('category')
        expected = pd.CategoricalIndex([pd.Period('2000-01-01', freq="D"),
                                        pd.Period('2000-01-02', freq="D")])
        tm.assert_index_equal(result, expected)

        result = obj._data.astype('category')
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        obj = pd.period_range("2000", periods=2)
        result = obj.astype(bool)
        expected = pd.Index(np.array([True, True]))
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
