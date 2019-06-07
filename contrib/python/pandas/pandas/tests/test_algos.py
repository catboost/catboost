# -*- coding: utf-8 -*-

from datetime import datetime
from itertools import permutations
import struct

import numpy as np
from numpy import nan
from numpy.random import RandomState
import pytest

from pandas._libs import (
    algos as libalgos, groupby as libgroupby, hashtable as ht)
from pandas.compat import lrange, range
from pandas.compat.numpy import np_array_datetime64_compat
import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import CategoricalDtype as CDT

import pandas as pd
from pandas import (
    Categorical, CategoricalIndex, DatetimeIndex, Index, IntervalIndex, Series,
    Timestamp, compat)
import pandas.core.algorithms as algos
from pandas.core.arrays import DatetimeArray
import pandas.core.common as com
import pandas.util.testing as tm
from pandas.util.testing import assert_almost_equal


class TestMatch(object):

    def test_ints(self):
        values = np.array([0, 2, 1])
        to_match = np.array([0, 1, 2, 2, 0, 1, 3, 0])

        result = algos.match(to_match, values)
        expected = np.array([0, 2, 1, 1, 0, 2, -1, 0], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        result = Series(algos.match(to_match, values, np.nan))
        expected = Series(np.array([0, 2, 1, 1, 0, 2, np.nan, 0]))
        tm.assert_series_equal(result, expected)

        s = Series(np.arange(5), dtype=np.float32)
        result = algos.match(s, [2, 4])
        expected = np.array([-1, -1, 0, -1, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        result = Series(algos.match(s, [2, 4], np.nan))
        expected = Series(np.array([np.nan, np.nan, 0, np.nan, 1]))
        tm.assert_series_equal(result, expected)

    def test_strings(self):
        values = ['foo', 'bar', 'baz']
        to_match = ['bar', 'foo', 'qux', 'foo', 'bar', 'baz', 'qux']

        result = algos.match(to_match, values)
        expected = np.array([1, 0, -1, 0, 1, 2, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        result = Series(algos.match(to_match, values, np.nan))
        expected = Series(np.array([1, 0, np.nan, 0, 1, 2, np.nan]))
        tm.assert_series_equal(result, expected)


class TestFactorize(object):

    def test_basic(self):

        labels, uniques = algos.factorize(['a', 'b', 'b', 'a', 'a', 'c', 'c',
                                           'c'])
        tm.assert_numpy_array_equal(
            uniques, np.array(['a', 'b', 'c'], dtype=object))

        labels, uniques = algos.factorize(['a', 'b', 'b', 'a',
                                           'a', 'c', 'c', 'c'], sort=True)
        exp = np.array([0, 1, 1, 0, 0, 2, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = np.array(['a', 'b', 'c'], dtype=object)
        tm.assert_numpy_array_equal(uniques, exp)

        labels, uniques = algos.factorize(list(reversed(range(5))))
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.int64)
        tm.assert_numpy_array_equal(uniques, exp)

        labels, uniques = algos.factorize(list(reversed(range(5))), sort=True)

        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        tm.assert_numpy_array_equal(uniques, exp)

        labels, uniques = algos.factorize(list(reversed(np.arange(5.))))
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = np.array([4., 3., 2., 1., 0.], dtype=np.float64)
        tm.assert_numpy_array_equal(uniques, exp)

        labels, uniques = algos.factorize(list(reversed(np.arange(5.))),
                                          sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = np.array([0., 1., 2., 3., 4.], dtype=np.float64)
        tm.assert_numpy_array_equal(uniques, exp)

    def test_mixed(self):

        # doc example reshaping.rst
        x = Series(['A', 'A', np.nan, 'B', 3.14, np.inf])
        labels, uniques = algos.factorize(x)

        exp = np.array([0, 0, -1, 1, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = Index(['A', 'B', 3.14, np.inf])
        tm.assert_index_equal(uniques, exp)

        labels, uniques = algos.factorize(x, sort=True)
        exp = np.array([2, 2, -1, 3, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = Index([3.14, np.inf, 'A', 'B'])
        tm.assert_index_equal(uniques, exp)

    def test_datelike(self):

        # M8
        v1 = Timestamp('20130101 09:00:00.00004')
        v2 = Timestamp('20130101')
        x = Series([v1, v1, v1, v2, v2, v1])
        labels, uniques = algos.factorize(x)

        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = DatetimeIndex([v1, v2])
        tm.assert_index_equal(uniques, exp)

        labels, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        exp = DatetimeIndex([v2, v1])
        tm.assert_index_equal(uniques, exp)

        # period
        v1 = pd.Period('201302', freq='M')
        v2 = pd.Period('201303', freq='M')
        x = Series([v1, v1, v1, v2, v2, v1])

        # periods are not 'sorted' as they are converted back into an index
        labels, uniques = algos.factorize(x)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        tm.assert_index_equal(uniques, pd.PeriodIndex([v1, v2]))

        labels, uniques = algos.factorize(x, sort=True)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        tm.assert_index_equal(uniques, pd.PeriodIndex([v1, v2]))

        # GH 5986
        v1 = pd.to_timedelta('1 day 1 min')
        v2 = pd.to_timedelta('1 day')
        x = Series([v1, v2, v1, v1, v2, v2, v1])
        labels, uniques = algos.factorize(x)
        exp = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        tm.assert_index_equal(uniques, pd.to_timedelta([v1, v2]))

        labels, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, exp)
        tm.assert_index_equal(uniques, pd.to_timedelta([v2, v1]))

    def test_factorize_nan(self):
        # nan should map to na_sentinel, not reverse_indexer[na_sentinel]
        # rizer.factorize should not raise an exception if na_sentinel indexes
        # outside of reverse_indexer
        key = np.array([1, 2, 1, np.nan], dtype='O')
        rizer = ht.Factorizer(len(key))
        for na_sentinel in (-1, 20):
            ids = rizer.factorize(key, sort=True, na_sentinel=na_sentinel)
            expected = np.array([0, 1, 0, na_sentinel], dtype='int32')
            assert len(set(key)) == len(set(expected))
            tm.assert_numpy_array_equal(pd.isna(key),
                                        expected == na_sentinel)

        # nan still maps to na_sentinel when sort=False
        key = np.array([0, np.nan, 1], dtype='O')
        na_sentinel = -1

        # TODO(wesm): unused?
        ids = rizer.factorize(key, sort=False, na_sentinel=na_sentinel)  # noqa

        expected = np.array([2, -1, 0], dtype='int32')
        assert len(set(key)) == len(set(expected))
        tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)

    @pytest.mark.parametrize("data,expected_label,expected_level", [
        (
            [(1, 1), (1, 2), (0, 0), (1, 2), 'nonsense'],
            [0, 1, 2, 1, 3],
            [(1, 1), (1, 2), (0, 0), 'nonsense']
        ),
        (
            [(1, 1), (1, 2), (0, 0), (1, 2), (1, 2, 3)],
            [0, 1, 2, 1, 3],
            [(1, 1), (1, 2), (0, 0), (1, 2, 3)]
        ),
        (
            [(1, 1), (1, 2), (0, 0), (1, 2)],
            [0, 1, 2, 1],
            [(1, 1), (1, 2), (0, 0)]
        )
    ])
    def test_factorize_tuple_list(self, data, expected_label, expected_level):
        # GH9454
        result = pd.factorize(data)

        tm.assert_numpy_array_equal(result[0],
                                    np.array(expected_label, dtype=np.intp))

        expected_level_array = com.asarray_tuplesafe(expected_level,
                                                     dtype=object)
        tm.assert_numpy_array_equal(result[1], expected_level_array)

    def test_complex_sorting(self):
        # gh 12666 - check no segfault
        x17 = np.array([complex(i) for i in range(17)], dtype=object)

        pytest.raises(TypeError, algos.factorize, x17[::-1], sort=True)

    def test_float64_factorize(self, writable):
        data = np.array([1.0, 1e8, 1.0, 1e-8, 1e8, 1.0], dtype=np.float64)
        data.setflags(write=writable)
        exp_labels = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        exp_uniques = np.array([1.0, 1e8, 1e-8], dtype=np.float64)

        labels, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(labels, exp_labels)
        tm.assert_numpy_array_equal(uniques, exp_uniques)

    def test_uint64_factorize(self, writable):
        data = np.array([2**64 - 1, 1, 2**64 - 1], dtype=np.uint64)
        data.setflags(write=writable)
        exp_labels = np.array([0, 1, 0], dtype=np.intp)
        exp_uniques = np.array([2**64 - 1, 1], dtype=np.uint64)

        labels, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(labels, exp_labels)
        tm.assert_numpy_array_equal(uniques, exp_uniques)

    def test_int64_factorize(self, writable):
        data = np.array([2**63 - 1, -2**63, 2**63 - 1], dtype=np.int64)
        data.setflags(write=writable)
        exp_labels = np.array([0, 1, 0], dtype=np.intp)
        exp_uniques = np.array([2**63 - 1, -2**63], dtype=np.int64)

        labels, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(labels, exp_labels)
        tm.assert_numpy_array_equal(uniques, exp_uniques)

    def test_string_factorize(self, writable):
        data = np.array(['a', 'c', 'a', 'b', 'c'],
                        dtype=object)
        data.setflags(write=writable)
        exp_labels = np.array([0, 1, 0, 2, 1], dtype=np.intp)
        exp_uniques = np.array(['a', 'c', 'b'], dtype=object)

        labels, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(labels, exp_labels)
        tm.assert_numpy_array_equal(uniques, exp_uniques)

    def test_object_factorize(self, writable):
        data = np.array(['a', 'c', None, np.nan, 'a', 'b', pd.NaT, 'c'],
                        dtype=object)
        data.setflags(write=writable)
        exp_labels = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
        exp_uniques = np.array(['a', 'c', 'b'], dtype=object)

        labels, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(labels, exp_labels)
        tm.assert_numpy_array_equal(uniques, exp_uniques)

    def test_deprecate_order(self):
        # gh 19727 - check warning is raised for deprecated keyword, order.
        # Test not valid once order keyword is removed.
        data = np.array([2**63, 1, 2**63], dtype=np.uint64)
        with tm.assert_produces_warning(expected_warning=FutureWarning):
            algos.factorize(data, order=True)
        with tm.assert_produces_warning(False):
            algos.factorize(data)

    @pytest.mark.parametrize('data', [
        np.array([0, 1, 0], dtype='u8'),
        np.array([-2**63, 1, -2**63], dtype='i8'),
        np.array(['__nan__', 'foo', '__nan__'], dtype='object'),
    ])
    def test_parametrized_factorize_na_value_default(self, data):
        # arrays that include the NA default for that type, but isn't used.
        l, u = algos.factorize(data)
        expected_uniques = data[[0, 1]]
        expected_labels = np.array([0, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(l, expected_labels)
        tm.assert_numpy_array_equal(u, expected_uniques)

    @pytest.mark.parametrize('data, na_value', [
        (np.array([0, 1, 0, 2], dtype='u8'), 0),
        (np.array([1, 0, 1, 2], dtype='u8'), 1),
        (np.array([-2**63, 1, -2**63, 0], dtype='i8'), -2**63),
        (np.array([1, -2**63, 1, 0], dtype='i8'), 1),
        (np.array(['a', '', 'a', 'b'], dtype=object), 'a'),
        (np.array([(), ('a', 1), (), ('a', 2)], dtype=object), ()),
        (np.array([('a', 1), (), ('a', 1), ('a', 2)], dtype=object),
         ('a', 1)),
    ])
    def test_parametrized_factorize_na_value(self, data, na_value):
        l, u = algos._factorize_array(data, na_value=na_value)
        expected_uniques = data[[1, 3]]
        expected_labels = np.array([-1, 0, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(l, expected_labels)
        tm.assert_numpy_array_equal(u, expected_uniques)

    @pytest.mark.parametrize('sort', [True, False])
    @pytest.mark.parametrize('na_sentinel', [-1, -10, 100])
    def test_factorize_na_sentinel(self, sort, na_sentinel):
        data = np.array(['b', 'a', None, 'b'], dtype=object)
        labels, uniques = algos.factorize(data, sort=sort,
                                          na_sentinel=na_sentinel)
        if sort:
            expected_labels = np.array([1, 0, na_sentinel, 1], dtype=np.intp)
            expected_uniques = np.array(['a', 'b'], dtype=object)
        else:
            expected_labels = np.array([0, 1, na_sentinel, 0], dtype=np.intp)
            expected_uniques = np.array(['b', 'a'], dtype=object)
        tm.assert_numpy_array_equal(labels, expected_labels)
        tm.assert_numpy_array_equal(uniques, expected_uniques)


class TestUnique(object):

    def test_ints(self):
        arr = np.random.randint(0, 100, size=50)

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_objects(self):
        arr = np.random.randint(0, 100, size=50).astype('O')

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_object_refcount_bug(self):
        lst = ['A', 'B', 'C', 'D', 'E']
        for i in range(1000):
            len(algos.unique(lst))

    def test_on_index_object(self):

        mindex = pd.MultiIndex.from_arrays([np.arange(5).repeat(5), np.tile(
            np.arange(5), 5)])
        expected = mindex.values
        expected.sort()

        mindex = mindex.repeat(2)

        result = pd.unique(mindex)
        result.sort()

        tm.assert_almost_equal(result, expected)

    def test_datetime64_dtype_array_returned(self):
        # GH 9431
        expected = np_array_datetime64_compat(
            ['2015-01-03T00:00:00.000000000+0000',
             '2015-01-01T00:00:00.000000000+0000'],
            dtype='M8[ns]')

        dt_index = pd.to_datetime(['2015-01-03T00:00:00.000000000',
                                   '2015-01-01T00:00:00.000000000',
                                   '2015-01-01T00:00:00.000000000'])
        result = algos.unique(dt_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(dt_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_timedelta64_dtype_array_returned(self):
        # GH 9431
        expected = np.array([31200, 45678, 10000], dtype='m8[ns]')

        td_index = pd.to_timedelta([31200, 45678, 31200, 10000, 45678])
        result = algos.unique(td_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(td_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_uint64_overflow(self):
        s = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.unique(s), exp)

    def test_nan_in_object_array(self):
        duplicated_items = ['a', np.nan, 'c', 'c']
        result = pd.unique(duplicated_items)
        expected = np.array(['a', np.nan, 'c'], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical(self):

        # we are expecting to return in the order
        # of appearance
        expected = Categorical(list('bac'), categories=list('bac'))

        # we are expecting to return in the order
        # of the categories
        expected_o = Categorical(
            list('bac'), categories=list('abc'), ordered=True)

        # GH 15939
        c = Categorical(list('baabc'))
        result = c.unique()
        tm.assert_categorical_equal(result, expected)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected)

        c = Categorical(list('baabc'), ordered=True)
        result = c.unique()
        tm.assert_categorical_equal(result, expected_o)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected_o)

        # Series of categorical dtype
        s = Series(Categorical(list('baabc')), name='foo')
        result = s.unique()
        tm.assert_categorical_equal(result, expected)

        result = pd.unique(s)
        tm.assert_categorical_equal(result, expected)

        # CI -> return CI
        ci = CategoricalIndex(Categorical(list('baabc'),
                                          categories=list('bac')))
        expected = CategoricalIndex(expected)
        result = ci.unique()
        tm.assert_index_equal(result, expected)

        result = pd.unique(ci)
        tm.assert_index_equal(result, expected)

    def test_datetime64tz_aware(self):
        # GH 15939

        result = Series(
            Index([Timestamp('20160101', tz='US/Eastern'),
                   Timestamp('20160101', tz='US/Eastern')])).unique()
        expected = DatetimeArray._from_sequence(np.array([
            Timestamp('2016-01-01 00:00:00-0500', tz="US/Eastern")
        ]))
        tm.assert_extension_array_equal(result, expected)

        result = Index([Timestamp('20160101', tz='US/Eastern'),
                        Timestamp('20160101', tz='US/Eastern')]).unique()
        expected = DatetimeIndex(['2016-01-01 00:00:00'],
                                 dtype='datetime64[ns, US/Eastern]', freq=None)
        tm.assert_index_equal(result, expected)

        result = pd.unique(
            Series(Index([Timestamp('20160101', tz='US/Eastern'),
                          Timestamp('20160101', tz='US/Eastern')])))
        expected = DatetimeArray._from_sequence(np.array([
            Timestamp('2016-01-01', tz="US/Eastern"),
        ]))
        tm.assert_extension_array_equal(result, expected)

        result = pd.unique(Index([Timestamp('20160101', tz='US/Eastern'),
                                  Timestamp('20160101', tz='US/Eastern')]))
        expected = DatetimeIndex(['2016-01-01 00:00:00'],
                                 dtype='datetime64[ns, US/Eastern]', freq=None)
        tm.assert_index_equal(result, expected)

    def test_order_of_appearance(self):
        # 9346
        # light testing of guarantee of order of appearance
        # these also are the doc-examples
        result = pd.unique(Series([2, 1, 3, 3]))
        tm.assert_numpy_array_equal(result,
                                    np.array([2, 1, 3], dtype='int64'))

        result = pd.unique(Series([2] + [1] * 5))
        tm.assert_numpy_array_equal(result,
                                    np.array([2, 1], dtype='int64'))

        result = pd.unique(Series([Timestamp('20160101'),
                                   Timestamp('20160101')]))
        expected = np.array(['2016-01-01T00:00:00.000000000'],
                            dtype='datetime64[ns]')
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(Index(
            [Timestamp('20160101', tz='US/Eastern'),
             Timestamp('20160101', tz='US/Eastern')]))
        expected = DatetimeIndex(['2016-01-01 00:00:00'],
                                 dtype='datetime64[ns, US/Eastern]',
                                 freq=None)
        tm.assert_index_equal(result, expected)

        result = pd.unique(list('aabc'))
        expected = np.array(['a', 'b', 'c'], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(Series(Categorical(list('aabc'))))
        expected = Categorical(list('abc'))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("arg ,expected", [
        (('1', '1', '2'), np.array(['1', '2'], dtype=object)),
        (('foo',), np.array(['foo'], dtype=object))
    ])
    def test_tuple_with_strings(self, arg, expected):
        # see GH 17108
        result = pd.unique(arg)
        tm.assert_numpy_array_equal(result, expected)

    def test_obj_none_preservation(self):
        # GH 20866
        arr = np.array(['foo', None], dtype=object)
        result = pd.unique(arr)
        expected = np.array(['foo', None], dtype=object)

        tm.assert_numpy_array_equal(result, expected, strict_nan=True)

    def test_signed_zero(self):
        # GH 21866
        a = np.array([-0.0, 0.0])
        result = pd.unique(a)
        expected = np.array([-0.0])  # 0.0 and -0.0 are equivalent
        tm.assert_numpy_array_equal(result, expected)

    def test_different_nans(self):
        # GH 21866
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2])  # NAN1 and NAN2 are equivalent
        result = pd.unique(a)
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

    def test_first_nan_kept(self):
        # GH 22295
        # create different nans from bit-patterns:
        bits_for_nan1 = 0xfff8000000000001
        bits_for_nan2 = 0x7ff8000000000001
        NAN1 = struct.unpack("d", struct.pack("=Q", bits_for_nan1))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", bits_for_nan2))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        for el_type in [np.float64, np.object]:
            a = np.array([NAN1, NAN2], dtype=el_type)
            result = pd.unique(a)
            assert result.size == 1
            # use bit patterns to identify which nan was kept:
            result_nan_bits = struct.unpack("=Q",
                                            struct.pack("d", result[0]))[0]
            assert result_nan_bits == bits_for_nan1

    def test_do_not_mangle_na_values(self, unique_nulls_fixture,
                                     unique_nulls_fixture2):
        # GH 22295
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values not unique
        a = np.array([unique_nulls_fixture,
                      unique_nulls_fixture2], dtype=np.object)
        result = pd.unique(a)
        assert result.size == 2
        assert a[0] is unique_nulls_fixture
        assert a[1] is unique_nulls_fixture2


class TestIsin(object):

    def test_invalid(self):

        pytest.raises(TypeError, lambda: algos.isin(1, 1))
        pytest.raises(TypeError, lambda: algos.isin(1, [1]))
        pytest.raises(TypeError, lambda: algos.isin([1], 1))

    def test_basic(self):

        result = algos.isin([1, 2], [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(['a', 'b'], ['a'])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(['a', 'b']), Series(['a']))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(['a', 'b']), {'a'})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(['a', 'b'], [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):

        arr = pd.date_range('20130101', periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        arr = pd.timedelta_range('1 day', periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_large(self):

        s = pd.date_range('20000101', periods=2000000, freq='s').values
        result = algos.isin(s, s[0:2])
        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical_from_codes(self):
        # GH 16639
        vals = np.array([0, 1, 2, 0])
        cats = ['a', 'b', 'c']
        Sd = Series(Categorical(1).from_codes(vals, cats))
        St = Series(Categorical(1).from_codes(np.array([0, 1]), cats))
        expected = np.array([True, True, False, True])
        result = algos.isin(Sd, St)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        # GH 22160
        # nan is special, because from " a is b" doesn't follow "a == b"
        # at least, isin() should follow python's "np.nan in [nan] == True"
        # casting to -> np.float64 -> another float-object somewher on
        # the way could lead jepardize this behavior
        comps = [np.nan]  # could be casted to float64
        values = [np.nan]
        expected = np.array([True])
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_object_is_in(self):
        # GH 22160
        # there could be special treatment for nans
        # the user however could define a custom class
        # with similar behavior, then we at least should
        # fall back to usual python's behavior: "a in [a] == True"
        class LikeNan(object):
            def __eq__(self):
                return False

            def __hash__(self):
                return 0

        a, b = LikeNan(), LikeNan()
        # same object -> True
        tm.assert_numpy_array_equal(algos.isin([a], [a]), np.array([True]))
        # different objects -> False
        tm.assert_numpy_array_equal(algos.isin([a], [b]), np.array([False]))

    def test_different_nans(self):
        # GH 22160
        # all nans are handled as equivalent

        comps = [float('nan')]
        values = [float('nan')]
        assert comps[0] is not values[0]  # different nan-objects

        # as list of python-objects:
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as object-array:
        result = algos.isin(np.asarray(comps, dtype=np.object),
                            np.asarray(values, dtype=np.object))
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as float64-array:
        result = algos.isin(np.asarray(comps, dtype=np.float64),
                            np.asarray(values, dtype=np.float64))
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        # GH 22160
        # ensure 42 is not casted to a string
        comps = ['ss', 42]
        values = ['42']
        expected = np.array([False, False])
        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("empty", [[], Series(), np.array([])])
    def test_empty(self, empty):
        # see gh-16991
        vals = Index(["a", "b"])
        expected = np.array([False, False])

        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        # GH 22119
        comps = np.array(['nan', np.nan * 1j, float('nan')], dtype=np.object)
        vals = np.array([float('nan')], dtype=np.object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        # GH 21866
        # create different nans from bit-patterns,
        # these nans will land in different buckets in the hash-table
        # if no special care is taken
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2

        # check that NAN1 and NAN2 are equivalent:
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)


class TestValueCounts(object):

    def test_value_counts(self):
        np.random.seed(1234)
        from pandas.core.reshape.tile import cut

        arr = np.random.randn(4)
        factor = cut(arr, 4)

        # assert isinstance(factor, n)
        result = algos.value_counts(factor)
        breaks = [-1.194, -0.535, 0.121, 0.777, 1.433]
        index = IntervalIndex.from_breaks(breaks).astype(CDT(ordered=True))
        expected = Series([1, 1, 1, 1], index=index)
        tm.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_value_counts_bins(self):
        s = [1, 2, 3, 4]
        result = algos.value_counts(s, bins=1)
        expected = Series([4],
                          index=IntervalIndex.from_tuples([(0.996, 4.0)]))
        tm.assert_series_equal(result, expected)

        result = algos.value_counts(s, bins=2, sort=False)
        expected = Series([2, 2],
                          index=IntervalIndex.from_tuples([(0.996, 2.5),
                                                           (2.5, 4.0)]))
        tm.assert_series_equal(result, expected)

    def test_value_counts_dtypes(self):
        result = algos.value_counts([1, 1.])
        assert len(result) == 1

        result = algos.value_counts([1, 1.], bins=1)
        assert len(result) == 1

        result = algos.value_counts(Series([1, 1., '1']))  # object
        assert len(result) == 2

        pytest.raises(TypeError, lambda s: algos.value_counts(s, bins=1),
                      ['1', 1])

    def test_value_counts_nat(self):
        td = Series([np.timedelta64(10000), pd.NaT], dtype='timedelta64[ns]')
        dt = pd.to_datetime(['NaT', '2014-01-01'])

        for s in [td, dt]:
            vc = algos.value_counts(s)
            vc_with_na = algos.value_counts(s, dropna=False)
            assert len(vc) == 1
            assert len(vc_with_na) == 2

        exp_dt = Series({Timestamp('2014-01-01 00:00:00'): 1})
        tm.assert_series_equal(algos.value_counts(dt), exp_dt)
        # TODO same for (timedelta)

    def test_value_counts_datetime_outofbounds(self):
        # GH 13663
        s = Series([datetime(3000, 1, 1), datetime(5000, 1, 1),
                    datetime(5000, 1, 1), datetime(6000, 1, 1),
                    datetime(3000, 1, 1), datetime(3000, 1, 1)])
        res = s.value_counts()

        exp_index = Index([datetime(3000, 1, 1), datetime(5000, 1, 1),
                           datetime(6000, 1, 1)], dtype=object)
        exp = Series([3, 2, 1], index=exp_index)
        tm.assert_series_equal(res, exp)

        # GH 12424
        res = pd.to_datetime(Series(['2362-01-01', np.nan]),
                             errors='ignore')
        exp = Series(['2362-01-01', np.nan], dtype=object)
        tm.assert_series_equal(res, exp)

    def test_categorical(self):
        s = Series(Categorical(list('aaabbc')))
        result = s.value_counts()
        expected = Series([3, 2, 1], index=CategoricalIndex(['a', 'b', 'c']))

        tm.assert_series_equal(result, expected, check_index_type=True)

        # preserve order?
        s = s.cat.as_ordered()
        result = s.value_counts()
        expected.index = expected.index.as_ordered()
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_nans(self):
        s = Series(Categorical(list('aaaaabbbcc')))  # 4,3,2,1 (nan)
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series([4, 3, 2], index=CategoricalIndex(
            ['a', 'b', 'c'], categories=['a', 'b', 'c']))
        tm.assert_series_equal(result, expected, check_index_type=True)
        result = s.value_counts(dropna=False)
        expected = Series([
            4, 3, 2, 1
        ], index=CategoricalIndex(['a', 'b', 'c', np.nan]))
        tm.assert_series_equal(result, expected, check_index_type=True)

        # out of order
        s = Series(Categorical(
            list('aaaaabbbcc'), ordered=True, categories=['b', 'a', 'c']))
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series([4, 3, 2], index=CategoricalIndex(
            ['a', 'b', 'c'], categories=['b', 'a', 'c'], ordered=True))
        tm.assert_series_equal(result, expected, check_index_type=True)

        result = s.value_counts(dropna=False)
        expected = Series([4, 3, 2, 1], index=CategoricalIndex(
            ['a', 'b', 'c', np.nan], categories=['b', 'a', 'c'], ordered=True))
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_zeroes(self):
        # keep the `d` category with 0
        s = Series(Categorical(
            list('bbbaac'), categories=list('abcd'), ordered=True))
        result = s.value_counts()
        expected = Series([3, 2, 1, 0], index=Categorical(
            ['b', 'a', 'c', 'd'], categories=list('abcd'), ordered=True))
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_dropna(self):
        # https://github.com/pandas-dev/pandas/issues/9443#issuecomment-73719328

        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=True),
            Series([2, 1], index=[True, False]))
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=False),
            Series([2, 1], index=[True, False]))

        tm.assert_series_equal(
            Series([True, True, False, None]).value_counts(dropna=True),
            Series([2, 1], index=[True, False]))
        tm.assert_series_equal(
            Series([True, True, False, None]).value_counts(dropna=False),
            Series([2, 1, 1], index=[True, False, np.nan]))
        tm.assert_series_equal(
            Series([10.3, 5., 5.]).value_counts(dropna=True),
            Series([2, 1], index=[5., 10.3]))
        tm.assert_series_equal(
            Series([10.3, 5., 5.]).value_counts(dropna=False),
            Series([2, 1], index=[5., 10.3]))

        tm.assert_series_equal(
            Series([10.3, 5., 5., None]).value_counts(dropna=True),
            Series([2, 1], index=[5., 10.3]))

        # 32-bit linux has a different ordering
        if not compat.is_platform_32bit():
            result = Series([10.3, 5., 5., None]).value_counts(dropna=False)
            expected = Series([2, 1, 1], index=[5., 10.3, np.nan])
            tm.assert_series_equal(result, expected)

    def test_value_counts_normalized(self):
        # GH12558
        s = Series([1, 2, np.nan, np.nan, np.nan])
        dtypes = (np.float64, np.object, 'M8[ns]')
        for t in dtypes:
            s_typed = s.astype(t)
            result = s_typed.value_counts(normalize=True, dropna=False)
            expected = Series([0.6, 0.2, 0.2],
                              index=Series([np.nan, 2.0, 1.0], dtype=t))
            tm.assert_series_equal(result, expected)

            result = s_typed.value_counts(normalize=True, dropna=True)
            expected = Series([0.5, 0.5],
                              index=Series([2.0, 1.0], dtype=t))
            tm.assert_series_equal(result, expected)

    def test_value_counts_uint64(self):
        arr = np.array([2**63], dtype=np.uint64)
        expected = Series([1], index=[2**63])
        result = algos.value_counts(arr)

        tm.assert_series_equal(result, expected)

        arr = np.array([-1, 2**63], dtype=object)
        expected = Series([1, 1], index=[-1, 2**63])
        result = algos.value_counts(arr)

        # 32-bit linux has a different ordering
        if not compat.is_platform_32bit():
            tm.assert_series_equal(result, expected)


class TestDuplicated(object):

    def test_duplicated_with_nas(self):
        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)

        result = algos.duplicated(keys)
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep='first')
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep='last')
        expected = np.array([True, False, True, False, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array([True, False, True, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        keys = np.empty(8, dtype=object)
        for i, t in enumerate(zip([0, 0, np.nan, np.nan] * 2,
                                  [0, np.nan, 0, np.nan] * 2)):
            keys[i] = t

        result = algos.duplicated(keys)
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep='last')
        expected = np.array(trues + falses)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array(trues + trues)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('case', [
        np.array([1, 2, 1, 5, 3,
                  2, 4, 1, 5, 6]),
        np.array([1.1, 2.2, 1.1, np.nan, 3.3,
                  2.2, 4.4, 1.1, np.nan, 6.6]),
        np.array([1 + 1j, 2 + 2j, 1 + 1j, 5 + 5j, 3 + 3j,
                  2 + 2j, 4 + 4j, 1 + 1j, 5 + 5j, 6 + 6j]),
        np.array(['a', 'b', 'a', 'e', 'c',
                  'b', 'd', 'a', 'e', 'f'], dtype=object),
        np.array([1, 2**63, 1, 3**5, 10, 2**63, 39, 1, 3**5, 7],
                 dtype=np.uint64),
    ])
    def test_numeric_object_likes(self, case):
        exp_first = np.array([False, False, True, False, False,
                              True, False, True, True, False])
        exp_last = np.array([True, True, True, True, False,
                             False, False, False, False, False])
        exp_false = exp_first | exp_last

        res_first = algos.duplicated(case, keep='first')
        tm.assert_numpy_array_equal(res_first, exp_first)

        res_last = algos.duplicated(case, keep='last')
        tm.assert_numpy_array_equal(res_last, exp_last)

        res_false = algos.duplicated(case, keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)

        # index
        for idx in [Index(case), Index(case, dtype='category')]:
            res_first = idx.duplicated(keep='first')
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = idx.duplicated(keep='last')
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = idx.duplicated(keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

        # series
        for s in [Series(case), Series(case, dtype='category')]:
            res_first = s.duplicated(keep='first')
            tm.assert_series_equal(res_first, Series(exp_first))

            res_last = s.duplicated(keep='last')
            tm.assert_series_equal(res_last, Series(exp_last))

            res_false = s.duplicated(keep=False)
            tm.assert_series_equal(res_false, Series(exp_false))

    def test_datetime_likes(self):

        dt = ['2011-01-01', '2011-01-02', '2011-01-01', 'NaT', '2011-01-03',
              '2011-01-02', '2011-01-04', '2011-01-01', 'NaT', '2011-01-06']
        td = ['1 days', '2 days', '1 days', 'NaT', '3 days',
              '2 days', '4 days', '1 days', 'NaT', '6 days']

        cases = [np.array([Timestamp(d) for d in dt]),
                 np.array([Timestamp(d, tz='US/Eastern') for d in dt]),
                 np.array([pd.Period(d, freq='D') for d in dt]),
                 np.array([np.datetime64(d) for d in dt]),
                 np.array([pd.Timedelta(d) for d in td])]

        exp_first = np.array([False, False, True, False, False,
                              True, False, True, True, False])
        exp_last = np.array([True, True, True, True, False,
                             False, False, False, False, False])
        exp_false = exp_first | exp_last

        for case in cases:
            res_first = algos.duplicated(case, keep='first')
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algos.duplicated(case, keep='last')
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [Index(case), Index(case, dtype='category'),
                        Index(case, dtype=object)]:
                res_first = idx.duplicated(keep='first')
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep='last')
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [Series(case), Series(case, dtype='category'),
                      Series(case, dtype=object)]:
                res_first = s.duplicated(keep='first')
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep='last')
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    def test_unique_index(self):
        cases = [Index([1, 2, 3]), pd.RangeIndex(0, 3)]
        for case in cases:
            assert case.is_unique is True
            tm.assert_numpy_array_equal(case.duplicated(),
                                        np.array([False, False, False]))

    @pytest.mark.parametrize('arr, unique', [
        ([(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)],
         [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ([('b', 'c'), ('a', 'b'), ('a', 'b'), ('b', 'c')],
         [('b', 'c'), ('a', 'b')]),
        ([('a', 1), ('b', 2), ('a', 3), ('a', 1)],
         [('a', 1), ('b', 2), ('a', 3)]),
    ])
    def test_unique_tuples(self, arr, unique):
        # https://github.com/pandas-dev/pandas/issues/16519
        expected = np.empty(len(unique), dtype=object)
        expected[:] = unique

        result = pd.unique(arr)
        tm.assert_numpy_array_equal(result, expected)


class GroupVarTestMixin(object):

    def test_group_var_generic_1d(self):
        prng = RandomState(1234)

        out = (np.nan * np.ones((5, 1))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.rand(15, 1).astype(self.dtype)
        labels = np.tile(np.arange(5), (3, )).astype('int64')

        expected_out = (np.squeeze(values)
                        .reshape((5, 3), order='F')
                        .std(axis=1, ddof=1) ** 2)[:, np.newaxis]
        expected_counts = counts + 3

        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_1d_flat_labels(self):
        prng = RandomState(1234)

        out = (np.nan * np.ones((1, 1))).astype(self.dtype)
        counts = np.zeros(1, dtype='int64')
        values = 10 * prng.rand(5, 1).astype(self.dtype)
        labels = np.zeros(5, dtype='int64')

        expected_out = np.array([[values.std(ddof=1) ** 2]])
        expected_counts = counts + 5

        self.algo(out, counts, values, labels)

        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_all_finite(self):
        prng = RandomState(1234)

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.rand(10, 2).astype(self.dtype)
        labels = np.tile(np.arange(5), (2, )).astype('int64')

        expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2
        expected_counts = counts + 2

        self.algo(out, counts, values, labels)
        assert np.allclose(out, expected_out, self.rtol)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_generic_2d_some_nan(self):
        prng = RandomState(1234)

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)
        counts = np.zeros(5, dtype='int64')
        values = 10 * prng.rand(10, 2).astype(self.dtype)
        values[:, 1] = np.nan
        labels = np.tile(np.arange(5), (2, )).astype('int64')

        expected_out = np.vstack([values[:, 0]
                                  .reshape(5, 2, order='F')
                                  .std(ddof=1, axis=1) ** 2,
                                  np.nan * np.ones(5)]).T.astype(self.dtype)
        expected_counts = counts + 2

        self.algo(out, counts, values, labels)
        tm.assert_almost_equal(out, expected_out, check_less_precise=6)
        tm.assert_numpy_array_equal(counts, expected_counts)

    def test_group_var_constant(self):
        # Regression test from GH 10448.

        out = np.array([[np.nan]], dtype=self.dtype)
        counts = np.array([0], dtype='int64')
        values = 0.832845131556193 * np.ones((3, 1), dtype=self.dtype)
        labels = np.zeros(3, dtype='int64')

        self.algo(out, counts, values, labels)

        assert counts[0] == 3
        assert out[0, 0] >= 0
        tm.assert_almost_equal(out[0, 0], 0.0)


class TestGroupVarFloat64(GroupVarTestMixin):
    __test__ = True

    algo = libgroupby.group_var_float64
    dtype = np.float64
    rtol = 1e-5

    def test_group_var_large_inputs(self):

        prng = RandomState(1234)

        out = np.array([[np.nan]], dtype=self.dtype)
        counts = np.array([0], dtype='int64')
        values = (prng.rand(10 ** 6) + 10 ** 12).astype(self.dtype)
        values.shape = (10 ** 6, 1)
        labels = np.zeros(10 ** 6, dtype='int64')

        self.algo(out, counts, values, labels)

        assert counts[0] == 10 ** 6
        tm.assert_almost_equal(out[0, 0], 1.0 / 12, check_less_precise=True)


class TestGroupVarFloat32(GroupVarTestMixin):
    __test__ = True

    algo = libgroupby.group_var_float32
    dtype = np.float32
    rtol = 1e-2


class TestHashTable(object):

    def test_lookup_nan(self, writable):
        xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.Float64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs),
                                                            dtype=np.int64))

    def test_add_signed_zeros(self):
        # GH 21866 inconsistent hash-function for float64
        # default hash-function would lead to different hash-buckets
        # for 0.0 and -0.0 if there are more than 2^30 hash-buckets
        # but this would mean 16GB
        N = 4  # 12 * 10**8 would trigger the error, if you have enough memory
        m = ht.Float64HashTable(N)
        m.set_item(0.0, 0)
        m.set_item(-0.0, 0)
        assert len(m) == 1  # 0.0 and -0.0 are equivalent

    def test_add_different_nans(self):
        # GH 21866 inconsistent hash-function for float64
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7ff8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # default hash function would lead to different hash-buckets
        # for NAN1 and NAN2 even if there are only 4 buckets:
        m = ht.Float64HashTable()
        m.set_item(NAN1, 0)
        m.set_item(NAN2, 0)
        assert len(m) == 1  # NAN1 and NAN2 are equivalent

    def test_lookup_overflow(self, writable):
        xs = np.array([1, 2, 2**63], dtype=np.uint64)
        # GH 21688 ensure we can deal with readonly memory views
        xs.setflags(write=writable)
        m = ht.UInt64HashTable()
        m.map_locations(xs)
        tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs),
                                                            dtype=np.int64))

    def test_get_unique(self):
        s = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(s.unique(), exp)

    @pytest.mark.parametrize('nvals', [0, 10])  # resizing to 0 is special case
    @pytest.mark.parametrize('htable, uniques, dtype, safely_resizes', [
        (ht.PyObjectHashTable, ht.ObjectVector, 'object', False),
        (ht.StringHashTable, ht.ObjectVector, 'object', True),
        (ht.Float64HashTable, ht.Float64Vector, 'float64', False),
        (ht.Int64HashTable, ht.Int64Vector, 'int64', False),
        (ht.UInt64HashTable, ht.UInt64Vector, 'uint64', False)])
    def test_vector_resize(self, writable, htable, uniques, dtype,
                           safely_resizes, nvals):
        # Test for memory errors after internal vector
        # reallocations (GH 7157)
        vals = np.array(np.random.randn(1000), dtype=dtype)

        # GH 21688 ensures we can deal with read-only memory views
        vals.setflags(write=writable)

        # initialise instances; cannot initialise in parametrization,
        # as otherwise external views would be held on the array (which is
        # one of the things this test is checking)
        htable = htable()
        uniques = uniques()

        # get_labels may append to uniques
        htable.get_labels(vals[:nvals], uniques, 0, -1)
        # to_array() sets an external_view_exists flag on uniques.
        tmp = uniques.to_array()
        oldshape = tmp.shape

        # subsequent get_labels() calls can no longer append to it
        # (except for StringHashTables + ObjectVector)
        if safely_resizes:
            htable.get_labels(vals, uniques, 0, -1)
        else:
            with pytest.raises(ValueError, match='external reference.*'):
                htable.get_labels(vals, uniques, 0, -1)

        uniques.to_array()   # should not raise here
        assert tmp.shape == oldshape

    @pytest.mark.parametrize('htable, tm_dtype', [
        (ht.PyObjectHashTable, 'String'),
        (ht.StringHashTable, 'String'),
        (ht.Float64HashTable, 'Float'),
        (ht.Int64HashTable, 'Int'),
        (ht.UInt64HashTable, 'UInt')])
    def test_hashtable_unique(self, htable, tm_dtype, writable):
        # output of maker has guaranteed unique elements
        maker = getattr(tm, 'make' + tm_dtype + 'Index')
        s = Series(maker(1000))
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, pd.NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.unique()
        expected_unique = s_duplicated.drop_duplicates(keep='first').values
        result_unique = htable().unique(s_duplicated.values)
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # test return_inverse=True
        # reconstruction can only succeed if the inverse is correct
        result_unique, result_inverse = htable().unique(s_duplicated.values,
                                                        return_inverse=True)
        tm.assert_numpy_array_equal(result_unique, expected_unique)
        reconstr = result_unique[result_inverse]
        tm.assert_numpy_array_equal(reconstr, s_duplicated.values)

    @pytest.mark.parametrize('htable, tm_dtype', [
        (ht.PyObjectHashTable, 'String'),
        (ht.StringHashTable, 'String'),
        (ht.Float64HashTable, 'Float'),
        (ht.Int64HashTable, 'Int'),
        (ht.UInt64HashTable, 'UInt')])
    def test_hashtable_factorize(self, htable, tm_dtype, writable):
        # output of maker has guaranteed unique elements
        maker = getattr(tm, 'make' + tm_dtype + 'Index')
        s = Series(maker(1000))
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, pd.NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)
        na_mask = s_duplicated.isna().values

        result_unique, result_inverse = htable().factorize(s_duplicated.values)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.factorize()
        # since factorize removes all NaNs, we do the same here
        expected_unique = s_duplicated.dropna().drop_duplicates().values
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # reconstruction can only succeed if the inverse is correct. Since
        # factorize removes the NaNs, those have to be excluded here as well
        result_reconstruct = result_unique[result_inverse[~na_mask]]
        expected_reconstruct = s_duplicated.dropna().values
        tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)

    @pytest.mark.parametrize('hashtable', [
        ht.PyObjectHashTable, ht.StringHashTable,
        ht.Float64HashTable, ht.Int64HashTable, ht.UInt64HashTable])
    def test_hashtable_large_sizehint(self, hashtable):
        # GH 22729
        size_hint = np.iinfo(np.uint32).max + 1
        tbl = hashtable(size_hint=size_hint) # noqa


def test_quantile():
    s = Series(np.random.randn(100))

    result = algos.quantile(s, [0, .25, .5, .75, 1.])
    expected = algos.quantile(s.values, [0, .25, .5, .75, 1.])
    tm.assert_almost_equal(result, expected)


def test_unique_label_indices():

    a = np.random.randint(1, 1 << 10, 1 << 15).astype('i8')

    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1]

    tm.assert_numpy_array_equal(left, right,
                                check_dtype=False)

    a[np.random.choice(len(a), 10)] = -1
    left = ht.unique_label_indices(a)
    right = np.unique(a, return_index=True)[1][1:]
    tm.assert_numpy_array_equal(left, right,
                                check_dtype=False)


class TestRank(object):

    @td.skip_if_no_scipy
    def test_scipy_compat(self):
        from scipy.stats import rankdata

        def _check(arr):
            mask = ~np.isfinite(arr)
            arr = arr.copy()
            result = libalgos.rank_1d_float64(arr)
            arr[mask] = np.inf
            exp = rankdata(arr)
            exp[mask] = nan
            assert_almost_equal(result, exp)

        _check(np.array([nan, nan, 5., 5., 5., nan, 1, 2, 3, nan]))
        _check(np.array([4., nan, 5., 5., 5., nan, 1, 2, 4., nan]))

    def test_basic(self):
        exp = np.array([1, 2], dtype=np.float64)

        for dtype in np.typecodes['AllInteger']:
            s = Series([1, 100], dtype=dtype)
            tm.assert_numpy_array_equal(algos.rank(s), exp)

    def test_uint64_overflow(self):
        exp = np.array([1, 2], dtype=np.float64)

        for dtype in [np.float64, np.uint64]:
            s = Series([1, 2**63], dtype=dtype)
            tm.assert_numpy_array_equal(algos.rank(s), exp)

    def test_too_many_ndims(self):
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        msg = "Array with ndim > 2 are not supported"

        with pytest.raises(TypeError, match=msg):
            algos.rank(arr)

    @pytest.mark.single
    @pytest.mark.parametrize('values', [
        np.arange(2**24 + 1),
        np.arange(2**25 + 2).reshape(2**24 + 1, 2)],
        ids=['1d', '2d'])
    def test_pct_max_many_rows(self, values):
        # GH 18271
        result = algos.rank(values, pct=True).max()
        assert result == 1


def test_pad_backfill_object_segfault():

    old = np.array([], dtype='O')
    new = np.array([datetime(2010, 12, 31)], dtype='O')

    result = libalgos.pad["object"](old, new)
    expected = np.array([-1], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.pad["object"](new, old)
    expected = np.array([], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.backfill["object"](old, new)
    expected = np.array([-1], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)

    result = libalgos.backfill["object"](new, old)
    expected = np.array([], dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)


def test_arrmap():
    values = np.array(['foo', 'foo', 'bar', 'bar', 'baz', 'qux'], dtype='O')
    result = libalgos.arrmap_object(values, lambda x: x in ['foo', 'bar'])
    assert (result.dtype == np.bool_)


class TestTseriesUtil(object):

    def test_combineFunc(self):
        pass

    def test_reindex(self):
        pass

    def test_isna(self):
        pass

    def test_groupby(self):
        pass

    def test_groupby_withnull(self):
        pass

    def test_backfill(self):
        old = Index([1, 5, 10])
        new = Index(lrange(12))

        filler = libalgos.backfill["int64_t"](old.values, new.values)

        expect_filler = np.array([0, 0, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = Index([1, 4])
        new = Index(lrange(5, 10))
        filler = libalgos.backfill["int64_t"](old.values, new.values)

        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad(self):
        old = Index([1, 5, 10])
        new = Index(lrange(12))

        filler = libalgos.pad["int64_t"](old.values, new.values)

        expect_filler = np.array([-1, 0, 0, 0, 0, 1,
                                  1, 1, 1, 1, 2, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = Index([5, 10])
        new = Index(lrange(5))
        filler = libalgos.pad["int64_t"](old.values, new.values)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(filler, expect_filler)


def test_is_lexsorted():
    failure = [
        np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  3, 3,
                  3, 3,
                  3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                  2, 2, 2, 2, 2, 2, 2,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1,
                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64'),
        np.array([30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                  15, 14,
                  13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 30, 29, 28,
                  27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                  12, 11,
                  10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 30, 29, 28, 27, 26, 25,
                  24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                  9, 8,
                  7, 6, 5, 4, 3, 2, 1, 0, 30, 29, 28, 27, 26, 25, 24, 23, 22,
                  21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                  6, 5,
                  4, 3, 2, 1, 0], dtype='int64')]

    assert (not libalgos.is_lexsorted(failure))


def test_groupsort_indexer():
    a = np.random.randint(0, 1000, 100).astype(np.int64)
    b = np.random.randint(0, 1000, 100).astype(np.int64)

    result = libalgos.groupsort_indexer(a, 1000)[0]

    # need to use a stable sort
    # np.argsort returns int, groupsort_indexer
    # always returns int64
    expected = np.argsort(a, kind='mergesort')
    expected = expected.astype(np.int64)

    tm.assert_numpy_array_equal(result, expected)

    # compare with lexsort
    # np.lexsort returns int, groupsort_indexer
    # always returns int64
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.int64)

    tm.assert_numpy_array_equal(result, expected)


def test_infinity_sort():
    # GH 13445
    # numpy's argsort can be unhappy if something is less than
    # itself.  Instead, let's give our infinities a self-consistent
    # ordering, but outside the float extended real line.

    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()

    ref_nums = [NegInf, float("-inf"), -1e100, 0, 1e100, float("inf"), Inf]

    assert all(Inf >= x for x in ref_nums)
    assert all(Inf > x or x is Inf for x in ref_nums)
    assert Inf >= Inf and Inf == Inf
    assert not Inf < Inf and not Inf > Inf
    assert libalgos.Infinity() == libalgos.Infinity()
    assert not libalgos.Infinity() != libalgos.Infinity()

    assert all(NegInf <= x for x in ref_nums)
    assert all(NegInf < x or x is NegInf for x in ref_nums)
    assert NegInf <= NegInf and NegInf == NegInf
    assert not NegInf < NegInf and not NegInf > NegInf
    assert libalgos.NegInfinity() == libalgos.NegInfinity()
    assert not libalgos.NegInfinity() != libalgos.NegInfinity()

    for perm in permutations(ref_nums):
        assert sorted(perm) == ref_nums

    # smoke tests
    np.array([libalgos.Infinity()] * 32).argsort()
    np.array([libalgos.NegInfinity()] * 32).argsort()


def test_infinity_against_nan():
    Inf = libalgos.Infinity()
    NegInf = libalgos.NegInfinity()

    assert not Inf > np.nan
    assert not Inf >= np.nan
    assert not Inf < np.nan
    assert not Inf <= np.nan
    assert not Inf == np.nan
    assert Inf != np.nan

    assert not NegInf > np.nan
    assert not NegInf >= np.nan
    assert not NegInf < np.nan
    assert not NegInf <= np.nan
    assert not NegInf == np.nan
    assert NegInf != np.nan


def test_ensure_platform_int():
    arr = np.arange(100, dtype=np.intp)

    result = libalgos.ensure_platform_int(arr)
    assert (result is arr)


def test_int64_add_overflow():
    # see gh-14068
    msg = "Overflow in int64 addition"
    m = np.iinfo(np.int64).max
    n = np.iinfo(np.int64).min

    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), m)
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([n, n]), n)
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([n, n]), np.array([n, n]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, n]), np.array([n, n]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                                   arr_mask=np.array([False, True]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                                   b_mask=np.array([False, True]))
    with pytest.raises(OverflowError, match=msg):
        algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                                   arr_mask=np.array([False, True]),
                                   b_mask=np.array([False, True]))
    with pytest.raises(OverflowError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            algos.checked_add_with_arr(np.array([m, m]),
                                       np.array([np.nan, m]))

    # Check that the nan boolean arrays override whether or not
    # the addition overflows. We don't check the result but just
    # the fact that an OverflowError is not raised.
    algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                               arr_mask=np.array([True, True]))
    algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                               b_mask=np.array([True, True]))
    algos.checked_add_with_arr(np.array([m, m]), np.array([m, m]),
                               arr_mask=np.array([True, False]),
                               b_mask=np.array([False, True]))


class TestMode(object):

    def test_no_mode(self):
        exp = Series([], dtype=np.float64)
        tm.assert_series_equal(algos.mode([]), exp)

    def test_mode_single(self):
        # GH 15714
        exp_single = [1]
        data_single = [1]

        exp_multi = [1]
        data_multi = [1, 1]

        for dt in np.typecodes['AllInteger'] + np.typecodes['Float']:
            s = Series(data_single, dtype=dt)
            exp = Series(exp_single, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

            s = Series(data_multi, dtype=dt)
            exp = Series(exp_multi, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

        exp = Series([1], dtype=np.int)
        tm.assert_series_equal(algos.mode([1]), exp)

        exp = Series(['a', 'b', 'c'], dtype=np.object)
        tm.assert_series_equal(algos.mode(['a', 'b', 'c']), exp)

    def test_number_mode(self):
        exp_single = [1]
        data_single = [1] * 5 + [2] * 3

        exp_multi = [1, 3]
        data_multi = [1] * 5 + [2] * 3 + [3] * 5

        for dt in np.typecodes['AllInteger'] + np.typecodes['Float']:
            s = Series(data_single, dtype=dt)
            exp = Series(exp_single, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

            s = Series(data_multi, dtype=dt)
            exp = Series(exp_multi, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

    def test_strobj_mode(self):
        exp = ['b']
        data = ['a'] * 2 + ['b'] * 3

        s = Series(data, dtype='c')
        exp = Series(exp, dtype='c')
        tm.assert_series_equal(algos.mode(s), exp)

        exp = ['bar']
        data = ['foo'] * 2 + ['bar'] * 3

        for dt in [str, object]:
            s = Series(data, dtype=dt)
            exp = Series(exp, dtype=dt)
            tm.assert_series_equal(algos.mode(s), exp)

    def test_datelike_mode(self):
        exp = Series(['1900-05-03', '2011-01-03',
                      '2013-01-02'], dtype="M8[ns]")
        s = Series(['2011-01-03', '2013-01-02',
                    '1900-05-03'], dtype='M8[ns]')
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series(['2011-01-03', '2013-01-02'], dtype='M8[ns]')
        s = Series(['2011-01-03', '2013-01-02', '1900-05-03',
                    '2011-01-03', '2013-01-02'], dtype='M8[ns]')
        tm.assert_series_equal(algos.mode(s), exp)

    def test_timedelta_mode(self):
        exp = Series(['-1 days', '0 days', '1 days'],
                     dtype='timedelta64[ns]')
        s = Series(['1 days', '-1 days', '0 days'],
                   dtype='timedelta64[ns]')
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series(['2 min', '1 day'], dtype='timedelta64[ns]')
        s = Series(['1 day', '1 day', '-1 day', '-1 day 2 min',
                    '2 min', '2 min'], dtype='timedelta64[ns]')
        tm.assert_series_equal(algos.mode(s), exp)

    def test_mixed_dtype(self):
        exp = Series(['foo'])
        s = Series([1, 'foo', 'foo'])
        tm.assert_series_equal(algos.mode(s), exp)

    def test_uint64_overflow(self):
        exp = Series([2**63], dtype=np.uint64)
        s = Series([1, 2**63, 2**63], dtype=np.uint64)
        tm.assert_series_equal(algos.mode(s), exp)

        exp = Series([1, 2**63], dtype=np.uint64)
        s = Series([1, 2**63], dtype=np.uint64)
        tm.assert_series_equal(algos.mode(s), exp)

    def test_categorical(self):
        c = Categorical([1, 2])
        exp = c
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

        c = Categorical([1, 'a', 'a'])
        exp = Categorical(['a'], categories=[1, 'a'])
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

        c = Categorical([1, 1, 2, 3, 3])
        exp = Categorical([1, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(algos.mode(c), exp)
        tm.assert_categorical_equal(c.mode(), exp)

    def test_index(self):
        idx = Index([1, 2, 3])
        exp = Series([1, 2, 3], dtype=np.int64)
        tm.assert_series_equal(algos.mode(idx), exp)

        idx = Index([1, 'a', 'a'])
        exp = Series(['a'], dtype=object)
        tm.assert_series_equal(algos.mode(idx), exp)

        idx = Index([1, 1, 2, 3, 3])
        exp = Series([1, 3], dtype=np.int64)
        tm.assert_series_equal(algos.mode(idx), exp)

        exp = Series(['2 min', '1 day'], dtype='timedelta64[ns]')
        idx = Index(['1 day', '1 day', '-1 day', '-1 day 2 min',
                     '2 min', '2 min'], dtype='timedelta64[ns]')
        tm.assert_series_equal(algos.mode(idx), exp)
