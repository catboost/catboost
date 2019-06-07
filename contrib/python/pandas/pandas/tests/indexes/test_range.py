# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pytest

from pandas.compat import PY3, range, u

import pandas as pd
from pandas import Float64Index, Index, Int64Index, RangeIndex, Series
import pandas.util.testing as tm

from .test_numeric import Numeric


class TestRangeIndex(Numeric):
    _holder = RangeIndex
    _compat_props = ['shape', 'ndim', 'size']

    def setup_method(self, method):
        self.indices = dict(index=RangeIndex(0, 20, 2, name='foo'),
                            index_dec=RangeIndex(18, -1, -2, name='bar'))
        self.setup_indices()

    def create_index(self):
        return RangeIndex(5)

    def test_can_hold_identifiers(self):
        idx = self.create_index()
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self):
        with pytest.raises(ValueError, match="^Length"):
            self.index.names = ["roger", "harold"]

    def test_constructor(self):
        index = RangeIndex(5)
        expected = np.arange(5, dtype=np.int64)
        assert isinstance(index, RangeIndex)
        assert index._start == 0
        assert index._stop == 5
        assert index._step == 1
        assert index.name is None
        tm.assert_index_equal(Index(expected), index)

        index = RangeIndex(1, 5)
        expected = np.arange(1, 5, dtype=np.int64)
        assert isinstance(index, RangeIndex)
        assert index._start == 1
        tm.assert_index_equal(Index(expected), index)

        index = RangeIndex(1, 5, 2)
        expected = np.arange(1, 5, 2, dtype=np.int64)
        assert isinstance(index, RangeIndex)
        assert index._step == 2
        tm.assert_index_equal(Index(expected), index)

        for index in [RangeIndex(0), RangeIndex(start=0), RangeIndex(stop=0),
                      RangeIndex(0, 0)]:
            expected = np.empty(0, dtype=np.int64)
            assert isinstance(index, RangeIndex)
            assert index._start == 0
            assert index._stop == 0
            assert index._step == 1
            tm.assert_index_equal(Index(expected), index)

        for index in [RangeIndex(0, name='Foo'),
                      RangeIndex(start=0, name='Foo'),
                      RangeIndex(stop=0, name='Foo'),
                      RangeIndex(0, 0, name='Foo')]:
            assert isinstance(index, RangeIndex)
            assert index.name == 'Foo'

        # we don't allow on a bare Index
        with pytest.raises(TypeError):
            Index(0, 1000)

    def test_constructor_invalid_args(self):
        msg = "RangeIndex\\(\\.\\.\\.\\) must be called with integers"
        with pytest.raises(TypeError, match=msg):
            RangeIndex()

        with pytest.raises(TypeError, match=msg):
            RangeIndex(name='Foo')

        # invalid args
        for i in [Index(['a', 'b']), Series(['a', 'b']), np.array(['a', 'b']),
                  [], 'foo', datetime(2000, 1, 1, 0, 0), np.arange(0, 10),
                  np.array([1]), [1]]:
            with pytest.raises(TypeError):
                RangeIndex(i)

    def test_constructor_same(self):

        # pass thru w and w/o copy
        index = RangeIndex(1, 5, 2)
        result = RangeIndex(index, copy=False)
        assert result.identical(index)

        result = RangeIndex(index, copy=True)
        tm.assert_index_equal(result, index, exact=True)

        result = RangeIndex(index)
        tm.assert_index_equal(result, index, exact=True)

        with pytest.raises(TypeError):
            RangeIndex(index, dtype='float64')

    def test_constructor_range(self):

        with pytest.raises(TypeError):
            RangeIndex(range(1, 5, 2))

        result = RangeIndex.from_range(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = RangeIndex.from_range(range(5, 6))
        expected = RangeIndex(5, 6, 1)
        tm.assert_index_equal(result, expected, exact=True)

        # an invalid range
        result = RangeIndex.from_range(range(5, 1))
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected, exact=True)

        result = RangeIndex.from_range(range(5))
        expected = RangeIndex(0, 5, 1)
        tm.assert_index_equal(result, expected, exact=True)

        result = Index(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        with pytest.raises(TypeError):
            Index(range(1, 5, 2), dtype='float64')

    def test_constructor_name(self):
        # GH12288
        orig = RangeIndex(10)
        orig.name = 'original'

        copy = RangeIndex(orig)
        copy.name = 'copy'

        assert orig.name == 'original'
        assert copy.name == 'copy'

        new = Index(copy)
        assert new.name == 'copy'

        new.name = 'new'
        assert orig.name == 'original'
        assert copy.name == 'copy'
        assert new.name == 'new'

    def test_constructor_corner(self):
        arr = np.array([1, 2, 3, 4], dtype=object)
        index = RangeIndex(1, 5)
        assert index.values.dtype == np.int64
        tm.assert_index_equal(index, Index(arr))

        # non-int raise Exception
        with pytest.raises(TypeError):
            RangeIndex('1', '10', '1')
        with pytest.raises(TypeError):
            RangeIndex(1.1, 10.2, 1.3)

        # invalid passed type
        with pytest.raises(TypeError):
            RangeIndex(1, 5, dtype='float64')

    def test_copy(self):
        i = RangeIndex(5, name='Foo')
        i_copy = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._start == 0
        assert i_copy._stop == 5
        assert i_copy._step == 1
        assert i_copy.name == 'Foo'

    def test_repr(self):
        i = RangeIndex(5, name='Foo')
        result = repr(i)
        if PY3:
            expected = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        else:
            expected = "RangeIndex(start=0, stop=5, step=1, name=u'Foo')"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = "RangeIndex(start=5, stop=0, step=-1)"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    def test_insert(self):

        idx = RangeIndex(5, name='Foo')
        result = idx[1:4]

        # test 0th element
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]))

        # GH 18295 (test missing)
        expected = Float64Index([0, np.nan, 1, 2, 3, 4])
        for na in (np.nan, pd.NaT, None):
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)

    def test_delete(self):

        idx = RangeIndex(5, name='Foo')
        expected = idx[1:].astype(int)
        result = idx.delete(0)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        expected = idx[:-1].astype(int)
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        with pytest.raises((IndexError, ValueError)):
            # either depending on numpy version
            result = idx.delete(len(idx))

    def test_view(self):
        i = RangeIndex(0, name='Foo')
        i_view = i.view()
        assert i_view.name == 'Foo'

        i_view = i.view('i8')
        tm.assert_numpy_array_equal(i.values, i_view)

        i_view = i.view(RangeIndex)
        tm.assert_index_equal(i, i_view)

    def test_dtype(self):
        assert self.index.dtype == np.int64

    def test_is_monotonic(self):
        assert self.index.is_monotonic is True
        assert self.index.is_monotonic_increasing is True
        assert self.index.is_monotonic_decreasing is False
        assert self.index._is_strictly_monotonic_increasing is True
        assert self.index._is_strictly_monotonic_decreasing is False

        index = RangeIndex(4, 0, -1)
        assert index.is_monotonic is False
        assert index._is_strictly_monotonic_increasing is False
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 2)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(2, 1)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 1)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    def test_equals_range(self):
        equiv_pairs = [(RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
                       (RangeIndex(0), RangeIndex(1, -1, 3)),
                       (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
                       (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2))]
        for left, right in equiv_pairs:
            assert left.equals(right)
            assert right.equals(left)

    def test_logical_compat(self):
        idx = self.create_index()
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self):
        i = Index(self.index.copy())
        assert i.identical(self.index)

        # we don't allow object dtype for RangeIndex
        if isinstance(self.index, RangeIndex):
            return

        same_values_different_type = Index(i, dtype=object)
        assert not i.identical(same_values_different_type)

        i = self.index.copy(dtype=object)
        i = i.rename('foo')
        same_values = Index(i, dtype=object)
        assert same_values.identical(self.index.copy(dtype=object))

        assert not i.identical(self.index)
        assert Index(same_values, name='foo', dtype=object).identical(i)

        assert not self.index.copy(dtype=object).identical(
            self.index.copy(dtype='int64'))

    def test_get_indexer(self):
        target = RangeIndex(10)
        indexer = self.index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_pad(self):
        target = RangeIndex(10)
        indexer = self.index.get_indexer(target, method='pad')
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_backfill(self):
        target = RangeIndex(10)
        indexer = self.index.get_indexer(target, method='backfill')
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_join_outer(self):
        # join with Int64Index
        other = Int64Index(np.arange(25, 14, -1))

        res, lidx, ridx = self.index.join(other, how='outer',
                                          return_indexers=True)
        noidx_res = self.index.join(other, how='outer')
        tm.assert_index_equal(res, noidx_res)

        eres = Int64Index([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25])
        elidx = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, 8, -1, 9,
                          -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)
        eridx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 10, 9, 8, 7, 6,
                          5, 4, 3, 2, 1, 0], dtype=np.intp)

        assert isinstance(res, Int64Index)
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # join with RangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = self.index.join(other, how='outer',
                                          return_indexers=True)
        noidx_res = self.index.join(other, how='outer')
        tm.assert_index_equal(res, noidx_res)

        assert isinstance(res, Int64Index)
        assert not isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_inner(self):
        # Join with non-RangeIndex
        other = Int64Index(np.arange(25, 14, -1))

        res, lidx, ridx = self.index.join(other, how='inner',
                                          return_indexers=True)

        # no guarantee of sortedness, so sort for comparison purposes
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        eres = Int64Index([16, 18])
        elidx = np.array([8, 9], dtype=np.intp)
        eridx = np.array([9, 7], dtype=np.intp)

        assert isinstance(res, Int64Index)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # Join two RangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = self.index.join(other, how='inner',
                                          return_indexers=True)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_left(self):
        # Join with Int64Index
        other = Int64Index(np.arange(25, 14, -1))

        res, lidx, ridx = self.index.join(other, how='left',
                                          return_indexers=True)
        eres = self.index
        eridx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 9, 7], dtype=np.intp)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # Join withRangeIndex
        other = Int64Index(np.arange(25, 14, -1))

        res, lidx, ridx = self.index.join(other, how='left',
                                          return_indexers=True)

        assert isinstance(res, RangeIndex)
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self):
        # Join with Int64Index
        other = Int64Index(np.arange(25, 14, -1))

        res, lidx, ridx = self.index.join(other, how='right',
                                          return_indexers=True)
        eres = other
        elidx = np.array([-1, -1, -1, -1, -1, -1, -1, 9, -1, 8, -1],
                         dtype=np.intp)

        assert isinstance(other, Int64Index)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

        # Join withRangeIndex
        other = RangeIndex(25, 14, -1)

        res, lidx, ridx = self.index.join(other, how='right',
                                          return_indexers=True)
        eres = other

        assert isinstance(other, RangeIndex)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

    def test_join_non_int_index(self):
        other = Index([3, 6, 7, 8, 10], dtype=object)

        outer = self.index.join(other, how='outer')
        outer2 = other.join(self.index, how='outer')
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)

        inner = self.index.join(other, how='inner')
        inner2 = other.join(self.index, how='inner')
        expected = Index([6, 8, 10])
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)

        left = self.index.join(other, how='left')
        tm.assert_index_equal(left, self.index.astype(object))

        left2 = other.join(self.index, how='left')
        tm.assert_index_equal(left2, other)

        right = self.index.join(other, how='right')
        tm.assert_index_equal(right, other)

        right2 = other.join(self.index, how='right')
        tm.assert_index_equal(right2, self.index.astype(object))

    def test_join_non_unique(self):
        other = Index([4, 4, 3, 3])

        res, lidx, ridx = self.index.join(other, return_indexers=True)

        eres = Int64Index([0, 2, 4, 4, 6, 8, 10, 12, 14, 16, 18])
        elidx = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp)
        eridx = np.array([-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1],
                         dtype=np.intp)

        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_self(self):
        kinds = 'outer', 'inner', 'left', 'right'
        for kind in kinds:
            joined = self.index.join(self.index, how=kind)
            assert self.index is joined

    @pytest.mark.parametrize("sort", [None, False])
    def test_intersection(self, sort):
        # intersect with Int64Index
        other = Index(np.arange(1, 6))
        result = self.index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(self.index.values,
                                                other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(self.index, sort=sort)
        expected = Index(np.sort(np.asarray(np.intersect1d(self.index.values,
                                                           other.values))))
        tm.assert_index_equal(result, expected)

        # intersect with increasing RangeIndex
        other = RangeIndex(1, 6)
        result = self.index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(self.index.values,
                                                other.values)))
        tm.assert_index_equal(result, expected)

        # intersect with decreasing RangeIndex
        other = RangeIndex(5, 0, -1)
        result = self.index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(self.index.values,
                                                other.values)))
        tm.assert_index_equal(result, expected)

        # reversed (GH 17296)
        result = other.intersection(self.index, sort=sort)
        tm.assert_index_equal(result, expected)

        # GH 17296: intersect two decreasing RangeIndexes
        first = RangeIndex(10, -2, -2)
        other = RangeIndex(5, -4, -1)
        expected = first.astype(int).intersection(other.astype(int), sort=sort)
        result = first.intersection(other, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        # reversed
        result = other.intersection(first, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        index = RangeIndex(5)

        # intersect of non-overlapping indices
        other = RangeIndex(5, 10, 1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        other = RangeIndex(-1, -5, -1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        # intersection of empty indices
        other = RangeIndex(0, 0, 1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected)

        # intersection of non-overlapping values based on start value and gcd
        index = RangeIndex(1, 10, 2)
        other = RangeIndex(0, 10, 4)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

    def test_union_noncomparable(self):
        from datetime import datetime, timedelta
        # corner case, non-Int64Index
        now = datetime.now()
        other = Index([now + timedelta(i) for i in range(4)], dtype=object)
        result = self.index.union(other)
        expected = Index(np.concatenate((self.index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(self.index)
        expected = Index(np.concatenate((other, self.index)))
        tm.assert_index_equal(result, expected)

    def test_union(self):
        RI = RangeIndex
        I64 = Int64Index
        cases = [(RI(0, 10, 1), RI(0, 10, 1), RI(0, 10, 1)),
                 (RI(0, 10, 1), RI(5, 20, 1), RI(0, 20, 1)),
                 (RI(0, 10, 1), RI(10, 20, 1), RI(0, 20, 1)),
                 (RI(0, -10, -1), RI(0, -10, -1), RI(0, -10, -1)),
                 (RI(0, -10, -1), RI(-10, -20, -1), RI(-19, 1, 1)),
                 (RI(0, 10, 2), RI(1, 10, 2), RI(0, 10, 1)),
                 (RI(0, 11, 2), RI(1, 12, 2), RI(0, 12, 1)),
                 (RI(0, 21, 4), RI(-2, 24, 4), RI(-2, 24, 2)),
                 (RI(0, -20, -2), RI(-1, -21, -2), RI(-19, 1, 1)),
                 (RI(0, 100, 5), RI(0, 100, 20), RI(0, 100, 5)),
                 (RI(0, -100, -5), RI(5, -100, -20), RI(-95, 10, 5)),
                 (RI(0, -11, -1), RI(1, -12, -4), RI(-11, 2, 1)),
                 (RI(0), RI(0), RI(0)),
                 (RI(0, -10, -2), RI(0), RI(0, -10, -2)),
                 (RI(0, 100, 2), RI(100, 150, 200), RI(0, 102, 2)),
                 (RI(0, -100, -2), RI(-100, 50, 102), RI(-100, 4, 2)),
                 (RI(0, -100, -1), RI(0, -50, -3), RI(-99, 1, 1)),
                 (RI(0, 1, 1), RI(5, 6, 10), RI(0, 6, 5)),
                 (RI(0, 10, 5), RI(-5, -6, -20), RI(-5, 10, 5)),
                 (RI(0, 3, 1), RI(4, 5, 1), I64([0, 1, 2, 4])),
                 (RI(0, 10, 1), I64([]), RI(0, 10, 1)),
                 (RI(0), I64([1, 5, 6]), I64([1, 5, 6]))]
        for idx1, idx2, expected in cases:
            res1 = idx1.union(idx2)
            res2 = idx2.union(idx1)
            res3 = idx1._int64index.union(idx2)
            tm.assert_index_equal(res1, expected, exact=True)
            tm.assert_index_equal(res2, expected, exact=True)
            tm.assert_index_equal(res3, expected)

    def test_nbytes(self):

        # memory savings vs int index
        i = RangeIndex(0, 1000)
        assert i.nbytes < i._int64index.nbytes / 10

        # constant memory usage
        i2 = RangeIndex(0, 10)
        assert i.nbytes == i2.nbytes

    def test_cant_or_shouldnt_cast(self):
        # can't
        with pytest.raises(TypeError):
            RangeIndex('foo', 'bar', 'baz')

        # shouldn't
        with pytest.raises(TypeError):
            RangeIndex('0', '1', '2')

    def test_view_Index(self):
        self.index.view(Index)

    def test_prevent_casting(self):
        result = self.index.astype('O')
        assert result.dtype == np.object_

    def test_take_preserve_name(self):
        index = RangeIndex(1, 5, name='foo')
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value(self):
        # GH 12631
        idx = pd.RangeIndex(1, 4, name='xxx')
        result = idx.take(np.array([1, 0, -1]))
        expected = pd.Int64Index([2, 1, 3], name='xxx')
        tm.assert_index_equal(result, expected)

        # fill_value
        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False,
                          fill_value=True)
        expected = pd.Int64Index([2, 1, 3], name='xxx')
        tm.assert_index_equal(result, expected)

        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        with pytest.raises(IndexError):
            idx.take(np.array([1, -5]))

    def test_print_unicode_columns(self):
        df = pd.DataFrame({u("\u05d0"): [1, 2, 3],
                           "\u05d1": [4, 5, 6],
                           "c": [7, 8, 9]})
        repr(df.columns)  # should not raise UnicodeDecodeError

    def test_repr_roundtrip(self):
        tm.assert_index_equal(eval(repr(self.index)), self.index)

    def test_slice_keep_name(self):
        idx = RangeIndex(1, 2, name='asdf')
        assert idx.name == idx[1:].name

    def test_explicit_conversions(self):

        # GH 8608
        # add/sub are overridden explicitly for Float/Int Index
        idx = RangeIndex(5)

        # float conversions
        arr = np.arange(5, dtype='int64') * 3.2
        expected = Float64Index(arr)
        fidx = idx * 3.2
        tm.assert_index_equal(fidx, expected)
        fidx = 3.2 * idx
        tm.assert_index_equal(fidx, expected)

        # interops with numpy arrays
        expected = Float64Index(arr)
        a = np.zeros(5, dtype='float64')
        result = fidx - a
        tm.assert_index_equal(result, expected)

        expected = Float64Index(-arr)
        a = np.zeros(5, dtype='float64')
        result = a - fidx
        tm.assert_index_equal(result, expected)

    def test_has_duplicates(self):
        for ind in self.indices:
            if not len(ind):
                continue
            idx = self.indices[ind]
            assert idx.is_unique
            assert not idx.has_duplicates

    def test_extended_gcd(self):
        result = self.index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]

        result = self.index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    def test_min_fitting_element(self):
        result = RangeIndex(0, 20, 2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(1, 6)._min_fitting_element(1)
        assert 1 == result

        result = RangeIndex(18, -2, -2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(5, 0, -1)._min_fitting_element(1)
        assert 1 == result

        big_num = 500000000000000000000000

        result = RangeIndex(5, big_num * 2, 1)._min_fitting_element(big_num)
        assert big_num == result

    def test_max_fitting_element(self):
        result = RangeIndex(0, 20, 2)._max_fitting_element(17)
        assert 16 == result

        result = RangeIndex(1, 6)._max_fitting_element(4)
        assert 4 == result

        result = RangeIndex(18, -2, -2)._max_fitting_element(17)
        assert 16 == result

        result = RangeIndex(5, 0, -1)._max_fitting_element(4)
        assert 4 == result

        big_num = 500000000000000000000000

        result = RangeIndex(5, big_num * 2, 1)._max_fitting_element(big_num)
        assert big_num == result

    def test_pickle_compat_construction(self):
        # RangeIndex() is a valid constructor
        pass

    def test_slice_specialised(self):

        # scalar indexing
        res = self.index[1]
        expected = 2
        assert res == expected

        res = self.index[-1]
        expected = 18
        assert res == expected

        # slicing
        # slice value completion
        index = self.index[:]
        expected = self.index
        tm.assert_index_equal(index, expected)

        # positive slice values
        index = self.index[7:10:2]
        expected = Index(np.array([14, 18]), name='foo')
        tm.assert_index_equal(index, expected)

        # negative slice values
        index = self.index[-1:-5:-2]
        expected = Index(np.array([18, 14]), name='foo')
        tm.assert_index_equal(index, expected)

        # stop overshoot
        index = self.index[2:100:4]
        expected = Index(np.array([4, 12]), name='foo')
        tm.assert_index_equal(index, expected)

        # reverse
        index = self.index[::-1]
        expected = Index(self.index.values[::-1], name='foo')
        tm.assert_index_equal(index, expected)

        index = self.index[-8::-1]
        expected = Index(np.array([4, 2, 0]), name='foo')
        tm.assert_index_equal(index, expected)

        index = self.index[-40::-1]
        expected = Index(np.array([], dtype=np.int64), name='foo')
        tm.assert_index_equal(index, expected)

        index = self.index[40::-1]
        expected = Index(self.index.values[40::-1], name='foo')
        tm.assert_index_equal(index, expected)

        index = self.index[10::-1]
        expected = Index(self.index.values[::-1], name='foo')
        tm.assert_index_equal(index, expected)

    def test_len_specialised(self):

        # make sure that our len is the same as
        # np.arange calc

        for step in np.arange(1, 6, 1):

            arr = np.arange(0, 5, step)
            i = RangeIndex(0, 5, step)
            assert len(i) == len(arr)

            i = RangeIndex(5, 0, step)
            assert len(i) == 0

        for step in np.arange(-6, -1, 1):

            arr = np.arange(5, 0, step)
            i = RangeIndex(5, 0, step)
            assert len(i) == len(arr)

            i = RangeIndex(0, 5, step)
            assert len(i) == 0

    def test_append(self):
        # GH16212
        RI = RangeIndex
        I64 = Int64Index
        F64 = Float64Index
        OI = Index
        cases = [([RI(1, 12, 5)], RI(1, 12, 5)),
                 ([RI(0, 6, 4)], RI(0, 6, 4)),
                 ([RI(1, 3), RI(3, 7)], RI(1, 7)),
                 ([RI(1, 5, 2), RI(5, 6)], RI(1, 6, 2)),
                 ([RI(1, 3, 2), RI(4, 7, 3)], RI(1, 7, 3)),
                 ([RI(-4, 3, 2), RI(4, 7, 2)], RI(-4, 7, 2)),
                 ([RI(-4, -8), RI(-8, -12)], RI(0, 0)),
                 ([RI(-4, -8), RI(3, -4)], RI(0, 0)),
                 ([RI(-4, -8), RI(3, 5)], RI(3, 5)),
                 ([RI(-4, -2), RI(3, 5)], I64([-4, -3, 3, 4])),
                 ([RI(-2,), RI(3, 5)], RI(3, 5)),
                 ([RI(2,), RI(2)], I64([0, 1, 0, 1])),
                 ([RI(2,), RI(2, 5), RI(5, 8, 4)], RI(0, 6)),
                 ([RI(2,), RI(3, 5), RI(5, 8, 4)], I64([0, 1, 3, 4, 5])),
                 ([RI(-2, 2), RI(2, 5), RI(5, 8, 4)], RI(-2, 6)),
                 ([RI(3,), I64([-1, 3, 15])], I64([0, 1, 2, -1, 3, 15])),
                 ([RI(3,), F64([-1, 3.1, 15.])], F64([0, 1, 2, -1, 3.1, 15.])),
                 ([RI(3,), OI(['a', None, 14])], OI([0, 1, 2, 'a', None, 14])),
                 ([RI(3, 1), OI(['a', None, 14])], OI(['a', None, 14]))
                 ]

        for indices, expected in cases:
            result = indices[0].append(indices[1:])
            tm.assert_index_equal(result, expected, exact=True)

            if len(indices) == 2:
                # Append single item rather than list
                result2 = indices[0].append(indices[1])
                tm.assert_index_equal(result2, expected, exact=True)
