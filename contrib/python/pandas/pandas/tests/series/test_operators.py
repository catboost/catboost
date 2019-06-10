# coding=utf-8
# pylint: disable-msg=E1101,W0612

from datetime import datetime, timedelta
import operator

import numpy as np
import pytest

import pandas.compat as compat
from pandas.compat import range

import pandas as pd
from pandas import (
    Categorical, DataFrame, Index, Series, bdate_range, date_range, isna)
from pandas.core import ops
import pandas.core.nanops as nanops
import pandas.util.testing as tm
from pandas.util.testing import (
    assert_almost_equal, assert_frame_equal, assert_series_equal)

from .common import TestData


class TestSeriesLogicalOps(object):
    @pytest.mark.parametrize('bool_op', [operator.and_,
                                         operator.or_, operator.xor])
    def test_bool_operators_with_nas(self, bool_op):
        # boolean &, |, ^ should work with object arrays and propagate NAs
        ser = Series(bdate_range('1/1/2000', periods=10), dtype=object)
        ser[::2] = np.nan

        mask = ser.isna()
        filled = ser.fillna(ser[0])

        result = bool_op(ser < ser[9], ser > ser[3])

        expected = bool_op(filled < filled[9], filled > filled[3])
        expected[mask] = False
        assert_series_equal(result, expected)

    def test_operators_bitwise(self):
        # GH#9016: support bitwise op for integer types
        index = list('bca')

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)
        s_tff = Series([True, False, False], index=index)
        s_empty = Series([])

        # TODO: unused
        # s_0101 = Series([0, 1, 0, 1])

        s_0123 = Series(range(4), dtype='int64')
        s_3333 = Series([3] * 4)
        s_4444 = Series([4] * 4)

        res = s_tft & s_empty
        expected = s_fff
        assert_series_equal(res, expected)

        res = s_tft | s_empty
        expected = s_tft
        assert_series_equal(res, expected)

        res = s_0123 & s_3333
        expected = Series(range(4), dtype='int64')
        assert_series_equal(res, expected)

        res = s_0123 | s_4444
        expected = Series(range(4, 8), dtype='int64')
        assert_series_equal(res, expected)

        s_a0b1c0 = Series([1], list('b'))

        res = s_tft & s_a0b1c0
        expected = s_tff.reindex(list('abc'))
        assert_series_equal(res, expected)

        res = s_tft | s_a0b1c0
        expected = s_tft.reindex(list('abc'))
        assert_series_equal(res, expected)

        n0 = 0
        res = s_tft & n0
        expected = s_fff
        assert_series_equal(res, expected)

        res = s_0123 & n0
        expected = Series([0] * 4)
        assert_series_equal(res, expected)

        n1 = 1
        res = s_tft & n1
        expected = s_tft
        assert_series_equal(res, expected)

        res = s_0123 & n1
        expected = Series([0, 1, 0, 1])
        assert_series_equal(res, expected)

        s_1111 = Series([1] * 4, dtype='int8')
        res = s_0123 & s_1111
        expected = Series([0, 1, 0, 1], dtype='int64')
        assert_series_equal(res, expected)

        res = s_0123.astype(np.int16) | s_1111.astype(np.int32)
        expected = Series([1, 1, 3, 3], dtype='int32')
        assert_series_equal(res, expected)

        with pytest.raises(TypeError):
            s_1111 & 'a'
        with pytest.raises(TypeError):
            s_1111 & ['a', 'b', 'c', 'd']
        with pytest.raises(TypeError):
            s_0123 & np.NaN
        with pytest.raises(TypeError):
            s_0123 & 3.14
        with pytest.raises(TypeError):
            s_0123 & [0.1, 4, 3.14, 2]

        # s_0123 will be all false now because of reindexing like s_tft
        exp = Series([False] * 7, index=[0, 1, 2, 3, 'a', 'b', 'c'])
        assert_series_equal(s_tft & s_0123, exp)

        # s_tft will be all false now because of reindexing like s_0123
        exp = Series([False] * 7, index=[0, 1, 2, 3, 'a', 'b', 'c'])
        assert_series_equal(s_0123 & s_tft, exp)

        assert_series_equal(s_0123 & False, Series([False] * 4))
        assert_series_equal(s_0123 ^ False, Series([False, True, True, True]))
        assert_series_equal(s_0123 & [False], Series([False] * 4))
        assert_series_equal(s_0123 & (False), Series([False] * 4))
        assert_series_equal(s_0123 & Series([False, np.NaN, False, False]),
                            Series([False] * 4))

        s_ftft = Series([False, True, False, True])
        assert_series_equal(s_0123 & Series([0.1, 4, -3.14, 2]), s_ftft)

        s_abNd = Series(['a', 'b', np.NaN, 'd'])
        res = s_0123 & s_abNd
        expected = s_ftft
        assert_series_equal(res, expected)

    def test_scalar_na_logical_ops_corners(self):
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])

        with pytest.raises(TypeError):
            s & datetime(2005, 1, 1)

        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        expected = Series(True, index=s.index)
        expected[::2] = False
        result = s & list(s)
        assert_series_equal(result, expected)

        d = DataFrame({'A': s})
        # TODO: Fix this exception - needs to be fixed! (see GH5035)
        # (previously this was a TypeError because series returned
        # NotImplemented

        # this is an alignment issue; these are equivalent
        # https://github.com/pandas-dev/pandas/issues/5284

        with pytest.raises(TypeError):
            d.__and__(s, axis='columns')

        with pytest.raises(TypeError):
            s & d

        # this is wrong as its not a boolean result
        # result = d.__and__(s,axis='index')

    @pytest.mark.parametrize('op', [
        operator.and_,
        operator.or_,
        operator.xor,

    ])
    def test_logical_ops_with_index(self, op):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

        result = op(ser, idx1)
        assert_series_equal(result, expected)

        expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))],
                          dtype=bool)

        result = op(ser, idx2)
        assert_series_equal(result, expected)

    @pytest.mark.parametrize("op, expected", [
        (ops.rand_, pd.Index([False, True])),
        (ops.ror_, pd.Index([False, True])),
        (ops.rxor, pd.Index([])),
    ])
    def test_reverse_ops_with_index(self, op, expected):
        # https://github.com/pandas-dev/pandas/pull/23628
        # multi-set Index ops are buggy, so let's avoid duplicates...
        ser = Series([True, False])
        idx = Index([False, True])
        result = op(ser, idx)
        tm.assert_index_equal(result, expected)

    def test_logical_ops_label_based(self):
        # GH#4947
        # logical ops should be label based

        a = Series([True, False, True], list('bca'))
        b = Series([False, True, False], list('abc'))

        expected = Series([False, True, False], list('abc'))
        result = a & b
        assert_series_equal(result, expected)

        expected = Series([True, True, False], list('abc'))
        result = a | b
        assert_series_equal(result, expected)

        expected = Series([True, False, False], list('abc'))
        result = a ^ b
        assert_series_equal(result, expected)

        # rhs is bigger
        a = Series([True, False, True], list('bca'))
        b = Series([False, True, False, True], list('abcd'))

        expected = Series([False, True, False, False], list('abcd'))
        result = a & b
        assert_series_equal(result, expected)

        expected = Series([True, True, False, False], list('abcd'))
        result = a | b
        assert_series_equal(result, expected)

        # filling

        # vs empty
        result = a & Series([])
        expected = Series([False, False, False], list('bca'))
        assert_series_equal(result, expected)

        result = a | Series([])
        expected = Series([True, False, True], list('bca'))
        assert_series_equal(result, expected)

        # vs non-matching
        result = a & Series([1], ['z'])
        expected = Series([False, False, False, False], list('abcz'))
        assert_series_equal(result, expected)

        result = a | Series([1], ['z'])
        expected = Series([True, True, False, False], list('abcz'))
        assert_series_equal(result, expected)

        # identity
        # we would like s[s|e] == s to hold for any e, whether empty or not
        for e in [Series([]), Series([1], ['z']),
                  Series(np.nan, b.index), Series(np.nan, a.index)]:
            result = a[a | e]
            assert_series_equal(result, a[a])

        for e in [Series(['z'])]:
            result = a[a | e]
            assert_series_equal(result, a[a])

        # vs scalars
        index = list('bca')
        t = Series([True, False, True])

        for v in [True, 1, 2]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, True, True], index=index)
            assert_series_equal(result, expected)

        for v in [np.nan, 'foo']:
            with pytest.raises(TypeError):
                t | v

        for v in [False, 0]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, False, True], index=index)
            assert_series_equal(result, expected)

        for v in [True, 1]:
            result = Series([True, False, True], index=index) & v
            expected = Series([True, False, True], index=index)
            assert_series_equal(result, expected)

        for v in [False, 0]:
            result = Series([True, False, True], index=index) & v
            expected = Series([False, False, False], index=index)
            assert_series_equal(result, expected)
        for v in [np.nan]:
            with pytest.raises(TypeError):
                t & v

    def test_logical_ops_df_compat(self):
        # GH#1134
        s1 = pd.Series([True, False, True], index=list('ABC'), name='x')
        s2 = pd.Series([True, True, False], index=list('ABD'), name='x')

        exp = pd.Series([True, False, False, False],
                        index=list('ABCD'), name='x')
        assert_series_equal(s1 & s2, exp)
        assert_series_equal(s2 & s1, exp)

        # True | np.nan => True
        exp = pd.Series([True, True, True, False],
                        index=list('ABCD'), name='x')
        assert_series_equal(s1 | s2, exp)
        # np.nan | True => np.nan, filled with False
        exp = pd.Series([True, True, False, False],
                        index=list('ABCD'), name='x')
        assert_series_equal(s2 | s1, exp)

        # DataFrame doesn't fill nan with False
        exp = pd.DataFrame({'x': [True, False, np.nan, np.nan]},
                           index=list('ABCD'))
        assert_frame_equal(s1.to_frame() & s2.to_frame(), exp)
        assert_frame_equal(s2.to_frame() & s1.to_frame(), exp)

        exp = pd.DataFrame({'x': [True, True, np.nan, np.nan]},
                           index=list('ABCD'))
        assert_frame_equal(s1.to_frame() | s2.to_frame(), exp)
        assert_frame_equal(s2.to_frame() | s1.to_frame(), exp)

        # different length
        s3 = pd.Series([True, False, True], index=list('ABC'), name='x')
        s4 = pd.Series([True, True, True, True], index=list('ABCD'), name='x')

        exp = pd.Series([True, False, True, False],
                        index=list('ABCD'), name='x')
        assert_series_equal(s3 & s4, exp)
        assert_series_equal(s4 & s3, exp)

        # np.nan | True => np.nan, filled with False
        exp = pd.Series([True, True, True, False],
                        index=list('ABCD'), name='x')
        assert_series_equal(s3 | s4, exp)
        # True | np.nan => True
        exp = pd.Series([True, True, True, True],
                        index=list('ABCD'), name='x')
        assert_series_equal(s4 | s3, exp)

        exp = pd.DataFrame({'x': [True, False, True, np.nan]},
                           index=list('ABCD'))
        assert_frame_equal(s3.to_frame() & s4.to_frame(), exp)
        assert_frame_equal(s4.to_frame() & s3.to_frame(), exp)

        exp = pd.DataFrame({'x': [True, True, True, np.nan]},
                           index=list('ABCD'))
        assert_frame_equal(s3.to_frame() | s4.to_frame(), exp)
        assert_frame_equal(s4.to_frame() | s3.to_frame(), exp)


class TestSeriesComparisons(object):
    def test_comparisons(self):
        left = np.random.randn(10)
        right = np.random.randn(10)
        left[:3] = np.nan

        result = nanops.nangt(left, right)
        with np.errstate(invalid='ignore'):
            expected = (left > right).astype('O')
        expected[:3] = np.nan

        assert_almost_equal(result, expected)

        s = Series(['a', 'b', 'c'])
        s2 = Series([False, True, False])

        # it works!
        exp = Series([False, False, False])
        assert_series_equal(s == s2, exp)
        assert_series_equal(s2 == s, exp)

    def test_categorical_comparisons(self):
        # GH 8938
        # allow equality comparisons
        a = Series(list('abc'), dtype="category")
        b = Series(list('abc'), dtype="object")
        c = Series(['a', 'b', 'cc'], dtype="object")
        d = Series(list('acb'), dtype="object")
        e = Categorical(list('abc'))
        f = Categorical(list('acb'))

        # vs scalar
        assert not (a == 'a').all()
        assert ((a != 'a') == ~(a == 'a')).all()

        assert not ('a' == a).all()
        assert (a == 'a')[0]
        assert ('a' == a)[0]
        assert not ('a' != a)[0]

        # vs list-like
        assert (a == a).all()
        assert not (a != a).all()

        assert (a == list(a)).all()
        assert (a == b).all()
        assert (b == a).all()
        assert ((~(a == b)) == (a != b)).all()
        assert ((~(b == a)) == (b != a)).all()

        assert not (a == c).all()
        assert not (c == a).all()
        assert not (a == d).all()
        assert not (d == a).all()

        # vs a cat-like
        assert (a == e).all()
        assert (e == a).all()
        assert not (a == f).all()
        assert not (f == a).all()

        assert ((~(a == e) == (a != e)).all())
        assert ((~(e == a) == (e != a)).all())
        assert ((~(a == f) == (a != f)).all())
        assert ((~(f == a) == (f != a)).all())

        # non-equality is not comparable
        with pytest.raises(TypeError):
            a < b
        with pytest.raises(TypeError):
            b < a
        with pytest.raises(TypeError):
            a > b
        with pytest.raises(TypeError):
            b > a

    def test_comparison_tuples(self):
        # GH11339
        # comparisons vs tuple
        s = Series([(1, 1), (1, 2)])

        result = s == (1, 2)
        expected = Series([False, True])
        assert_series_equal(result, expected)

        result = s != (1, 2)
        expected = Series([True, False])
        assert_series_equal(result, expected)

        result = s == (0, 0)
        expected = Series([False, False])
        assert_series_equal(result, expected)

        result = s != (0, 0)
        expected = Series([True, True])
        assert_series_equal(result, expected)

        s = Series([(1, 1), (1, 1)])

        result = s == (1, 1)
        expected = Series([True, True])
        assert_series_equal(result, expected)

        result = s != (1, 1)
        expected = Series([False, False])
        assert_series_equal(result, expected)

        s = Series([frozenset([1]), frozenset([1, 2])])

        result = s == frozenset([1])
        expected = Series([True, False])
        assert_series_equal(result, expected)

    def test_comparison_operators_with_nas(self):
        ser = Series(bdate_range('1/1/2000', periods=10), dtype=object)
        ser[::2] = np.nan

        # test that comparisons work
        ops = ['lt', 'le', 'gt', 'ge', 'eq', 'ne']
        for op in ops:
            val = ser[5]

            f = getattr(operator, op)
            result = f(ser, val)

            expected = f(ser.dropna(), val).reindex(ser.index)

            if op == 'ne':
                expected = expected.fillna(True).astype(bool)
            else:
                expected = expected.fillna(False).astype(bool)

            assert_series_equal(result, expected)

            # fffffffuuuuuuuuuuuu
            # result = f(val, s)
            # expected = f(val, s.dropna()).reindex(s.index)
            # assert_series_equal(result, expected)

    def test_unequal_categorical_comparison_raises_type_error(self):
        # unequal comparison should raise for unordered cats
        cat = Series(Categorical(list("abc")))
        with pytest.raises(TypeError):
            cat > "b"

        cat = Series(Categorical(list("abc"), ordered=False))
        with pytest.raises(TypeError):
            cat > "b"

        # https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        # and following comparisons with scalars not in categories should raise
        # for unequal comps, but not for equal/not equal
        cat = Series(Categorical(list("abc"), ordered=True))

        with pytest.raises(TypeError):
            cat < "d"
        with pytest.raises(TypeError):
            cat > "d"
        with pytest.raises(TypeError):
            "d" < cat
        with pytest.raises(TypeError):
            "d" > cat

        tm.assert_series_equal(cat == "d", Series([False, False, False]))
        tm.assert_series_equal(cat != "d", Series([True, True, True]))

    def test_ne(self):
        ts = Series([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], dtype=float)
        expected = [True, True, False, True, True]
        assert tm.equalContents(ts.index != 5, expected)
        assert tm.equalContents(~(ts.index == 5), expected)

    def test_comp_ops_df_compat(self):
        # GH 1134
        s1 = pd.Series([1, 2, 3], index=list('ABC'), name='x')
        s2 = pd.Series([2, 2, 2], index=list('ABD'), name='x')

        s3 = pd.Series([1, 2, 3], index=list('ABC'), name='x')
        s4 = pd.Series([2, 2, 2, 2], index=list('ABCD'), name='x')

        for left, right in [(s1, s2), (s2, s1), (s3, s4), (s4, s3)]:

            msg = "Can only compare identically-labeled Series objects"
            with pytest.raises(ValueError, match=msg):
                left == right

            with pytest.raises(ValueError, match=msg):
                left != right

            with pytest.raises(ValueError, match=msg):
                left < right

            msg = "Can only compare identically-labeled DataFrame objects"
            with pytest.raises(ValueError, match=msg):
                left.to_frame() == right.to_frame()

            with pytest.raises(ValueError, match=msg):
                left.to_frame() != right.to_frame()

            with pytest.raises(ValueError, match=msg):
                left.to_frame() < right.to_frame()

    def test_compare_series_interval_keyword(self):
        # GH 25338
        s = Series(['IntervalA', 'IntervalB', 'IntervalC'])
        result = s == 'IntervalA'
        expected = Series([True, False, False])
        assert_series_equal(result, expected)


class TestSeriesFlexComparisonOps(object):

    def test_comparison_flex_alignment(self):
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2, 2], index=list('bcd'))

        exp = pd.Series([False, False, True, False], index=list('abcd'))
        assert_series_equal(left.eq(right), exp)

        exp = pd.Series([True, True, False, True], index=list('abcd'))
        assert_series_equal(left.ne(right), exp)

        exp = pd.Series([False, False, True, False], index=list('abcd'))
        assert_series_equal(left.le(right), exp)

        exp = pd.Series([False, False, False, False], index=list('abcd'))
        assert_series_equal(left.lt(right), exp)

        exp = pd.Series([False, True, True, False], index=list('abcd'))
        assert_series_equal(left.ge(right), exp)

        exp = pd.Series([False, True, False, False], index=list('abcd'))
        assert_series_equal(left.gt(right), exp)

    def test_comparison_flex_alignment_fill(self):
        left = Series([1, 3, 2], index=list('abc'))
        right = Series([2, 2, 2], index=list('bcd'))

        exp = pd.Series([False, False, True, True], index=list('abcd'))
        assert_series_equal(left.eq(right, fill_value=2), exp)

        exp = pd.Series([True, True, False, False], index=list('abcd'))
        assert_series_equal(left.ne(right, fill_value=2), exp)

        exp = pd.Series([False, False, True, True], index=list('abcd'))
        assert_series_equal(left.le(right, fill_value=0), exp)

        exp = pd.Series([False, False, False, True], index=list('abcd'))
        assert_series_equal(left.lt(right, fill_value=0), exp)

        exp = pd.Series([True, True, True, False], index=list('abcd'))
        assert_series_equal(left.ge(right, fill_value=0), exp)

        exp = pd.Series([True, True, False, False], index=list('abcd'))
        assert_series_equal(left.gt(right, fill_value=0), exp)


class TestSeriesOperators(TestData):

    def test_operators_empty_int_corner(self):
        s1 = Series([], [], dtype=np.int32)
        s2 = Series({'x': 0.})
        assert_series_equal(s1 * s2, Series([np.nan], index=['x']))

    def test_ops_datetimelike_align(self):
        # GH 7500
        # datetimelike ops need to align
        dt = Series(date_range('2012-1-1', periods=3, freq='D'))
        dt.iloc[2] = np.nan
        dt2 = dt[::-1]

        expected = Series([timedelta(0), timedelta(0), pd.NaT])
        # name is reset
        result = dt2 - dt
        assert_series_equal(result, expected)

        expected = Series(expected, name=0)
        result = (dt2.to_frame() - dt.to_frame())[0]
        assert_series_equal(result, expected)

    def test_operators_corner(self):
        series = self.ts

        empty = Series([], index=Index([]))

        result = series + empty
        assert np.isnan(result).all()

        result = empty + Series([], index=Index([]))
        assert len(result) == 0

        # TODO: this returned NotImplemented earlier, what to do?
        # deltas = Series([timedelta(1)] * 5, index=np.arange(5))
        # sub_deltas = deltas[::2]
        # deltas5 = deltas * 5
        # deltas = deltas + sub_deltas

        # float + int
        int_ts = self.ts.astype(int)[:-5]
        added = self.ts + int_ts
        expected = Series(self.ts.values[:-5] + int_ts.values,
                          index=self.ts.index[:-5], name='ts')
        tm.assert_series_equal(added[:-5], expected)

    pairings = []
    for op in ['add', 'sub', 'mul', 'pow', 'truediv', 'floordiv']:
        fv = 0
        lop = getattr(Series, op)
        lequiv = getattr(operator, op)
        rop = getattr(Series, 'r' + op)
        # bind op at definition time...
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        pairings.append((lop, lequiv, fv))
        pairings.append((rop, requiv, fv))
    if compat.PY3:
        pairings.append((Series.div, operator.truediv, 1))
        pairings.append((Series.rdiv, lambda x, y: operator.truediv(y, x), 1))
    else:
        pairings.append((Series.div, operator.div, 1))
        pairings.append((Series.rdiv, lambda x, y: operator.div(y, x), 1))

    @pytest.mark.parametrize('op, equiv_op, fv', pairings)
    def test_operators_combine(self, op, equiv_op, fv):
        def _check_fill(meth, op, a, b, fill_value=0):
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)

            amask = isna(a)
            bmask = isna(b)

            exp_values = []
            for i in range(len(exp_index)):
                with np.errstate(all='ignore'):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))

            result = meth(a, b, fill_value=fill_value)
            expected = Series(exp_values, exp_index)
            assert_series_equal(result, expected)

        a = Series([np.nan, 1., 2., 3., np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.], index=np.arange(6))

        result = op(a, b)
        exp = equiv_op(a, b)
        assert_series_equal(result, exp)
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        # should accept axis=0 or axis='rows'
        op(a, b, axis=0)

    def test_operators_na_handling(self):
        from decimal import Decimal
        from datetime import date
        s = Series([Decimal('1.3'), Decimal('2.3')],
                   index=[date(2012, 1, 1), date(2012, 1, 2)])

        result = s + s.shift(1)
        result2 = s.shift(1) + s
        assert isna(result[0])
        assert isna(result2[0])

    def test_op_duplicate_index(self):
        # GH14227
        s1 = Series([1, 2], index=[1, 1])
        s2 = Series([10, 10], index=[1, 2])
        result = s1 + s2
        expected = pd.Series([11, 12, np.nan], index=[1, 1, 2])
        assert_series_equal(result, expected)


class TestSeriesUnaryOps(object):
    # __neg__, __pos__, __inv__

    def test_neg(self):
        ser = tm.makeStringSeries()
        ser.name = 'series'
        assert_series_equal(-ser, -1 * ser)

    def test_invert(self):
        ser = tm.makeStringSeries()
        ser.name = 'series'
        assert_series_equal(-(ser < 0), ~(ser < 0))
