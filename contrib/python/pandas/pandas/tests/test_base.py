# -*- coding: utf-8 -*-
from __future__ import print_function

from datetime import datetime, timedelta
import re
import sys

import numpy as np
import pytest

from pandas._libs.tslib import iNaT
import pandas.compat as compat
from pandas.compat import PYPY, StringIO, long
from pandas.compat.numpy import np_array_datetime64_compat

from pandas.core.dtypes.common import (
    is_datetime64_dtype, is_datetime64tz_dtype, is_object_dtype,
    is_timedelta64_dtype, needs_i8_conversion)
from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    CategoricalIndex, DataFrame, DatetimeIndex, Index, Interval, IntervalIndex,
    Panel, PeriodIndex, Series, Timedelta, TimedeltaIndex, Timestamp)
from pandas.core.accessor import PandasDelegate
from pandas.core.arrays import DatetimeArray, PandasArray, TimedeltaArray
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
import pandas.util.testing as tm


class CheckStringMixin(object):

    def test_string_methods_dont_fail(self):
        repr(self.container)
        str(self.container)
        bytes(self.container)
        if not compat.PY3:
            unicode(self.container)  # noqa

    def test_tricky_container(self):
        if not hasattr(self, 'unicode_container'):
            pytest.skip('Need unicode_container to test with this')
        repr(self.unicode_container)
        str(self.unicode_container)
        bytes(self.unicode_container)
        if not compat.PY3:
            unicode(self.unicode_container)  # noqa


class CheckImmutable(object):
    mutable_regex = re.compile('does not support mutable operations')

    def check_mutable_error(self, *args, **kwargs):
        # Pass whatever function you normally would to pytest.raises
        # (after the Exception kind).
        with pytest.raises(TypeError):
            self.mutable_regex(*args, **kwargs)

    def test_no_mutable_funcs(self):
        def setitem():
            self.container[0] = 5

        self.check_mutable_error(setitem)

        def setslice():
            self.container[1:2] = 3

        self.check_mutable_error(setslice)

        def delitem():
            del self.container[0]

        self.check_mutable_error(delitem)

        def delslice():
            del self.container[0:3]

        self.check_mutable_error(delslice)
        mutable_methods = getattr(self, "mutable_methods", [])

        for meth in mutable_methods:
            self.check_mutable_error(getattr(self.container, meth))

    def test_slicing_maintains_type(self):
        result = self.container[1:2]
        expected = self.lst[1:2]
        self.check_result(result, expected)

    def check_result(self, result, expected, klass=None):
        klass = klass or self.klass
        assert isinstance(result, klass)
        assert result == expected


class TestPandasDelegate(object):

    class Delegator(object):
        _properties = ['foo']
        _methods = ['bar']

        def _set_foo(self, value):
            self.foo = value

        def _get_foo(self):
            return self.foo

        foo = property(_get_foo, _set_foo, doc="foo property")

        def bar(self, *args, **kwargs):
            """ a test bar method """
            pass

    class Delegate(PandasDelegate, PandasObject):

        def __init__(self, obj):
            self.obj = obj

    def setup_method(self, method):
        pass

    def test_invalid_delegation(self):
        # these show that in order for the delegation to work
        # the _delegate_* methods need to be overridden to not raise
        # a TypeError

        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._properties,
            typ='property'
        )
        self.Delegate._add_delegate_accessors(
            delegate=self.Delegator,
            accessors=self.Delegator._methods,
            typ='method'
        )

        delegate = self.Delegate(self.Delegator())

        with pytest.raises(TypeError):
            delegate.foo

        with pytest.raises(TypeError):
            delegate.foo = 5

        with pytest.raises(TypeError):
            delegate.foo()

    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        # Delegate does not implement memory_usage.
        # Check that we fall back to in-built `__sizeof__`
        # GH 12924
        delegate = self.Delegate(self.Delegator())
        sys.getsizeof(delegate)


class Ops(object):

    def _allow_na_ops(self, obj):
        """Whether to skip test cases including NaN"""
        if (isinstance(obj, Index) and
                (obj.is_boolean() or not obj._can_hold_na)):
            # don't test boolean / int64 index
            return False
        return True

    def setup_method(self, method):
        self.bool_index = tm.makeBoolIndex(10, name='a')
        self.int_index = tm.makeIntIndex(10, name='a')
        self.float_index = tm.makeFloatIndex(10, name='a')
        self.dt_index = tm.makeDateIndex(10, name='a')
        self.dt_tz_index = tm.makeDateIndex(10, name='a').tz_localize(
            tz='US/Eastern')
        self.period_index = tm.makePeriodIndex(10, name='a')
        self.string_index = tm.makeStringIndex(10, name='a')
        self.unicode_index = tm.makeUnicodeIndex(10, name='a')

        arr = np.random.randn(10)
        self.bool_series = Series(arr, index=self.bool_index, name='a')
        self.int_series = Series(arr, index=self.int_index, name='a')
        self.float_series = Series(arr, index=self.float_index, name='a')
        self.dt_series = Series(arr, index=self.dt_index, name='a')
        self.dt_tz_series = self.dt_tz_index.to_series(keep_tz=True)
        self.period_series = Series(arr, index=self.period_index, name='a')
        self.string_series = Series(arr, index=self.string_index, name='a')
        self.unicode_series = Series(arr, index=self.unicode_index, name='a')

        types = ['bool', 'int', 'float', 'dt', 'dt_tz', 'period', 'string',
                 'unicode']
        self.indexes = [getattr(self, '{}_index'.format(t)) for t in types]
        self.series = [getattr(self, '{}_series'.format(t)) for t in types]
        self.objs = self.indexes + self.series

    def check_ops_properties(self, props, filter=None, ignore_failures=False):
        for op in props:
            for o in self.is_valid_objs:

                # if a filter, skip if it doesn't match
                if filter is not None:
                    filt = o.index if isinstance(o, Series) else o
                    if not filter(filt):
                        continue

                try:
                    if isinstance(o, Series):
                        expected = Series(
                            getattr(o.index, op), index=o.index, name='a')
                    else:
                        expected = getattr(o, op)
                except (AttributeError):
                    if ignore_failures:
                        continue

                result = getattr(o, op)

                # these couuld be series, arrays or scalars
                if isinstance(result, Series) and isinstance(expected, Series):
                    tm.assert_series_equal(result, expected)
                elif isinstance(result, Index) and isinstance(expected, Index):
                    tm.assert_index_equal(result, expected)
                elif isinstance(result, np.ndarray) and isinstance(expected,
                                                                   np.ndarray):
                    tm.assert_numpy_array_equal(result, expected)
                else:
                    assert result == expected

            # freq raises AttributeError on an Int64Index because its not
            # defined we mostly care about Series here anyhow
            if not ignore_failures:
                for o in self.not_valid_objs:

                    # an object that is datetimelike will raise a TypeError,
                    # otherwise an AttributeError
                    err = AttributeError
                    if issubclass(type(o), DatetimeIndexOpsMixin):
                        err = TypeError

                    with pytest.raises(err):
                        getattr(o, op)

    @pytest.mark.parametrize('klass', [Series, DataFrame, Panel])
    def test_binary_ops_docs(self, klass):
        op_map = {'add': '+',
                  'sub': '-',
                  'mul': '*',
                  'mod': '%',
                  'pow': '**',
                  'truediv': '/',
                  'floordiv': '//'}
        for op_name in op_map:
            operand1 = klass.__name__.lower()
            operand2 = 'other'
            op = op_map[op_name]
            expected_str = ' '.join([operand1, op, operand2])
            assert expected_str in getattr(klass, op_name).__doc__

            # reverse version of the binary ops
            expected_str = ' '.join([operand2, op, operand1])
            assert expected_str in getattr(klass, 'r' + op_name).__doc__


class TestIndexOps(Ops):

    def setup_method(self, method):
        super(TestIndexOps, self).setup_method(method)
        self.is_valid_objs = self.objs
        self.not_valid_objs = []

    def test_none_comparison(self):

        # bug brought up by #1079
        # changed from TypeError in 0.17.0
        for o in self.is_valid_objs:
            if isinstance(o, Series):

                o[0] = np.nan

                # noinspection PyComparisonWithNone
                result = o == None  # noqa
                assert not result.iat[0]
                assert not result.iat[1]

                # noinspection PyComparisonWithNone
                result = o != None  # noqa
                assert result.iat[0]
                assert result.iat[1]

                result = None == o  # noqa
                assert not result.iat[0]
                assert not result.iat[1]

                result = None != o  # noqa
                assert result.iat[0]
                assert result.iat[1]

                if (is_datetime64_dtype(o) or is_datetime64tz_dtype(o)):
                    # Following DatetimeIndex (and Timestamp) convention,
                    # inequality comparisons with Series[datetime64] raise
                    with pytest.raises(TypeError):
                        None > o
                    with pytest.raises(TypeError):
                        o > None
                else:
                    result = None > o
                    assert not result.iat[0]
                    assert not result.iat[1]

                    result = o < None
                    assert not result.iat[0]
                    assert not result.iat[1]

    def test_ndarray_compat_properties(self):

        for o in self.objs:
            # Check that we work.
            for p in ['shape', 'dtype', 'T', 'nbytes']:
                assert getattr(o, p, None) is not None

            # deprecated properties
            for p in ['flags', 'strides', 'itemsize']:
                with tm.assert_produces_warning(FutureWarning):
                    assert getattr(o, p, None) is not None

            with tm.assert_produces_warning(FutureWarning):
                assert hasattr(o, 'base')

            # If we have a datetime-like dtype then needs a view to work
            # but the user is responsible for that
            try:
                with tm.assert_produces_warning(FutureWarning):
                    assert o.data is not None
            except ValueError:
                pass

            with pytest.raises(ValueError):
                o.item()  # len > 1

            assert o.ndim == 1
            assert o.size == len(o)

        assert Index([1]).item() == 1
        assert Series([1]).item() == 1

    def test_value_counts_unique_nunique(self):
        for orig in self.objs:
            o = orig.copy()
            klass = type(o)
            values = o._values

            if isinstance(values, Index):
                # reset name not to affect latter process
                values.name = None

            # create repeated values, 'n'th element is repeated by n+1 times
            # skip boolean, because it only has 2 values at most
            if isinstance(o, Index) and o.is_boolean():
                continue
            elif isinstance(o, Index):
                expected_index = Index(o[::-1])
                expected_index.name = None
                o = o.repeat(range(1, len(o) + 1))
                o.name = 'a'
            else:
                expected_index = Index(values[::-1])
                idx = o.index.repeat(range(1, len(o) + 1))
                # take-based repeat
                indices = np.repeat(np.arange(len(o)), range(1, len(o) + 1))
                rep = values.take(indices)
                o = klass(rep, index=idx, name='a')

            # check values has the same dtype as the original
            assert o.dtype == orig.dtype

            expected_s = Series(range(10, 0, -1), index=expected_index,
                                dtype='int64', name='a')

            result = o.value_counts()
            tm.assert_series_equal(result, expected_s)
            assert result.index.name is None
            assert result.name == 'a'

            result = o.unique()
            if isinstance(o, Index):
                assert isinstance(result, o.__class__)
                tm.assert_index_equal(result, orig)
            elif is_datetime64tz_dtype(o):
                # datetimetz Series returns array of Timestamp
                assert result[0] == orig[0]
                for r in result:
                    assert isinstance(r, Timestamp)

                tm.assert_numpy_array_equal(
                    result.astype(object),
                    orig._values.astype(object))
            else:
                tm.assert_numpy_array_equal(result, orig.values)

            assert o.nunique() == len(np.unique(o.values))

    @pytest.mark.parametrize('null_obj', [np.nan, None])
    def test_value_counts_unique_nunique_null(self, null_obj):

        for orig in self.objs:
            o = orig.copy()
            klass = type(o)
            values = o._ndarray_values

            if not self._allow_na_ops(o):
                continue

            # special assign to the numpy array
            if is_datetime64tz_dtype(o):
                if isinstance(o, DatetimeIndex):
                    v = o.asi8
                    v[0:2] = iNaT
                    values = o._shallow_copy(v)
                else:
                    o = o.copy()
                    o[0:2] = iNaT
                    values = o._values

            elif needs_i8_conversion(o):
                values[0:2] = iNaT
                values = o._shallow_copy(values)
            else:
                values[0:2] = null_obj
            # check values has the same dtype as the original

            assert values.dtype == o.dtype

            # create repeated values, 'n'th element is repeated by n+1
            # times
            if isinstance(o, (DatetimeIndex, PeriodIndex)):
                expected_index = o.copy()
                expected_index.name = None

                # attach name to klass
                o = klass(values.repeat(range(1, len(o) + 1)))
                o.name = 'a'
            else:
                if isinstance(o, DatetimeIndex):
                    expected_index = orig._values._shallow_copy(values)
                else:
                    expected_index = Index(values)
                expected_index.name = None
                o = o.repeat(range(1, len(o) + 1))
                o.name = 'a'

            # check values has the same dtype as the original
            assert o.dtype == orig.dtype
            # check values correctly have NaN
            nanloc = np.zeros(len(o), dtype=np.bool)
            nanloc[:3] = True
            if isinstance(o, Index):
                tm.assert_numpy_array_equal(pd.isna(o), nanloc)
            else:
                exp = Series(nanloc, o.index, name='a')
                tm.assert_series_equal(pd.isna(o), exp)

            expected_s_na = Series(list(range(10, 2, -1)) + [3],
                                   index=expected_index[9:0:-1],
                                   dtype='int64', name='a')
            expected_s = Series(list(range(10, 2, -1)),
                                index=expected_index[9:1:-1],
                                dtype='int64', name='a')

            result_s_na = o.value_counts(dropna=False)
            tm.assert_series_equal(result_s_na, expected_s_na)
            assert result_s_na.index.name is None
            assert result_s_na.name == 'a'
            result_s = o.value_counts()
            tm.assert_series_equal(o.value_counts(), expected_s)
            assert result_s.index.name is None
            assert result_s.name == 'a'

            result = o.unique()
            if isinstance(o, Index):
                tm.assert_index_equal(result,
                                      Index(values[1:], name='a'))
            elif is_datetime64tz_dtype(o):
                # unable to compare NaT / nan
                tm.assert_extension_array_equal(result[1:], values[2:])
                assert result[0] is pd.NaT
            else:
                tm.assert_numpy_array_equal(result[1:], values[2:])

                assert pd.isna(result[0])
                assert result.dtype == orig.dtype

            assert o.nunique() == 8
            assert o.nunique(dropna=False) == 9

    @pytest.mark.parametrize('klass', [Index, Series])
    def test_value_counts_inferred(self, klass):
        s_values = ['a', 'b', 'b', 'b', 'b', 'c', 'd', 'd', 'a', 'a']
        s = klass(s_values)
        expected = Series([4, 3, 2, 1], index=['b', 'a', 'd', 'c'])
        tm.assert_series_equal(s.value_counts(), expected)

        if isinstance(s, Index):
            exp = Index(np.unique(np.array(s_values, dtype=np.object_)))
            tm.assert_index_equal(s.unique(), exp)
        else:
            exp = np.unique(np.array(s_values, dtype=np.object_))
            tm.assert_numpy_array_equal(s.unique(), exp)

        assert s.nunique() == 4
        # don't sort, have to sort after the fact as not sorting is
        # platform-dep
        hist = s.value_counts(sort=False).sort_values()
        expected = Series([3, 1, 4, 2], index=list('acbd')).sort_values()
        tm.assert_series_equal(hist, expected)

        # sort ascending
        hist = s.value_counts(ascending=True)
        expected = Series([1, 2, 3, 4], index=list('cdab'))
        tm.assert_series_equal(hist, expected)

        # relative histogram.
        hist = s.value_counts(normalize=True)
        expected = Series([.4, .3, .2, .1], index=['b', 'a', 'd', 'c'])
        tm.assert_series_equal(hist, expected)

    @pytest.mark.parametrize('klass', [Index, Series])
    def test_value_counts_bins(self, klass):
        s_values = ['a', 'b', 'b', 'b', 'b', 'c', 'd', 'd', 'a', 'a']
        s = klass(s_values)

        # bins
        with pytest.raises(TypeError):
            s.value_counts(bins=1)

        s1 = Series([1, 1, 2, 3])
        res1 = s1.value_counts(bins=1)
        exp1 = Series({Interval(0.997, 3.0): 4})
        tm.assert_series_equal(res1, exp1)
        res1n = s1.value_counts(bins=1, normalize=True)
        exp1n = Series({Interval(0.997, 3.0): 1.0})
        tm.assert_series_equal(res1n, exp1n)

        if isinstance(s1, Index):
            tm.assert_index_equal(s1.unique(), Index([1, 2, 3]))
        else:
            exp = np.array([1, 2, 3], dtype=np.int64)
            tm.assert_numpy_array_equal(s1.unique(), exp)

        assert s1.nunique() == 3

        # these return the same
        res4 = s1.value_counts(bins=4, dropna=True)
        intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
        exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 3, 1, 2]))
        tm.assert_series_equal(res4, exp4)

        res4 = s1.value_counts(bins=4, dropna=False)
        intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
        exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 3, 1, 2]))
        tm.assert_series_equal(res4, exp4)

        res4n = s1.value_counts(bins=4, normalize=True)
        exp4n = Series([0.5, 0.25, 0.25, 0],
                       index=intervals.take([0, 3, 1, 2]))
        tm.assert_series_equal(res4n, exp4n)

        # handle NA's properly
        s_values = ['a', 'b', 'b', 'b', np.nan, np.nan,
                    'd', 'd', 'a', 'a', 'b']
        s = klass(s_values)
        expected = Series([4, 3, 2], index=['b', 'a', 'd'])
        tm.assert_series_equal(s.value_counts(), expected)

        if isinstance(s, Index):
            exp = Index(['a', 'b', np.nan, 'd'])
            tm.assert_index_equal(s.unique(), exp)
        else:
            exp = np.array(['a', 'b', np.nan, 'd'], dtype=object)
            tm.assert_numpy_array_equal(s.unique(), exp)
        assert s.nunique() == 3

        s = klass({})
        expected = Series([], dtype=np.int64)
        tm.assert_series_equal(s.value_counts(), expected,
                               check_index_type=False)
        # returned dtype differs depending on original
        if isinstance(s, Index):
            tm.assert_index_equal(s.unique(), Index([]), exact=False)
        else:
            tm.assert_numpy_array_equal(s.unique(), np.array([]),
                                        check_dtype=False)

        assert s.nunique() == 0

    @pytest.mark.parametrize('klass', [Index, Series])
    def test_value_counts_datetime64(self, klass):

        # GH 3002, datetime64[ns]
        # don't test names though
        txt = "\n".join(['xxyyzz20100101PIE', 'xxyyzz20100101GUM',
                         'xxyyzz20100101EGG', 'xxyyww20090101EGG',
                         'foofoo20080909PIE', 'foofoo20080909GUM'])
        f = StringIO(txt)
        df = pd.read_fwf(f, widths=[6, 8, 3],
                         names=["person_id", "dt", "food"],
                         parse_dates=["dt"])

        s = klass(df['dt'].copy())
        s.name = None
        idx = pd.to_datetime(['2010-01-01 00:00:00',
                              '2008-09-09 00:00:00',
                              '2009-01-01 00:00:00'])
        expected_s = Series([3, 2, 1], index=idx)
        tm.assert_series_equal(s.value_counts(), expected_s)

        expected = np_array_datetime64_compat(['2010-01-01 00:00:00',
                                               '2009-01-01 00:00:00',
                                               '2008-09-09 00:00:00'],
                                              dtype='datetime64[ns]')
        if isinstance(s, Index):
            tm.assert_index_equal(s.unique(), DatetimeIndex(expected))
        else:
            tm.assert_numpy_array_equal(s.unique(), expected)

        assert s.nunique() == 3

        # with NaT
        s = df['dt'].copy()
        s = klass([v for v in s.values] + [pd.NaT])

        result = s.value_counts()
        assert result.index.dtype == 'datetime64[ns]'
        tm.assert_series_equal(result, expected_s)

        result = s.value_counts(dropna=False)
        expected_s[pd.NaT] = 1
        tm.assert_series_equal(result, expected_s)

        unique = s.unique()
        assert unique.dtype == 'datetime64[ns]'

        # numpy_array_equal cannot compare pd.NaT
        if isinstance(s, Index):
            exp_idx = DatetimeIndex(expected.tolist() + [pd.NaT])
            tm.assert_index_equal(unique, exp_idx)
        else:
            tm.assert_numpy_array_equal(unique[:3], expected)
            assert pd.isna(unique[3])

        assert s.nunique() == 3
        assert s.nunique(dropna=False) == 4

        # timedelta64[ns]
        td = df.dt - df.dt + timedelta(1)
        td = klass(td, name='dt')

        result = td.value_counts()
        expected_s = Series([6], index=[Timedelta('1day')], name='dt')
        tm.assert_series_equal(result, expected_s)

        expected = TimedeltaIndex(['1 days'], name='dt')
        if isinstance(td, Index):
            tm.assert_index_equal(td.unique(), expected)
        else:
            tm.assert_numpy_array_equal(td.unique(), expected.values)

        td2 = timedelta(1) + (df.dt - df.dt)
        td2 = klass(td2, name='dt')
        result2 = td2.value_counts()
        tm.assert_series_equal(result2, expected_s)

    def test_factorize(self):
        for orig in self.objs:
            o = orig.copy()

            if isinstance(o, Index) and o.is_boolean():
                exp_arr = np.array([0, 1] + [0] * 8, dtype=np.intp)
                exp_uniques = o
                exp_uniques = Index([False, True])
            else:
                exp_arr = np.array(range(len(o)), dtype=np.intp)
                exp_uniques = o
            labels, uniques = o.factorize()

            tm.assert_numpy_array_equal(labels, exp_arr)
            if isinstance(o, Series):
                tm.assert_index_equal(uniques, Index(orig),
                                      check_names=False)
            else:
                # factorize explicitly resets name
                tm.assert_index_equal(uniques, exp_uniques,
                                      check_names=False)

    def test_factorize_repeated(self):
        for orig in self.objs:
            o = orig.copy()

            # don't test boolean
            if isinstance(o, Index) and o.is_boolean():
                continue

            # sort by value, and create duplicates
            if isinstance(o, Series):
                o = o.sort_values()
                n = o.iloc[5:].append(o)
            else:
                indexer = o.argsort()
                o = o.take(indexer)
                n = o[5:].append(o)

            exp_arr = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               dtype=np.intp)
            labels, uniques = n.factorize(sort=True)

            tm.assert_numpy_array_equal(labels, exp_arr)
            if isinstance(o, Series):
                tm.assert_index_equal(uniques, Index(orig).sort_values(),
                                      check_names=False)
            else:
                tm.assert_index_equal(uniques, o, check_names=False)

            exp_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                               np.intp)
            labels, uniques = n.factorize(sort=False)
            tm.assert_numpy_array_equal(labels, exp_arr)

            if isinstance(o, Series):
                expected = Index(o.iloc[5:10].append(o.iloc[:5]))
                tm.assert_index_equal(uniques, expected, check_names=False)
            else:
                expected = o[5:10].append(o[:5])
                tm.assert_index_equal(uniques, expected, check_names=False)

    def test_duplicated_drop_duplicates_index(self):
        # GH 4060
        for original in self.objs:
            if isinstance(original, Index):

                # special case
                if original.is_boolean():
                    result = original.drop_duplicates()
                    expected = Index([False, True], name='a')
                    tm.assert_index_equal(result, expected)
                    continue

                # original doesn't have duplicates
                expected = np.array([False] * len(original), dtype=bool)
                duplicated = original.duplicated()
                tm.assert_numpy_array_equal(duplicated, expected)
                assert duplicated.dtype == bool
                result = original.drop_duplicates()
                tm.assert_index_equal(result, original)
                assert result is not original

                # has_duplicates
                assert not original.has_duplicates

                # create repeated values, 3rd and 5th values are duplicated
                idx = original[list(range(len(original))) + [5, 3]]
                expected = np.array([False] * len(original) + [True, True],
                                    dtype=bool)
                duplicated = idx.duplicated()
                tm.assert_numpy_array_equal(duplicated, expected)
                assert duplicated.dtype == bool
                tm.assert_index_equal(idx.drop_duplicates(), original)

                base = [False] * len(idx)
                base[3] = True
                base[5] = True
                expected = np.array(base)

                duplicated = idx.duplicated(keep='last')
                tm.assert_numpy_array_equal(duplicated, expected)
                assert duplicated.dtype == bool
                result = idx.drop_duplicates(keep='last')
                tm.assert_index_equal(result, idx[~expected])

                base = [False] * len(original) + [True, True]
                base[3] = True
                base[5] = True
                expected = np.array(base)

                duplicated = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(duplicated, expected)
                assert duplicated.dtype == bool
                result = idx.drop_duplicates(keep=False)
                tm.assert_index_equal(result, idx[~expected])

                with pytest.raises(TypeError,
                                   match=(r"drop_duplicates\(\) got an "
                                          r"unexpected keyword argument")):
                    idx.drop_duplicates(inplace=True)

            else:
                expected = Series([False] * len(original),
                                  index=original.index, name='a')
                tm.assert_series_equal(original.duplicated(), expected)
                result = original.drop_duplicates()
                tm.assert_series_equal(result, original)
                assert result is not original

                idx = original.index[list(range(len(original))) + [5, 3]]
                values = original._values[list(range(len(original))) + [5, 3]]
                s = Series(values, index=idx, name='a')

                expected = Series([False] * len(original) + [True, True],
                                  index=idx, name='a')
                tm.assert_series_equal(s.duplicated(), expected)
                tm.assert_series_equal(s.drop_duplicates(), original)

                base = [False] * len(idx)
                base[3] = True
                base[5] = True
                expected = Series(base, index=idx, name='a')

                tm.assert_series_equal(s.duplicated(keep='last'), expected)
                tm.assert_series_equal(s.drop_duplicates(keep='last'),
                                       s[~np.array(base)])

                base = [False] * len(original) + [True, True]
                base[3] = True
                base[5] = True
                expected = Series(base, index=idx, name='a')

                tm.assert_series_equal(s.duplicated(keep=False), expected)
                tm.assert_series_equal(s.drop_duplicates(keep=False),
                                       s[~np.array(base)])

                s.drop_duplicates(inplace=True)
                tm.assert_series_equal(s, original)

    def test_drop_duplicates_series_vs_dataframe(self):
        # GH 14192
        df = pd.DataFrame({'a': [1, 1, 1, 'one', 'one'],
                           'b': [2, 2, np.nan, np.nan, np.nan],
                           'c': [3, 3, np.nan, np.nan, 'three'],
                           'd': [1, 2, 3, 4, 4],
                           'e': [datetime(2015, 1, 1), datetime(2015, 1, 1),
                                 datetime(2015, 2, 1), pd.NaT, pd.NaT]
                           })
        for column in df.columns:
            for keep in ['first', 'last', False]:
                dropped_frame = df[[column]].drop_duplicates(keep=keep)
                dropped_series = df[column].drop_duplicates(keep=keep)
                tm.assert_frame_equal(dropped_frame, dropped_series.to_frame())

    def test_fillna(self):
        # # GH 11343
        # though Index.fillna and Series.fillna has separate impl,
        # test here to confirm these works as the same

        for orig in self.objs:

            o = orig.copy()
            values = o.values

            # values will not be changed
            result = o.fillna(o.astype(object).values[0])
            if isinstance(o, Index):
                tm.assert_index_equal(o, result)
            else:
                tm.assert_series_equal(o, result)
            # check shallow_copied
            assert o is not result

        for null_obj in [np.nan, None]:
            for orig in self.objs:
                o = orig.copy()
                klass = type(o)

                if not self._allow_na_ops(o):
                    continue

                if needs_i8_conversion(o):

                    values = o.astype(object).values
                    fill_value = values[0]
                    values[0:2] = pd.NaT
                else:
                    values = o.values.copy()
                    fill_value = o.values[0]
                    values[0:2] = null_obj

                expected = [fill_value] * 2 + list(values[2:])

                expected = klass(expected)
                o = klass(values)

                # check values has the same dtype as the original
                assert o.dtype == orig.dtype

                result = o.fillna(fill_value)
                if isinstance(o, Index):
                    tm.assert_index_equal(result, expected)
                else:
                    tm.assert_series_equal(result, expected)
                # check shallow_copied
                assert o is not result

    @pytest.mark.skipif(PYPY, reason="not relevant for PyPy")
    def test_memory_usage(self):
        for o in self.objs:
            res = o.memory_usage()
            res_deep = o.memory_usage(deep=True)

            if (is_object_dtype(o) or (isinstance(o, Series) and
                                       is_object_dtype(o.index))):
                # if there are objects, only deep will pick them up
                assert res_deep > res
            else:
                assert res == res_deep

            if isinstance(o, Series):
                assert ((o.memory_usage(index=False) +
                         o.index.memory_usage()) ==
                        o.memory_usage(index=True))

            # sys.getsizeof will call the .memory_usage with
            # deep=True, and add on some GC overhead
            diff = res_deep - sys.getsizeof(o)
            assert abs(diff) < 100

    def test_searchsorted(self):
        # See gh-12238
        for o in self.objs:
            index = np.searchsorted(o, max(o))
            assert 0 <= index <= len(o)

            index = np.searchsorted(o, max(o), sorter=range(len(o)))
            assert 0 <= index <= len(o)

    def test_validate_bool_args(self):
        invalid_values = [1, "True", [1, 2, 3], 5.0]

        for value in invalid_values:
            with pytest.raises(ValueError):
                self.int_series.drop_duplicates(inplace=value)

    def test_getitem(self):
        for i in self.indexes:
            s = pd.Series(i)

            assert i[0] == s.iloc[0]
            assert i[5] == s.iloc[5]
            assert i[-1] == s.iloc[-1]

            assert i[-1] == i[9]

            with pytest.raises(IndexError):
                i[20]
            with pytest.raises(IndexError):
                s.iloc[20]

    @pytest.mark.parametrize('indexer_klass', [list, pd.Index])
    @pytest.mark.parametrize('indexer', [[True] * 10, [False] * 10,
                                         [True, False, True, True, False,
                                          False, True, True, False, True]])
    def test_bool_indexing(self, indexer_klass, indexer):
        # GH 22533
        for idx in self.indexes:
            exp_idx = [i for i in range(len(indexer)) if indexer[i]]
            tm.assert_index_equal(idx[indexer_klass(indexer)], idx[exp_idx])
            s = pd.Series(idx)
            tm.assert_series_equal(s[indexer_klass(indexer)], s.iloc[exp_idx])


class TestTranspose(Ops):
    errmsg = "the 'axes' parameter is not supported"

    def test_transpose(self):
        for obj in self.objs:
            tm.assert_equal(obj.transpose(), obj)

    def test_transpose_non_default_axes(self):
        for obj in self.objs:
            with pytest.raises(ValueError, match=self.errmsg):
                obj.transpose(1)
            with pytest.raises(ValueError, match=self.errmsg):
                obj.transpose(axes=1)

    def test_numpy_transpose(self):
        for obj in self.objs:
            tm.assert_equal(np.transpose(obj), obj)

            with pytest.raises(ValueError, match=self.errmsg):
                np.transpose(obj, axes=1)


class TestNoNewAttributesMixin(object):

    def test_mixin(self):
        class T(NoNewAttributesMixin):
            pass

        t = T()
        assert not hasattr(t, "__frozen")

        t.a = "test"
        assert t.a == "test"

        t._freeze()
        assert "__frozen" in dir(t)
        assert getattr(t, "__frozen")

        with pytest.raises(AttributeError):
            t.b = "test"

        assert not hasattr(t, "b")


class TestToIterable(object):
    # test that we convert an iterable to python types

    dtypes = [
        ('int8', (int, long)),
        ('int16', (int, long)),
        ('int32', (int, long)),
        ('int64', (int, long)),
        ('uint8', (int, long)),
        ('uint16', (int, long)),
        ('uint32', (int, long)),
        ('uint64', (int, long)),
        ('float16', float),
        ('float32', float),
        ('float64', float),
        ('datetime64[ns]', Timestamp),
        ('datetime64[ns, US/Eastern]', Timestamp),
        ('timedelta64[ns]', Timedelta)]

    @pytest.mark.parametrize(
        'dtype, rdtype', dtypes)
    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ], ids=['tolist', 'to_list', 'list', 'iter'])
    @pytest.mark.parametrize('typ', [Series, Index])
    @pytest.mark.filterwarnings("ignore:\\n    Passing:FutureWarning")
    # TODO(GH-24559): Remove the filterwarnings
    def test_iterable(self, typ, method, dtype, rdtype):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        s = typ([1], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        'dtype, rdtype, obj',
        [
            ('object', object, 'a'),
            ('object', (int, long), 1),
            ('category', object, 'a'),
            ('category', (int, long), 1)])
    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ], ids=['tolist', 'to_list', 'list', 'iter'])
    @pytest.mark.parametrize('typ', [Series, Index])
    def test_iterable_object_and_category(self, typ, method,
                                          dtype, rdtype, obj):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        s = typ([obj], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        'dtype, rdtype', dtypes)
    def test_iterable_items(self, dtype, rdtype):
        # gh-13258
        # test items / iteritems yields the correct boxed scalars
        # this only applies to series
        s = Series([1], dtype=dtype)
        _, result = list(s.items())[0]
        assert isinstance(result, rdtype)

        _, result = list(s.iteritems())[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        'dtype, rdtype',
        dtypes + [
            ('object', (int, long)),
            ('category', (int, long))])
    @pytest.mark.parametrize('typ', [Series, Index])
    @pytest.mark.filterwarnings("ignore:\\n    Passing:FutureWarning")
    # TODO(GH-24559): Remove the filterwarnings
    def test_iterable_map(self, typ, dtype, rdtype):
        # gh-13236
        # coerce iteration to underlying python / pandas types
        s = typ([1], dtype=dtype)
        result = s.map(type)[0]
        if not isinstance(rdtype, tuple):
            rdtype = tuple([rdtype])
        assert result in rdtype

    @pytest.mark.parametrize(
        'method',
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ], ids=['tolist', 'to_list', 'list', 'iter'])
    def test_categorial_datetimelike(self, method):
        i = CategoricalIndex([Timestamp('1999-12-31'),
                              Timestamp('2000-12-31')])

        result = method(i)[0]
        assert isinstance(result, Timestamp)

    def test_iter_box(self):
        vals = [Timestamp('2011-01-01'), Timestamp('2011-01-02')]
        s = Series(vals)
        assert s.dtype == 'datetime64[ns]'
        for res, exp in zip(s, vals):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp

        vals = [Timestamp('2011-01-01', tz='US/Eastern'),
                Timestamp('2011-01-02', tz='US/Eastern')]
        s = Series(vals)

        assert s.dtype == 'datetime64[ns, US/Eastern]'
        for res, exp in zip(s, vals):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp

        # timedelta
        vals = [Timedelta('1 days'), Timedelta('2 days')]
        s = Series(vals)
        assert s.dtype == 'timedelta64[ns]'
        for res, exp in zip(s, vals):
            assert isinstance(res, Timedelta)
            assert res == exp

        # period
        vals = [pd.Period('2011-01-01', freq='M'),
                pd.Period('2011-01-02', freq='M')]
        s = Series(vals)
        assert s.dtype == 'Period[M]'
        for res, exp in zip(s, vals):
            assert isinstance(res, pd.Period)
            assert res.freq == 'M'
            assert res == exp


@pytest.mark.parametrize('array, expected_type, dtype', [
    (np.array([0, 1], dtype=np.int64), np.ndarray, 'int64'),
    (np.array(['a', 'b']), np.ndarray, 'object'),
    (pd.Categorical(['a', 'b']), pd.Categorical, 'category'),
    (pd.DatetimeIndex(['2017', '2018'], tz="US/Central"), DatetimeArray,
     'datetime64[ns, US/Central]'),

    (pd.PeriodIndex([2018, 2019], freq='A'), pd.core.arrays.PeriodArray,
     pd.core.dtypes.dtypes.PeriodDtype("A-DEC")),
    (pd.IntervalIndex.from_breaks([0, 1, 2]), pd.core.arrays.IntervalArray,
     'interval'),

    # This test is currently failing for datetime64[ns] and timedelta64[ns].
    # The NumPy type system is sufficient for representing these types, so
    # we just use NumPy for Series / DataFrame columns of these types (so
    # we get consolidation and so on).
    # However, DatetimeIndex and TimedeltaIndex use the DateLikeArray
    # abstraction to for code reuse.
    # At the moment, we've judged that allowing this test to fail is more
    # practical that overriding Series._values to special case
    # Series[M8[ns]] and Series[m8[ns]] to return a DateLikeArray.
    pytest.param(
        pd.DatetimeIndex(['2017', '2018']), np.ndarray, 'datetime64[ns]',
        marks=[pytest.mark.xfail(reason="datetime _values", strict=True)]
    ),
    pytest.param(
        pd.TimedeltaIndex([10**10]), np.ndarray, 'm8[ns]',
        marks=[pytest.mark.xfail(reason="timedelta _values", strict=True)]
    ),

])
def test_values_consistent(array, expected_type, dtype):
    l_values = pd.Series(array)._values
    r_values = pd.Index(array)._values
    assert type(l_values) is expected_type
    assert type(l_values) is type(r_values)

    tm.assert_equal(l_values, r_values)


@pytest.mark.parametrize('array, expected', [
    (np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
    (np.array(['0', '1']), np.array(['0', '1'], dtype=object)),
    (pd.Categorical(['a', 'a']), np.array([0, 0], dtype='int8')),
    (pd.DatetimeIndex(['2017-01-01T00:00:00']),
     np.array(['2017-01-01T00:00:00'], dtype='M8[ns]')),
    (pd.DatetimeIndex(['2017-01-01T00:00:00'], tz="US/Eastern"),
     np.array(['2017-01-01T05:00:00'], dtype='M8[ns]')),
    (pd.TimedeltaIndex([10**10]), np.array([10**10], dtype='m8[ns]')),
    (pd.PeriodIndex(['2017', '2018'], freq='D'),
     np.array([17167, 17532], dtype=np.int64)),
])
def test_ndarray_values(array, expected):
    l_values = pd.Series(array)._ndarray_values
    r_values = pd.Index(array)._ndarray_values
    tm.assert_numpy_array_equal(l_values, r_values)
    tm.assert_numpy_array_equal(l_values, expected)


@pytest.mark.parametrize("arr", [
    np.array([1, 2, 3]),
])
def test_numpy_array(arr):
    ser = pd.Series(arr)
    result = ser.array
    expected = PandasArray(arr)
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_all_dtypes(any_numpy_dtype):
    ser = pd.Series(dtype=any_numpy_dtype)
    result = ser.array
    if is_datetime64_dtype(any_numpy_dtype):
        assert isinstance(result, DatetimeArray)
    elif is_timedelta64_dtype(any_numpy_dtype):
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, PandasArray)


@pytest.mark.parametrize("array, attr", [
    (pd.Categorical(['a', 'b']), '_codes'),
    (pd.core.arrays.period_array(['2000', '2001'], freq='D'), '_data'),
    (pd.core.arrays.integer_array([0, np.nan]), '_data'),
    (pd.core.arrays.IntervalArray.from_breaks([0, 1]), '_left'),
    (pd.SparseArray([0, 1]), '_sparse_values'),
    (DatetimeArray(np.array([1, 2], dtype="datetime64[ns]")), "_data"),
    # tz-aware Datetime
    (DatetimeArray(np.array(['2000-01-01T12:00:00',
                             '2000-01-02T12:00:00'],
                            dtype='M8[ns]'),
                   dtype=DatetimeTZDtype(tz="US/Central")),
     '_data'),
])
@pytest.mark.parametrize('box', [pd.Series, pd.Index])
def test_array(array, attr, box):
    if array.dtype.name in ('Int64', 'Sparse[int64, 0]') and box is pd.Index:
        pytest.skip("No index type for {}".format(array.dtype))
    result = box(array, copy=False).array

    if attr:
        array = getattr(array, attr)
        result = getattr(result, attr)

    assert result is array


def test_array_multiindex_raises():
    idx = pd.MultiIndex.from_product([['A'], ['a', 'b']])
    with pytest.raises(ValueError, match='MultiIndex'):
        idx.array


@pytest.mark.parametrize('array, expected', [
    (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
    (pd.Categorical(['a', 'b']), np.array(['a', 'b'], dtype=object)),
    (pd.core.arrays.period_array(['2000', '2001'], freq='D'),
     np.array([pd.Period('2000', freq="D"), pd.Period('2001', freq='D')])),
    (pd.core.arrays.integer_array([0, np.nan]),
     np.array([0, np.nan], dtype=object)),
    (pd.core.arrays.IntervalArray.from_breaks([0, 1, 2]),
     np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object)),
    (pd.SparseArray([0, 1]), np.array([0, 1], dtype=np.int64)),

    # tz-naive datetime
    (DatetimeArray(np.array(['2000', '2001'], dtype='M8[ns]')),
     np.array(['2000', '2001'], dtype='M8[ns]')),

    # tz-aware stays tz`-aware
    (DatetimeArray(np.array(['2000-01-01T06:00:00',
                             '2000-01-02T06:00:00'],
                            dtype='M8[ns]'),
                   dtype=DatetimeTZDtype(tz='US/Central')),
     np.array([pd.Timestamp('2000-01-01', tz='US/Central'),
               pd.Timestamp('2000-01-02', tz='US/Central')])),

    # Timedelta
    (TimedeltaArray(np.array([0, 3600000000000], dtype='i8'), freq='H'),
     np.array([0, 3600000000000], dtype='m8[ns]')),
])
@pytest.mark.parametrize('box', [pd.Series, pd.Index])
def test_to_numpy(array, expected, box):
    thing = box(array)

    if array.dtype.name in ('Int64', 'Sparse[int64, 0]') and box is pd.Index:
        pytest.skip("No index type for {}".format(array.dtype))

    result = thing.to_numpy()
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize("arr", [
    np.array([1, 2, 3], dtype="int64"),
    np.array(['a', 'b', 'c'], dtype=object),
])
def test_to_numpy_copy(arr, as_series):
    obj = pd.Index(arr, copy=False)
    if as_series:
        obj = pd.Series(obj.values, copy=False)

    # no copy by default
    result = obj.to_numpy()
    assert np.shares_memory(arr, result) is True

    result = obj.to_numpy(copy=False)
    assert np.shares_memory(arr, result) is True

    # copy=True
    result = obj.to_numpy(copy=True)
    assert np.shares_memory(arr, result) is False


@pytest.mark.parametrize("as_series", [True, False])
def test_to_numpy_dtype(as_series):
    tz = "US/Eastern"
    obj = pd.DatetimeIndex(['2000', '2001'], tz=tz)
    if as_series:
        obj = pd.Series(obj)

    # preserve tz by default
    result = obj.to_numpy()
    expected = np.array([pd.Timestamp('2000', tz=tz),
                         pd.Timestamp('2001', tz=tz)],
                        dtype=object)
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="M8[ns]")
    expected = np.array(['2000-01-01T05', '2001-01-01T05'],
                        dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
