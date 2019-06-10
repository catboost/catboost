from __future__ import division

from functools import partial

import numpy as np
import pytest

from pandas.compat import lzip

from pandas.core.dtypes.common import is_categorical_dtype
from pandas.core.dtypes.dtypes import IntervalDtype

from pandas import (
    Categorical, CategoricalIndex, Float64Index, Index, Int64Index, Interval,
    IntervalIndex, date_range, notna, period_range, timedelta_range)
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
import pandas.util.testing as tm


@pytest.fixture(params=[None, 'foo'])
def name(request):
    return request.param


class Base(object):
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    @pytest.mark.parametrize('breaks', [
        [3, 14, 15, 92, 653],
        np.arange(10, dtype='int64'),
        Int64Index(range(-10, 11)),
        Float64Index(np.arange(20, 30, 0.5)),
        date_range('20180101', periods=10),
        date_range('20180101', periods=10, tz='US/Eastern'),
        timedelta_range('1 day', periods=10)])
    def test_constructor(self, constructor, breaks, closed, name):
        result_kwargs = self.get_kwargs_from_breaks(breaks, closed)
        result = constructor(closed=closed, name=name, **result_kwargs)

        assert result.closed == closed
        assert result.name == name
        assert result.dtype.subtype == getattr(breaks, 'dtype', 'int64')
        tm.assert_index_equal(result.left, Index(breaks[:-1]))
        tm.assert_index_equal(result.right, Index(breaks[1:]))

    @pytest.mark.parametrize('breaks, subtype', [
        (Int64Index([0, 1, 2, 3, 4]), 'float64'),
        (Int64Index([0, 1, 2, 3, 4]), 'datetime64[ns]'),
        (Int64Index([0, 1, 2, 3, 4]), 'timedelta64[ns]'),
        (Float64Index([0, 1, 2, 3, 4]), 'int64'),
        (date_range('2017-01-01', periods=5), 'int64'),
        (timedelta_range('1 day', periods=5), 'int64')])
    def test_constructor_dtype(self, constructor, breaks, subtype):
        # GH 19262: conversion via dtype parameter
        expected_kwargs = self.get_kwargs_from_breaks(breaks.astype(subtype))
        expected = constructor(**expected_kwargs)

        result_kwargs = self.get_kwargs_from_breaks(breaks)
        iv_dtype = IntervalDtype(subtype)
        for dtype in (iv_dtype, str(iv_dtype)):
            result = constructor(dtype=dtype, **result_kwargs)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('breaks', [
        [np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_subtype = np.float64
        expected_values = np.array(breaks[:-1], dtype=object)

        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(result._ndarray_values, expected_values)

    @pytest.mark.parametrize('breaks', [
        [],
        np.array([], dtype='int64'),
        np.array([], dtype='float64'),
        np.array([], dtype='datetime64[ns]'),
        np.array([], dtype='timedelta64[ns]')])
    def test_constructor_empty(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_values = np.array([], dtype=object)
        expected_subtype = getattr(breaks, 'dtype', np.int64)

        assert result.empty
        assert result.closed == closed
        assert result.dtype.subtype == expected_subtype
        tm.assert_numpy_array_equal(result._ndarray_values, expected_values)

    @pytest.mark.parametrize('breaks', [
        tuple('0123456789'),
        list('abcdefghij'),
        np.array(list('abcdefghij'), dtype=object),
        np.array(list('abcdefghij'), dtype='<U1')])
    def test_constructor_string(self, constructor, breaks):
        # GH 19016
        msg = ('category, object, and string subtypes are not supported '
               'for IntervalIndex')
        with pytest.raises(TypeError, match=msg):
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize('cat_constructor', [
        Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
        # GH 21243/21253
        if isinstance(constructor, partial) and constructor.func is Index:
            # Index is defined to create CategoricalIndex from categorical data
            pytest.skip()

        breaks = np.arange(10, dtype='int64')
        expected = IntervalIndex.from_breaks(breaks)

        cat_breaks = cat_constructor(breaks)
        result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
        result = constructor(**result_kwargs)
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor):
        # filler input data to be used when supplying invalid kwargs
        filler = self.get_kwargs_from_breaks(range(10))

        # invalid closed
        msg = "invalid option for 'closed': invalid"
        with pytest.raises(ValueError, match=msg):
            constructor(closed='invalid', **filler)

        # unsupported dtype
        msg = 'dtype must be an IntervalDtype, got int64'
        with pytest.raises(TypeError, match=msg):
            constructor(dtype='int64', **filler)

        # invalid dtype
        msg = "data type 'invalid' not understood"
        with pytest.raises(TypeError, match=msg):
            constructor(dtype='invalid', **filler)

        # no point in nesting periods in an IntervalIndex
        periods = period_range('2000-01-01', periods=10)
        periods_kwargs = self.get_kwargs_from_breaks(periods)
        msg = 'Period dtypes are not supported, use a PeriodIndex instead'
        with pytest.raises(ValueError, match=msg):
            constructor(**periods_kwargs)

        # decreasing values
        decreasing_kwargs = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = 'left side of interval must be <= right side'
        with pytest.raises(ValueError, match=msg):
            constructor(**decreasing_kwargs)


class TestFromArrays(Base):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {'left': breaks[:-1], 'right': breaks[1:]}

    def test_constructor_errors(self):
        # GH 19016: categorical data
        data = Categorical(list('01234abcde'), ordered=True)
        msg = ('category, object, and string subtypes are not supported '
               'for IntervalIndex')
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])

        # unequal length
        left = [0, 1, 2]
        right = [2, 3]
        msg = 'left and right must have the same length'
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize('left_subtype, right_subtype', [
        (np.int64, np.float64), (np.float64, np.int64)])
    def test_mixed_float_int(self, left_subtype, right_subtype):
        """mixed int/float left/right results in float for both sides"""
        left = np.arange(9, dtype=left_subtype)
        right = np.arange(1, 10, dtype=right_subtype)
        result = IntervalIndex.from_arrays(left, right)

        expected_left = Float64Index(left)
        expected_right = Float64Index(right)
        expected_subtype = np.float64

        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        assert result.dtype.subtype == expected_subtype


class TestFromBreaks(Base):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_breaks

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_breaks
        """
        return {'breaks': breaks}

    def test_constructor_errors(self):
        # GH 19016: categorical data
        data = Categorical(list('01234abcde'), ordered=True)
        msg = ('category, object, and string subtypes are not supported '
               'for IntervalIndex')
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    def test_length_one(self):
        """breaks of length one produce an empty IntervalIndex"""
        breaks = [0]
        result = IntervalIndex.from_breaks(breaks)
        expected = IntervalIndex.from_breaks([])
        tm.assert_index_equal(result, expected)


class TestFromTuples(Base):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        if len(breaks) == 0:
            return {'data': breaks}

        tuples = lzip(breaks[:-1], breaks[1:])
        if isinstance(breaks, (list, tuple)):
            return {'data': tuples}
        elif is_categorical_dtype(breaks):
            return {'data': breaks._constructor(tuples)}
        return {'data': com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self):
        # non-tuple
        tuples = [(0, 1), 2, (3, 4)]
        msg = 'IntervalIndex.from_tuples received an invalid item, 2'
        with pytest.raises(TypeError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        # too few/many items
        tuples = [(0, 1), (2,), (3, 4)]
        msg = 'IntervalIndex.from_tuples requires tuples of length 2, got {t}'
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self):
        # tuple (NA, NA) evaluates the same as NA as an elemenent
        na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
        idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
        idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        tm.assert_index_equal(idx_na_tuple, idx_na_element)


class TestClassConstructors(Base):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture(params=[IntervalIndex, partial(Index, dtype='interval')],
                    ids=['IntervalIndex', 'Index'])
    def constructor(self, request):
        return request.param

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by the IntervalIndex/Index constructors
        """
        if len(breaks) == 0:
            return {'data': breaks}

        ivs = [Interval(l, r, closed) if notna(l) else l
               for l, r in zip(breaks[:-1], breaks[1:])]

        if isinstance(breaks, list):
            return {'data': ivs}
        elif is_categorical_dtype(breaks):
            return {'data': breaks._constructor(ivs)}
        return {'data': np.array(ivs, dtype=object)}

    def test_generic_errors(self, constructor):
        """
        override the base class implementation since errors are handled
        differently; checks unnecessary since caught at the Interval level
        """
        pass

    def test_constructor_errors(self, constructor):
        # mismatched closed within intervals with no constructor override
        ivs = [Interval(0, 1, closed='right'), Interval(2, 3, closed='left')]
        msg = 'intervals must all be closed on the same side'
        with pytest.raises(ValueError, match=msg):
            constructor(ivs)

        # scalar
        msg = (r'IntervalIndex\(...\) must be called with a collection of '
               'some kind, 5 was passed')
        with pytest.raises(TypeError, match=msg):
            constructor(5)

        # not an interval
        msg = ("type <(class|type) 'numpy.int64'> with value 0 "
               "is not an interval")
        with pytest.raises(TypeError, match=msg):
            constructor([0, 1])

    @pytest.mark.parametrize('data, closed', [
        ([], 'both'),
        ([np.nan, np.nan], 'neither'),
        ([Interval(0, 3, closed='neither'),
          Interval(2, 5, closed='neither')], 'left'),
        ([Interval(0, 3, closed='left'),
          Interval(2, 5, closed='right')], 'neither'),
        (IntervalIndex.from_breaks(range(5), closed='both'), 'right')])
    def test_override_inferred_closed(self, constructor, data, closed):
        # GH 19370
        if isinstance(data, IntervalIndex):
            tuples = data.to_tuples()
        else:
            tuples = [(iv.left, iv.right) if notna(iv) else iv for iv in data]
        expected = IntervalIndex.from_tuples(tuples, closed=closed)
        result = constructor(data, closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('values_constructor', [
        list, np.array, IntervalIndex, IntervalArray])
    def test_index_object_dtype(self, values_constructor):
        # Index(intervals, dtype=object) is an Index (not an IntervalIndex)
        intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
        values = values_constructor(intervals)
        result = Index(values, dtype=object)

        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))


class TestFromIntervals(TestClassConstructors):
    """
    Tests for IntervalIndex.from_intervals, which is deprecated in favor of the
    IntervalIndex constructor.  Same tests as the IntervalIndex constructor,
    plus deprecation test.  Should only need to delete this class when removed.
    """

    @pytest.fixture
    def constructor(self):
        def from_intervals_ignore_warnings(*args, **kwargs):
            with tm.assert_produces_warning(FutureWarning,
                                            check_stacklevel=False):
                return IntervalIndex.from_intervals(*args, **kwargs)
        return from_intervals_ignore_warnings

    def test_deprecated(self):
        ivs = [Interval(0, 1), Interval(1, 2)]
        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            IntervalIndex.from_intervals(ivs)

    @pytest.mark.skip(reason='parent class test that is not applicable')
    def test_index_object_dtype(self):
        pass
