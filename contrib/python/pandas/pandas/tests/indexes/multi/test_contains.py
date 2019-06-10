# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pandas.compat import PYPY

import pandas as pd
from pandas import MultiIndex
import pandas.util.testing as tm


def test_contains_top_level():
    midx = MultiIndex.from_product([['A', 'B'], [1, 2]])
    assert 'A' in midx
    assert 'A' not in midx._engine


def test_contains_with_nat():
    # MI with a NaT
    mi = MultiIndex(levels=[['C'],
                            pd.date_range('2012-01-01', periods=5)],
                    codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
                    names=[None, 'B'])
    assert ('C', pd.Timestamp('2012-01-01')) in mi
    for val in mi.values:
        assert val in mi


def test_contains(idx):
    assert ('foo', 'two') in idx
    assert ('bar', 'two') not in idx
    assert None not in idx


@pytest.mark.skipif(not PYPY, reason="tuples cmp recursively on PyPy")
def test_isin_nan_pypy():
    idx = MultiIndex.from_arrays([['foo', 'bar'], [1.0, np.nan]])
    tm.assert_numpy_array_equal(idx.isin([('bar', np.nan)]),
                                np.array([False, True]))
    tm.assert_numpy_array_equal(idx.isin([('bar', float('nan'))]),
                                np.array([False, True]))


def test_isin():
    values = [('foo', 2), ('bar', 3), ('quux', 4)]

    idx = MultiIndex.from_arrays([
        ['qux', 'baz', 'foo', 'bar'],
        np.arange(4)
    ])
    result = idx.isin(values)
    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(result, expected)

    # empty, return dtype bool
    idx = MultiIndex.from_arrays([[], []])
    result = idx.isin(values)
    assert len(result) == 0
    assert result.dtype == np.bool_


@pytest.mark.skipif(PYPY, reason="tuples cmp recursively on PyPy")
def test_isin_nan_not_pypy():
    idx = MultiIndex.from_arrays([['foo', 'bar'], [1.0, np.nan]])
    tm.assert_numpy_array_equal(idx.isin([('bar', np.nan)]),
                                np.array([False, False]))
    tm.assert_numpy_array_equal(idx.isin([('bar', float('nan'))]),
                                np.array([False, False]))


def test_isin_level_kwarg():
    idx = MultiIndex.from_arrays([['qux', 'baz', 'foo', 'bar'], np.arange(
        4)])

    vals_0 = ['foo', 'bar', 'quux']
    vals_1 = [2, 3, 10]

    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=0))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=-2))

    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=1))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=-1))

    pytest.raises(IndexError, idx.isin, vals_0, level=5)
    pytest.raises(IndexError, idx.isin, vals_0, level=-5)

    pytest.raises(KeyError, idx.isin, vals_0, level=1.0)
    pytest.raises(KeyError, idx.isin, vals_1, level=-1.0)
    pytest.raises(KeyError, idx.isin, vals_1, level='A')

    idx.names = ['A', 'B']
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level='A'))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level='B'))

    pytest.raises(KeyError, idx.isin, vals_1, level='C')
