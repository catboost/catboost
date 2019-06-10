# -*- coding: utf-8 -*-


import numpy as np
import pytest

from pandas.compat import lrange
from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import Index, MultiIndex
import pandas.util.testing as tm


def test_drop(idx):
    dropped = idx.drop([('foo', 'two'), ('qux', 'one')])

    index = MultiIndex.from_tuples([('foo', 'two'), ('qux', 'one')])
    dropped2 = idx.drop(index)

    expected = idx[[0, 2, 3, 5]]
    tm.assert_index_equal(dropped, expected)
    tm.assert_index_equal(dropped2, expected)

    dropped = idx.drop(['bar'])
    expected = idx[[0, 1, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop('foo')
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    index = MultiIndex.from_tuples([('bar', 'two')])
    pytest.raises(KeyError, idx.drop, [('bar', 'two')])
    pytest.raises(KeyError, idx.drop, index)
    pytest.raises(KeyError, idx.drop, ['foo', 'two'])

    # partially correct argument
    mixed_index = MultiIndex.from_tuples([('qux', 'one'), ('bar', 'two')])
    pytest.raises(KeyError, idx.drop, mixed_index)

    # error='ignore'
    dropped = idx.drop(index, errors='ignore')
    expected = idx[[0, 1, 2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop(mixed_index, errors='ignore')
    expected = idx[[0, 1, 2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    dropped = idx.drop(['foo', 'two'], errors='ignore')
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # mixed partial / full drop
    dropped = idx.drop(['foo', ('qux', 'one')])
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    # mixed partial / full drop / error='ignore'
    mixed_index = ['foo', ('qux', 'one'), 'two']
    pytest.raises(KeyError, idx.drop, mixed_index)
    dropped = idx.drop(mixed_index, errors='ignore')
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)


def test_droplevel_with_names(idx):
    index = idx[idx.get_loc('foo')]
    dropped = index.droplevel(0)
    assert dropped.name == 'second'

    index = MultiIndex(
        levels=[Index(lrange(4)), Index(lrange(4)), Index(lrange(4))],
        codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array(
            [0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])],
        names=['one', 'two', 'three'])
    dropped = index.droplevel(0)
    assert dropped.names == ('two', 'three')

    dropped = index.droplevel('two')
    expected = index.droplevel(1)
    assert dropped.equals(expected)


def test_droplevel_list():
    index = MultiIndex(
        levels=[Index(lrange(4)), Index(lrange(4)), Index(lrange(4))],
        codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array(
            [0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])],
        names=['one', 'two', 'three'])

    dropped = index[:2].droplevel(['three', 'one'])
    expected = index[:2].droplevel(2).droplevel(0)
    assert dropped.equals(expected)

    dropped = index[:2].droplevel([])
    expected = index[:2]
    assert dropped.equals(expected)

    with pytest.raises(ValueError):
        index[:2].droplevel(['one', 'two', 'three'])

    with pytest.raises(KeyError):
        index[:2].droplevel(['one', 'four'])


def test_drop_not_lexsorted():
    # GH 12078

    # define the lexsorted version of the multi-index
    tuples = [('a', ''), ('b1', 'c1'), ('b2', 'c2')]
    lexsorted_mi = MultiIndex.from_tuples(tuples, names=['b', 'c'])
    assert lexsorted_mi.is_lexsorted()

    # and the not-lexsorted version
    df = pd.DataFrame(columns=['a', 'b', 'c', 'd'],
                      data=[[1, 'b1', 'c1', 3], [1, 'b2', 'c2', 4]])
    df = df.pivot_table(index='a', columns=['b', 'c'], values='d')
    df = df.reset_index()
    not_lexsorted_mi = df.columns
    assert not not_lexsorted_mi.is_lexsorted()

    # compare the results
    tm.assert_index_equal(lexsorted_mi, not_lexsorted_mi)
    with tm.assert_produces_warning(PerformanceWarning):
        tm.assert_index_equal(lexsorted_mi.drop('a'),
                              not_lexsorted_mi.drop('a'))
