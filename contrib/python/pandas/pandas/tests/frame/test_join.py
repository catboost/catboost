# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pandas import DataFrame, Index, period_range
from pandas.tests.frame.common import TestData
import pandas.util.testing as tm


@pytest.fixture
def frame_with_period_index():
    return DataFrame(
        data=np.arange(20).reshape(4, 5),
        columns=list('abcde'),
        index=period_range(start='2000', freq='A', periods=4))


@pytest.fixture
def frame():
    return TestData().frame


@pytest.fixture
def left():
    return DataFrame({'a': [20, 10, 0]}, index=[2, 1, 0])


@pytest.fixture
def right():
    return DataFrame({'b': [300, 100, 200]}, index=[3, 1, 2])


@pytest.mark.parametrize(
    "how, sort, expected",
    [('inner', False, DataFrame({'a': [20, 10],
                                 'b': [200, 100]},
                                index=[2, 1])),
     ('inner', True, DataFrame({'a': [10, 20],
                                'b': [100, 200]},
                               index=[1, 2])),
     ('left', False, DataFrame({'a': [20, 10, 0],
                                'b': [200, 100, np.nan]},
                               index=[2, 1, 0])),
     ('left', True, DataFrame({'a': [0, 10, 20],
                               'b': [np.nan, 100, 200]},
                              index=[0, 1, 2])),
     ('right', False, DataFrame({'a': [np.nan, 10, 20],
                                 'b': [300, 100, 200]},
                                index=[3, 1, 2])),
     ('right', True, DataFrame({'a': [10, 20, np.nan],
                                'b': [100, 200, 300]},
                               index=[1, 2, 3])),
     ('outer', False, DataFrame({'a': [0, 10, 20, np.nan],
                                 'b': [np.nan, 100, 200, 300]},
                                index=[0, 1, 2, 3])),
     ('outer', True, DataFrame({'a': [0, 10, 20, np.nan],
                                'b': [np.nan, 100, 200, 300]},
                               index=[0, 1, 2, 3]))])
def test_join(left, right, how, sort, expected):

    result = left.join(right, how=how, sort=sort)
    tm.assert_frame_equal(result, expected)


def test_join_index(frame):
    # left / right

    f = frame.loc[frame.index[:10], ['A', 'B']]
    f2 = frame.loc[frame.index[5:], ['C', 'D']].iloc[::-1]

    joined = f.join(f2)
    tm.assert_index_equal(f.index, joined.index)
    expected_columns = Index(['A', 'B', 'C', 'D'])
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how='left')
    tm.assert_index_equal(joined.index, f.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how='right')
    tm.assert_index_equal(joined.index, f2.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    # inner

    joined = f.join(f2, how='inner')
    tm.assert_index_equal(joined.index, f.index[5:10])
    tm.assert_index_equal(joined.columns, expected_columns)

    # outer

    joined = f.join(f2, how='outer')
    tm.assert_index_equal(joined.index, frame.index.sort_values())
    tm.assert_index_equal(joined.columns, expected_columns)

    with pytest.raises(ValueError, match='join method'):
        f.join(f2, how='foo')

    # corner case - overlapping columns
    msg = 'columns overlap but no suffix'
    for how in ('outer', 'left', 'inner'):
        with pytest.raises(ValueError, match=msg):
            frame.join(frame, how=how)


def test_join_index_more(frame):
    af = frame.loc[:, ['A', 'B']]
    bf = frame.loc[::2, ['C', 'D']]

    expected = af.copy()
    expected['C'] = frame['C'][::2]
    expected['D'] = frame['D'][::2]

    result = af.join(bf)
    tm.assert_frame_equal(result, expected)

    result = af.join(bf, how='right')
    tm.assert_frame_equal(result, expected[::2])

    result = bf.join(af, how='right')
    tm.assert_frame_equal(result, expected.loc[:, result.columns])


def test_join_index_series(frame):
    df = frame.copy()
    s = df.pop(frame.columns[-1])
    joined = df.join(s)

    # TODO should this check_names ?
    tm.assert_frame_equal(joined, frame, check_names=False)

    s.name = None
    with pytest.raises(ValueError, match='must have a name'):
        df.join(s)


def test_join_overlap(frame):
    df1 = frame.loc[:, ['A', 'B', 'C']]
    df2 = frame.loc[:, ['B', 'C', 'D']]

    joined = df1.join(df2, lsuffix='_df1', rsuffix='_df2')
    df1_suf = df1.loc[:, ['B', 'C']].add_suffix('_df1')
    df2_suf = df2.loc[:, ['B', 'C']].add_suffix('_df2')

    no_overlap = frame.loc[:, ['A', 'D']]
    expected = df1_suf.join(df2_suf).join(no_overlap)

    # column order not necessarily sorted
    tm.assert_frame_equal(joined, expected.loc[:, joined.columns])


def test_join_period_index(frame_with_period_index):
    other = frame_with_period_index.rename(
        columns=lambda x: '{key}{key}'.format(key=x))

    joined_values = np.concatenate(
        [frame_with_period_index.values] * 2, axis=1)

    joined_cols = frame_with_period_index.columns.append(other.columns)

    joined = frame_with_period_index.join(other)
    expected = DataFrame(
        data=joined_values,
        columns=joined_cols,
        index=frame_with_period_index.index)

    tm.assert_frame_equal(joined, expected)


def test_join_left_sequence_non_unique_index():
    # https://github.com/pandas-dev/pandas/issues/19607
    df1 = DataFrame({'a': [0, 10, 20]}, index=[1, 2, 3])
    df2 = DataFrame({'b': [100, 200, 300]}, index=[4, 3, 2])
    df3 = DataFrame({'c': [400, 500, 600]}, index=[2, 2, 4])

    joined = df1.join([df2, df3], how='left')

    expected = DataFrame({
        'a': [0, 10, 10, 20],
        'b': [np.nan, 300, 300, 200],
        'c': [np.nan, 400, 500, np.nan]
    }, index=[1, 2, 2, 3])

    tm.assert_frame_equal(joined, expected)
