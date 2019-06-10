import numpy as np
import pytest

from pandas import DataFrame, Series, SparseDataFrame, bdate_range
from pandas.core import nanops
from pandas.core.sparse.api import SparseDtype
from pandas.util import testing as tm


@pytest.fixture
def dates():
    return bdate_range('1/1/2011', periods=10)


@pytest.fixture
def empty():
    return SparseDataFrame()


@pytest.fixture
def frame(dates):
    data = {'A': [np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6],
            'B': [0, 1, 2, np.nan, np.nan, np.nan, 3, 4, 5, 6],
            'C': np.arange(10, dtype=np.float64),
            'D': [0, 1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan]}

    return SparseDataFrame(data, index=dates)


@pytest.fixture
def fill_frame(frame):
    values = frame.values.copy()
    values[np.isnan(values)] = 2

    return SparseDataFrame(values, columns=['A', 'B', 'C', 'D'],
                           default_fill_value=2,
                           index=frame.index)


def test_apply(frame):
    applied = frame.apply(np.sqrt)
    assert isinstance(applied, SparseDataFrame)
    tm.assert_almost_equal(applied.values, np.sqrt(frame.values))

    # agg / broadcast
    with tm.assert_produces_warning(FutureWarning):
        broadcasted = frame.apply(np.sum, broadcast=True)
    assert isinstance(broadcasted, SparseDataFrame)

    with tm.assert_produces_warning(FutureWarning):
        exp = frame.to_dense().apply(np.sum, broadcast=True)
    tm.assert_frame_equal(broadcasted.to_dense(), exp)

    applied = frame.apply(np.sum)
    tm.assert_series_equal(applied,
                           frame.to_dense().apply(nanops.nansum).to_sparse())


def test_apply_fill(fill_frame):
    applied = fill_frame.apply(np.sqrt)
    assert applied['A'].fill_value == np.sqrt(2)


def test_apply_empty(empty):
    assert empty.apply(np.sqrt) is empty


def test_apply_nonuq():
    orig = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     index=['a', 'a', 'c'])
    sparse = orig.to_sparse()
    res = sparse.apply(lambda s: s[0], axis=1)
    exp = orig.apply(lambda s: s[0], axis=1)

    # dtype must be kept
    assert res.dtype == SparseDtype(np.int64)

    # ToDo: apply must return subclassed dtype
    assert isinstance(res, Series)
    tm.assert_series_equal(res.to_dense(), exp)

    # df.T breaks
    sparse = orig.T.to_sparse()
    res = sparse.apply(lambda s: s[0], axis=0)  # noqa
    exp = orig.T.apply(lambda s: s[0], axis=0)

    # TODO: no non-unique columns supported in sparse yet
    # tm.assert_series_equal(res.to_dense(), exp)


def test_applymap(frame):
    # just test that it works
    result = frame.applymap(lambda x: x * 2)
    assert isinstance(result, SparseDataFrame)


def test_apply_keep_sparse_dtype():
    # GH 23744
    sdf = SparseDataFrame(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]]),
                          columns=['b', 'a', 'c'], default_fill_value=1)
    df = DataFrame(sdf)

    expected = sdf.apply(np.exp)
    result = df.apply(np.exp)
    tm.assert_frame_equal(expected, result)
