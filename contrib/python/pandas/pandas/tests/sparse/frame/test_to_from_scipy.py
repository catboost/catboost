from distutils.version import LooseVersion

import numpy as np
import pytest

from pandas.core.dtypes.common import is_bool_dtype

import pandas as pd
from pandas import SparseDataFrame, SparseSeries
from pandas.core.sparse.api import SparseDtype
from pandas.util import testing as tm

scipy = pytest.importorskip('scipy')
ignore_matrix_warning = pytest.mark.filterwarnings(
    "ignore:the matrix subclass:PendingDeprecationWarning"
)


@pytest.mark.parametrize('index', [None, list('abc')])  # noqa: F811
@pytest.mark.parametrize('columns', [None, list('def')])
@pytest.mark.parametrize('fill_value', [None, 0, np.nan])
@pytest.mark.parametrize('dtype', [bool, int, float, np.uint16])
@ignore_matrix_warning
def test_from_to_scipy(spmatrix, index, columns, fill_value, dtype):
    # GH 4343
    # Make one ndarray and from it one sparse matrix, both to be used for
    # constructing frames and comparing results
    arr = np.eye(3, dtype=dtype)
    # GH 16179
    arr[0, 1] = dtype(2)
    try:
        spm = spmatrix(arr)
        assert spm.dtype == arr.dtype
    except (TypeError, AssertionError):
        # If conversion to sparse fails for this spmatrix type and arr.dtype,
        # then the combination is not currently supported in NumPy, so we
        # can just skip testing it thoroughly
        return

    sdf = SparseDataFrame(spm, index=index, columns=columns,
                          default_fill_value=fill_value)

    # Expected result construction is kind of tricky for all
    # dtype-fill_value combinations; easiest to cast to something generic
    # and except later on
    rarr = arr.astype(object)
    rarr[arr == 0] = np.nan
    expected = SparseDataFrame(rarr, index=index, columns=columns).fillna(
        fill_value if fill_value is not None else np.nan)

    # Assert frame is as expected
    sdf_obj = sdf.astype(object)
    tm.assert_sp_frame_equal(sdf_obj, expected)
    tm.assert_frame_equal(sdf_obj.to_dense(), expected.to_dense())

    # Assert spmatrices equal
    assert dict(sdf.to_coo().todok()) == dict(spm.todok())

    # Ensure dtype is preserved if possible
    # XXX: verify this
    res_dtype = bool if is_bool_dtype(dtype) else dtype
    tm.assert_contains_all(sdf.dtypes.apply(lambda dtype: dtype.subtype),
                           {np.dtype(res_dtype)})
    assert sdf.to_coo().dtype == res_dtype

    # However, adding a str column results in an upcast to object
    sdf['strings'] = np.arange(len(sdf)).astype(str)
    assert sdf.to_coo().dtype == np.object_


@pytest.mark.parametrize('fill_value', [None, 0, np.nan])  # noqa: F811
@ignore_matrix_warning
@pytest.mark.filterwarnings("ignore:object dtype is not supp:UserWarning")
def test_from_to_scipy_object(spmatrix, fill_value):
    # GH 4343
    dtype = object
    columns = list('cd')
    index = list('ab')

    if (spmatrix is scipy.sparse.dok_matrix and LooseVersion(
            scipy.__version__) >= LooseVersion('0.19.0')):
        pytest.skip("dok_matrix from object does not work in SciPy >= 0.19")

    # Make one ndarray and from it one sparse matrix, both to be used for
    # constructing frames and comparing results
    arr = np.eye(2, dtype=dtype)
    try:
        spm = spmatrix(arr)
        assert spm.dtype == arr.dtype
    except (TypeError, AssertionError):
        # If conversion to sparse fails for this spmatrix type and arr.dtype,
        # then the combination is not currently supported in NumPy, so we
        # can just skip testing it thoroughly
        return

    sdf = SparseDataFrame(spm, index=index, columns=columns,
                          default_fill_value=fill_value)

    # Expected result construction is kind of tricky for all
    # dtype-fill_value combinations; easiest to cast to something generic
    # and except later on
    rarr = arr.astype(object)
    rarr[arr == 0] = np.nan
    expected = SparseDataFrame(rarr, index=index, columns=columns).fillna(
        fill_value if fill_value is not None else np.nan)

    # Assert frame is as expected
    sdf_obj = sdf.astype(SparseDtype(object, fill_value))
    tm.assert_sp_frame_equal(sdf_obj, expected)
    tm.assert_frame_equal(sdf_obj.to_dense(), expected.to_dense())

    # Assert spmatrices equal
    assert dict(sdf.to_coo().todok()) == dict(spm.todok())

    # Ensure dtype is preserved if possible
    res_dtype = object
    tm.assert_contains_all(sdf.dtypes.apply(lambda dtype: dtype.subtype),
                           {np.dtype(res_dtype)})
    assert sdf.to_coo().dtype == res_dtype


@ignore_matrix_warning
def test_from_scipy_correct_ordering(spmatrix):
    # GH 16179
    arr = np.arange(1, 5).reshape(2, 2)
    try:
        spm = spmatrix(arr)
        assert spm.dtype == arr.dtype
    except (TypeError, AssertionError):
        # If conversion to sparse fails for this spmatrix type and arr.dtype,
        # then the combination is not currently supported in NumPy, so we
        # can just skip testing it thoroughly
        return

    sdf = SparseDataFrame(spm)
    expected = SparseDataFrame(arr)
    tm.assert_sp_frame_equal(sdf, expected)
    tm.assert_frame_equal(sdf.to_dense(), expected.to_dense())


@ignore_matrix_warning
def test_from_scipy_fillna(spmatrix):
    # GH 16112
    arr = np.eye(3)
    arr[1:, 0] = np.nan

    try:
        spm = spmatrix(arr)
        assert spm.dtype == arr.dtype
    except (TypeError, AssertionError):
        # If conversion to sparse fails for this spmatrix type and arr.dtype,
        # then the combination is not currently supported in NumPy, so we
        # can just skip testing it thoroughly
        return

    sdf = SparseDataFrame(spm).fillna(-1.0)

    # Returning frame should fill all nan values with -1.0
    expected = SparseDataFrame({
        0: SparseSeries([1., -1, -1]),
        1: SparseSeries([np.nan, 1, np.nan]),
        2: SparseSeries([np.nan, np.nan, 1]),
    }, default_fill_value=-1)

    # fill_value is expected to be what .fillna() above was called with
    # We don't use -1 as initial fill_value in expected SparseSeries
    # construction because this way we obtain "compressed" SparseArrays,
    # avoiding having to construct them ourselves
    for col in expected:
        expected[col].fill_value = -1

    tm.assert_sp_frame_equal(sdf, expected)


def test_index_names_multiple_nones():
    # https://github.com/pandas-dev/pandas/pull/24092
    sparse = pytest.importorskip("scipy.sparse")

    s = (pd.Series(1, index=pd.MultiIndex.from_product([['A', 'B'], [0, 1]]))
           .to_sparse())
    result, _, _ = s.to_coo()
    assert isinstance(result, sparse.coo_matrix)
    result = result.toarray()
    expected = np.ones((2, 2), dtype="int64")
    tm.assert_numpy_array_equal(result, expected)
