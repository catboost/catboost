import pytest
import numpy as np
import scipy.sparse as sp

from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.random import RandomState

from sklearn.datasets import make_classification
from sklearn.utils.sparsefuncs import (mean_variance_axis,
                                       incr_mean_variance_axis,
                                       inplace_column_scale,
                                       inplace_row_scale,
                                       inplace_swap_row, inplace_swap_column,
                                       min_max_axis,
                                       count_nonzero, csc_median_axis_0)
from sklearn.utils.sparsefuncs_fast import (assign_rows_csr,
                                            inplace_csr_row_normalize_l1,
                                            inplace_csr_row_normalize_l2)
from sklearn.utils._testing import assert_raises
from sklearn.utils._testing import assert_allclose


def test_mean_variance_axis0():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = sp.lil_matrix(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    assert_raises(TypeError, mean_variance_axis, X_lil, axis=0)

    X_csr = sp.csr_matrix(X_lil)
    X_csc = sp.csc_matrix(X_lil)

    expected_dtypes = [(np.float32, np.float32),
                       (np.float64, np.float64),
                       (np.int32, np.float64),
                       (np.int64, np.float64)]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


def test_mean_variance_axis1():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_lil = sp.lil_matrix(X)
    X_lil[1, 0] = 0
    X[1, 0] = 0

    assert_raises(TypeError, mean_variance_axis, X_lil, axis=1)

    X_csr = sp.csr_matrix(X_lil)
    X_csc = sp.csc_matrix(X_lil)

    expected_dtypes = [(np.float32, np.float32),
                       (np.float64, np.float64),
                       (np.int32, np.float64),
                       (np.int64, np.float64)]

    for input_dtype, output_dtype in expected_dtypes:
        X_test = X.astype(input_dtype)
        for X_sparse in (X_csr, X_csc):
            X_sparse = X_sparse.astype(input_dtype)
            X_means, X_vars = mean_variance_axis(X_sparse, axis=0)
            assert X_means.dtype == output_dtype
            assert X_vars.dtype == output_dtype
            assert_array_almost_equal(X_means, np.mean(X_test, axis=0))
            assert_array_almost_equal(X_vars, np.var(X_test, axis=0))


def test_incr_mean_variance_axis():
    for axis in [0, 1]:
        rng = np.random.RandomState(0)
        n_features = 50
        n_samples = 10
        data_chunks = [rng.randint(0, 2, size=n_features)
                       for i in range(n_samples)]

        # default params for incr_mean_variance
        last_mean = np.zeros(n_features)
        last_var = np.zeros_like(last_mean)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        # Test errors
        X = np.array(data_chunks[0])
        X = np.atleast_2d(X)
        X_lil = sp.lil_matrix(X)
        X_csr = sp.csr_matrix(X_lil)
        assert_raises(TypeError, incr_mean_variance_axis, axis,
                      last_mean, last_var, last_n)
        assert_raises(TypeError, incr_mean_variance_axis, axis,
                      last_mean, last_var, last_n)
        assert_raises(TypeError, incr_mean_variance_axis, X_lil, axis,
                      last_mean, last_var, last_n)

        # Test _incr_mean_and_var with a 1 row input
        X_means, X_vars = mean_variance_axis(X_csr, axis)
        X_means_incr, X_vars_incr, n_incr = \
            incr_mean_variance_axis(X_csr, axis, last_mean, last_var, last_n)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        # X.shape[axis] picks # samples
        assert_array_equal(X.shape[axis], n_incr)

        X_csc = sp.csc_matrix(X_lil)
        X_means, X_vars = mean_variance_axis(X_csc, axis)
        assert_array_almost_equal(X_means, X_means_incr)
        assert_array_almost_equal(X_vars, X_vars_incr)
        assert_array_equal(X.shape[axis], n_incr)

        # Test _incremental_mean_and_var with whole data
        X = np.vstack(data_chunks)
        X_lil = sp.lil_matrix(X)
        X_csr = sp.csr_matrix(X_lil)
        X_csc = sp.csc_matrix(X_lil)

        expected_dtypes = [(np.float32, np.float32),
                           (np.float64, np.float64),
                           (np.int32, np.float64),
                           (np.int64, np.float64)]

        for input_dtype, output_dtype in expected_dtypes:
            for X_sparse in (X_csr, X_csc):
                X_sparse = X_sparse.astype(input_dtype)
                last_mean = last_mean.astype(output_dtype)
                last_var = last_var.astype(output_dtype)
                X_means, X_vars = mean_variance_axis(X_sparse, axis)
                X_means_incr, X_vars_incr, n_incr = \
                    incr_mean_variance_axis(X_sparse, axis, last_mean,
                                            last_var, last_n)
                assert X_means_incr.dtype == output_dtype
                assert X_vars_incr.dtype == output_dtype
                assert_array_almost_equal(X_means, X_means_incr)
                assert_array_almost_equal(X_vars, X_vars_incr)
                assert_array_equal(X.shape[axis], n_incr)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sparse_constructor", [sp.csc_matrix, sp.csr_matrix])
def test_incr_mean_variance_axis_ignore_nan(axis, sparse_constructor):
    old_means = np.array([535., 535., 535., 535.])
    old_variances = np.array([4225., 4225., 4225., 4225.])
    old_sample_count = np.array([2, 2, 2, 2], dtype=np.int64)

    X = sparse_constructor(
        np.array([[170, 170, 170, 170],
                  [430, 430, 430, 430],
                  [300, 300, 300, 300]]))

    X_nan = sparse_constructor(
        np.array([[170, np.nan, 170, 170],
                  [np.nan, 170, 430, 430],
                  [430, 430, np.nan, 300],
                  [300, 300, 300, np.nan]]))

    # we avoid creating specific data for axis 0 and 1: translating the data is
    # enough.
    if axis:
        X = X.T
        X_nan = X_nan.T

    # take a copy of the old statistics since they are modified in place.
    X_means, X_vars, X_sample_count = incr_mean_variance_axis(
        X, axis, old_means.copy(), old_variances.copy(),
        old_sample_count.copy())
    X_nan_means, X_nan_vars, X_nan_sample_count = incr_mean_variance_axis(
        X_nan, axis, old_means.copy(), old_variances.copy(),
        old_sample_count.copy())

    assert_allclose(X_nan_means, X_means)
    assert_allclose(X_nan_vars, X_vars)
    assert_allclose(X_nan_sample_count, X_sample_count)


def test_mean_variance_illegal_axis():
    X, _ = make_classification(5, 4, random_state=0)
    # Sparsify the array a little bit
    X[0, 0] = 0
    X[2, 1] = 0
    X[4, 3] = 0
    X_csr = sp.csr_matrix(X)
    assert_raises(ValueError, mean_variance_axis, X_csr, axis=-3)
    assert_raises(ValueError, mean_variance_axis, X_csr, axis=2)
    assert_raises(ValueError, mean_variance_axis, X_csr, axis=-1)

    assert_raises(ValueError, incr_mean_variance_axis, X_csr, axis=-3,
                  last_mean=None, last_var=None, last_n=None)
    assert_raises(ValueError, incr_mean_variance_axis, X_csr, axis=2,
                  last_mean=None, last_var=None, last_n=None)
    assert_raises(ValueError, incr_mean_variance_axis, X_csr, axis=-1,
                  last_mean=None, last_var=None, last_n=None)


def test_densify_rows():
    for dtype in (np.float32, np.float64):
        X = sp.csr_matrix([[0, 3, 0],
                        [2, 4, 0],
                        [0, 0, 0],
                        [9, 8, 7],
                        [4, 0, 5]], dtype=dtype)
        X_rows = np.array([0, 2, 3], dtype=np.intp)
        out = np.ones((6, X.shape[1]), dtype=dtype)
        out_rows = np.array([1, 3, 4], dtype=np.intp)

        expect = np.ones_like(out)
        expect[out_rows] = X[X_rows, :].toarray()

        assign_rows_csr(X, X_rows, out_rows, out)
        assert_array_equal(out, expect)


def test_inplace_column_scale():
    rng = np.random.RandomState(0)
    X = sp.rand(100, 200, 0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(200)
    XA *= scale

    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    assert_raises(TypeError, inplace_column_scale, X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale
    inplace_column_scale(Xc, scale)
    inplace_column_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    assert_raises(TypeError, inplace_column_scale, X.tolil(), scale)


def test_inplace_row_scale():
    rng = np.random.RandomState(0)
    X = sp.rand(100, 200, 0.05)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    scale = rng.rand(100)
    XA *= scale.reshape(-1, 1)

    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    assert_raises(TypeError, inplace_column_scale, X.tolil(), scale)

    X = X.astype(np.float32)
    scale = scale.astype(np.float32)
    Xr = X.tocsr()
    Xc = X.tocsc()
    XA = X.toarray()
    XA *= scale.reshape(-1, 1)
    inplace_row_scale(Xc, scale)
    inplace_row_scale(Xr, scale)
    assert_array_almost_equal(Xr.toarray(), Xc.toarray())
    assert_array_almost_equal(XA, Xc.toarray())
    assert_array_almost_equal(XA, Xr.toarray())
    assert_raises(TypeError, inplace_column_scale, X.tolil(), scale)


def test_inplace_swap_row():
    X = np.array([[0, 3, 0],
                  [2, 4, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float64)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)

    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    assert_raises(TypeError, inplace_swap_row, X_csr.tolil())

    X = np.array([[0, 3, 0],
                  [2, 4, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float32)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[0], X[-1] = swap(X[0], X[-1])
    inplace_swap_row(X_csr, 0, -1)
    inplace_swap_row(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[2], X[3] = swap(X[2], X[3])
    inplace_swap_row(X_csr, 2, 3)
    inplace_swap_row(X_csc, 2, 3)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    assert_raises(TypeError, inplace_swap_row, X_csr.tolil())


def test_inplace_swap_column():
    X = np.array([[0, 3, 0],
                  [2, 4, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float64)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)

    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())

    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    assert_raises(TypeError, inplace_swap_column, X_csr.tolil())

    X = np.array([[0, 3, 0],
                  [2, 4, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float32)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    swap = linalg.get_blas_funcs(('swap',), (X,))
    swap = swap[0]
    X[:, 0], X[:, -1] = swap(X[:, 0], X[:, -1])
    inplace_swap_column(X_csr, 0, -1)
    inplace_swap_column(X_csc, 0, -1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    X[:, 0], X[:, 1] = swap(X[:, 0], X[:, 1])
    inplace_swap_column(X_csr, 0, 1)
    inplace_swap_column(X_csc, 0, 1)
    assert_array_equal(X_csr.toarray(), X_csc.toarray())
    assert_array_equal(X, X_csc.toarray())
    assert_array_equal(X, X_csr.toarray())
    assert_raises(TypeError, inplace_swap_column, X_csr.tolil())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("sparse_format", [sp.csr_matrix, sp.csc_matrix])
@pytest.mark.parametrize(
    "missing_values, min_func, max_func, ignore_nan",
    [(0, np.min, np.max, False),
     (np.nan, np.nanmin, np.nanmax, True)]
)
@pytest.mark.parametrize("large_indices", [True, False])
def test_min_max(dtype, axis, sparse_format, missing_values, min_func,
                 max_func, ignore_nan, large_indices):
    X = np.array([[0, 3, 0],
                  [2, -1, missing_values],
                  [0, 0, 0],
                  [9, missing_values, 7],
                  [4, 0, 5]], dtype=dtype)
    X_sparse = sparse_format(X)
    if large_indices:
        X_sparse.indices = X_sparse.indices.astype('int64')
        X_sparse.indptr = X_sparse.indptr.astype('int64')

    mins_sparse, maxs_sparse = min_max_axis(X_sparse, axis=axis,
                                            ignore_nan=ignore_nan)
    assert_array_equal(mins_sparse, min_func(X, axis=axis))
    assert_array_equal(maxs_sparse, max_func(X, axis=axis))


def test_min_max_axis_errors():
    X = np.array([[0, 3, 0],
                  [2, -1, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float64)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    assert_raises(TypeError, min_max_axis, X_csr.tolil(), axis=0)
    assert_raises(ValueError, min_max_axis, X_csr, axis=2)
    assert_raises(ValueError, min_max_axis, X_csc, axis=-3)


def test_count_nonzero():
    X = np.array([[0, 3, 0],
                  [2, -1, 0],
                  [0, 0, 0],
                  [9, 8, 7],
                  [4, 0, 5]], dtype=np.float64)
    X_csr = sp.csr_matrix(X)
    X_csc = sp.csc_matrix(X)
    X_nonzero = X != 0
    sample_weight = [.5, .2, .3, .1, .1]
    X_nonzero_weighted = X_nonzero * np.array(sample_weight)[:, None]

    for axis in [0, 1, -1, -2, None]:
        assert_array_almost_equal(count_nonzero(X_csr, axis=axis),
                                  X_nonzero.sum(axis=axis))
        assert_array_almost_equal(count_nonzero(X_csr, axis=axis,
                                                sample_weight=sample_weight),
                                  X_nonzero_weighted.sum(axis=axis))

    assert_raises(TypeError, count_nonzero, X_csc)
    assert_raises(ValueError, count_nonzero, X_csr, axis=2)

    assert (count_nonzero(X_csr, axis=0).dtype ==
            count_nonzero(X_csr, axis=1).dtype)
    assert (count_nonzero(X_csr, axis=0, sample_weight=sample_weight).dtype ==
            count_nonzero(X_csr, axis=1, sample_weight=sample_weight).dtype)

    # Check dtypes with large sparse matrices too
    # XXX: test fails on Appveyor (python3.5 32bit)
    try:
        X_csr.indices = X_csr.indices.astype(np.int64)
        X_csr.indptr = X_csr.indptr.astype(np.int64)
        assert (count_nonzero(X_csr, axis=0).dtype ==
                count_nonzero(X_csr, axis=1).dtype)
        assert (count_nonzero(X_csr, axis=0,
                              sample_weight=sample_weight).dtype ==
                count_nonzero(X_csr, axis=1,
                              sample_weight=sample_weight).dtype)
    except TypeError as e:
        if ("according to the rule 'safe'" in e.args[0] and
                np.intp().nbytes < 8):
            pass
        else:
            raise


def test_csc_row_median():
    # Test csc_row_median actually calculates the median.

    # Test that it gives the same output when X is dense.
    rng = np.random.RandomState(0)
    X = rng.rand(100, 50)
    dense_median = np.median(X, axis=0)
    csc = sp.csc_matrix(X)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test that it gives the same output when X is sparse
    X = rng.rand(51, 100)
    X[X < 0.7] = 0.0
    ind = rng.randint(0, 50, 10)
    X[ind] = -X[ind]
    csc = sp.csc_matrix(X)
    dense_median = np.median(X, axis=0)
    sparse_median = csc_median_axis_0(csc)
    assert_array_equal(sparse_median, dense_median)

    # Test for toy data.
    X = [[0, -2], [-1, -1], [1, 0], [2, 1]]
    csc = sp.csc_matrix(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0.5, -0.5]))
    X = [[0, -2], [-1, -5], [1, -3]]
    csc = sp.csc_matrix(X)
    assert_array_equal(csc_median_axis_0(csc), np.array([0., -3]))

    # Test that it raises an Error for non-csc matrices.
    assert_raises(TypeError, csc_median_axis_0, sp.csr_matrix(X))


def test_inplace_normalize():
    ones = np.ones((10, 1))
    rs = RandomState(10)

    for inplace_csr_row_normalize in (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2):
        for dtype in (np.float64, np.float32):
            X = rs.randn(10, 5).astype(dtype)
            X_csr = sp.csr_matrix(X)
            for index_dtype in [np.int32, np.int64]:
                # csr_matrix will use int32 indices by default,
                # up-casting those to int64 when necessary
                if index_dtype is np.int64:
                    X_csr.indptr = X_csr.indptr.astype(index_dtype)
                    X_csr.indices = X_csr.indices.astype(index_dtype)
                assert X_csr.indices.dtype == index_dtype
                assert X_csr.indptr.dtype == index_dtype
                inplace_csr_row_normalize(X_csr)
                assert X_csr.dtype == dtype
                if inplace_csr_row_normalize is inplace_csr_row_normalize_l2:
                    X_csr.data **= 2
                assert_array_almost_equal(np.abs(X_csr).sum(axis=1), ones)
