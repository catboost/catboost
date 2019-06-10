# -*- coding: utf-8 -*-
from datetime import datetime
import re

import numpy as np
import pytest

from pandas._libs.tslib import iNaT
from pandas.compat import long

import pandas.core.algorithms as algos
import pandas.util.testing as tm


@pytest.fixture(params=[True, False])
def writeable(request):
    return request.param


# Check that take_nd works both with writeable arrays
# (in which case fast typed memory-views implementation)
# and read-only arrays alike.
@pytest.fixture(params=[
    (np.float64, True),
    (np.float32, True),
    (np.uint64, False),
    (np.uint32, False),
    (np.uint16, False),
    (np.uint8, False),
    (np.int64, False),
    (np.int32, False),
    (np.int16, False),
    (np.int8, False),
    (np.object_, True),
    (np.bool, False),
])
def dtype_can_hold_na(request):
    return request.param


@pytest.fixture(params=[
    (np.int8, np.int16(127), np.int8),
    (np.int8, np.int16(128), np.int16),
    (np.int32, 1, np.int32),
    (np.int32, 2.0, np.float64),
    (np.int32, 3.0 + 4.0j, np.complex128),
    (np.int32, True, np.object_),
    (np.int32, "", np.object_),
    (np.float64, 1, np.float64),
    (np.float64, 2.0, np.float64),
    (np.float64, 3.0 + 4.0j, np.complex128),
    (np.float64, True, np.object_),
    (np.float64, "", np.object_),
    (np.complex128, 1, np.complex128),
    (np.complex128, 2.0, np.complex128),
    (np.complex128, 3.0 + 4.0j, np.complex128),
    (np.complex128, True, np.object_),
    (np.complex128, "", np.object_),
    (np.bool_, 1, np.object_),
    (np.bool_, 2.0, np.object_),
    (np.bool_, 3.0 + 4.0j, np.object_),
    (np.bool_, True, np.bool_),
    (np.bool_, '', np.object_),
])
def dtype_fill_out_dtype(request):
    return request.param


class TestTake(object):
    # Standard incompatible fill error.
    fill_error = re.compile("Incompatible type for fill_value")

    def test_1d_with_out(self, dtype_can_hold_na, writeable):
        dtype, can_hold_na = dtype_can_hold_na

        data = np.random.randint(0, 2, 4).astype(dtype)
        data.flags.writeable = writeable

        indexer = [2, 1, 0, 1]
        out = np.empty(4, dtype=dtype)
        algos.take_1d(data, indexer, out=out)

        expected = data.take(indexer)
        tm.assert_almost_equal(out, expected)

        indexer = [2, 1, 0, -1]
        out = np.empty(4, dtype=dtype)

        if can_hold_na:
            algos.take_1d(data, indexer, out=out)
            expected = data.take(indexer)
            expected[3] = np.nan
            tm.assert_almost_equal(out, expected)
        else:
            with pytest.raises(TypeError, match=self.fill_error):
                algos.take_1d(data, indexer, out=out)

            # No Exception otherwise.
            data.take(indexer, out=out)

    def test_1d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.randint(0, 2, 4).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_1d(data, indexer, fill_value=fill_value)
        assert ((result[[0, 1, 2]] == data[[2, 1, 0]]).all())
        assert (result[3] == fill_value)
        assert (result.dtype == out_dtype)

        indexer = [2, 1, 0, 1]

        result = algos.take_1d(data, indexer, fill_value=fill_value)
        assert ((result[[0, 1, 2, 3]] == data[indexer]).all())
        assert (result.dtype == dtype)

    def test_2d_with_out(self, dtype_can_hold_na, writeable):
        dtype, can_hold_na = dtype_can_hold_na

        data = np.random.randint(0, 2, (5, 3)).astype(dtype)
        data.flags.writeable = writeable

        indexer = [2, 1, 0, 1]
        out0 = np.empty((4, 3), dtype=dtype)
        out1 = np.empty((5, 4), dtype=dtype)
        algos.take_nd(data, indexer, out=out0, axis=0)
        algos.take_nd(data, indexer, out=out1, axis=1)

        expected0 = data.take(indexer, axis=0)
        expected1 = data.take(indexer, axis=1)
        tm.assert_almost_equal(out0, expected0)
        tm.assert_almost_equal(out1, expected1)

        indexer = [2, 1, 0, -1]
        out0 = np.empty((4, 3), dtype=dtype)
        out1 = np.empty((5, 4), dtype=dtype)

        if can_hold_na:
            algos.take_nd(data, indexer, out=out0, axis=0)
            algos.take_nd(data, indexer, out=out1, axis=1)

            expected0 = data.take(indexer, axis=0)
            expected1 = data.take(indexer, axis=1)
            expected0[3, :] = np.nan
            expected1[:, 3] = np.nan

            tm.assert_almost_equal(out0, expected0)
            tm.assert_almost_equal(out1, expected1)
        else:
            for i, out in enumerate([out0, out1]):
                with pytest.raises(TypeError, match=self.fill_error):
                    algos.take_nd(data, indexer, out=out, axis=i)

                # No Exception otherwise.
                data.take(indexer, out=out, axis=i)

    def test_2d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        data = np.random.randint(0, 2, (5, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0,
                               fill_value=fill_value)
        assert ((result[[0, 1, 2], :] == data[[2, 1, 0], :]).all())
        assert ((result[3, :] == fill_value).all())
        assert (result.dtype == out_dtype)

        result = algos.take_nd(data, indexer, axis=1,
                               fill_value=fill_value)
        assert ((result[:, [0, 1, 2]] == data[:, [2, 1, 0]]).all())
        assert ((result[:, 3] == fill_value).all())
        assert (result.dtype == out_dtype)

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0,
                               fill_value=fill_value)
        assert ((result[[0, 1, 2, 3], :] == data[indexer, :]).all())
        assert (result.dtype == dtype)

        result = algos.take_nd(data, indexer, axis=1,
                               fill_value=fill_value)
        assert ((result[:, [0, 1, 2, 3]] == data[:, indexer]).all())
        assert (result.dtype == dtype)

    def test_3d_with_out(self, dtype_can_hold_na):
        dtype, can_hold_na = dtype_can_hold_na

        data = np.random.randint(0, 2, (5, 4, 3)).astype(dtype)
        indexer = [2, 1, 0, 1]

        out0 = np.empty((4, 4, 3), dtype=dtype)
        out1 = np.empty((5, 4, 3), dtype=dtype)
        out2 = np.empty((5, 4, 4), dtype=dtype)

        algos.take_nd(data, indexer, out=out0, axis=0)
        algos.take_nd(data, indexer, out=out1, axis=1)
        algos.take_nd(data, indexer, out=out2, axis=2)

        expected0 = data.take(indexer, axis=0)
        expected1 = data.take(indexer, axis=1)
        expected2 = data.take(indexer, axis=2)

        tm.assert_almost_equal(out0, expected0)
        tm.assert_almost_equal(out1, expected1)
        tm.assert_almost_equal(out2, expected2)

        indexer = [2, 1, 0, -1]
        out0 = np.empty((4, 4, 3), dtype=dtype)
        out1 = np.empty((5, 4, 3), dtype=dtype)
        out2 = np.empty((5, 4, 4), dtype=dtype)

        if can_hold_na:
            algos.take_nd(data, indexer, out=out0, axis=0)
            algos.take_nd(data, indexer, out=out1, axis=1)
            algos.take_nd(data, indexer, out=out2, axis=2)

            expected0 = data.take(indexer, axis=0)
            expected1 = data.take(indexer, axis=1)
            expected2 = data.take(indexer, axis=2)

            expected0[3, :, :] = np.nan
            expected1[:, 3, :] = np.nan
            expected2[:, :, 3] = np.nan

            tm.assert_almost_equal(out0, expected0)
            tm.assert_almost_equal(out1, expected1)
            tm.assert_almost_equal(out2, expected2)
        else:
            for i, out in enumerate([out0, out1, out2]):
                with pytest.raises(TypeError, match=self.fill_error):
                    algos.take_nd(data, indexer, out=out, axis=i)

                # No Exception otherwise.
                data.take(indexer, out=out, axis=i)

    def test_3d_fill_nonna(self, dtype_fill_out_dtype):
        dtype, fill_value, out_dtype = dtype_fill_out_dtype

        data = np.random.randint(0, 2, (5, 4, 3)).astype(dtype)
        indexer = [2, 1, 0, -1]

        result = algos.take_nd(data, indexer, axis=0,
                               fill_value=fill_value)
        assert ((result[[0, 1, 2], :, :] == data[[2, 1, 0], :, :]).all())
        assert ((result[3, :, :] == fill_value).all())
        assert (result.dtype == out_dtype)

        result = algos.take_nd(data, indexer, axis=1,
                               fill_value=fill_value)
        assert ((result[:, [0, 1, 2], :] == data[:, [2, 1, 0], :]).all())
        assert ((result[:, 3, :] == fill_value).all())
        assert (result.dtype == out_dtype)

        result = algos.take_nd(data, indexer, axis=2,
                               fill_value=fill_value)
        assert ((result[:, :, [0, 1, 2]] == data[:, :, [2, 1, 0]]).all())
        assert ((result[:, :, 3] == fill_value).all())
        assert (result.dtype == out_dtype)

        indexer = [2, 1, 0, 1]
        result = algos.take_nd(data, indexer, axis=0,
                               fill_value=fill_value)
        assert ((result[[0, 1, 2, 3], :, :] == data[indexer, :, :]).all())
        assert (result.dtype == dtype)

        result = algos.take_nd(data, indexer, axis=1,
                               fill_value=fill_value)
        assert ((result[:, [0, 1, 2, 3], :] == data[:, indexer, :]).all())
        assert (result.dtype == dtype)

        result = algos.take_nd(data, indexer, axis=2,
                               fill_value=fill_value)
        assert ((result[:, :, [0, 1, 2, 3]] == data[:, :, indexer]).all())
        assert (result.dtype == dtype)

    def test_1d_other_dtypes(self):
        arr = np.random.randn(10).astype(np.float32)

        indexer = [1, 2, 3, -1]
        result = algos.take_1d(arr, indexer)
        expected = arr.take(indexer)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_other_dtypes(self):
        arr = np.random.randn(10, 5).astype(np.float32)

        indexer = [1, 2, 3, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected[-1] = np.nan
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, -1] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_1d_bool(self):
        arr = np.array([0, 1, 0], dtype=bool)

        result = algos.take_1d(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_1d(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_bool(self):
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)

        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1], axis=0)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, 2, 1], axis=1)
        expected = arr.take([0, 2, 2, 1], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.take_nd(arr, [0, 2, -1])
        assert result.dtype == np.object_

    def test_2d_float32(self):
        arr = np.random.randn(4, 3).astype(np.float32)
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, axis=0, out=result2)
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = np.nan
        tm.assert_almost_equal(result, expected)

        # this now accepts a float32! # test with float64 out buffer
        out = np.empty((len(indexer), arr.shape[1]), dtype='float32')
        algos.take_nd(arr, indexer, out=out)  # it works!

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, axis=1, out=result2)
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = np.nan
        tm.assert_almost_equal(result, expected)

    def test_2d_datetime64(self):
        # 2005/01/01 - 2006/01/01
        arr = np.random.randint(
            long(11045376), long(11360736), (5, 3)) * 100000000000
        arr = arr.view(dtype='datetime64[ns]')
        indexer = [0, 2, -1, 1, -1]

        # axis=0
        result = algos.take_nd(arr, indexer, axis=0)
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, axis=0, out=result2)
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=0)
        expected.view(np.int64)[[2, 4], :] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=0,
                               fill_value=datetime(2007, 1, 1))
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, out=result2, axis=0,
                      fill_value=datetime(2007, 1, 1))
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

        # axis=1
        result = algos.take_nd(arr, indexer, axis=1)
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, axis=1, out=result2)
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=1)
        expected.view(np.int64)[:, [2, 4]] = iNaT
        tm.assert_almost_equal(result, expected)

        result = algos.take_nd(arr, indexer, axis=1,
                               fill_value=datetime(2007, 1, 1))
        result2 = np.empty_like(result)
        algos.take_nd(arr, indexer, out=result2, axis=1,
                      fill_value=datetime(2007, 1, 1))
        tm.assert_almost_equal(result, result2)

        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

    def test_take_axis_0(self):
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1])
        expected = np.array([[0, 1, 2], [9, 10, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], allow_fill=True, fill_value=0)
        expected = np.array([[0, 1, 2], [0, 0, 0]])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_axis_1(self):
        arr = np.arange(12).reshape(4, 3)
        result = algos.take(arr, [0, -1], axis=1)
        expected = np.array([[0, 2], [3, 5], [6, 8], [9, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # allow_fill=True
        result = algos.take(arr, [0, -1], axis=1, allow_fill=True,
                            fill_value=0)
        expected = np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
        tm.assert_numpy_array_equal(result, expected)


class TestExtensionTake(object):
    # The take method found in pd.api.extensions

    def test_bounds_check_large(self):
        arr = np.array([1, 2])
        with pytest.raises(IndexError):
            algos.take(arr, [2, 3], allow_fill=True)

        with pytest.raises(IndexError):
            algos.take(arr, [2, 3], allow_fill=False)

    def test_bounds_check_small(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        indexer = [0, -1, -2]
        with pytest.raises(ValueError):
            algos.take(arr, indexer, allow_fill=True)

        result = algos.take(arr, indexer)
        expected = np.array([1, 3, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('allow_fill', [True, False])
    def test_take_empty(self, allow_fill):
        arr = np.array([], dtype=np.int64)
        # empty take is ok
        result = algos.take(arr, [], allow_fill=allow_fill)
        tm.assert_numpy_array_equal(arr, result)

        with pytest.raises(IndexError):
            algos.take(arr, [0], allow_fill=allow_fill)

    def test_take_na_empty(self):
        result = algos.take(np.array([]), [-1, -1], allow_fill=True,
                            fill_value=0.0)
        expected = np.array([0., 0.])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_coerces_list(self):
        arr = [1, 2, 3]
        result = algos.take(arr, [0, 0])
        expected = np.array([1, 1])
        tm.assert_numpy_array_equal(result, expected)
