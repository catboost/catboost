"""
Template for each `dtype` helper function using 1-d template

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_float64(ndarray[float64_t, ndim=2] arr,
                     ndarray[float64_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_float32(ndarray[float32_t, ndim=2] arr,
                     ndarray[float32_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_int8(ndarray[int8_t, ndim=2] arr,
                     ndarray[float32_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_int16(ndarray[int16_t, ndim=2] arr,
                     ndarray[float32_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_int32(ndarray[int32_t, ndim=2] arr,
                     ndarray[float64_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_2d_int64(ndarray[int64_t, ndim=2] arr,
                     ndarray[float64_t, ndim=2] out,
                     Py_ssize_t periods, int axis):
    cdef:
        Py_ssize_t i, j, sx, sy

    sx, sy = (<object>arr).shape
    if arr.flags.f_contiguous:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for j in range(sy):
                for i in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for j in range(start, stop):
                for i in range(sx):
                    out[i, j] = arr[i, j] - arr[i, j - periods]
    else:
        if axis == 0:
            if periods >= 0:
                start, stop = periods, sx
            else:
                start, stop = 0, sx + periods
            for i in range(start, stop):
                for j in range(sy):
                    out[i, j] = arr[i, j] - arr[i - periods, j]
        else:
            if periods >= 0:
                start, stop = periods, sy
            else:
                start, stop = 0, sy + periods
            for i in range(sx):
                for j in range(start, stop):
                    out[i, j] = arr[i, j] - arr[i, j - periods]

# ----------------------------------------------------------------------
# ensure_dtype
# ----------------------------------------------------------------------

cdef int PLATFORM_INT = (<ndarray>np.arange(0, dtype=np.intp)).descr.type_num


def ensure_platform_int(object arr):
    # GH3033, GH1392
    # platform int is the size of the int pointer, e.g. np.intp
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == PLATFORM_INT:
            return arr
        else:
            return arr.astype(np.intp)
    else:
        return np.array(arr, dtype=np.intp)


def ensure_object(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_OBJECT:
            return arr
        else:
            return arr.astype(np.object_)
    else:
        return np.array(arr, dtype=np.object_)


def ensure_float64(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_FLOAT64:
            return arr
        else:
            return arr.astype(np.float64, copy=copy)
    else:
        return np.array(arr, dtype=np.float64)


def ensure_float32(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_FLOAT32:
            return arr
        else:
            return arr.astype(np.float32, copy=copy)
    else:
        return np.array(arr, dtype=np.float32)


def ensure_int8(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT8:
            return arr
        else:
            return arr.astype(np.int8, copy=copy)
    else:
        return np.array(arr, dtype=np.int8)


def ensure_int16(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT16:
            return arr
        else:
            return arr.astype(np.int16, copy=copy)
    else:
        return np.array(arr, dtype=np.int16)


def ensure_int32(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT32:
            return arr
        else:
            return arr.astype(np.int32, copy=copy)
    else:
        return np.array(arr, dtype=np.int32)


def ensure_int64(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT64:
            return arr
        else:
            return arr.astype(np.int64, copy=copy)
    else:
        return np.array(arr, dtype=np.int64)


def ensure_uint8(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_UINT8:
            return arr
        else:
            return arr.astype(np.uint8, copy=copy)
    else:
        return np.array(arr, dtype=np.uint8)


def ensure_uint16(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_UINT16:
            return arr
        else:
            return arr.astype(np.uint16, copy=copy)
    else:
        return np.array(arr, dtype=np.uint16)


def ensure_uint32(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_UINT32:
            return arr
        else:
            return arr.astype(np.uint32, copy=copy)
    else:
        return np.array(arr, dtype=np.uint32)


def ensure_uint64(object arr, copy=True):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_UINT64:
            return arr
        else:
            return arr.astype(np.uint64, copy=copy)
    else:
        return np.array(arr, dtype=np.uint64)
