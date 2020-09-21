"""
Template for each `dtype` helper function using 1-d template

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

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
