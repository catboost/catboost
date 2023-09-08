"""
Template for each `dtype` helper function using 1-d template

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# ensure_dtype
# ----------------------------------------------------------------------


def ensure_platform_int(object arr):
    # GH3033, GH1392
    # platform int is the size of the int pointer, e.g. np.intp
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == cnp.NPY_INTP:
            return arr
        else:
            # equiv: arr.astype(np.intp)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_INTP)
    else:
        return np.array(arr, dtype=np.intp)


def ensure_object(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_OBJECT:
            return arr
        else:
            # equiv: arr.astype(object)
            return cnp.PyArray_Cast(<ndarray>arr, NPY_OBJECT)
    else:
        return np.array(arr, dtype=np.object_)


def ensure_float64(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_FLOAT64:
            return arr
        else:
            # equiv: arr.astype(np.float64)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_FLOAT64)
    else:
        return np.asarray(arr, dtype=np.float64)


def ensure_int8(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT8:
            return arr
        else:
            # equiv: arr.astype(np.int8)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_INT8)
    else:
        return np.asarray(arr, dtype=np.int8)


def ensure_int16(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT16:
            return arr
        else:
            # equiv: arr.astype(np.int16)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_INT16)
    else:
        return np.asarray(arr, dtype=np.int16)


def ensure_int32(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT32:
            return arr
        else:
            # equiv: arr.astype(np.int32)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_INT32)
    else:
        return np.asarray(arr, dtype=np.int32)


def ensure_int64(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_INT64:
            return arr
        else:
            # equiv: arr.astype(np.int64)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_INT64)
    else:
        return np.asarray(arr, dtype=np.int64)


def ensure_uint64(object arr):
    if util.is_array(arr):
        if (<ndarray>arr).descr.type_num == NPY_UINT64:
            return arr
        else:
            # equiv: arr.astype(np.uint64)
            return cnp.PyArray_Cast(<ndarray>arr, cnp.NPY_UINT64)
    else:
        return np.asarray(arr, dtype=np.uint64)
