# -*- cython -*-
#
# Tempita-templated Cython file
#
"""
Fast snippets for LIL matrices.
"""

cimport cython
cimport numpy as cnp
import numpy as np


cnp.import_array()


@cython.wraparound(False)
cpdef lil_get1(cnp.npy_intp M, cnp.npy_intp N, object[:] rows, object[:] datas,
               cnp.npy_intp i, cnp.npy_intp j):
    """
    Get a single item from LIL matrix.

    Doesn't do output type conversion. Checks for bounds errors.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get

    Returns
    -------
    x
        Value at indices.

    """
    cdef list row, data

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]
    cdef cnp.npy_intp pos = bisect_left(row, j)

    if pos != len(data) and row[pos] == j:
        return data[pos]
    else:
        return 0


@cython.wraparound(False)
cpdef int lil_insert(cnp.npy_intp M, cnp.npy_intp N, object[:] rows,
                     object[:] datas, cnp.npy_intp i, cnp.npy_intp j,
                     object x) except -1:
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    cdef list row, data

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]

    cdef cnp.npy_intp pos = bisect_left(row, j)
    if x == 0:
        if pos < len(row) and row[pos] == j:
            del row[pos]
            del data[pos]
    else:
        if pos == len(row):
            row.append(j)
            data.append(x)
        elif row[pos] != j:
            row.insert(pos, j)
            data.insert(pos, x)
        else:
            data[pos] = x

def lil_get_lengths(object[:] input,
                    cnp.ndarray output):
    return _LIL_GET_LENGTHS_DISPATCH[output.dtype](input, output)
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_get_lengths_int32(object[:] input,
                    cnp.ndarray[cnp.npy_int32] output):
    for i in range(len(input)):
        output[i] = len(input[i])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_get_lengths_int64(object[:] input,
                    cnp.ndarray[cnp.npy_int64] output):
    for i in range(len(input)):
        output[i] = len(input[i])

cdef dict _LIL_GET_LENGTHS_DISPATCH = {

np.dtype(np.int32): _lil_get_lengths_int32,
np.dtype(np.int64): _lil_get_lengths_int64,
}



# We define the fuse type below because Cython does not currently allow to
# declare object memory views (cf. https://github.com/cython/cython/issues/2485)
# We can track the support of object memory views in
# https://github.com/cython/cython/pull/4712
ctypedef fused obj_fused:
    object
    double

def lil_flatten_to_array(const obj_fused[:] input,
                         cnp.ndarray output):
    return _LIL_FLATTEN_TO_ARRAY_DISPATCH[output.dtype](input, output)
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_bool_(object[:] input not None, cnp.ndarray[cnp.npy_bool] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_int8(object[:] input not None, cnp.ndarray[cnp.npy_int8] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_uint8(object[:] input not None, cnp.ndarray[cnp.npy_uint8] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_int16(object[:] input not None, cnp.ndarray[cnp.npy_int16] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_uint16(object[:] input not None, cnp.ndarray[cnp.npy_uint16] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_int32(object[:] input not None, cnp.ndarray[cnp.npy_int32] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_uint32(object[:] input not None, cnp.ndarray[cnp.npy_uint32] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_int64(object[:] input not None, cnp.ndarray[cnp.npy_int64] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_uint64(object[:] input not None, cnp.ndarray[cnp.npy_uint64] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_float32(object[:] input not None, cnp.ndarray[cnp.npy_float32] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_float64(object[:] input not None, cnp.ndarray[cnp.npy_float64] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_longdouble(object[:] input not None, cnp.ndarray[long double] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_complex64(object[:] input not None, cnp.ndarray[float complex] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_complex128(object[:] input not None, cnp.ndarray[double complex] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_flatten_to_array_clongdouble(object[:] input not None, cnp.ndarray[long double complex] output not None):
    cdef list row
    cdef size_t pos = 0
    for i in range(len(input)):
        row = input[i]
        for j in range(len(row)):
            output[pos] = row[j]
            pos += 1

cdef dict _LIL_FLATTEN_TO_ARRAY_DISPATCH = {

np.dtype(np.bool_): _lil_flatten_to_array_bool_,
np.dtype(np.int8): _lil_flatten_to_array_int8,
np.dtype(np.uint8): _lil_flatten_to_array_uint8,
np.dtype(np.int16): _lil_flatten_to_array_int16,
np.dtype(np.uint16): _lil_flatten_to_array_uint16,
np.dtype(np.int32): _lil_flatten_to_array_int32,
np.dtype(np.uint32): _lil_flatten_to_array_uint32,
np.dtype(np.int64): _lil_flatten_to_array_int64,
np.dtype(np.uint64): _lil_flatten_to_array_uint64,
np.dtype(np.float32): _lil_flatten_to_array_float32,
np.dtype(np.float64): _lil_flatten_to_array_float64,
np.dtype(np.longdouble): _lil_flatten_to_array_longdouble,
np.dtype(np.complex64): _lil_flatten_to_array_complex64,
np.dtype(np.complex128): _lil_flatten_to_array_complex128,
np.dtype(np.clongdouble): _lil_flatten_to_array_clongdouble,
}



def lil_fancy_get(cnp.npy_intp M, cnp.npy_intp N,
                  object[:] rows,
                  object[:] datas,
                  object[:] new_rows,
                  object[:] new_datas,
                  cnp.ndarray i_idx,
                  cnp.ndarray j_idx):
    """
    Get multiple items at given indices in LIL matrix and store to
    another LIL.

    Parameters
    ----------
    M, N, rows, data
        LIL matrix data, initially empty
    new_rows, new_idx
        Data for LIL matrix to insert to.
        Must be preallocated to shape `i_idx.shape`!
    i_idx, j_idx
        Indices of elements to insert to the new LIL matrix.

    """
    return _LIL_FANCY_GET_DISPATCH[i_idx.dtype](M, N, rows, datas, new_rows, new_datas, i_idx, j_idx)

def _lil_fancy_get_int32(cnp.npy_intp M, cnp.npy_intp N,
                            object[:] rows,
                            object[:] datas,
                            object[:] new_rows,
                            object[:] new_datas,
                            cnp.npy_int32[:,:] i_idx,
                            cnp.npy_int32[:,:] j_idx):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j
    cdef object value
    cdef list new_row
    cdef list new_data

    for x in range(i_idx.shape[0]):
        new_row = []
        new_data = []

        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]

            value = lil_get1(M, N, rows, datas, i, j)

            if value is not 0:
                # Object identity as shortcut
                new_row.append(y)
                new_data.append(value)

        new_rows[x] = new_row
        new_datas[x] = new_data
def _lil_fancy_get_int64(cnp.npy_intp M, cnp.npy_intp N,
                            object[:] rows,
                            object[:] datas,
                            object[:] new_rows,
                            object[:] new_datas,
                            cnp.npy_int64[:,:] i_idx,
                            cnp.npy_int64[:,:] j_idx):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j
    cdef object value
    cdef list new_row
    cdef list new_data

    for x in range(i_idx.shape[0]):
        new_row = []
        new_data = []

        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]

            value = lil_get1(M, N, rows, datas, i, j)

            if value is not 0:
                # Object identity as shortcut
                new_row.append(y)
                new_data.append(value)

        new_rows[x] = new_row
        new_datas[x] = new_data


cdef dict _LIL_FANCY_GET_DISPATCH = {

np.dtype(np.int32): _lil_fancy_get_int32,
np.dtype(np.int64): _lil_fancy_get_int64,
}




def lil_fancy_set(cnp.npy_intp M, cnp.npy_intp N,
                  object[:] rows,
                  object[:] data,
                  cnp.ndarray i_idx,
                  cnp.ndarray j_idx,
                  cnp.ndarray values):
    """
    Set multiple items to a LIL matrix.

    Checks for zero elements and deletes them.

    Parameters
    ----------
    M, N, rows, data
        LIL matrix data
    i_idx, j_idx
        Indices of elements to insert to the new LIL matrix.
    values
        Values of items to set.

    """
    if values.dtype == np.bool_:
        # Cython doesn't support np.bool_ as a memoryview type
        values = values.view(dtype=np.uint8)

    assert i_idx.shape[0] == j_idx.shape[0] and i_idx.shape[1] == j_idx.shape[1]
    return _LIL_FANCY_SET_DISPATCH[i_idx.dtype, values.dtype](M, N, rows, data, i_idx, j_idx, values)

@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_bool_(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_bool[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_int8(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_int8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_uint8(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_uint8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_int16(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_int16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_uint16(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_uint16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_int32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_int32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_uint32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_uint32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_int64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_int64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_uint64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_uint64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_float32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_float32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_float64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         cnp.npy_float64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_longdouble(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         long double[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_complex64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         float complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_complex128(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int32_clongdouble(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int32[:,:] i_idx,
                                         cnp.npy_int32[:,:] j_idx,
                                         long double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_bool_(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_bool[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_int8(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_int8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_uint8(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_uint8[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_int16(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_int16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_uint16(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_uint16[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_int32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_int32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_uint32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_uint32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_int64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_int64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_uint64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_uint64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_float32(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_float32[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_float64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         cnp.npy_float64[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_longdouble(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         long double[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_complex64(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         float complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_complex128(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])
@cython.boundscheck(False)
@cython.wraparound(False)
def _lil_fancy_set_int64_clongdouble(cnp.npy_intp M, cnp.npy_intp N,
                                         object[:] rows,
                                         object[:] data,
                                         cnp.npy_int64[:,:] i_idx,
                                         cnp.npy_int64[:,:] j_idx,
                                         long double complex[:,:] values):
    cdef cnp.npy_intp x, y
    cdef cnp.npy_intp i, j

    for x in range(i_idx.shape[0]):
        for y in range(i_idx.shape[1]):
            i = i_idx[x,y]
            j = j_idx[x,y]
            lil_insert(M, N, rows, data, i, j, values[x, y])


cdef dict _LIL_FANCY_SET_DISPATCH = {

(np.dtype(np.int32), np.dtype(np.bool_)): _lil_fancy_set_int32_bool_,
(np.dtype(np.int32), np.dtype(np.int8)): _lil_fancy_set_int32_int8,
(np.dtype(np.int32), np.dtype(np.uint8)): _lil_fancy_set_int32_uint8,
(np.dtype(np.int32), np.dtype(np.int16)): _lil_fancy_set_int32_int16,
(np.dtype(np.int32), np.dtype(np.uint16)): _lil_fancy_set_int32_uint16,
(np.dtype(np.int32), np.dtype(np.int32)): _lil_fancy_set_int32_int32,
(np.dtype(np.int32), np.dtype(np.uint32)): _lil_fancy_set_int32_uint32,
(np.dtype(np.int32), np.dtype(np.int64)): _lil_fancy_set_int32_int64,
(np.dtype(np.int32), np.dtype(np.uint64)): _lil_fancy_set_int32_uint64,
(np.dtype(np.int32), np.dtype(np.float32)): _lil_fancy_set_int32_float32,
(np.dtype(np.int32), np.dtype(np.float64)): _lil_fancy_set_int32_float64,
(np.dtype(np.int32), np.dtype(np.longdouble)): _lil_fancy_set_int32_longdouble,
(np.dtype(np.int32), np.dtype(np.complex64)): _lil_fancy_set_int32_complex64,
(np.dtype(np.int32), np.dtype(np.complex128)): _lil_fancy_set_int32_complex128,
(np.dtype(np.int32), np.dtype(np.clongdouble)): _lil_fancy_set_int32_clongdouble,
(np.dtype(np.int64), np.dtype(np.bool_)): _lil_fancy_set_int64_bool_,
(np.dtype(np.int64), np.dtype(np.int8)): _lil_fancy_set_int64_int8,
(np.dtype(np.int64), np.dtype(np.uint8)): _lil_fancy_set_int64_uint8,
(np.dtype(np.int64), np.dtype(np.int16)): _lil_fancy_set_int64_int16,
(np.dtype(np.int64), np.dtype(np.uint16)): _lil_fancy_set_int64_uint16,
(np.dtype(np.int64), np.dtype(np.int32)): _lil_fancy_set_int64_int32,
(np.dtype(np.int64), np.dtype(np.uint32)): _lil_fancy_set_int64_uint32,
(np.dtype(np.int64), np.dtype(np.int64)): _lil_fancy_set_int64_int64,
(np.dtype(np.int64), np.dtype(np.uint64)): _lil_fancy_set_int64_uint64,
(np.dtype(np.int64), np.dtype(np.float32)): _lil_fancy_set_int64_float32,
(np.dtype(np.int64), np.dtype(np.float64)): _lil_fancy_set_int64_float64,
(np.dtype(np.int64), np.dtype(np.longdouble)): _lil_fancy_set_int64_longdouble,
(np.dtype(np.int64), np.dtype(np.complex64)): _lil_fancy_set_int64_complex64,
(np.dtype(np.int64), np.dtype(np.complex128)): _lil_fancy_set_int64_complex128,
(np.dtype(np.int64), np.dtype(np.clongdouble)): _lil_fancy_set_int64_clongdouble,
}




def lil_get_row_ranges(cnp.npy_intp M, cnp.npy_intp N,
                       const obj_fused[:] rows, const obj_fused[:] datas,
                       object[:] new_rows, object[:] new_datas,
                       object irows,
                       cnp.npy_intp j_start,
                       cnp.npy_intp j_stop,
                       cnp.npy_intp j_stride,
                       cnp.npy_intp nj):
    """
    Column-slicing fast path for LIL matrices.
    Extracts values from rows/datas and inserts in to
    new_rows/new_datas.
    Parameters
    ----------
    M, N
         Shape of input array
    rows, datas
         LIL data for input array, shape (M, N)
    new_rows, new_datas
         LIL data for output array, shape (len(irows), nj)
    irows : iterator
         Iterator yielding row indices
    j_start, j_stop, j_stride
         Column range(j_start, j_stop, j_stride) to get
    nj : int
         Number of columns corresponding to j_* variables.
    """
    cdef cnp.npy_intp nk, k, j, a, b, m, r, p
    cdef list cur_row, cur_data, new_row, new_data

    if j_stride == 0:
        raise ValueError("cannot index with zero stride")

    for nk, k in enumerate(irows):
        if k >= M or k < -M:
            raise ValueError("row index %d out of bounds" % (k,))
        if k < 0:
            k += M

        if j_stride == 1 and nj == N:
            # full row slice
            new_rows[nk] = list(rows[k])
            new_datas[nk] = list(datas[k])
        else:
            # partial row slice
            cur_row = rows[k]
            cur_data = datas[k]
            new_row = new_rows[nk]
            new_data = new_datas[nk]

            if j_stride > 0:
                a = bisect_left(cur_row, j_start)
                for m in range(a, len(cur_row)):
                    j = cur_row[m]
                    if j >= j_stop:
                        break
                    r = (j - j_start) % j_stride
                    if r != 0:
                        continue
                    p = (j - j_start) // j_stride
                    new_row.append(p)
                    new_data.append(cur_data[m])
            else:
                a = bisect_right(cur_row, j_stop)
                for m in range(a, len(cur_row)):
                    j = cur_row[m]
                    if j > j_start:
                        break
                    r = (j - j_start) % j_stride
                    if r != 0:
                        continue
                    p = (j - j_start) // j_stride
                    new_row.insert(0, p)
                    new_data.insert(0, cur_data[m])


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.npy_intp bisect_left(list a, cnp.npy_intp x) except -1:
    """
    Bisection search in a sorted list.

    List is assumed to contain objects castable to integers.

    Parameters
    ----------
    a
        List to search in
    x
        Value to search for

    Returns
    -------
    j : int
        Index at value (if present), or at the point to which
        it can be inserted maintaining order.

    """
    cdef Py_ssize_t hi = len(a)
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t mid, v

    while lo < hi:
        mid = lo + (hi - lo) // 2
        v = a[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline cnp.npy_intp bisect_right(list a, cnp.npy_intp x) except -1:
    """
    Bisection search in a sorted list.

    List is assumed to contain objects castable to integers.

    Parameters
    ----------
    a
        List to search in
    x
        Value to search for
    Returns
    -------
    j : int
        Index immediately at the right of the value (if present), or at
        the point to which it can be inserted maintaining order.
    """
    cdef cnp.npy_intp hi = len(a)
    cdef cnp.npy_intp lo = 0
    cdef cnp.npy_intp mid, v

    while lo < hi:
        mid = (lo + hi) // 2
        v = a[mid]
        if x < v:
            hi = mid
        else:
            lo = mid + 1
    return lo


cdef _fill_dtype_map(map, chars):
    """
    Fill in Numpy dtype chars for problematic types, working around
    Numpy < 1.6 bugs.
    """
    for c in chars:
        if c in "SUVO":
            continue
        dt = np.dtype(c)
        if dt not in map:
            for k, v in map.items():
                if k.kind == dt.kind and k.itemsize == dt.itemsize:
                    map[dt] = v
                    break


cdef _fill_dtype_map2(map):
    """
    Fill in Numpy dtype chars for problematic types, working around
    Numpy < 1.6 bugs.
    """
    for c1 in np.typecodes['Integer']:
        for c2 in np.typecodes['All']:
            if c2 in "SUVO":
                continue
            dt1 = np.dtype(c1)
            dt2 = np.dtype(c2)
            if (dt1, dt2) not in map:
                for k, v in map.items():
                    if (k[0].kind == dt1.kind and k[0].itemsize == dt1.itemsize and
                        k[1].kind == dt2.kind and k[1].itemsize == dt2.itemsize):
                        map[(dt1, dt2)] = v
                        break

_fill_dtype_map(_LIL_FANCY_GET_DISPATCH, np.typecodes['Integer'])
_fill_dtype_map2(_LIL_FANCY_SET_DISPATCH)
