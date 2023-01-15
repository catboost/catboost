"""
Template for each `dtype` helper function for take

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# take_1d, take_2d
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_bool_bool(const uint8_t[:] values,
                              const int64_t[:] indexer,
                              uint8_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        uint8_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_bool_bool(const uint8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    uint8_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        uint8_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const uint8_t *v
            uint8_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(uint8_t) and
            sizeof(uint8_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(uint8_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_bool_bool(const uint8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    uint8_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        uint8_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_bool_bool(ndarray[uint8_t, ndim=2] values,
                                    indexer,
                                    ndarray[uint8_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        uint8_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_bool_object(const uint8_t[:] values,
                              const int64_t[:] indexer,
                              object[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        object fv

    n = indexer.shape[0]

    fv = fill_value

    if True:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = True if values[idx] > 0 else False


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_bool_object(const uint8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    object[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        object fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const object *v
            object *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(object) and
            sizeof(object) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(object) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = True if values[idx, j] > 0 else False


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_bool_object(const uint8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    object[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        object fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = True if values[i, idx] > 0 else False


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_bool_object(ndarray[uint8_t, ndim=2] values,
                                    indexer,
                                    ndarray[object, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        object fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = True if values[idx, idx1[j]] > 0 else False


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int8_int8(const int8_t[:] values,
                              const int64_t[:] indexer,
                              int8_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int8_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int8_int8(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int8_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int8_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const int8_t *v
            int8_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int8_t) and
            sizeof(int8_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int8_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int8_int8(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int8_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int8_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int8_int8(ndarray[int8_t, ndim=2] values,
                                    indexer,
                                    ndarray[int8_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int8_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int8_int32(const int8_t[:] values,
                              const int64_t[:] indexer,
                              int32_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int32_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int8_int32(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const int32_t *v
            int32_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int32_t) and
            sizeof(int32_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int32_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int8_int32(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int8_int32(ndarray[int8_t, ndim=2] values,
                                    indexer,
                                    ndarray[int32_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int32_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int8_int64(const int8_t[:] values,
                              const int64_t[:] indexer,
                              int64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int8_int64(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const int64_t *v
            int64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int64_t) and
            sizeof(int64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int8_int64(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int8_int64(ndarray[int8_t, ndim=2] values,
                                    indexer,
                                    ndarray[int64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int8_float64(const int8_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int8_float64(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int8_float64(const int8_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int8_float64(ndarray[int8_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int16_int16(const int16_t[:] values,
                              const int64_t[:] indexer,
                              int16_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int16_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int16_int16(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int16_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int16_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const int16_t *v
            int16_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int16_t) and
            sizeof(int16_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int16_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int16_int16(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int16_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int16_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int16_int16(ndarray[int16_t, ndim=2] values,
                                    indexer,
                                    ndarray[int16_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int16_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int16_int32(const int16_t[:] values,
                              const int64_t[:] indexer,
                              int32_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int32_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int16_int32(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const int32_t *v
            int32_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int32_t) and
            sizeof(int32_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int32_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int16_int32(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int16_int32(ndarray[int16_t, ndim=2] values,
                                    indexer,
                                    ndarray[int32_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int32_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int16_int64(const int16_t[:] values,
                              const int64_t[:] indexer,
                              int64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int16_int64(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const int64_t *v
            int64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int64_t) and
            sizeof(int64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int16_int64(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int16_int64(ndarray[int16_t, ndim=2] values,
                                    indexer,
                                    ndarray[int64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int16_float64(const int16_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int16_float64(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int16_float64(const int16_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int16_float64(ndarray[int16_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int32_int32(const int32_t[:] values,
                              const int64_t[:] indexer,
                              int32_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int32_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int32_int32(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const int32_t *v
            int32_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int32_t) and
            sizeof(int32_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int32_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int32_int32(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int32_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int32_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int32_int32(ndarray[int32_t, ndim=2] values,
                                    indexer,
                                    ndarray[int32_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int32_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int32_int64(const int32_t[:] values,
                              const int64_t[:] indexer,
                              int64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int32_int64(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const int64_t *v
            int64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int64_t) and
            sizeof(int64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int32_int64(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int32_int64(ndarray[int32_t, ndim=2] values,
                                    indexer,
                                    ndarray[int64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int32_float64(const int32_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int32_float64(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int32_float64(const int32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int32_float64(ndarray[int32_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int64_int64(const int64_t[:] values,
                              const int64_t[:] indexer,
                              int64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        int64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int64_int64(const int64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const int64_t *v
            int64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(int64_t) and
            sizeof(int64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(int64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int64_int64(const int64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    int64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        int64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int64_int64(ndarray[int64_t, ndim=2] values,
                                    indexer,
                                    ndarray[int64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        int64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_int64_float64(const int64_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_int64_float64(const int64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_int64_float64(const int64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_int64_float64(ndarray[int64_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_float32_float32(const float32_t[:] values,
                              const int64_t[:] indexer,
                              float32_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float32_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_float32_float32(const float32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float32_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float32_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const float32_t *v
            float32_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float32_t) and
            sizeof(float32_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float32_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_float32_float32(const float32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float32_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float32_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_float32_float32(ndarray[float32_t, ndim=2] values,
                                    indexer,
                                    ndarray[float32_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float32_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_float32_float64(const float32_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_float32_float64(const float32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_float32_float64(const float32_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_float32_float64(ndarray[float32_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_float64_float64(const float64_t[:] values,
                              const int64_t[:] indexer,
                              float64_t[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        float64_t fv

    n = indexer.shape[0]

    fv = fill_value

    with nogil:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_float64_float64(const float64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF True:
        cdef:
            const float64_t *v
            float64_t *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(float64_t) and
            sizeof(float64_t) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(float64_t) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_float64_float64(const float64_t[:, :] values,
                                    ndarray[int64_t] indexer,
                                    float64_t[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        float64_t fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_float64_float64(ndarray[float64_t, ndim=2] values,
                                    indexer,
                                    ndarray[float64_t, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        float64_t fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_1d_object_object(ndarray[object, ndim=1] values,
                              const int64_t[:] indexer,
                              object[:] out,
                              fill_value=np.nan):

    cdef:
        Py_ssize_t i, n, idx
        object fv

    n = indexer.shape[0]

    fv = fill_value

    if True:
        for i in range(n):
            idx = indexer[i]
            if idx == -1:
                out[i] = fv
            else:
                out[i] = values[idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis0_object_object(ndarray[object, ndim=2] values,
                                    ndarray[int64_t] indexer,
                                    object[:, :] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        object fv

    n = len(indexer)
    k = values.shape[1]

    fv = fill_value

    IF False:
        cdef:
            const object *v
            object *o

        # GH#3130
        if (values.strides[1] == out.strides[1] and
            values.strides[1] == sizeof(object) and
            sizeof(object) * n >= 256):

            for i in range(n):
                idx = indexer[i]
                if idx == -1:
                    for j in range(k):
                        out[i, j] = fv
                else:
                    v = &values[idx, 0]
                    o = &out[i, 0]
                    memmove(o, v, <size_t>(sizeof(object) * k))
            return

    for i in range(n):
        idx = indexer[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                out[i, j] = values[idx, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_axis1_object_object(ndarray[object, ndim=2] values,
                                    ndarray[int64_t] indexer,
                                    object[:, :] out,
                                    fill_value=np.nan):

    cdef:
        Py_ssize_t i, j, k, n, idx
        object fv

    n = len(values)
    k = len(indexer)

    if n == 0 or k == 0:
        return

    fv = fill_value

    for i in range(n):
        for j in range(k):
            idx = indexer[j]
            if idx == -1:
                out[i, j] = fv
            else:
                out[i, j] = values[i, idx]


@cython.wraparound(False)
@cython.boundscheck(False)
def take_2d_multi_object_object(ndarray[object, ndim=2] values,
                                    indexer,
                                    ndarray[object, ndim=2] out,
                                    fill_value=np.nan):
    cdef:
        Py_ssize_t i, j, k, n, idx
        ndarray[int64_t] idx0 = indexer[0]
        ndarray[int64_t] idx1 = indexer[1]
        object fv

    n = len(idx0)
    k = len(idx1)

    fv = fill_value
    for i in range(n):
        idx = idx0[i]
        if idx == -1:
            for j in range(k):
                out[i, j] = fv
        else:
            for j in range(k):
                if idx1[j] == -1:
                    out[i, j] = fv
                else:
                    out[i, j] = values[idx, idx1[j]]

# ----------------------------------------------------------------------
# take_2d internal function
# ----------------------------------------------------------------------

ctypedef fused take_t:
    float64_t
    uint64_t
    int64_t
    object


cdef _take_2d(ndarray[take_t, ndim=2] values, object idx):
    cdef:
        Py_ssize_t i, j, N, K
        ndarray[Py_ssize_t, ndim=2, cast=True] indexer = idx
        ndarray[take_t, ndim=2] result

    N, K = (<object>values).shape

    if take_t is object:
        # evaluated at compile-time
        result = values.copy()
    else:
        result = np.empty_like(values)

    for i in range(N):
        for j in range(K):
            result[i, j] = values[i, indexer[i, j]]
    return result
