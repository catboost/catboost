"""
Template for each `dtype` helper function for hashtable

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_float64(float64_t[:] values,
                                 kh_float64_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        float64_t val

        int ret = 0

    with nogil:
        kh_resize_float64(table, n)

        for i in range(n):
            val = values[i]

            if val == val or not dropna:
                k = kh_get_float64(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_float64(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_float64(float64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_float64_t *table

        float64_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_float64()
    build_count_table_float64(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'float64')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_float64(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_float64(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_float64(const float64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        float64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_float64_t *table = kh_init_float64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_float64(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_float64(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_float64(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_float64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_float64(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_float64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_float64(const float64_t[:] arr, const float64_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : float64 ndarray
    values : float64 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        float64_t val
        kh_float64_t *table = kh_init_float64()

    # construct the table
    n = len(values)
    kh_resize_float64(table, n)

    with nogil:
        for i in range(n):
            kh_put_float64(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_float64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_float64(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_float32(float32_t[:] values,
                                 kh_float32_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        float32_t val

        int ret = 0

    with nogil:
        kh_resize_float32(table, n)

        for i in range(n):
            val = values[i]

            if val == val or not dropna:
                k = kh_get_float32(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_float32(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_float32(float32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_float32_t *table

        float32_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_float32()
    build_count_table_float32(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'float32')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_float32(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_float32(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_float32(const float32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        float32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_float32_t *table = kh_init_float32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_float32(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_float32(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_float32(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_float32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_float32(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_float32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_float32(const float32_t[:] arr, const float32_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : float32 ndarray
    values : float32 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        float32_t val
        kh_float32_t *table = kh_init_float32()

    # construct the table
    n = len(values)
    kh_resize_float32(table, n)

    with nogil:
        for i in range(n):
            kh_put_float32(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_float32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_float32(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_uint64(uint64_t[:] values,
                                 kh_uint64_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        uint64_t val

        int ret = 0

    with nogil:
        kh_resize_uint64(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_uint64(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_uint64(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_uint64(uint64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_uint64_t *table

        uint64_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_uint64()
    build_count_table_uint64(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'uint64')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint64(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_uint64(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_uint64(const uint64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint64_t *table = kh_init_uint64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint64(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_uint64(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_uint64(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_uint64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint64(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_uint64(const uint64_t[:] arr, const uint64_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : uint64 ndarray
    values : uint64 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        uint64_t val
        kh_uint64_t *table = kh_init_uint64()

    # construct the table
    n = len(values)
    kh_resize_uint64(table, n)

    with nogil:
        for i in range(n):
            kh_put_uint64(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_uint64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint64(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_uint32(uint32_t[:] values,
                                 kh_uint32_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        uint32_t val

        int ret = 0

    with nogil:
        kh_resize_uint32(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_uint32(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_uint32(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_uint32(uint32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_uint32_t *table

        uint32_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_uint32()
    build_count_table_uint32(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'uint32')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint32(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_uint32(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_uint32(const uint32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint32_t *table = kh_init_uint32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint32(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_uint32(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_uint32(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_uint32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint32(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_uint32(const uint32_t[:] arr, const uint32_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : uint32 ndarray
    values : uint32 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        uint32_t val
        kh_uint32_t *table = kh_init_uint32()

    # construct the table
    n = len(values)
    kh_resize_uint32(table, n)

    with nogil:
        for i in range(n):
            kh_put_uint32(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_uint32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint32(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_uint16(uint16_t[:] values,
                                 kh_uint16_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        uint16_t val

        int ret = 0

    with nogil:
        kh_resize_uint16(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_uint16(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_uint16(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_uint16(uint16_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_uint16_t *table

        uint16_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_uint16()
    build_count_table_uint16(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'uint16')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint16(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_uint16(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_uint16(const uint16_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint16_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint16_t *table = kh_init_uint16()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint16(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_uint16(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_uint16(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_uint16(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint16(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint16(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_uint16(const uint16_t[:] arr, const uint16_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : uint16 ndarray
    values : uint16 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        uint16_t val
        kh_uint16_t *table = kh_init_uint16()

    # construct the table
    n = len(values)
    kh_resize_uint16(table, n)

    with nogil:
        for i in range(n):
            kh_put_uint16(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_uint16(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint16(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_uint8(uint8_t[:] values,
                                 kh_uint8_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        uint8_t val

        int ret = 0

    with nogil:
        kh_resize_uint8(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_uint8(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_uint8(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_uint8(uint8_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_uint8_t *table

        uint8_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_uint8()
    build_count_table_uint8(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'uint8')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint8(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_uint8(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_uint8(const uint8_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint8_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint8_t *table = kh_init_uint8()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint8(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_uint8(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_uint8(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_uint8(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint8(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint8(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_uint8(const uint8_t[:] arr, const uint8_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : uint8 ndarray
    values : uint8 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        uint8_t val
        kh_uint8_t *table = kh_init_uint8()

    # construct the table
    n = len(values)
    kh_resize_uint8(table, n)

    with nogil:
        for i in range(n):
            kh_put_uint8(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_uint8(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint8(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_object(ndarray[object] values,
                                 kh_pymap_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        object val

        int ret = 0

    kh_resize_pymap(table, n // 10)

    for i in range(n):
        val = values[i]

        if not checknull(val) or not dropna:
            k = kh_get_pymap(table, <PyObject*>val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_pymap(table, <PyObject*>val, &ret)
                table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_object(ndarray[object] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_pymap_t *table


        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_pymap()
    build_count_table_object(values, table, 1)

    result_keys = np.empty(table.n_occupied, 'object')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    for k in range(table.n_buckets):
        if kh_exist_pymap(table, k):
            result_keys[i] = <object>table.keys[k]
            result_counts[i] = table.vals[k]
            i += 1

    kh_destroy_pymap(table)

    return result_keys, result_counts


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_object(ndarray[object] values, object keep='first'):
    cdef:
        int ret = 0
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_pymap_t *table = kh_init_pymap()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_pymap(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        for i in range(n - 1, -1, -1):
            # equivalent: range(n)[::-1], which cython doesn't like in nogil
            kh_put_pymap(table, <PyObject*>values[i], &ret)
            out[i] = ret == 0
    elif keep == 'first':
        for i in range(n):
            kh_put_pymap(table, <PyObject*>values[i], &ret)
            out[i] = ret == 0
    else:
        for i in range(n):
            value = values[i]
            k = kh_get_pymap(table, <PyObject*>value)
            if k != table.n_buckets:
                out[table.vals[k]] = 1
                out[i] = 1
            else:
                k = kh_put_pymap(table, <PyObject*>value, &ret)
                table.keys[k] = <PyObject*>value
                table.vals[k] = i
                out[i] = 0
    kh_destroy_pymap(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_object(ndarray[object] arr, ndarray[object] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : object ndarray
    values : object ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        object val
        kh_pymap_t *table = kh_init_pymap()

    # construct the table
    n = len(values)
    kh_resize_pymap(table, n)

    for i in range(n):
        kh_put_pymap(table, <PyObject*>values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    for i in range(n):
        val = arr[i]
        k = kh_get_pymap(table, <PyObject*>val)
        result[i] = (k != table.n_buckets)

    kh_destroy_pymap(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_int64(int64_t[:] values,
                                 kh_int64_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        int64_t val

        int ret = 0

    with nogil:
        kh_resize_int64(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_int64(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_int64(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_int64(int64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_int64_t *table

        int64_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_int64()
    build_count_table_int64(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'int64')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int64(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_int64(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_int64(const int64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int64_t *table = kh_init_int64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int64(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_int64(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_int64(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_int64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int64(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_int64(const int64_t[:] arr, const int64_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : int64 ndarray
    values : int64 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        int64_t val
        kh_int64_t *table = kh_init_int64()

    # construct the table
    n = len(values)
    kh_resize_int64(table, n)

    with nogil:
        for i in range(n):
            kh_put_int64(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_int64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int64(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_int32(int32_t[:] values,
                                 kh_int32_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        int32_t val

        int ret = 0

    with nogil:
        kh_resize_int32(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_int32(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_int32(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_int32(int32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_int32_t *table

        int32_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_int32()
    build_count_table_int32(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'int32')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int32(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_int32(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_int32(const int32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int32_t *table = kh_init_int32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int32(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_int32(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_int32(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_int32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int32(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_int32(const int32_t[:] arr, const int32_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : int32 ndarray
    values : int32 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        int32_t val
        kh_int32_t *table = kh_init_int32()

    # construct the table
    n = len(values)
    kh_resize_int32(table, n)

    with nogil:
        for i in range(n):
            kh_put_int32(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_int32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int32(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_int16(int16_t[:] values,
                                 kh_int16_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        int16_t val

        int ret = 0

    with nogil:
        kh_resize_int16(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_int16(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_int16(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_int16(int16_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_int16_t *table

        int16_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_int16()
    build_count_table_int16(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'int16')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int16(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_int16(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_int16(const int16_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int16_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int16_t *table = kh_init_int16()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int16(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_int16(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_int16(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_int16(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int16(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int16(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_int16(const int16_t[:] arr, const int16_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : int16 ndarray
    values : int16 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        int16_t val
        kh_int16_t *table = kh_init_int16()

    # construct the table
    n = len(values)
    kh_resize_int16(table, n)

    with nogil:
        for i in range(n):
            kh_put_int16(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_int16(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int16(table)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef build_count_table_int8(int8_t[:] values,
                                 kh_int8_t *table, bint dropna):
    cdef:
        khiter_t k
        Py_ssize_t i, n = len(values)

        int8_t val

        int ret = 0

    with nogil:
        kh_resize_int8(table, n)

        for i in range(n):
            val = values[i]

            if True:
                k = kh_get_int8(table, val)
                if k != table.n_buckets:
                    table.vals[k] += 1
                else:
                    k = kh_put_int8(table, val, &ret)
                    table.vals[k] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef value_count_int8(int8_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        kh_int8_t *table

        int8_t[:] result_keys
        int64_t[:] result_counts

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k

    table = kh_init_int8()
    build_count_table_int8(values, table, dropna)

    result_keys = np.empty(table.n_occupied, 'int8')
    result_counts = np.zeros(table.n_occupied, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int8(table, k):
                result_keys[i] = table.keys[k]
                result_counts[i] = table.vals[k]
                i += 1

    kh_destroy_int8(table)

    return np.asarray(result_keys), np.asarray(result_counts)


@cython.wraparound(False)
@cython.boundscheck(False)
def duplicated_int8(const int8_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int8_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int8_t *table = kh_init_int8()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int8(table, min(n, SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                kh_put_int8(table, values[i], &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                kh_put_int8(table, values[i], &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = values[i]
                k = kh_get_int8(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int8(table, value, &ret)
                    table.keys[k] = value
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int8(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
def ismember_int8(const int8_t[:] arr, const int8_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : int8 ndarray
    values : int8 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        int8_t val
        kh_int8_t *table = kh_init_int8()

    # construct the table
    n = len(values)
    kh_resize_int8(table, n)

    with nogil:
        for i in range(n):
            kh_put_int8(table, values[i], &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = arr[i]
            k = kh_get_int8(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int8(table)
    return result.view(np.bool_)


# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_float64(float64_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_float64_t *table
        ndarray[float64_t] modes

    table = kh_init_float64()
    build_count_table_float64(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.float64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_float64(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_float64(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_float32(float32_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_float32_t *table
        ndarray[float32_t] modes

    table = kh_init_float32()
    build_count_table_float32(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.float32)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_float32(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_float32(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_int64(int64_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_int64_t *table
        ndarray[int64_t] modes

    table = kh_init_int64()
    build_count_table_int64(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.int64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int64(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_int64(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_int32(int32_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_int32_t *table
        ndarray[int32_t] modes

    table = kh_init_int32()
    build_count_table_int32(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.int32)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int32(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_int32(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_int16(int16_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_int16_t *table
        ndarray[int16_t] modes

    table = kh_init_int16()
    build_count_table_int16(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.int16)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int16(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_int16(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_int8(int8_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_int8_t *table
        ndarray[int8_t] modes

    table = kh_init_int8()
    build_count_table_int8(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.int8)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_int8(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_int8(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_uint64(uint64_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_uint64_t *table
        ndarray[uint64_t] modes

    table = kh_init_uint64()
    build_count_table_uint64(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.uint64)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint64(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_uint64(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_uint32(uint32_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_uint32_t *table
        ndarray[uint32_t] modes

    table = kh_init_uint32()
    build_count_table_uint32(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.uint32)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint32(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_uint32(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_uint16(uint16_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_uint16_t *table
        ndarray[uint16_t] modes

    table = kh_init_uint16()
    build_count_table_uint16(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.uint16)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint16(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_uint16(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_uint8(uint8_t[:] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_uint8_t *table
        ndarray[uint8_t] modes

    table = kh_init_uint8()
    build_count_table_uint8(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.uint8)

    with nogil:
        for k in range(table.n_buckets):
            if kh_exist_uint8(table, k):
                count = table.vals[k]
                if count == max_count:
                    j += 1
                elif count > max_count:
                    max_count = count
                    j = 0
                else:
                    continue

                modes[j] = table.keys[k]

    kh_destroy_uint8(table)

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)


def mode_object(ndarray[object] values, bint dropna):
    cdef:
        int count, max_count = 1
        int j = -1  # so you can do +=
        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        kh_pymap_t *table
        ndarray[object] modes

    table = kh_init_pymap()
    build_count_table_object(values, table, dropna)

    modes = np.empty(table.n_buckets, dtype=np.object_)

    for k in range(table.n_buckets):
        if kh_exist_pymap(table, k):
            count = table.vals[k]

            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = <object>table.keys[k]

    kh_destroy_pymap(table)

    return modes[:j + 1]
