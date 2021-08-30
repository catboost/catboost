"""
Template for each `dtype` helper function for hashtable

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_complex128(const complex128_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_complex128_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        khcomplex128_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Complex128Vector()
    table = kh_init_complex128()

    kh_resize_complex128(table, n)

    for i in range(n):
        val = to_khcomplex128_t(values[i])

        if not is_nan_khcomplex128_t(val) or not dropna:
            k = kh_get_complex128(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_complex128(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_complex128(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_complex128(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_complex128(const complex128_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        khcomplex128_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_complex128_t *table = kh_init_complex128()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_complex128(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = to_khcomplex128_t(values[i])
                kh_put_complex128(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = to_khcomplex128_t(values[i])
                kh_put_complex128(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = to_khcomplex128_t(values[i])
                k = kh_get_complex128(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_complex128(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_complex128(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_complex128(const complex128_t[:] arr, const complex128_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : complex128 ndarray
    values : complex128 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        khcomplex128_t val
        kh_complex128_t *table = kh_init_complex128()

    # construct the table
    n = len(values)
    kh_resize_complex128(table, n)

    with nogil:
        for i in range(n):
            val = to_khcomplex128_t(values[i])
            kh_put_complex128(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = to_khcomplex128_t(arr[i])
            k = kh_get_complex128(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_complex128(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_complex128(const complex128_t[:] values, bint dropna):
    cdef:
        complex128_t[:] keys
        ndarray[complex128_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_complex128(values, dropna)

    modes = np.empty(len(keys), dtype=np.complex128)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_complex64(const complex64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_complex64_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        khcomplex64_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Complex64Vector()
    table = kh_init_complex64()

    kh_resize_complex64(table, n)

    for i in range(n):
        val = to_khcomplex64_t(values[i])

        if not is_nan_khcomplex64_t(val) or not dropna:
            k = kh_get_complex64(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_complex64(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_complex64(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_complex64(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_complex64(const complex64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        khcomplex64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_complex64_t *table = kh_init_complex64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_complex64(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = to_khcomplex64_t(values[i])
                kh_put_complex64(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = to_khcomplex64_t(values[i])
                kh_put_complex64(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = to_khcomplex64_t(values[i])
                k = kh_get_complex64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_complex64(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_complex64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_complex64(const complex64_t[:] arr, const complex64_t[:] values):
    """
    Return boolean of values in arr on an
    element by-element basis

    Parameters
    ----------
    arr : complex64 ndarray
    values : complex64 ndarray

    Returns
    -------
    boolean ndarry len of (arr)
    """
    cdef:
        Py_ssize_t i, n
        khiter_t k
        int ret = 0
        ndarray[uint8_t] result
        khcomplex64_t val
        kh_complex64_t *table = kh_init_complex64()

    # construct the table
    n = len(values)
    kh_resize_complex64(table, n)

    with nogil:
        for i in range(n):
            val = to_khcomplex64_t(values[i])
            kh_put_complex64(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = to_khcomplex64_t(arr[i])
            k = kh_get_complex64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_complex64(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_complex64(const complex64_t[:] values, bint dropna):
    cdef:
        complex64_t[:] keys
        ndarray[complex64_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_complex64(values, dropna)

    modes = np.empty(len(keys), dtype=np.complex64)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_float64(const float64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_float64_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        float64_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Float64Vector()
    table = kh_init_float64()

    kh_resize_float64(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_float64_t(val) or not dropna:
            k = kh_get_float64(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_float64(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_float64(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_float64(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_float64(const float64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        float64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_float64_t *table = kh_init_float64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_float64(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_float64(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_float64(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_float64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_float64(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_float64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_float64(const float64_t[:] arr, const float64_t[:] values):
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
            val = (values[i])
            kh_put_float64(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_float64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_float64(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_float64(const float64_t[:] values, bint dropna):
    cdef:
        float64_t[:] keys
        ndarray[float64_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_float64(values, dropna)

    modes = np.empty(len(keys), dtype=np.float64)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_float32(const float32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_float32_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        float32_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Float32Vector()
    table = kh_init_float32()

    kh_resize_float32(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_float32_t(val) or not dropna:
            k = kh_get_float32(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_float32(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_float32(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_float32(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_float32(const float32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        float32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_float32_t *table = kh_init_float32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_float32(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_float32(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_float32(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_float32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_float32(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_float32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_float32(const float32_t[:] arr, const float32_t[:] values):
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
            val = (values[i])
            kh_put_float32(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_float32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_float32(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_float32(const float32_t[:] values, bint dropna):
    cdef:
        float32_t[:] keys
        ndarray[float32_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_float32(values, dropna)

    modes = np.empty(len(keys), dtype=np.float32)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_uint64(const uint64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_uint64_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        uint64_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = UInt64Vector()
    table = kh_init_uint64()

    kh_resize_uint64(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_uint64_t(val) or not dropna:
            k = kh_get_uint64(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_uint64(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_uint64(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_uint64(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_uint64(const uint64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint64_t *table = kh_init_uint64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint64(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_uint64(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_uint64(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_uint64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint64(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_uint64(const uint64_t[:] arr, const uint64_t[:] values):
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
            val = (values[i])
            kh_put_uint64(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_uint64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint64(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_uint64(const uint64_t[:] values, bint dropna):
    cdef:
        uint64_t[:] keys
        ndarray[uint64_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_uint64(values, dropna)

    modes = np.empty(len(keys), dtype=np.uint64)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_uint32(const uint32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_uint32_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        uint32_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = UInt32Vector()
    table = kh_init_uint32()

    kh_resize_uint32(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_uint32_t(val) or not dropna:
            k = kh_get_uint32(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_uint32(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_uint32(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_uint32(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_uint32(const uint32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint32_t *table = kh_init_uint32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint32(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_uint32(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_uint32(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_uint32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint32(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_uint32(const uint32_t[:] arr, const uint32_t[:] values):
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
            val = (values[i])
            kh_put_uint32(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_uint32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint32(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_uint32(const uint32_t[:] values, bint dropna):
    cdef:
        uint32_t[:] keys
        ndarray[uint32_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_uint32(values, dropna)

    modes = np.empty(len(keys), dtype=np.uint32)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_uint16(const uint16_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_uint16_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        uint16_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = UInt16Vector()
    table = kh_init_uint16()

    kh_resize_uint16(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_uint16_t(val) or not dropna:
            k = kh_get_uint16(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_uint16(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_uint16(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_uint16(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_uint16(const uint16_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint16_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint16_t *table = kh_init_uint16()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint16(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_uint16(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_uint16(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_uint16(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint16(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint16(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_uint16(const uint16_t[:] arr, const uint16_t[:] values):
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
            val = (values[i])
            kh_put_uint16(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_uint16(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint16(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_uint16(const uint16_t[:] values, bint dropna):
    cdef:
        uint16_t[:] keys
        ndarray[uint16_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_uint16(values, dropna)

    modes = np.empty(len(keys), dtype=np.uint16)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_uint8(const uint8_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_uint8_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        uint8_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = UInt8Vector()
    table = kh_init_uint8()

    kh_resize_uint8(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_uint8_t(val) or not dropna:
            k = kh_get_uint8(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_uint8(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_uint8(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_uint8(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_uint8(const uint8_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        uint8_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_uint8_t *table = kh_init_uint8()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_uint8(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_uint8(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_uint8(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_uint8(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_uint8(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_uint8(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_uint8(const uint8_t[:] arr, const uint8_t[:] values):
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
            val = (values[i])
            kh_put_uint8(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_uint8(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_uint8(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_uint8(const uint8_t[:] values, bint dropna):
    cdef:
        uint8_t[:] keys
        ndarray[uint8_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_uint8(values, dropna)

    modes = np.empty(len(keys), dtype=np.uint8)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_object(ndarray[object] values, bint dropna, navalue=np.NaN):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_pymap_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        object val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = ObjectVector()
    table = kh_init_pymap()

    kh_resize_pymap(table, n // 10)

    for i in range(n):
        val = values[i]
        is_null = checknull(val)
        if not is_null or not dropna:
            # all nas become the same representative:
            if is_null:
                val = navalue
            k = kh_get_pymap(table, <PyObject*>val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_pymap(table, <PyObject*>val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_pymap(table, result_keys.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_pymap(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_object(ndarray[object] values, object keep='first'):
    cdef:
        int ret = 0
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_pymap_t *table = kh_init_pymap()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_pymap(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

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
                table.vals[k] = i
                out[i] = 0
    kh_destroy_pymap(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_object(ndarray[object] arr, ndarray[object] values):
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

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_object(ndarray[object] values, bint dropna):
    cdef:
        ndarray[object] keys
        ndarray[object] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_object(values, dropna)

    modes = np.empty(len(keys), dtype=np.object_)
    for k in range(len(keys)):
        count = counts[k]
        if count == max_count:
            j += 1
        elif count > max_count:
            max_count = count
            j = 0
        else:
            continue

        modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_int64(const int64_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_int64_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        int64_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Int64Vector()
    table = kh_init_int64()

    kh_resize_int64(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_int64_t(val) or not dropna:
            k = kh_get_int64(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_int64(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_int64(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_int64(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_int64(const int64_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int64_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int64_t *table = kh_init_int64()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int64(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_int64(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_int64(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_int64(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int64(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int64(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_int64(const int64_t[:] arr, const int64_t[:] values):
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
            val = (values[i])
            kh_put_int64(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_int64(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int64(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_int64(const int64_t[:] values, bint dropna):
    cdef:
        int64_t[:] keys
        ndarray[int64_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_int64(values, dropna)

    modes = np.empty(len(keys), dtype=np.int64)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_int32(const int32_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_int32_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        int32_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Int32Vector()
    table = kh_init_int32()

    kh_resize_int32(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_int32_t(val) or not dropna:
            k = kh_get_int32(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_int32(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_int32(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_int32(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_int32(const int32_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int32_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int32_t *table = kh_init_int32()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int32(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_int32(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_int32(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_int32(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int32(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int32(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_int32(const int32_t[:] arr, const int32_t[:] values):
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
            val = (values[i])
            kh_put_int32(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_int32(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int32(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_int32(const int32_t[:] values, bint dropna):
    cdef:
        int32_t[:] keys
        ndarray[int32_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_int32(values, dropna)

    modes = np.empty(len(keys), dtype=np.int32)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_int16(const int16_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_int16_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        int16_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Int16Vector()
    table = kh_init_int16()

    kh_resize_int16(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_int16_t(val) or not dropna:
            k = kh_get_int16(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_int16(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_int16(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_int16(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_int16(const int16_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int16_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int16_t *table = kh_init_int16()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int16(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_int16(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_int16(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_int16(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int16(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int16(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_int16(const int16_t[:] arr, const int16_t[:] values):
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
            val = (values[i])
            kh_put_int16(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_int16(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int16(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_int16(const int16_t[:] values, bint dropna):
    cdef:
        int16_t[:] keys
        ndarray[int16_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_int16(values, dropna)

    modes = np.empty(len(keys), dtype=np.int16)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef value_count_int8(const int8_t[:] values, bint dropna):
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = len(values)
        kh_int8_t *table

        # Don't use Py_ssize_t, since table.n_buckets is unsigned
        khiter_t k
        bint is_null

        int8_t val

        int ret = 0

    # we track the order in which keys are first seen (GH39009),
    # khash-map isn't insertion-ordered, thus:
    #    table maps keys to counts
    #    result_keys remembers the original order of keys

    result_keys = Int8Vector()
    table = kh_init_int8()

    kh_resize_int8(table, n)

    for i in range(n):
        val = (values[i])

        if not is_nan_int8_t(val) or not dropna:
            k = kh_get_int8(table, val)
            if k != table.n_buckets:
                table.vals[k] += 1
            else:
                k = kh_put_int8(table, val, &ret)
                table.vals[k] = 1
                result_keys.append(val)

    # collect counts in the order corresponding to result_keys:
    cdef int64_t[:] result_counts = np.empty(table.size, dtype=np.int64)
    for i in range(table.size):
        k = kh_get_int8(table, result_keys.data.data[i])
        result_counts[i] = table.vals[k]

    kh_destroy_int8(table)

    return result_keys.to_array(), result_counts.base


@cython.wraparound(False)
@cython.boundscheck(False)
cdef duplicated_int8(const int8_t[:] values, object keep='first'):
    cdef:
        int ret = 0
        int8_t value
        Py_ssize_t i, n = len(values)
        khiter_t k
        kh_int8_t *table = kh_init_int8()
        ndarray[uint8_t, ndim=1, cast=True] out = np.empty(n, dtype='bool')

    kh_resize_int8(table, min(kh_needed_n_buckets(n), SIZE_HINT_LIMIT))

    if keep not in ('last', 'first', False):
        raise ValueError('keep must be either "first", "last" or False')

    if keep == 'last':
        with nogil:
            for i in range(n - 1, -1, -1):
                # equivalent: range(n)[::-1], which cython doesn't like in nogil
                value = (values[i])
                kh_put_int8(table, value, &ret)
                out[i] = ret == 0
    elif keep == 'first':
        with nogil:
            for i in range(n):
                value = (values[i])
                kh_put_int8(table, value, &ret)
                out[i] = ret == 0
    else:
        with nogil:
            for i in range(n):
                value = (values[i])
                k = kh_get_int8(table, value)
                if k != table.n_buckets:
                    out[table.vals[k]] = 1
                    out[i] = 1
                else:
                    k = kh_put_int8(table, value, &ret)
                    table.vals[k] = i
                    out[i] = 0
    kh_destroy_int8(table)
    return out


# ----------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef ismember_int8(const int8_t[:] arr, const int8_t[:] values):
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
            val = (values[i])
            kh_put_int8(table, val, &ret)

    # test membership
    n = len(arr)
    result = np.empty(n, dtype=np.uint8)

    with nogil:
        for i in range(n):
            val = (arr[i])
            k = kh_get_int8(table, val)
            result[i] = (k != table.n_buckets)

    kh_destroy_int8(table)
    return result.view(np.bool_)

# ----------------------------------------------------------------------
# Mode Computations
# ----------------------------------------------------------------------


@cython.wraparound(False)
@cython.boundscheck(False)
cdef mode_int8(const int8_t[:] values, bint dropna):
    cdef:
        int8_t[:] keys
        ndarray[int8_t] modes
        int64_t[:] counts
        int64_t count, max_count = -1
        Py_ssize_t k, j = 0

    keys, counts = value_count_int8(values, dropna)

    modes = np.empty(len(keys), dtype=np.int8)
    with nogil:
        for k in range(len(keys)):
            count = counts[k]
            if count == max_count:
                j += 1
            elif count > max_count:
                max_count = count
                j = 0
            else:
                continue

            modes[j] = keys[k]

    return modes[:j + 1]


ctypedef fused htfunc_t:
    complex128_t
    complex64_t
    float64_t
    float32_t
    uint64_t
    uint32_t
    uint16_t
    uint8_t
    int64_t
    int32_t
    int16_t
    int8_t
    object


cpdef value_count(ndarray[htfunc_t] values, bint dropna):
    if htfunc_t is object:
        return value_count_object(values, dropna)

    elif htfunc_t is int8_t:
        return value_count_int8(values, dropna)
    elif htfunc_t is int16_t:
        return value_count_int16(values, dropna)
    elif htfunc_t is int32_t:
        return value_count_int32(values, dropna)
    elif htfunc_t is int64_t:
        return value_count_int64(values, dropna)

    elif htfunc_t is uint8_t:
        return value_count_uint8(values, dropna)
    elif htfunc_t is uint16_t:
        return value_count_uint16(values, dropna)
    elif htfunc_t is uint32_t:
        return value_count_uint32(values, dropna)
    elif htfunc_t is uint64_t:
        return value_count_uint64(values, dropna)

    elif htfunc_t is float64_t:
        return value_count_float64(values, dropna)
    elif htfunc_t is float32_t:
        return value_count_float32(values, dropna)

    elif htfunc_t is complex128_t:
        return value_count_complex128(values, dropna)
    elif htfunc_t is complex64_t:
        return value_count_complex64(values, dropna)

    else:
        raise TypeError(values.dtype)


cpdef duplicated(ndarray[htfunc_t] values, object keep="first"):
    if htfunc_t is object:
        return duplicated_object(values, keep)

    elif htfunc_t is int8_t:
        return duplicated_int8(values, keep)
    elif htfunc_t is int16_t:
        return duplicated_int16(values, keep)
    elif htfunc_t is int32_t:
        return duplicated_int32(values, keep)
    elif htfunc_t is int64_t:
        return duplicated_int64(values, keep)

    elif htfunc_t is uint8_t:
        return duplicated_uint8(values, keep)
    elif htfunc_t is uint16_t:
        return duplicated_uint16(values, keep)
    elif htfunc_t is uint32_t:
        return duplicated_uint32(values, keep)
    elif htfunc_t is uint64_t:
        return duplicated_uint64(values, keep)

    elif htfunc_t is float64_t:
        return duplicated_float64(values, keep)
    elif htfunc_t is float32_t:
        return duplicated_float32(values, keep)

    elif htfunc_t is complex128_t:
        return duplicated_complex128(values, keep)
    elif htfunc_t is complex64_t:
        return duplicated_complex64(values, keep)

    else:
        raise TypeError(values.dtype)


cpdef ismember(ndarray[htfunc_t] arr, ndarray[htfunc_t] values):
    if htfunc_t is object:
        return ismember_object(arr, values)

    elif htfunc_t is int8_t:
        return ismember_int8(arr, values)
    elif htfunc_t is int16_t:
        return ismember_int16(arr, values)
    elif htfunc_t is int32_t:
        return ismember_int32(arr, values)
    elif htfunc_t is int64_t:
        return ismember_int64(arr, values)

    elif htfunc_t is uint8_t:
        return ismember_uint8(arr, values)
    elif htfunc_t is uint16_t:
        return ismember_uint16(arr, values)
    elif htfunc_t is uint32_t:
        return ismember_uint32(arr, values)
    elif htfunc_t is uint64_t:
        return ismember_uint64(arr, values)

    elif htfunc_t is float64_t:
        return ismember_float64(arr, values)
    elif htfunc_t is float32_t:
        return ismember_float32(arr, values)

    elif htfunc_t is complex128_t:
        return ismember_complex128(arr, values)
    elif htfunc_t is complex64_t:
        return ismember_complex64(arr, values)

    else:
        raise TypeError(values.dtype)


cpdef mode(ndarray[htfunc_t] values, bint dropna):
    if htfunc_t is object:
        return mode_object(values, dropna)

    elif htfunc_t is int8_t:
        return mode_int8(values, dropna)
    elif htfunc_t is int16_t:
        return mode_int16(values, dropna)
    elif htfunc_t is int32_t:
        return mode_int32(values, dropna)
    elif htfunc_t is int64_t:
        return mode_int64(values, dropna)

    elif htfunc_t is uint8_t:
        return mode_uint8(values, dropna)
    elif htfunc_t is uint16_t:
        return mode_uint16(values, dropna)
    elif htfunc_t is uint32_t:
        return mode_uint32(values, dropna)
    elif htfunc_t is uint64_t:
        return mode_uint64(values, dropna)

    elif htfunc_t is float64_t:
        return mode_float64(values, dropna)
    elif htfunc_t is float32_t:
        return mode_float32(values, dropna)

    elif htfunc_t is complex128_t:
        return mode_complex128(values, dropna)
    elif htfunc_t is complex64_t:
        return mode_complex64(values, dropna)

    else:
        raise TypeError(values.dtype)
