"""
Template for each `dtype` helper function in `np.random.randint`.
"""

def _rand_bool(npy_bool low, npy_bool high, size, rngstate):
    """
    _rand_bool(low, high, size, rngstate)

    Return random np.bool_ integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.bool_ type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.bool_
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_bool off, rng, buf
    cdef npy_bool *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_bool>(low)
    rng = <npy_bool>(high) - <npy_bool>(low)

    if size is None:
        rk_random_bool(off, rng, 1, &buf, state)
        return np.bool_(<npy_bool>buf)
    else:
        array = <ndarray>np.empty(size, np.bool_)
        cnt = PyArray_SIZE(array)
        array_data = <npy_bool *>PyArray_DATA(array)
        with nogil:
            rk_random_bool(off, rng, cnt, array_data, state)
        return array

def _rand_int8(npy_int8 low, npy_int8 high, size, rngstate):
    """
    _rand_int8(low, high, size, rngstate)

    Return random np.int8 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.int8 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.int8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint8 off, rng, buf
    cdef npy_uint8 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint8>(low)
    rng = <npy_uint8>(high) - <npy_uint8>(low)

    if size is None:
        rk_random_uint8(off, rng, 1, &buf, state)
        return np.int8(<npy_int8>buf)
    else:
        array = <ndarray>np.empty(size, np.int8)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint8 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint8(off, rng, cnt, array_data, state)
        return array

def _rand_int16(npy_int16 low, npy_int16 high, size, rngstate):
    """
    _rand_int16(low, high, size, rngstate)

    Return random np.int16 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.int16 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.int16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint16 off, rng, buf
    cdef npy_uint16 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint16>(low)
    rng = <npy_uint16>(high) - <npy_uint16>(low)

    if size is None:
        rk_random_uint16(off, rng, 1, &buf, state)
        return np.int16(<npy_int16>buf)
    else:
        array = <ndarray>np.empty(size, np.int16)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint16 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint16(off, rng, cnt, array_data, state)
        return array

def _rand_int32(npy_int32 low, npy_int32 high, size, rngstate):
    """
    _rand_int32(low, high, size, rngstate)

    Return random np.int32 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.int32 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.int32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint32 off, rng, buf
    cdef npy_uint32 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint32>(low)
    rng = <npy_uint32>(high) - <npy_uint32>(low)

    if size is None:
        rk_random_uint32(off, rng, 1, &buf, state)
        return np.int32(<npy_int32>buf)
    else:
        array = <ndarray>np.empty(size, np.int32)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint32 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint32(off, rng, cnt, array_data, state)
        return array

def _rand_int64(npy_int64 low, npy_int64 high, size, rngstate):
    """
    _rand_int64(low, high, size, rngstate)

    Return random np.int64 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.int64 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.int64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint64 off, rng, buf
    cdef npy_uint64 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint64>(low)
    rng = <npy_uint64>(high) - <npy_uint64>(low)

    if size is None:
        rk_random_uint64(off, rng, 1, &buf, state)
        return np.int64(<npy_int64>buf)
    else:
        array = <ndarray>np.empty(size, np.int64)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint64 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint64(off, rng, cnt, array_data, state)
        return array

def _rand_uint8(npy_uint8 low, npy_uint8 high, size, rngstate):
    """
    _rand_uint8(low, high, size, rngstate)

    Return random np.uint8 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.uint8 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.uint8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint8 off, rng, buf
    cdef npy_uint8 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint8>(low)
    rng = <npy_uint8>(high) - <npy_uint8>(low)

    if size is None:
        rk_random_uint8(off, rng, 1, &buf, state)
        return np.uint8(<npy_uint8>buf)
    else:
        array = <ndarray>np.empty(size, np.uint8)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint8 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint8(off, rng, cnt, array_data, state)
        return array

def _rand_uint16(npy_uint16 low, npy_uint16 high, size, rngstate):
    """
    _rand_uint16(low, high, size, rngstate)

    Return random np.uint16 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.uint16 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.uint16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint16 off, rng, buf
    cdef npy_uint16 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint16>(low)
    rng = <npy_uint16>(high) - <npy_uint16>(low)

    if size is None:
        rk_random_uint16(off, rng, 1, &buf, state)
        return np.uint16(<npy_uint16>buf)
    else:
        array = <ndarray>np.empty(size, np.uint16)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint16 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint16(off, rng, cnt, array_data, state)
        return array

def _rand_uint32(npy_uint32 low, npy_uint32 high, size, rngstate):
    """
    _rand_uint32(low, high, size, rngstate)

    Return random np.uint32 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.uint32 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.uint32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint32 off, rng, buf
    cdef npy_uint32 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint32>(low)
    rng = <npy_uint32>(high) - <npy_uint32>(low)

    if size is None:
        rk_random_uint32(off, rng, 1, &buf, state)
        return np.uint32(<npy_uint32>buf)
    else:
        array = <ndarray>np.empty(size, np.uint32)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint32 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint32(off, rng, cnt, array_data, state)
        return array

def _rand_uint64(npy_uint64 low, npy_uint64 high, size, rngstate):
    """
    _rand_uint64(low, high, size, rngstate)

    Return random np.uint64 integers between ``low`` and ``high``, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [``low``, ``high``). On entry the arguments are presumed
    to have been validated for size and order for the np.uint64 type.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Highest (signed) integer to be drawn from the distribution.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    rngstate : encapsulated pointer to rk_state
        The specific type depends on the python version. In Python 2 it is
        a PyCObject, in Python 3 a PyCapsule object.

    Returns
    -------
    out : python integer or ndarray of np.uint64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    """
    cdef npy_uint64 off, rng, buf
    cdef npy_uint64 *out
    cdef ndarray array "arrayObject"
    cdef npy_intp cnt
    cdef rk_state *state = <rk_state *>PyCapsule_GetPointer(rngstate, NULL)

    off = <npy_uint64>(low)
    rng = <npy_uint64>(high) - <npy_uint64>(low)

    if size is None:
        rk_random_uint64(off, rng, 1, &buf, state)
        return np.uint64(<npy_uint64>buf)
    else:
        array = <ndarray>np.empty(size, np.uint64)
        cnt = PyArray_SIZE(array)
        array_data = <npy_uint64 *>PyArray_DATA(array)
        with nogil:
            rk_random_uint64(off, rng, cnt, array_data, state)
        return array
