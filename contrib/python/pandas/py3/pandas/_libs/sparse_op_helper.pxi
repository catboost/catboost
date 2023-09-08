"""
Template for each `dtype` helper function for sparse ops

WARNING: DO NOT edit .pxi FILE directly, .pxi is generated from .pxi.in
"""

# ----------------------------------------------------------------------
# Sparse op
# ----------------------------------------------------------------------

ctypedef fused sparse_t:
    float64_t
    int64_t


cdef float64_t __div__(sparse_t a, sparse_t b):
    if b == 0:
        if a > 0:
            return INF
        elif a < 0:
            return -INF
        else:
            return NaN
    else:
        return float(a) / b


cdef float64_t __truediv__(sparse_t a, sparse_t b):
    return __div__(a, b)


cdef sparse_t __mod__(sparse_t a, sparse_t b):
    if b == 0:
        if sparse_t is float64_t:
            return NaN
        else:
            return 0
    else:
        return a % b


cdef sparse_t __floordiv__(sparse_t a, sparse_t b):
    if b == 0:
        if sparse_t is float64_t:
            # Match non-sparse Series behavior implemented in mask_zero_div_zero
            if a > 0:
                return INF
            elif a < 0:
                return -INF
            return NaN
        else:
            return 0
    else:
        return a // b


# ----------------------------------------------------------------------
# sparse array op
# ----------------------------------------------------------------------
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_add_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] + y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill + yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_add_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] + y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

    return out, out_index, xfill + yfill


cpdef sparse_add_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_add_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_add_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_add_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] + y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill + yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_add_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] + y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] + yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill + y[yi]
            yi += 1

    return out, out_index, xfill + yfill


cpdef sparse_add_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_add_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_add_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_sub_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] - y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill - yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_sub_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] - y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

    return out, out_index, xfill - yfill


cpdef sparse_sub_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_sub_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_sub_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_sub_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] - y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill - yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_sub_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] - y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] - yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill - y[yi]
            yi += 1

    return out, out_index, xfill - yfill


cpdef sparse_sub_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_sub_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_sub_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_mul_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] * y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill * yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_mul_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] * y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

    return out, out_index, xfill * yfill


cpdef sparse_mul_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_mul_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_mul_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_mul_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] * y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill * yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_mul_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] * y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] * yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill * y[yi]
            yi += 1

    return out, out_index, xfill * yfill


cpdef sparse_mul_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_mul_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_mul_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_div_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __div__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __div__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_div_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __div__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

    return out, out_index, __div__(xfill, yfill)


cpdef sparse_div_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_div_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_div_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_div_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __div__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __div__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_div_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __div__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __div__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __div__(xfill, y[yi])
            yi += 1

    return out, out_index, __div__(xfill, yfill)


cpdef sparse_div_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_div_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_div_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_mod_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __mod__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __mod__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_mod_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __mod__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

    return out, out_index, __mod__(xfill, yfill)


cpdef sparse_mod_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_mod_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_mod_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_mod_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __mod__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __mod__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_mod_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __mod__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __mod__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __mod__(xfill, y[yi])
            yi += 1

    return out, out_index, __mod__(xfill, yfill)


cpdef sparse_mod_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_mod_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_mod_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_truediv_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __truediv__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __truediv__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_truediv_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __truediv__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

    return out, out_index, __truediv__(xfill, yfill)


cpdef sparse_truediv_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_truediv_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_truediv_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_truediv_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __truediv__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __truediv__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_truediv_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __truediv__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __truediv__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __truediv__(xfill, y[yi])
            yi += 1

    return out, out_index, __truediv__(xfill, yfill)


cpdef sparse_truediv_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_truediv_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_truediv_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_floordiv_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __floordiv__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __floordiv__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_floordiv_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __floordiv__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

    return out, out_index, __floordiv__(xfill, yfill)


cpdef sparse_floordiv_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_floordiv_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_floordiv_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_floordiv_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __floordiv__(x[xi], y[yi])
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, __floordiv__(xfill, yfill)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_floordiv_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = __floordiv__(x[xi], y[yi])
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = __floordiv__(x[xi], yfill)
            xi += 1
        else:
            # use x fill value
            out[out_i] = __floordiv__(xfill, y[yi])
            yi += 1

    return out, out_index, __floordiv__(xfill, yfill)


cpdef sparse_floordiv_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_floordiv_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_floordiv_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.cpow(True) # Cython 3 matches Python pow, which isn't what we want here
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_pow_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ** y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill ** yfill
@cython.cpow(True) # Cython 3 matches Python pow, which isn't what we want here
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_pow_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[float64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.float64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ** y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

    return out, out_index, xfill ** yfill


cpdef sparse_pow_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_pow_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_pow_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.cpow(True) # Cython 3 matches Python pow, which isn't what we want here
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_pow_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ** y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill ** yfill
@cython.cpow(True) # Cython 3 matches Python pow, which isn't what we want here
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_pow_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[int64_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.int64)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ** y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ** yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill ** y[yi]
            yi += 1

    return out, out_index, xfill ** yfill


cpdef sparse_pow_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_pow_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_pow_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_eq_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] == y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill == yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_eq_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] == y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

    return out, out_index, xfill == yfill


cpdef sparse_eq_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_eq_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_eq_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_eq_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] == y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill == yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_eq_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] == y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] == yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill == y[yi]
            yi += 1

    return out, out_index, xfill == yfill


cpdef sparse_eq_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_eq_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_eq_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_ne_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] != y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill != yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_ne_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] != y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

    return out, out_index, xfill != yfill


cpdef sparse_ne_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_ne_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_ne_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_ne_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] != y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill != yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_ne_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] != y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] != yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill != y[yi]
            yi += 1

    return out, out_index, xfill != yfill


cpdef sparse_ne_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_ne_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_ne_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_lt_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] < y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill < yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_lt_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] < y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

    return out, out_index, xfill < yfill


cpdef sparse_lt_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_lt_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_lt_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_lt_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] < y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill < yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_lt_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] < y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] < yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill < y[yi]
            yi += 1

    return out, out_index, xfill < yfill


cpdef sparse_lt_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_lt_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_lt_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_gt_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] > y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill > yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_gt_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] > y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

    return out, out_index, xfill > yfill


cpdef sparse_gt_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_gt_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_gt_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_gt_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] > y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill > yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_gt_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] > y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] > yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill > y[yi]
            yi += 1

    return out, out_index, xfill > yfill


cpdef sparse_gt_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_gt_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_gt_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_le_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] <= y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill <= yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_le_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] <= y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

    return out, out_index, xfill <= yfill


cpdef sparse_le_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_le_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_le_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_le_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] <= y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill <= yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_le_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] <= y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] <= yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill <= y[yi]
            yi += 1

    return out, out_index, xfill <= yfill


cpdef sparse_le_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_le_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_le_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_ge_float64(float64_t[:] x_,
                                                BlockIndex xindex,
                                                float64_t xfill,
                                                float64_t[:] y_,
                                                BlockIndex yindex,
                                                float64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] >= y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill >= yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_ge_float64(float64_t[:] x_,
                                              IntIndex xindex,
                                              float64_t xfill,
                                              float64_t[:] y_,
                                              IntIndex yindex,
                                              float64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        float64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] >= y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

    return out, out_index, xfill >= yfill


cpdef sparse_ge_float64(float64_t[:] x,
                                  SparseIndex xindex, float64_t xfill,
                                  float64_t[:] y,
                                  SparseIndex yindex, float64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_ge_float64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_ge_float64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_ge_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] >= y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill >= yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_ge_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] >= y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] >= yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill >= y[yi]
            yi += 1

    return out, out_index, xfill >= yfill


cpdef sparse_ge_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_ge_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_ge_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_and_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] & y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill & yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_and_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] & y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

    return out, out_index, xfill & yfill


cpdef sparse_and_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_and_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_and_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_and_uint8(uint8_t[:] x_,
                                                BlockIndex xindex,
                                                uint8_t xfill,
                                                uint8_t[:] y_,
                                                BlockIndex yindex,
                                                uint8_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] & y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill & yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_and_uint8(uint8_t[:] x_,
                                              IntIndex xindex,
                                              uint8_t xfill,
                                              uint8_t[:] y_,
                                              IntIndex yindex,
                                              uint8_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] & y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] & yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill & y[yi]
            yi += 1

    return out, out_index, xfill & yfill


cpdef sparse_and_uint8(uint8_t[:] x,
                                  SparseIndex xindex, uint8_t xfill,
                                  uint8_t[:] y,
                                  SparseIndex yindex, uint8_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_and_uint8(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_and_uint8(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_or_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] | y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill | yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_or_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] | y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

    return out, out_index, xfill | yfill


cpdef sparse_or_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_or_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_or_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_or_uint8(uint8_t[:] x_,
                                                BlockIndex xindex,
                                                uint8_t xfill,
                                                uint8_t[:] y_,
                                                BlockIndex yindex,
                                                uint8_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] | y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill | yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_or_uint8(uint8_t[:] x_,
                                              IntIndex xindex,
                                              uint8_t xfill,
                                              uint8_t[:] y_,
                                              IntIndex yindex,
                                              uint8_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] | y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] | yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill | y[yi]
            yi += 1

    return out, out_index, xfill | yfill


cpdef sparse_or_uint8(uint8_t[:] x,
                                  SparseIndex xindex, uint8_t xfill,
                                  uint8_t[:] y,
                                  SparseIndex yindex, uint8_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_or_uint8(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_or_uint8(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_xor_int64(int64_t[:] x_,
                                                BlockIndex xindex,
                                                int64_t xfill,
                                                int64_t[:] y_,
                                                BlockIndex yindex,
                                                int64_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ^ y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill ^ yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_xor_int64(int64_t[:] x_,
                                              IntIndex xindex,
                                              int64_t xfill,
                                              int64_t[:] y_,
                                              IntIndex yindex,
                                              int64_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        int64_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ^ y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

    return out, out_index, xfill ^ yfill


cpdef sparse_xor_int64(int64_t[:] x,
                                  SparseIndex xindex, int64_t xfill,
                                  int64_t[:] y,
                                  SparseIndex yindex, int64_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_xor_int64(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_xor_int64(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple block_op_xor_uint8(uint8_t[:] x_,
                                                BlockIndex xindex,
                                                uint8_t xfill,
                                                uint8_t[:] y_,
                                                BlockIndex yindex,
                                                uint8_t yfill):
    """
    Binary operator on BlockIndex objects with fill values
    """

    cdef:
        BlockIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xbp = 0, ybp = 0  # block positions
        int32_t xloc, yloc
        Py_ssize_t xblock = 0, yblock = 0  # block numbers

        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # to suppress Cython warning
    x = x_
    y = y_

    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    # Wow, what a hack job. Need to do something about this

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if yblock == yindex.nblocks:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
            continue

        if xblock == xindex.nblocks:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0
            continue

        yloc = yindex.locbuf[yblock] + ybp
        xloc = xindex.locbuf[xblock] + xbp

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ^ y[yi]
            xi += 1
            yi += 1

            # advance both locations
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0

            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1

            # advance x location
            xbp += 1
            if xbp == xindex.lenbuf[xblock]:
                xblock += 1
                xbp = 0
        else:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

            # advance y location
            ybp += 1
            if ybp == yindex.lenbuf[yblock]:
                yblock += 1
                ybp = 0

    return out, out_index, xfill ^ yfill
@cython.wraparound(False)
@cython.boundscheck(False)
cdef tuple int_op_xor_uint8(uint8_t[:] x_,
                                              IntIndex xindex,
                                              uint8_t xfill,
                                              uint8_t[:] y_,
                                              IntIndex yindex,
                                              uint8_t yfill):
    cdef:
        IntIndex out_index
        Py_ssize_t xi = 0, yi = 0, out_i = 0  # fp buf indices
        int32_t xloc, yloc
        int32_t[:] xindices, yindices, out_indices
        uint8_t[:] x, y
        ndarray[uint8_t, ndim=1] out

    # suppress Cython compiler warnings due to inlining
    x = x_
    y = y_

    # need to do this first to know size of result array
    out_index = xindex.make_union(yindex)
    out = np.empty(out_index.npoints, dtype=np.uint8)

    xindices = xindex.indices
    yindices = yindex.indices
    out_indices = out_index.indices

    # walk the two SparseVectors, adding matched locations...
    for out_i in range(out_index.npoints):
        if xi == xindex.npoints:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1
            continue

        if yi == yindex.npoints:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1
            continue

        xloc = xindices[xi]
        yloc = yindices[yi]

        # each index in the out_index had to come from either x, y, or both
        if xloc == yloc:
            out[out_i] = x[xi] ^ y[yi]
            xi += 1
            yi += 1
        elif xloc < yloc:
            # use y fill value
            out[out_i] = x[xi] ^ yfill
            xi += 1
        else:
            # use x fill value
            out[out_i] = xfill ^ y[yi]
            yi += 1

    return out, out_index, xfill ^ yfill


cpdef sparse_xor_uint8(uint8_t[:] x,
                                  SparseIndex xindex, uint8_t xfill,
                                  uint8_t[:] y,
                                  SparseIndex yindex, uint8_t yfill):

    if isinstance(xindex, BlockIndex):
        return block_op_xor_uint8(x, xindex.to_block_index(), xfill,
                                             y, yindex.to_block_index(), yfill)
    elif isinstance(xindex, IntIndex):
        return int_op_xor_uint8(x, xindex.to_int_index(), xfill,
                                           y, yindex.to_int_index(), yfill)
    else:
        raise NotImplementedError
