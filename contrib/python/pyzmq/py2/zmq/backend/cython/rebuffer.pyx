"""
Utility for changing itemsize of memoryviews, and getting
numpy arrays from byte-arrays that should be interpreted with a different
itemsize.

Authors
-------
* MinRK
"""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010-2012 Brian Granger, Min Ragan-Kelley
#
#  This file is part of pyzmq
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------

from libc.stdlib cimport malloc
from zmq.utils.buffers cimport *

cdef inline object _rebuffer(object obj, char * format, int itemsize):
    """clobber the format & itemsize of a 1-D

    This is the Python 3 model, but will work on Python >= 2.6. Currently,
    we use it only on >= 3.0.
    """
    cdef Py_buffer view
    cdef int flags = PyBUF_SIMPLE
    cdef int mode = 0
    # cdef Py_ssize_t *shape, *strides, *suboffsets
    
    mode = check_buffer(obj)
    if mode == 0:
        raise TypeError("%r does not provide a buffer interface."%obj)

    if mode == 3:
        flags = PyBUF_ANY_CONTIGUOUS
        if format:
            flags |= PyBUF_FORMAT
        PyObject_GetBuffer(obj, &view, flags)
        assert view.ndim <= 1, "Can only reinterpret 1-D memoryviews"
        assert view.len % itemsize == 0, "Buffer of length %i not divisible into items of size %i"%(view.len, itemsize)
        # hack the format
        view.ndim = 1
        view.format = format
        view.itemsize = itemsize
        view.strides = <Py_ssize_t *>malloc(sizeof(Py_ssize_t))
        view.strides[0] = itemsize
        view.shape = <Py_ssize_t *>malloc(sizeof(Py_ssize_t))
        view.shape[0] = view.len/itemsize
        view.suboffsets = <Py_ssize_t *>malloc(sizeof(Py_ssize_t))
        view.suboffsets[0] = 0
        # for debug: make buffer writable, for zero-copy testing
        # view.readonly = 0
        
        return PyMemoryView_FromBuffer(&view)
    else:
        raise TypeError("This funciton is only for new-style buffer objects.")

def rebuffer(obj, format, itemsize):
    """Change the itemsize of a memoryview.
    
    Only for 1D contiguous buffers.
    """
    return _rebuffer(obj, format, itemsize)

def array_from_buffer(view, dtype, shape):
    """Get a numpy array from a memoryview, regardless of the itemsize of the original
    memoryview.  This is important, because pyzmq does not send memoryview shape data
    over the wire, so we need to change the memoryview itemsize before calling
    asarray.
    """
    import numpy
    A = numpy.array([],dtype=dtype)
    ref = viewfromobject(A,0)
    fmt = ref.format.encode()
    buf = viewfromobject(view, 0)
    buf = _rebuffer(view, fmt, ref.itemsize)
    return numpy.asarray(buf, dtype=dtype).reshape(shape)

def print_view_info(obj):
    """simple utility for printing info on a new-style buffer object"""
    cdef Py_buffer view
    cdef int flags = PyBUF_ANY_CONTIGUOUS|PyBUF_FORMAT
    cdef int mode = 0
    
    mode = check_buffer(obj)
    if mode == 0:
        raise TypeError("%r does not provide a buffer interface."%obj)

    if mode == 3:
        PyObject_GetBuffer(obj, &view, flags)
        print <size_t>view.buf, view.len, view.format, view.ndim,
        if view.ndim:
            if view.shape:
                print view.shape[0],
            if view.strides:
                print view.strides[0],
            if view.suboffsets:
                print view.suboffsets[0],
        print
