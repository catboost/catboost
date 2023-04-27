"""Python version-independent methods for C/Python buffers.

This file was copied and adapted from mpi4py.

Authors
-------
* MinRK
"""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010 Lisandro Dalcin
#  All rights reserved.
#  Used under BSD License: http://www.opensource.org/licenses/bsd-license.php
#
#  Retrieval:
#  Jul 23, 2010 18:00 PST (r539)
#  http://code.google.com/p/mpi4py/source/browse/trunk/src/MPI/asbuffer.pxi
#
#  Modifications from original:
#  Copyright (c) 2010-2012 Brian Granger, Min Ragan-Kelley
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Python includes.
#-----------------------------------------------------------------------------

# get version-independent aliases:
cdef extern from "pyversion_compat.h":
    pass

# Python 3 buffer interface (PEP 3118)
cdef extern from "Python.h":
    int PY_MAJOR_VERSION
    int PY_MINOR_VERSION
    ctypedef int Py_ssize_t
    ctypedef struct PyMemoryViewObject:
        pass
    ctypedef struct Py_buffer:
        void *buf
        Py_ssize_t len
        int readonly
        char *format
        int ndim
        Py_ssize_t *shape
        Py_ssize_t *strides
        Py_ssize_t *suboffsets
        Py_ssize_t itemsize
        void *internal
    cdef enum:
        PyBUF_SIMPLE
        PyBUF_WRITABLE
        PyBUF_FORMAT
        PyBUF_ANY_CONTIGUOUS
    int  PyObject_CheckBuffer(object)
    int  PyObject_GetBuffer(object, Py_buffer *, int) except -1
    void PyBuffer_Release(Py_buffer *)
    
    int PyBuffer_FillInfo(Py_buffer *view, object obj, void *buf,
                Py_ssize_t len, int readonly, int infoflags) except -1
    object PyMemoryView_FromBuffer(Py_buffer *info)
    
    object PyMemoryView_FromObject(object)

# Python 2 buffer interface (legacy)
cdef extern from "Python.h":
    Py_ssize_t Py_END_OF_BUFFER
    int PyObject_CheckReadBuffer(object)
    int PyObject_AsReadBuffer (object, const void **, Py_ssize_t *) except -1
    int PyObject_AsWriteBuffer(object, void **, Py_ssize_t *) except -1
    
    object PyBuffer_FromMemory(void *ptr, Py_ssize_t s)
    object PyBuffer_FromReadWriteMemory(void *ptr, Py_ssize_t s)

    object PyBuffer_FromObject(object, Py_ssize_t offset, Py_ssize_t size)
    object PyBuffer_FromReadWriteObject(object, Py_ssize_t offset, Py_ssize_t size)


#-----------------------------------------------------------------------------
# asbuffer: C buffer from python object
#-----------------------------------------------------------------------------


cdef inline int memoryview_available():
    return PY_MAJOR_VERSION >= 3 or (PY_MAJOR_VERSION >=2 and PY_MINOR_VERSION >= 7)

cdef inline int oldstyle_available():
    return PY_MAJOR_VERSION < 3


cdef inline int check_buffer(object ob):
    """Version independent check for whether an object is a buffer.
    
    Parameters
    ----------
    object : object
        Any Python object

    Returns
    -------
    int : 0 if no buffer interface, 3 if newstyle buffer interface, 2 if oldstyle.
    """
    if PyObject_CheckBuffer(ob):
        return 3
    if oldstyle_available():
        return PyObject_CheckReadBuffer(ob) and 2
    return 0


cdef inline object asbuffer(object ob, int writable, int format,
                            void **base, Py_ssize_t *size,
                            Py_ssize_t *itemsize):
    """Turn an object into a C buffer in a Python version-independent way.
    
    Parameters
    ----------
    ob : object
        The object to be turned into a buffer.
        Must provide a Python Buffer interface
    writable : int
        Whether the resulting buffer should be allowed to write
        to the object.
    format : int
        The format of the buffer.  See Python buffer docs.
    base : void **
        The pointer that will be used to store the resulting C buffer.
    size : Py_ssize_t *
        The size of the buffer(s).
    itemsize : Py_ssize_t *
        The size of an item, if the buffer is non-contiguous.
    
    Returns
    -------
    An object describing the buffer format. Generally a str, such as 'B'.
    """

    cdef void *bptr = NULL
    cdef Py_ssize_t blen = 0, bitemlen = 0
    cdef Py_buffer view
    cdef int flags = PyBUF_SIMPLE
    cdef int mode = 0
    
    bfmt = None

    mode = check_buffer(ob)
    if mode == 0:
        raise TypeError("%r does not provide a buffer interface."%ob)

    if mode == 3:
        flags = PyBUF_ANY_CONTIGUOUS
        if writable:
            flags |= PyBUF_WRITABLE
        if format:
            flags |= PyBUF_FORMAT
        PyObject_GetBuffer(ob, &view, flags)
        bptr = view.buf
        blen = view.len
        if format:
            if view.format != NULL:
                bfmt = view.format
                bitemlen = view.itemsize
        PyBuffer_Release(&view)
    else: # oldstyle
        if writable:
            PyObject_AsWriteBuffer(ob, &bptr, &blen)
        else:
            PyObject_AsReadBuffer(ob, <const void **>&bptr, &blen)
        if format:
            try: # numpy.ndarray
                dtype = ob.dtype
                bfmt = dtype.char
                bitemlen = dtype.itemsize
            except AttributeError:
                try: # array.array
                    bfmt = ob.typecode
                    bitemlen = ob.itemsize
                except AttributeError:
                    if isinstance(ob, bytes):
                        bfmt = b"B"
                        bitemlen = 1
                    else:
                        # nothing found
                        bfmt = None
                        bitemlen = 0
    if base: base[0] = <void *>bptr
    if size: size[0] = <Py_ssize_t>blen
    if itemsize: itemsize[0] = <Py_ssize_t>bitemlen
    
    if PY_MAJOR_VERSION >= 3 and bfmt is not None:
        return bfmt.decode('ascii')
    return bfmt


cdef inline object asbuffer_r(object ob, void **base, Py_ssize_t *size):
    """Wrapper for standard calls to asbuffer with a readonly buffer."""
    asbuffer(ob, 0, 0, base, size, NULL)
    return ob


cdef inline object asbuffer_w(object ob, void **base, Py_ssize_t *size):
    """Wrapper for standard calls to asbuffer with a writable buffer."""
    asbuffer(ob, 1, 0, base, size, NULL)
    return ob

#------------------------------------------------------------------------------
# frombuffer: python buffer/view from C buffer
#------------------------------------------------------------------------------


cdef inline object frombuffer_3(void *ptr, Py_ssize_t s, int readonly):
    """Python 3 version of frombuffer.

    This is the Python 3 model, but will work on Python >= 2.6. Currently,
    we use it only on >= 3.0.
    """
    cdef Py_buffer pybuf
    cdef Py_ssize_t *shape = [s]
    cdef str astr=""
    PyBuffer_FillInfo(&pybuf, astr, ptr, s, readonly, PyBUF_SIMPLE)
    pybuf.format = "B"
    pybuf.shape = shape
    pybuf.ndim = 1
    return PyMemoryView_FromBuffer(&pybuf)


cdef inline object frombuffer_2(void *ptr, Py_ssize_t s, int readonly):
    """Python 2 version of frombuffer. 

    This must be used for Python <= 2.6, but we use it for all Python < 3.
    """
    
    if oldstyle_available():
        if readonly:
            return PyBuffer_FromMemory(ptr, s)
        else:
            return PyBuffer_FromReadWriteMemory(ptr, s)
    else:
        raise NotImplementedError("Old style buffers not available.")


cdef inline object frombuffer(void *ptr, Py_ssize_t s, int readonly):
    """Create a Python Buffer/View of a C array. 
    
    Parameters
    ----------
    ptr : void *
        Pointer to the array to be copied.
    s : size_t
        Length of the buffer.
    readonly : int
        whether the resulting object should be allowed to write to the buffer.
    
    Returns
    -------
    Python Buffer/View of the C buffer.
    """
    # oldstyle first priority for now
    if oldstyle_available():
        return frombuffer_2(ptr, s, readonly)
    else:
        return frombuffer_3(ptr, s, readonly)


cdef inline object frombuffer_r(void *ptr, Py_ssize_t s):
    """Wrapper for readonly view frombuffer."""
    return frombuffer(ptr, s, 1)


cdef inline object frombuffer_w(void *ptr, Py_ssize_t s):
    """Wrapper for writable view frombuffer."""
    return frombuffer(ptr, s, 0)

#------------------------------------------------------------------------------
# viewfromobject: python buffer/view from python object, refcounts intact
# frombuffer(asbuffer(obj)) would lose track of refs
#------------------------------------------------------------------------------

cdef inline object viewfromobject(object obj, int readonly):
    """Construct a Python Buffer/View object from another Python object.

    This work in a Python version independent manner.
    
    Parameters
    ----------
    obj : object
        The input object to be cast as a buffer
    readonly : int
        Whether the result should be prevented from overwriting the original.
    
    Returns
    -------
    Buffer/View of the original object.
    """
    if not memoryview_available():
        if readonly:
            return PyBuffer_FromObject(obj, 0, Py_END_OF_BUFFER)
        else:
            return PyBuffer_FromReadWriteObject(obj, 0, Py_END_OF_BUFFER)
    else:
        return PyMemoryView_FromObject(obj)


cdef inline object viewfromobject_r(object obj):
    """Wrapper for readonly viewfromobject."""
    return viewfromobject(obj, 1)


cdef inline object viewfromobject_w(object obj):
    """Wrapper for writable viewfromobject."""
    return viewfromobject(obj, 0)
