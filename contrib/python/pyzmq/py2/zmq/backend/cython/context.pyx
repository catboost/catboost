"""0MQ Context class."""
# coding: utf-8

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Lesser GNU Public License (LGPL).

from libc.stdlib cimport free, malloc, realloc

from .libzmq cimport *

cdef extern from "getpid_compat.h":
    int getpid()

from zmq.error import ZMQError, InterruptedSystemCall
from .checkrc cimport _check_rc


_instance = None

cdef class Context:
    """Context(io_threads=1)

    Manage the lifecycle of a 0MQ context.

    Parameters
    ----------
    io_threads : int
        The number of IO threads.
    """
    
    # no-op for the signature
    def __init__(self, io_threads=1, shadow=0):
        pass
    
    def __cinit__(self, int io_threads=1, size_t shadow=0, **kwargs):
        self.handle = NULL
        if shadow:
            self.handle = <void *>shadow
            self._shadow = True
        else:
            self._shadow = False
            if ZMQ_VERSION_MAJOR >= 3:
                self.handle = zmq_ctx_new()
            else:
                self.handle = zmq_init(io_threads)

        if self.handle == NULL:
            raise ZMQError()

        cdef int rc = 0
        if ZMQ_VERSION_MAJOR >= 3 and not self._shadow:
            rc = zmq_ctx_set(self.handle, ZMQ_IO_THREADS, io_threads)
            _check_rc(rc)

        self.closed = False
        self._pid = getpid()

    def __dealloc__(self):
        """don't touch members in dealloc, just cleanup allocations"""
        cdef int rc

        # we can't call object methods in dealloc as it
        # might already be partially deleted
        if not self._shadow:
            self._term()

    @property
    def underlying(self):
        """The address of the underlying libzmq context"""
        return <size_t> self.handle

    cdef inline int _term(self):
        cdef int rc=0
        if self.handle != NULL and not self.closed and getpid() == self._pid:
            with nogil:
                rc = zmq_ctx_destroy(self.handle)
        self.handle = NULL
        return rc
    
    def term(self):
        """ctx.term()

        Close or terminate the context.
        
        This can be called to close the context by hand. If this is not called,
        the context will automatically be closed when it is garbage collected.
        """
        cdef int rc=0
        rc = self._term()
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            # ignore interrupted term
            # see PEP 475 notes about close & EINTR for why
            pass
        
        self.closed = True
    
    def set(self, int option, optval):
        """ctx.set(option, optval)

        Set a context option.

        See the 0MQ API documentation for zmq_ctx_set
        for details on specific options.
        
        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        option : int
            The option to set.  Available values will depend on your
            version of libzmq.  Examples include::
            
                zmq.IO_THREADS, zmq.MAX_SOCKETS
        
        optval : int
            The value of the option to set.
        """
        cdef int optval_int_c
        cdef int rc
        cdef char* optval_c

        if self.closed:
            raise RuntimeError("Context has been destroyed")
        
        if not isinstance(optval, int):
            raise TypeError('expected int, got: %r' % optval)
        optval_int_c = optval
        rc = zmq_ctx_set(self.handle, option, optval_int_c)
        _check_rc(rc)

    def get(self, int option):
        """ctx.get(option)

        Get the value of a context option.

        See the 0MQ API documentation for zmq_ctx_get
        for details on specific options.
        
        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        option : int
            The option to get.  Available values will depend on your
            version of libzmq.  Examples include::
            
                zmq.IO_THREADS, zmq.MAX_SOCKETS
            
        Returns
        -------
        optval : int
            The value of the option as an integer.
        """
        cdef int optval_int_c
        cdef size_t sz
        cdef int rc

        if self.closed:
            raise RuntimeError("Context has been destroyed")

        rc = zmq_ctx_get(self.handle, option)
        _check_rc(rc)

        return rc


__all__ = ['Context']
