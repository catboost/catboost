"""0MQ Message related classes."""

#
#    Copyright (c) 2013 Brian E. Granger & Min Ragan-Kelley
#
#    This file is part of pyzmq.
#
#    pyzmq is free software; you can redistribute it and/or modify it under
#    the terms of the Lesser GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    pyzmq is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    Lesser GNU General Public License for more details.
#
#    You should have received a copy of the Lesser GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# get version-independent aliases:
cdef extern from "pyversion_compat.h":
    pass


from cpython cimport Py_DECREF, Py_INCREF

from zmq.utils.buffers cimport asbuffer_r


cdef extern from "Python.h":
    ctypedef int Py_ssize_t

cdef extern from "mutex.h" nogil:
    ctypedef struct mutex_t:
        pass
    cdef mutex_t* mutex_allocate()
    cdef void mutex_dallocate(mutex_t*)
    cdef int mutex_lock(mutex_t*)
    cdef int mutex_unlock(mutex_t*)

from libc.stdio cimport fprintf
from libc.stdio cimport stderr as cstderr
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from .libzmq cimport *

import time
from weakref import ref

try:
    # below 3.3
    from threading import _Event as Event
except (ImportError, AttributeError):
    # python throws ImportError, cython throws AttributeError
    from threading import Event

import zmq
from zmq.error import _check_version

from .checkrc cimport _check_rc

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

ctypedef struct zhint:
    void *sock
    mutex_t *mutex
    size_t id

cdef void free_python_msg(void *data, void *vhint) nogil:
    """A pure-C function for DECREF'ing Python-owned message data.

    Sends a message on a PUSH socket

    The hint is a `zhint` struct with two values:

    sock (void *): pointer to the Garbage Collector's PUSH socket
    id (size_t): the id to be used to construct a zmq_msg_t that should be sent on a PUSH socket,
       signaling the Garbage Collector to remove its reference to the object.

    When the Garbage Collector's PULL socket receives the message,
    it deletes its reference to the object,
    allowing Python to free the memory.
    """
    cdef zmq_msg_t msg
    cdef zhint *hint = <zhint *> vhint
    cdef int rc

    if hint != NULL:
        zmq_msg_init_size(&msg, sizeof(size_t))
        memcpy(zmq_msg_data(&msg), &hint.id, sizeof(size_t))
        rc = mutex_lock(hint.mutex)
        if rc != 0:
            fprintf(cstderr, "pyzmq-gc mutex lock failed rc=%d\n", rc)
        rc = zmq_msg_send(&msg, hint.sock, 0)
        if rc < 0:
            # gc socket could have been closed, e.g. during process teardown.
            # If so, ignore the failure because there's nothing to do.
            if zmq_errno() != ZMQ_ENOTSOCK:
                fprintf(cstderr, "pyzmq-gc send failed: %s\n", zmq_strerror(zmq_errno()))
        rc = mutex_unlock(hint.mutex)
        if rc != 0:
            fprintf(cstderr, "pyzmq-gc mutex unlock failed rc=%d\n", rc)

        zmq_msg_close(&msg)
        free(hint)


gc = None

cdef class Frame:
    def __cinit__(self, object data=None, track=False, copy=None, copy_threshold=None, **kwargs):
        cdef int rc
        cdef char *data_c = NULL
        cdef Py_ssize_t data_len_c=0
        cdef zhint *hint
        if copy_threshold is None:
            copy_threshold = zmq.COPY_THRESHOLD

        # init more as False
        self.more = False

        # Save the data object in case the user wants the the data as a str.
        self._data = data
        self._failed_init = True  # bool switch for dealloc
        self._buffer = None       # buffer view of data
        self._bytes = None        # bytes copy of data

        self.tracker_event = None
        self.tracker = None
        # self.tracker should start finished
        # except in the case where we are sharing memory with libzmq
        if track:
            self.tracker = zmq._FINISHED_TRACKER

        if isinstance(data, str):
            raise TypeError("Unicode objects not allowed. Only: str/bytes, buffer interfaces.")

        if data is None:
            rc = zmq_msg_init(&self.zmq_msg)
            _check_rc(rc)
            self._failed_init = False
            return

        asbuffer_r(data, <void **>&data_c, &data_len_c)

        # copy unspecified, apply copy_threshold
        if copy is None:
            if copy_threshold and data_len_c < copy_threshold:
                copy = True
            else:
                copy = False

        if copy:
            # copy message data instead of sharing memory
            rc = zmq_msg_init_size(&self.zmq_msg, data_len_c)
            _check_rc(rc)
            memcpy(zmq_msg_data(&self.zmq_msg), data_c, data_len_c)
            self._failed_init = False
            return

        # Getting here means that we are doing a true zero-copy Frame,
        # where libzmq and Python are sharing memory.
        # Hook up garbage collection with MessageTracker and zmq_free_fn

        # Event and MessageTracker for monitoring when zmq is done with data:
        if track:
            evt = Event()
            self.tracker_event = evt
            self.tracker = zmq.MessageTracker(evt)
        # create the hint for zmq_free_fn
        # two pointers: the gc context and a message to be sent to the gc PULL socket
        # allows libzmq to signal to Python when it is done with Python-owned memory.
        global gc
        if gc is None:
            from zmq.utils.garbage import gc

        hint = <zhint *> malloc(sizeof(zhint))
        hint.id = gc.store(data, self.tracker_event)
        if not gc._push_mutex:
            hint.mutex = mutex_allocate()
            gc._push_mutex = <size_t> hint.mutex
        else:
            hint.mutex = <mutex_t *> <size_t> gc._push_mutex
        hint.sock = <void *> <size_t> gc._push_socket.underlying

        rc = zmq_msg_init_data(
                &self.zmq_msg, <void *>data_c, data_len_c,
                <zmq_free_fn *>free_python_msg, <void *>hint
            )
        if rc != 0:
            free(hint)
            _check_rc(rc)
        self._failed_init = False

    def __init__(self, object data=None, track=False, copy=False, copy_threshold=None):
        """Enforce signature"""
        pass

    def __dealloc__(self):
        cdef int rc
        if self._failed_init:
            return
        # This simply decreases the 0MQ ref-count of zmq_msg.
        with nogil:
            rc = zmq_msg_close(&self.zmq_msg)
        _check_rc(rc)

    # buffer interface code adapted from petsc4py by Lisandro Dalcin, a BSD project

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        # new-style (memoryview) buffer interface
        buffer.buf = zmq_msg_data(&self.zmq_msg)
        buffer.len = zmq_msg_size(&self.zmq_msg)

        buffer.obj = self
        buffer.readonly = 0
        buffer.format = "B"
        buffer.ndim = 1
        buffer.shape = &(buffer.len)
        buffer.strides = NULL
        buffer.suboffsets = NULL
        buffer.itemsize = 1
        buffer.internal = NULL

    # end buffer interface

    def __copy__(self):
        """Create a shallow copy of the message.

        This does not copy the contents of the Frame, just the pointer.
        This will increment the 0MQ ref count of the message, but not
        the ref count of the Python object. That is only done once when
        the Python is first turned into a 0MQ message.
        """
        return self.fast_copy()

    cdef Frame fast_copy(self):
        """Fast, cdef'd version of shallow copy of the Frame."""
        cdef Frame new_msg
        new_msg = Frame()
        # This does not copy the contents, but just increases the ref-count
        # of the zmq_msg by one.
        zmq_msg_copy(&new_msg.zmq_msg, &self.zmq_msg)
        # Copy the ref to data so the copy won't create a copy when str is
        # called.
        if self._data is not None:
            new_msg._data = self._data
        if self._buffer is not None:
            new_msg._buffer = self._buffer
        if self._bytes is not None:
            new_msg._bytes = self._bytes

        # Frame copies share the tracker and tracker_event
        new_msg.tracker_event = self.tracker_event
        new_msg.tracker = self.tracker

        return new_msg

    def __len__(self):
        """Return the length of the message in bytes."""
        cdef size_t sz
        sz = zmq_msg_size(&self.zmq_msg)
        return sz
        # return <int>zmq_msg_size(&self.zmq_msg)

    def __str__(self):
        """Return the str form of the message."""
        if isinstance(self._data, bytes):
            b = self._data
        else:
            b = self.bytes
        if str is unicode:
            return b.decode()
        else:
            return b

    cdef inline object _getbuffer(self):
        """Deprecated alias for memoryview(self)"""
        return memoryview(self)

    @property
    def buffer(self):
        """A memoryview of the message contents."""
        _buffer = self._buffer and self._buffer()
        if _buffer is not None:
            return _buffer
        _buffer = memoryview(self)
        self._buffer = ref(_buffer)
        return _buffer

    @property
    def bytes(self):
        """The message content as a Python bytes object.

        The first time this property is accessed, a copy of the message
        contents is made. From then on that same copy of the message is
        returned.
        """
        if self._bytes is None:
            self._bytes = copy_zmq_msg_bytes(&self.zmq_msg)
        return self._bytes

    def set(self, option, value):
        """Frame.set(option, value)

        Set a Frame option.

        See the 0MQ API documentation for zmq_msg_set
        for details on specific options.

        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0
        .. versionchanged:: 17.0
            Added support for `routing_id` and `group`.
            Only available if draft API is enabled
            with libzmq >= 4.2.
        """
        cdef int rc
        cdef uint32_t routing_id

        if option == 'routing_id':
            routing_id = value
            rc = zmq_msg_set_routing_id(&self.zmq_msg, routing_id)
            _check_rc(rc)
            return
        elif option == 'group':
            if isinstance(value, str):
                value = value.encode('utf8')
            rc = zmq_msg_set_group(&self.zmq_msg, value)
            _check_rc(rc)
            return

        rc = zmq_msg_set(&self.zmq_msg, option, value)
        _check_rc(rc)

    def get(self, option):
        """Frame.get(option)

        Get a Frame option or property.

        See the 0MQ API documentation for zmq_msg_get and zmq_msg_gets
        for details on specific options.

        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        .. versionchanged:: 14.3
            add support for zmq_msg_gets (requires libzmq-4.1)
            All message properties are strings.

        .. versionchanged:: 17.0
            Added support for `routing_id` and `group`.
            Only available if draft API is enabled
            with libzmq >= 4.2.
        """
        cdef int rc = 0
        cdef char *property_c = NULL
        cdef Py_ssize_t property_len_c = 0
        cdef uint32_t routing_id

        # zmq_msg_get
        if isinstance(option, int):
            rc = zmq_msg_get(&self.zmq_msg, option)
            _check_rc(rc)
            return rc

        if option == 'routing_id':
            routing_id = zmq_msg_routing_id(&self.zmq_msg)
            if (routing_id == 0):
                _check_rc(-1)
            return routing_id
        elif option == 'group':
            buf = zmq_msg_group(&self.zmq_msg)
            if buf == NULL:
                _check_rc(-1)
            return buf.decode('utf8')

        # zmq_msg_gets
        _check_version((4,1), "get string properties")
        if isinstance(option, str):
            option = option.encode('utf8')

        if not isinstance(option, bytes):
            raise TypeError("expected str, got: %r" % option)

        property_c = option

        cdef const char *result = <char *>zmq_msg_gets(&self.zmq_msg, property_c)
        if result == NULL:
            _check_rc(-1)
        return result.decode('utf8')

# legacy Message name
Message = Frame

__all__ = ['Frame', 'Message']
