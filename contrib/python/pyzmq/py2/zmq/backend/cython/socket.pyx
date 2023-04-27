"""0MQ Socket class."""

#
#    Copyright (c) 2010-2011 Brian E. Granger & Min Ragan-Kelley
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
# Cython Imports
#-----------------------------------------------------------------------------

# get version-independent aliases:
cdef extern from "pyversion_compat.h":
    pass

from libc.errno cimport ENAMETOOLONG, ENOENT, ENOTSOCK
from libc.string cimport memcpy

from cpython cimport PyBytes_FromStringAndSize
from cpython cimport PyBytes_AsString, PyBytes_Size
from cpython cimport Py_DECREF, Py_INCREF, PY_VERSION_HEX

from zmq.utils.buffers cimport asbuffer_r, viewfromobject_r

from .libzmq cimport (
    fd_t,
    int64_t,

    zmq_errno,

    zmq_msg_t,
    zmq_msg_init,
    zmq_msg_init_size,
    zmq_msg_close,
    zmq_msg_data,
    zmq_msg_size,
    zmq_msg_send,
    zmq_msg_recv,

    zmq_socket,
    zmq_socket_monitor,
    zmq_connect,
    zmq_disconnect,
    zmq_bind,
    zmq_unbind,
    zmq_setsockopt,
    zmq_getsockopt,
    zmq_close,
    zmq_join,
    zmq_leave,

    ZMQ_EVENT_ALL,
    ZMQ_IDENTITY,
    ZMQ_LINGER,
    ZMQ_TYPE,
)
from .message cimport Frame, copy_zmq_msg_bytes

from .context cimport Context

cdef extern from "Python.h":
    ctypedef int Py_ssize_t

cdef extern from "ipcmaxlen.h":
    int get_ipc_path_max_len()

cdef extern from "getpid_compat.h":
    int getpid()


#-----------------------------------------------------------------------------
# Python Imports
#-----------------------------------------------------------------------------

import copy as copy_mod
import time
import sys
import random
import struct
import codecs

try:
    import cPickle
    pickle = cPickle
except:
    cPickle = None
    import pickle

import zmq
from zmq.backend.cython import constants
from .checkrc cimport _check_rc
from zmq.error import ZMQError, ZMQBindError, InterruptedSystemCall, _check_version

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

IPC_PATH_MAX_LEN = get_ipc_path_max_len()

# inline some small socket submethods:
# true methods frequently cannot be inlined, acc. Cython docs

cdef inline Py_ssize_t nbytes(buf) except -1:
    """get n bytes"""
    if PY_VERSION_HEX >= 0x03030000:
        return buf.nbytes

    cdef Py_ssize_t n = buf.itemsize
    cdef Py_ssize_t ndim = buf.ndim
    cdef Py_ssize_t i
    for i in range(ndim):
        n *= buf.shape[i]
    return n

cdef inline _check_closed(Socket s):
    """raise ENOTSUP if socket is closed
    
    Does not do a deep check
    """
    if s._closed:
        raise ZMQError(ENOTSOCK)

cdef inline _check_closed_deep(Socket s):
    """thorough check of whether the socket has been closed,
    even if by another entity (e.g. ctx.destroy).
    
    Only used by the `closed` property.
    
    returns True if closed, False otherwise
    """
    cdef int rc
    cdef int errno
    cdef int stype
    cdef size_t sz=sizeof(int)
    if s._closed:
        return True
    else:
        rc = zmq_getsockopt(s.handle, ZMQ_TYPE, <void *>&stype, &sz)
        if rc < 0 and zmq_errno() == ENOTSOCK:
            s._closed = True
            return True
        else:
            _check_rc(rc)
    return False

cdef inline Frame _recv_frame(void *handle, int flags=0, track=False):
    """Receive a message in a non-copying manner and return a Frame."""
    cdef int rc
    msg = zmq.Frame(track=track)
    cdef Frame cmsg = msg
    
    while True:
        with nogil:
            rc = zmq_msg_recv(&cmsg.zmq_msg, handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return msg

cdef inline object _recv_copy(void *handle, int flags=0):
    """Receive a message and return a copy"""
    cdef zmq_msg_t zmq_msg
    rc = zmq_msg_init (&zmq_msg)
    _check_rc(rc)
    while True:
        with nogil:
            rc = zmq_msg_recv(&zmq_msg, handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        except Exception:
            zmq_msg_close(&zmq_msg) # ensure msg is closed on failure
            raise
        else:
            break
    
    msg_bytes = copy_zmq_msg_bytes(&zmq_msg)
    zmq_msg_close(&zmq_msg)
    return msg_bytes

cdef inline object _send_frame(void *handle, Frame msg, int flags=0):
    """Send a Frame on this socket in a non-copy manner."""
    cdef int rc
    cdef Frame msg_copy

    # Always copy so the original message isn't garbage collected.
    # This doesn't do a real copy, just a reference.
    msg_copy = msg.fast_copy()
    
    while True:
        with nogil:
            rc = zmq_msg_send(&msg_copy.zmq_msg, handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break

    return msg.tracker


cdef inline object _send_copy(void *handle, object msg, int flags=0):
    """Send a message on this socket by copying its content."""
    cdef int rc
    cdef zmq_msg_t data
    cdef char *msg_c
    cdef Py_ssize_t msg_c_len=0

    # copy to c array:
    asbuffer_r(msg, <void **>&msg_c, &msg_c_len)

    # Copy the msg before sending. This avoids any complications with
    # the GIL, etc.
    # If zmq_msg_init_* fails we must not call zmq_msg_close (Bus Error)
    rc = zmq_msg_init_size(&data, msg_c_len)
    _check_rc(rc)
    
    while True:
        with nogil:
            memcpy(zmq_msg_data(&data), msg_c, zmq_msg_size(&data))
            rc = zmq_msg_send(&data, handle, flags)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        except Exception:
            zmq_msg_close(&data) # close the unused msg
            raise # raise original exception
        else:
            rc = zmq_msg_close(&data)
            _check_rc(rc)
            break

cdef inline object _getsockopt(void *handle, int option, void *optval, size_t *sz):
    """getsockopt, retrying interrupted calls
    
    checks rc, raising ZMQError on failure.
    """
    cdef int rc=0
    while True:
        rc = zmq_getsockopt(handle, option, optval, sz)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break

cdef inline object _setsockopt(void *handle, int option, void *optval, size_t sz):
    """setsockopt, retrying interrupted calls
    
    checks rc, raising ZMQError on failure.
    """
    cdef int rc=0
    while True:
        rc = zmq_setsockopt(handle, option, optval, sz)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break


cdef class Socket:
    """Socket(context, socket_type)

    A 0MQ socket.

    These objects will generally be constructed via the socket() method of a Context object.
    
    Note: 0MQ Sockets are *not* threadsafe. **DO NOT** share them across threads.
    
    Parameters
    ----------
    context : Context
        The 0MQ Context this Socket belongs to.
    socket_type : int
        The socket type, which can be any of the 0MQ socket types: 
        REQ, REP, PUB, SUB, PAIR, DEALER, ROUTER, PULL, PUSH, XPUB, XSUB.
    
    See Also
    --------
    .Context.socket : method for creating a socket bound to a Context.
    """

    def __init__(self, context=None, socket_type=-1, shadow=0, copy_threshold=None):
        if copy_threshold is None:
            copy_threshold = zmq.COPY_THRESHOLD
        self.copy_threshold = copy_threshold

        self.handle = NULL
        self.context = context
        cdef size_t c_shadow
        if shadow:
            if isinstance(shadow, Socket):
                shadow = shadow.underlying
            c_shadow = shadow
            self._shadow = True
            self.handle = <void *>c_shadow
        else:
            if context is None:
                raise TypeError("context must be specified")
            if socket_type < 0:
                raise TypeError("socket_type must be specified")
            self._shadow = False
            self.handle = zmq_socket(self.context.handle, socket_type)
        if self.handle == NULL:
            raise ZMQError()
        self._closed = False
        self._pid = getpid()

    def __cinit__(self, *args, **kwargs):
        # basic init
        self.handle = NULL
        self._pid = 0
        self._shadow = False
        self.context = None

    def __dealloc__(self):
        """remove from context's list

        But be careful that context might not exist if called during gc
        """
        if self.handle != NULL and not self._shadow and getpid() == self._pid:
            self._c_close()

    @property
    def underlying(self):
        """The address of the underlying libzmq socket"""
        return <size_t> self.handle

    @property
    def closed(self):
        return _check_closed_deep(self)

    def close(self, linger=None):
        """s.close(linger=None)

        Close the socket.

        If linger is specified, LINGER sockopt will be set prior to closing.

        This can be called to close the socket by hand. If this is not
        called, the socket will automatically be closed when it is
        garbage collected.
        """
        cdef int rc=0
        cdef int linger_c
        cdef bint setlinger=False

        if linger is not None:
            linger_c = linger
            setlinger=True

        if self.handle != NULL and not self._closed and getpid() == self._pid:
            if setlinger:
                zmq_setsockopt(self.handle, ZMQ_LINGER, &linger_c, sizeof(int))
            self._c_close()

    cdef void _c_close(self):
        """Close underlying socket and unregister with self.context"""
        rc = zmq_close(self.handle)
        if rc < 0 and zmq_errno() != ENOTSOCK:
            # ignore ENOTSOCK (closed by Context)
            _check_rc(rc)
        self._closed = True
        self.handle = NULL

    def set(self, int option, optval):
        """s.set(option, optval)

        Set socket options.

        See the 0MQ API documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to set.  Available values will depend on your
            version of libzmq.  Examples include::

                zmq.SUBSCRIBE, UNSUBSCRIBE, IDENTITY, HWM, LINGER, FD

        optval : int or bytes
            The value of the option to set.

        Notes
        -----
        .. warning::

            All options other than zmq.SUBSCRIBE, zmq.UNSUBSCRIBE and
            zmq.LINGER only take effect for subsequent socket bind/connects.
        """
        cdef int64_t optval_int64_c
        cdef int optval_int_c
        cdef char* optval_c
        cdef Py_ssize_t sz

        _check_closed(self)
        if isinstance(optval, unicode):
            raise TypeError("unicode not allowed, use setsockopt_string")

        if option in zmq.constants.bytes_sockopts:
            if not isinstance(optval, bytes):
                raise TypeError('expected bytes, got: %r' % optval)
            optval_c = PyBytes_AsString(optval)
            sz = PyBytes_Size(optval)
            _setsockopt(self.handle, option, optval_c, sz)
        elif option in zmq.constants.int64_sockopts:
            if not isinstance(optval, int):
                raise TypeError('expected int, got: %r' % optval)
            optval_int64_c = optval
            _setsockopt(self.handle, option, &optval_int64_c, sizeof(int64_t))
        else:
            # default is to assume int, which is what most new sockopts will be
            # this lets pyzmq work with newer libzmq which may add constants
            # pyzmq has not yet added, rather than artificially raising. Invalid
            # sockopts will still raise just the same, but it will be libzmq doing
            # the raising.
            if not isinstance(optval, int):
                raise TypeError('expected int, got: %r' % optval)
            optval_int_c = optval
            _setsockopt(self.handle, option, &optval_int_c, sizeof(int))

    def get(self, int option):
        """s.get(option)

        Get the value of a socket option.

        See the 0MQ API documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to get.  Available values will depend on your
            version of libzmq.  Examples include::
            
                zmq.IDENTITY, HWM, LINGER, FD, EVENTS

        Returns
        -------
        optval : int or bytes
            The value of the option as a bytestring or int.
        """
        cdef int64_t optval_int64_c
        cdef int optval_int_c
        cdef fd_t optval_fd_c
        cdef char identity_str_c [255]
        cdef size_t sz
        cdef int rc

        _check_closed(self)

        if option in zmq.constants.bytes_sockopts:
            sz = 255
            _getsockopt(self.handle, option, <void *>identity_str_c, &sz)
            # strip null-terminated strings *except* identity
            if option != ZMQ_IDENTITY and sz > 0 and (<char *>identity_str_c)[sz-1] == b'\0':
                sz -= 1
            result = PyBytes_FromStringAndSize(<char *>identity_str_c, sz)
        elif option in zmq.constants.int64_sockopts:
            sz = sizeof(int64_t)
            _getsockopt(self.handle, option, <void *>&optval_int64_c, &sz)
            result = optval_int64_c
        elif option in zmq.constants.fd_sockopts:
            sz = sizeof(fd_t)
            _getsockopt(self.handle, option, <void *>&optval_fd_c, &sz)
            result = optval_fd_c
        else:
            # default is to assume int, which is what most new sockopts will be
            # this lets pyzmq work with newer libzmq which may add constants
            # pyzmq has not yet added, rather than artificially raising. Invalid
            # sockopts will still raise just the same, but it will be libzmq doing
            # the raising.
            sz = sizeof(int)
            _getsockopt(self.handle, option, <void *>&optval_int_c, &sz)
            result = optval_int_c

        return result
    
    def bind(self, addr):
        """s.bind(addr)

        Bind the socket to an address.

        This causes the socket to listen on a network port. Sockets on the
        other side of this connection will use ``Socket.connect(addr)`` to
        connect to this socket.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported include
            tcp, udp, pgm, epgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        cdef int rc
        cdef char* c_addr

        _check_closed(self)
        if isinstance(addr, unicode):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        rc = zmq_bind(self.handle, c_addr)
        if rc != 0:
            if IPC_PATH_MAX_LEN and zmq_errno() == ENAMETOOLONG:
                # py3compat: addr is bytes, but msg wants str
                if str is unicode:
                    addr = addr.decode('utf-8', 'replace')
                path = addr.split('://', 1)[-1]
                msg = ('ipc path "{0}" is longer than {1} '
                                'characters (sizeof(sockaddr_un.sun_path)). '
                                'zmq.IPC_PATH_MAX_LEN constant can be used '
                                'to check addr length (if it is defined).'
                                .format(path, IPC_PATH_MAX_LEN))
                raise ZMQError(msg=msg)
            elif zmq_errno() == ENOENT:
                # py3compat: address is bytes, but msg wants str
                if str is unicode:
                    addr = addr.decode('utf-8', 'replace')
                path = addr.split('://', 1)[-1]
                msg = ('No such file or directory for ipc path "{0}".'.format(
                       path))
                raise ZMQError(msg=msg)
        while True:
            try:
                _check_rc(rc)
            except InterruptedSystemCall:
                rc = zmq_bind(self.handle, c_addr)
                continue
            else:
                break

    def connect(self, addr):
        """s.connect(addr)

        Connect to a remote 0MQ socket.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, upd, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        cdef int rc
        cdef char* c_addr

        _check_closed(self)
        if isinstance(addr, unicode):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        
        while True:
            try:
                rc = zmq_connect(self.handle, c_addr)
                _check_rc(rc)
            except InterruptedSystemCall:
                # retry syscall
                continue
            else:
                break

    def unbind(self, addr):
        """s.unbind(addr)
        
        Unbind from an address (undoes a call to bind).
        
        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, upd, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        cdef int rc
        cdef char* c_addr

        _check_version((3,2), "unbind")
        _check_closed(self)
        if isinstance(addr, unicode):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        
        rc = zmq_unbind(self.handle, c_addr)
        if rc != 0:
            raise ZMQError()

    def disconnect(self, addr):
        """s.disconnect(addr)

        Disconnect from a remote 0MQ socket (undoes a call to connect).
        
        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, upd, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        cdef int rc
        cdef char* c_addr

        _check_version((3,2), "disconnect")
        _check_closed(self)
        if isinstance(addr, unicode):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr

        rc = zmq_disconnect(self.handle, c_addr)
        if rc != 0:
            raise ZMQError()

    def monitor(self, addr, int events=ZMQ_EVENT_ALL):
        """s.monitor(addr, flags)

        Start publishing socket events on inproc.
        See libzmq docs for zmq_monitor for details.

        While this function is available from libzmq 3.2,
        pyzmq cannot parse monitor messages from libzmq prior to 4.0.

        .. versionadded: libzmq-3.2
        .. versionadded: 14.0

        Parameters
        ----------
        addr : str
            The inproc url used for monitoring. Passing None as
            the addr will cause an existing socket monitor to be
            deregistered.
        events : int [default: zmq.EVENT_ALL]
            The zmq event bitmask for which events will be sent to the monitor.
        """
        cdef int rc, c_flags
        cdef char* c_addr = NULL

        _check_version((3,2), "monitor")
        if addr is not None:
            if isinstance(addr, unicode):
                addr = addr.encode('utf-8')
            if not isinstance(addr, bytes):
                raise TypeError('expected str, got: %r' % addr)
            c_addr = addr
        c_flags = events
        rc = zmq_socket_monitor(self.handle, c_addr, c_flags)
        _check_rc(rc)

    def join(self, group):
        """join(group)

        Join a RADIO-DISH group

        Only for DISH sockets.

        libzmq and pyzmq must have been built with ZMQ_BUILD_DRAFT_API

        .. versionadded:: 17
        """
        _check_version((4,2), "RADIO-DISH")
        if not zmq.has('draft'):
            raise RuntimeError("libzmq must be built with draft support")
        if isinstance(group, unicode):
            group = group.encode('utf8')
        cdef int rc = zmq_join(self.handle, group)
        _check_rc(rc)

    def leave(self, group):
        """leave(group)

        Leave a RADIO-DISH group

        Only for DISH sockets.

        libzmq and pyzmq must have been built with ZMQ_BUILD_DRAFT_API

        .. versionadded:: 17
        """
        _check_version((4,2), "RADIO-DISH")
        if not zmq.has('draft'):
            raise RuntimeError("libzmq must be built with draft support")
        cdef int rc = zmq_leave(self.handle, group)
        _check_rc(rc)
        

    #-------------------------------------------------------------------------
    # Sending and receiving messages
    #-------------------------------------------------------------------------

    cpdef send(self, object data, int flags=0, copy=True, track=False):
        """Send a single zmq message frame on this socket.

        This queues the message to be sent by the IO thread at a later time.

        With flags=NOBLOCK, this raises :class:`ZMQError` if the queue is full;
        otherwise, this waits until space is available.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        data : bytes, Frame, memoryview
            The content of the message. This can be any object that provides
            the Python buffer API (`memoryview(data)` can be called).
        flags : int
            0, NOBLOCK, SNDMORE, or NOBLOCK|SNDMORE.
        copy : bool
            Should the message be sent in a copying or non-copying manner.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        None : if `copy` or not track
            None if message was sent, raises an exception otherwise.
        MessageTracker : if track and not copy
            a MessageTracker object, whose `pending` property will
            be True until the send is completed.
        
        Raises
        ------
        TypeError
            If a unicode object is passed
        ValueError
            If `track=True`, but an untracked Frame is passed.
        ZMQError
            for any of the reasons zmq_msg_send might fail (including
            if NOBLOCK is set and the outgoing queue is full).
        
        """
        _check_closed(self)

        if isinstance(data, unicode):
            raise TypeError("unicode not allowed, use send_string")

        if copy and not isinstance(data, Frame):
            return _send_copy(self.handle, data, flags)
        else:
            if isinstance(data, Frame):
                if track and not data.tracker:
                    raise ValueError('Not a tracked message')
                msg = data
            else:
                if self.copy_threshold:
                    buf = memoryview(data)
                    # always copy messages smaller than copy_threshold
                    if nbytes(buf) < self.copy_threshold:
                        _send_copy(self.handle, buf, flags)
                        return zmq._FINISHED_TRACKER
                msg = Frame(data, track=track, copy_threshold=self.copy_threshold)
            return _send_frame(self.handle, msg, flags)

    cpdef recv(self, int flags=0, copy=True, track=False):
        """s.recv(flags=0, copy=True, track=False)

        Receive a message.

        With flags=NOBLOCK, this raises :class:`ZMQError` if no messages have
        arrived; otherwise, this waits until a message arrives.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        flags : int
            0 or NOBLOCK.
        copy : bool
            Should the message be received in a copying or non-copying manner?
            If False a Frame object is returned, if True a string copy of
            message is returned.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        msg : bytes or Frame
            The received message frame.  If `copy` is False, then it will be a Frame,
            otherwise it will be bytes.

        Raises
        ------
        ZMQError
            for any of the reasons zmq_msg_recv might fail (including if
            NOBLOCK is set and no new messages have arrived).
        """
        _check_closed(self)
        
        if copy:
            return _recv_copy(self.handle, flags)
        else:
            frame = _recv_frame(self.handle, flags, track)
            frame.more = self.get(zmq.RCVMORE)
            return frame
    

__all__ = ['Socket', 'IPC_PATH_MAX_LEN']
