"""MonitoredQueue classes and functions.

Authors
-------
* MinRK
* Brian Granger
"""

#-----------------------------------------------------------------------------
#  Copyright (c) 2010-2012 Brian Granger, Min Ragan-Kelley
#
#  This file is part of pyzmq
#
#  Distributed under the terms of the New BSD License.  The full license is in
#  the file COPYING.BSD, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

cdef extern from "Python.h":
    ctypedef int Py_ssize_t

from libc.string cimport memcpy

from zmq.utils.buffers cimport asbuffer_r
from zmq.backend.cython.libzmq cimport *

from zmq.backend.cython.socket cimport Socket
from zmq.backend.cython.checkrc cimport _check_rc

from zmq import ROUTER, ZMQError
from zmq.error import InterruptedSystemCall

#-----------------------------------------------------------------------------
# MonitoredQueue functions
#-----------------------------------------------------------------------------


def monitored_queue(Socket in_socket, Socket out_socket, Socket mon_socket,
                    bytes in_prefix=b'in', bytes out_prefix=b'out'):
    """monitored_queue(in_socket, out_socket, mon_socket,
                       in_prefix=b'in', out_prefix=b'out')
    
    Start a monitored queue device.
    
    A monitored queue is very similar to the zmq.proxy device (monitored queue came first).
    
    Differences from zmq.proxy:
    
    - monitored_queue supports both in and out being ROUTER sockets
      (via swapping IDENTITY prefixes).
    - monitor messages are prefixed, making in and out messages distinguishable.
    
    Parameters
    ----------
    in_socket : Socket
        One of the sockets to the Queue. Its messages will be prefixed with
        'in'.
    out_socket : Socket
        One of the sockets to the Queue. Its messages will be prefixed with
        'out'. The only difference between in/out socket is this prefix.
    mon_socket : Socket
        This socket sends out every message received by each of the others
        with an in/out prefix specifying which one it was.
    in_prefix : str
        Prefix added to broadcast messages from in_socket.
    out_prefix : str
        Prefix added to broadcast messages from out_socket.
    """
    
    cdef void *ins=in_socket.handle
    cdef void *outs=out_socket.handle
    cdef void *mons=mon_socket.handle
    cdef zmq_msg_t in_msg
    cdef zmq_msg_t out_msg
    cdef bint swap_ids
    cdef char *msg_c = NULL
    cdef Py_ssize_t msg_c_len
    cdef int rc

    # force swap_ids if both ROUTERs
    swap_ids = (in_socket.type == ROUTER and out_socket.type == ROUTER)
    
    # build zmq_msg objects from str prefixes
    asbuffer_r(in_prefix, <void **>&msg_c, &msg_c_len)
    rc = zmq_msg_init_size(&in_msg, msg_c_len)
    _check_rc(rc)
    
    memcpy(zmq_msg_data(&in_msg), msg_c, zmq_msg_size(&in_msg))
    
    asbuffer_r(out_prefix, <void **>&msg_c, &msg_c_len)
    
    rc = zmq_msg_init_size(&out_msg, msg_c_len)
    _check_rc(rc)
    
    while True:
        with nogil:
            memcpy(zmq_msg_data(&out_msg), msg_c, zmq_msg_size(&out_msg))
            rc = c_monitored_queue(ins, outs, mons, &in_msg, &out_msg, swap_ids)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc

__all__ = ['monitored_queue']
