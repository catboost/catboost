"""Python binding for 0MQ device function."""

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
# Imports
#-----------------------------------------------------------------------------

from .libzmq cimport zmq_device, zmq_proxy, ZMQ_VERSION_MAJOR
from .socket cimport Socket as cSocket
from .checkrc cimport _check_rc

from zmq.error import InterruptedSystemCall

#-----------------------------------------------------------------------------
# Basic device API
#-----------------------------------------------------------------------------

def device(int device_type, cSocket frontend, cSocket backend=None):
    """device(device_type, frontend, backend)

    Start a zeromq device.
    
    .. deprecated:: libzmq-3.2
        Use zmq.proxy

    Parameters
    ----------
    device_type : (QUEUE, FORWARDER, STREAMER)
        The type of device to start.
    frontend : Socket
        The Socket instance for the incoming traffic.
    backend : Socket
        The Socket instance for the outbound traffic.
    """
    if ZMQ_VERSION_MAJOR >= 3:
        return proxy(frontend, backend)

    cdef int rc = 0
    while True:
        with nogil:
            rc = zmq_device(device_type, frontend.handle, backend.handle)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc

def proxy(cSocket frontend, cSocket backend, cSocket capture=None):
    """proxy(frontend, backend, capture)
    
    Start a zeromq proxy (replacement for device).
    
    .. versionadded:: libzmq-3.2
    .. versionadded:: 13.0
    
    Parameters
    ----------
    frontend : Socket
        The Socket instance for the incoming traffic.
    backend : Socket
        The Socket instance for the outbound traffic.
    capture : Socket (optional)
        The Socket instance for capturing traffic.
    """
    cdef int rc = 0
    cdef void* capture_handle
    if isinstance(capture, cSocket):
        capture_handle = capture.handle
    else:
        capture_handle = NULL
    while True:
        with nogil:
            rc = zmq_proxy(frontend.handle, backend.handle, capture_handle)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc

__all__ = ['device', 'proxy']

