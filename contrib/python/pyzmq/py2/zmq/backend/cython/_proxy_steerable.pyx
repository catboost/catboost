"""Python binding for ZMQ steerable proxy function."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from .libzmq cimport zmq_proxy_steerable
from .socket cimport Socket as cSocket
from .checkrc cimport _check_rc

from zmq.error import InterruptedSystemCall


def proxy_steerable(
        cSocket frontend,
        cSocket backend,
        cSocket capture=None,
        cSocket control=None,
    ):
    """proxy_steerable(frontend, backend, capture, control)

    Start a zeromq proxy with control flow.

    .. versionadded:: libzmq-4.1
    .. versionadded:: 18.0

    Parameters
    ----------
    frontend : Socket
        The Socket instance for the incoming traffic.
    backend : Socket
        The Socket instance for the outbound traffic.
    capture : Socket (optional)
        The Socket instance for capturing traffic.
    control : Socket (optional)
        The Socket instance for control flow.
    """
    cdef int rc = 0
    cdef void* capture_handle
    if isinstance(capture, cSocket):
        capture_handle = capture.handle
    else:
        capture_handle = NULL
    if isinstance(control, cSocket):
        control_handle = control.handle
    else:
        control_handle = NULL
    while True:
        with nogil:
            rc = zmq_proxy_steerable(
                    frontend.handle,
                    backend.handle,
                    capture_handle,
                    control_handle
                )
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc

__all__ = ['proxy_steerable']
