from libc.errno cimport EINTR, EAGAIN
from cpython cimport PyErr_CheckSignals

from .libzmq cimport zmq_errno, ZMQ_ETERM


cdef inline int _check_rc(int rc) except -1:
    """internal utility for checking zmq return condition

    and raising the appropriate Exception class
    """
    cdef int errno = zmq_errno()
    PyErr_CheckSignals()
    if rc == -1: # if rc < -1, it's a bug in libzmq. Should we warn?
        if errno == EINTR:
            from zmq.error import InterruptedSystemCall
            raise InterruptedSystemCall(errno)
        elif errno == EAGAIN:
            from zmq.error import Again
            raise Again(errno)
        elif errno == ZMQ_ETERM:
            from zmq.error import ContextTerminated
            raise ContextTerminated(errno)
        else:
            from zmq.error import ZMQError
            raise ZMQError(errno)
    return 0
