cimport cython

from gevent._gevent_c_greenlet_primitives cimport SwitchOutGreenletWithLoop
from gevent._gevent_c_hub_local cimport get_hub_noargs as get_hub

from gevent._gevent_c_waiter cimport Waiter
from gevent._gevent_c_waiter cimport MultipleWaiter

cdef InvalidSwitchError
cdef _waiter
cdef _greenlet_primitives
cdef traceback
cdef _timeout_error
cdef Timeout


cdef extern from "greenlet/greenlet.h":

    ctypedef class greenlet.greenlet [object PyGreenlet]:
        pass

    # These are actually macros and so much be included
    # (defined) in each .pxd, as are the two functions
    # that call them.
    greenlet PyGreenlet_GetCurrent()
    void PyGreenlet_Import()

@cython.final
cdef inline greenlet getcurrent():
    return PyGreenlet_GetCurrent()

cdef bint _greenlet_imported

cdef inline void greenlet_init():
    global _greenlet_imported
    if not _greenlet_imported:
        PyGreenlet_Import()
        _greenlet_imported = True


cdef class WaitOperationsGreenlet(SwitchOutGreenletWithLoop):
    # The Hub will extend this class.
    cpdef wait(self, watcher)
    cpdef cancel_wait(self, watcher, error, close_watcher=*)
    cpdef _cancel_wait(self, watcher, error, close_watcher)

cdef class _WaitIterator:
    cdef SwitchOutGreenletWithLoop _hub
    cdef MultipleWaiter _waiter
    cdef _switch
    cdef _timeout
    cdef _objects
    cdef _timer
    cdef Py_ssize_t _count
    cdef bint _begun


    cdef _begin(self)
    cdef _cleanup(self)

    cpdef __enter__(self)
    cpdef __exit__(self, typ, value, tb)


cpdef iwait_on_objects(objects, timeout=*, count=*)
cpdef wait_on_objects(objects=*, timeout=*, count=*)

cdef _primitive_wait(watcher, timeout, timeout_exc, WaitOperationsGreenlet hub)
cpdef wait_on_watcher(watcher, timeout=*, timeout_exc=*, WaitOperationsGreenlet hub=*)
cpdef wait_read(fileno, timeout=*, timeout_exc=*)
cpdef wait_write(fileno, timeout=*, timeout_exc=*, event=*)
cpdef wait_readwrite(fileno, timeout=*, timeout_exc=*, event=*)
cpdef wait_on_socket(socket, watcher, timeout_exc=*)
