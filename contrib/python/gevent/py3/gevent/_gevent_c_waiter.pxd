cimport cython

from gevent._gevent_c_greenlet_primitives cimport SwitchOutGreenletWithLoop
from gevent._gevent_c_hub_local cimport get_hub_noargs as get_hub

cdef sys
cdef ConcurrentObjectUseError


cdef bint _greenlet_imported
cdef _NONE

cdef extern from "greenlet/greenlet.h":

    ctypedef class greenlet.greenlet [object PyGreenlet]:
        pass

    # These are actually macros and so much be included
    # (defined) in each .pxd, as are the two functions
    # that call them.
    greenlet PyGreenlet_GetCurrent()
    void PyGreenlet_Import()

cdef inline greenlet getcurrent():
    return PyGreenlet_GetCurrent()

cdef inline void greenlet_init():
    global _greenlet_imported
    if not _greenlet_imported:
        PyGreenlet_Import()
        _greenlet_imported = True

cdef class Waiter:
    cdef readonly SwitchOutGreenletWithLoop hub
    cdef readonly greenlet greenlet
    cdef readonly value
    cdef _exception

    cpdef get(self)
    cpdef clear(self)

    # cpdef of switch leads to parameter errors...
    #cpdef switch(self, value)

@cython.final
@cython.internal
cdef class MultipleWaiter(Waiter):
    cdef list _values
