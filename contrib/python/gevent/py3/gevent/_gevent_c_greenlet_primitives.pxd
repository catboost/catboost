cimport cython

# This file must not cimport anything from gevent.
cdef get_objects
cdef wref

cdef BlockingSwitchOutError


cdef extern from "greenlet/greenlet.h":

    ctypedef class greenlet.greenlet [object PyGreenlet]:
        pass

    # These are actually macros and so much be included
    # (defined) in each .pxd, as are the two functions
    # that call them.
    greenlet PyGreenlet_GetCurrent()
    object PyGreenlet_Switch(greenlet self, void* args, void* kwargs)
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

cdef inline object _greenlet_switch(greenlet self):
    return PyGreenlet_Switch(self, NULL, NULL)

cdef class TrackedRawGreenlet(greenlet):
    pass

cdef class SwitchOutGreenletWithLoop(TrackedRawGreenlet):
    cdef public loop

    cpdef switch(self)
    cpdef switch_out(self)


cpdef list get_reachable_greenlets()

cdef type _memoryview
cdef type _buffer

cpdef get_memory(data)
