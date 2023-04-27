cimport cython
from gevent._gevent_cgreenlet cimport Greenlet
from gevent._gevent_c_semaphore cimport Semaphore
from gevent._gevent_cqueue cimport UnboundQueue

@cython.freelist(100)
@cython.internal
@cython.final
cdef class Failure:
    cdef readonly exc
    cdef raise_exception

cdef inline _raise_exc(Failure failure)

cdef class IMapUnordered(Greenlet):
    cdef bint _zipped
    cdef func
    cdef iterable
    cdef spawn
    cdef Semaphore _result_semaphore
    cdef int _outstanding_tasks
    cdef int _max_index

    cdef readonly UnboundQueue queue
    cdef readonly bint finished

    cdef _inext(self)
    cdef _ispawn(self, func, item, int item_index)

    # Passed to greenlet.link
    cpdef _on_result(self, greenlet)
    # Called directly
    cdef _on_finish(self, exception)

    cdef _iqueue_value_for_success(self, greenlet)
    cdef _iqueue_value_for_failure(self, greenlet)
    cdef _iqueue_value_for_self_finished(self)
    cdef _iqueue_value_for_self_failure(self, exception)

cdef class IMap(IMapUnordered):
    cdef int index
    cdef dict _results

    @cython.locals(index=int)
    cdef _inext(self)
