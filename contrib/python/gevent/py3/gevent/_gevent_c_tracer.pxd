cimport cython

cdef sys
cdef traceback

cdef settrace
cdef getcurrent

cdef format_run_info

cdef perf_counter
cdef gmctime


cdef class GreenletTracer:
    cdef readonly object active_greenlet
    cdef readonly object previous_trace_function
    cdef readonly Py_ssize_t greenlet_switch_counter

    cdef bint _killed

    cpdef _trace(self, str event, tuple args)

    @cython.locals(did_switch=bint)
    cpdef did_block_hub(self, hub)

    cpdef kill(self)

@cython.internal
cdef class _HubTracer(GreenletTracer):
    cdef readonly object hub
    cdef readonly double max_blocking_time


cdef class HubSwitchTracer(_HubTracer):
    cdef readonly double last_entered_hub

cdef class MaxSwitchTracer(_HubTracer):
    cdef readonly double max_blocking
    cdef readonly double last_switch

    @cython.locals(switched_at=double)
    cpdef _trace(self, str event, tuple args)
