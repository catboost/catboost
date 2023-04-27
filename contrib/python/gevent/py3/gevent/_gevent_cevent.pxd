cimport cython

from gevent._gevent_c_hub_local cimport get_hub_noargs as get_hub
from gevent._gevent_c_abstract_linkable cimport AbstractLinkable

cdef _None
cdef reraise
cdef dump_traceback
cdef load_traceback

cdef Timeout

cdef class Event(AbstractLinkable):
   cdef bint _flag

cdef class AsyncResult(AbstractLinkable):
    cdef readonly _value
    cdef readonly tuple _exc_info

    # For the use of _imap.py
    cdef public int _imap_task_index

    cpdef get(self, block=*, timeout=*)
    cpdef bint successful(self)

    cpdef wait(self, timeout=*)
    cpdef bint done(self)

    cpdef bint cancel(self)
    cpdef bint cancelled(self)
