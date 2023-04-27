# cython: auto_pickle=False

cimport cython
from cpython.ref cimport Py_DECREF

from gevent._gevent_c_ident cimport IdentRegistry
from gevent._gevent_c_hub_local cimport get_hub_noargs as get_hub
from gevent._gevent_c_waiter cimport Waiter
from gevent._gevent_c_greenlet_primitives cimport SwitchOutGreenletWithLoop

cdef bint _PYPY
cdef sys_getframe
cdef sys_exc_info
cdef Timeout
cdef GreenletExit
cdef InvalidSwitchError

cdef extern from "greenlet/greenlet.h":

    ctypedef class greenlet.greenlet [object PyGreenlet]:
        # Defining this as a void* means we can't access it as a python attribute
        # in the Python code; but we can't define it as a greenlet because that doesn't
        # properly handle the case that it can be NULL. So instead we inline a getparent
        # function that does the same thing as the green_getparent accessor but without
        # going through the overhead of generic attribute lookup.
        #cdef void* parent
        pass

    # These are actually macros and so must be included
    # (defined) in each .pxd, as are the two functions
    # that call them.
    greenlet PyGreenlet_GetCurrent()
    void* PyGreenlet_GetParent(greenlet)
    void PyGreenlet_Import()

@cython.final
cdef inline greenlet getcurrent():
    return PyGreenlet_GetCurrent()

cdef inline object get_generic_parent(greenlet s):
    # We don't use any typed functions on the return of this,
    # so save the type check by making it just an object.
    cdef object result
    cdef void* parent = PyGreenlet_GetParent(s)
    if parent != NULL:
        # The cast will perform an incref; but the GetParent
        # function already did an incref if we got it (and not NULL).
        # Therefore, we must DECREF immediately.
        result = <object>parent
        Py_DECREF(result)
        return result

cdef inline SwitchOutGreenletWithLoop get_my_hub(greenlet s):
    # This one we do want type checked on the return value.
    # Must not be called with s = None
    cdef object result
    cdef void* parent = PyGreenlet_GetParent(s)
    if parent != NULL:
        result = <object>parent
        # See above
        Py_DECREF(result)
        return result

cdef bint _greenlet_imported

cdef inline void greenlet_init():
    global _greenlet_imported
    if not _greenlet_imported:
        PyGreenlet_Import()
        _greenlet_imported = True

ctypedef object CodeType
ctypedef object FrameType

cdef extern from "_compat.h":
    int Gevent_PyFrame_GetLineNumber(FrameType frame)
    CodeType Gevent_PyFrame_GetCode(FrameType frame)
    object Gevent_PyFrame_GetBack(FrameType frame)
    # We don't do this:
    #
    # ctypedef class types.FrameType [object PyFrameObject]:
    #     pass
    #
    # to avoid "RuntimeWarning: types.FrameType size changed, may
    # indicate binary incompatibility. Expected 56 from C header, got
    # 120 from PyObject" on Python 3.11. That makes the functions that
    # really require that kind of object not safe and capable of crashing the
    # interpreter.
    #
    # However, as of cython 3.0a11, that results in a failure to compile if
    # we have a local variable typed as FrameType, so we can't do that.
    #
    # Also, it removes a layer of type checking and makes it possible to crash
    # the interpreter if you call these functions with something that's not a PyFrameObject.
    # Don't do that.


cdef void _init()

cdef class SpawnedLink:
    cdef public object callback


@cython.final
cdef class SuccessSpawnedLink(SpawnedLink):
    pass

@cython.final
cdef class FailureSpawnedLink(SpawnedLink):
    pass

@cython.final
@cython.internal
@cython.freelist(1000)
cdef class _Frame:
    cdef readonly CodeType f_code
    cdef readonly int f_lineno
    cdef readonly _Frame f_back


@cython.final
@cython.locals(# frame=FrameType, # See above about why we cannot do this
               newest_Frame=_Frame,
               newer_Frame=_Frame,
               older_Frame=_Frame)
cdef _Frame _extract_stack(int limit)

cdef class Greenlet(greenlet):
    cdef readonly object value
    cdef readonly tuple args
    cdef readonly dict kwargs
    cdef readonly object spawning_greenlet
    cdef readonly _Frame spawning_stack
    cdef public dict spawn_tree_locals

    cdef list _links
    cdef tuple _exc_info
    cdef object _notifier
    cdef object _start_event
    cdef str _formatted_info
    cdef object _ident

    cpdef bint has_links(self)
    cpdef join(self, timeout=*)
    cpdef kill(self, exception=*, block=*, timeout=*)
    cpdef bint ready(self)
    cpdef bint successful(self)
    cpdef rawlink(self, object callback)
    cpdef str _formatinfo(self)

    # Helper for killall()
    cpdef bint _maybe_kill_before_start(self, exception)

    # This is a helper function for a @property getter;
    # defining locals() for a @property doesn't seem to work.
    @cython.locals(reg=IdentRegistry)
    cdef _get_minimal_ident(self)

    cdef bint __started_but_aborted(self)
    cdef bint __start_cancelled_by_kill(self)
    cdef bint __start_pending(self)
    cdef bint __never_started_or_killed(self)
    cdef bint __start_completed(self)
    cdef __handle_death_before_start(self, tuple args)
    @cython.final
    cdef inline void __free(self)

    cdef __cancel_start(self)

    cdef inline __report_result(self, object result)
    cdef inline __report_error(self, tuple exc_info)
    # This is used as the target of a callback
    # from the loop, and so needs to be a cpdef
    cpdef _notify_links(self)

    # Hmm, declaring _raise_exception causes issues when _imap
    # is also compiled.
    # TypeError: wrap() takes exactly one argument (0 given)
    # cpdef _raise_exception(self)

# Declare a bunch of imports as cdefs so they can
# be accessed directly as static vars without
# doing a module global lookup. This is especially important
# for spawning greenlets.
cdef _greenlet__init__
cdef _threadlocal
cdef get_hub_class
cdef wref

cdef dump_traceback
cdef load_traceback
cdef Waiter
cdef wait
cdef iwait
cdef reraise
cdef GEVENT_CONFIG


@cython.final
@cython.internal
cdef class _dummy_event:
    cdef readonly bint pending
    cdef readonly bint active

    cpdef stop(self)
    cpdef start(self, cb)
    cpdef close(self)

cdef _dummy_event _cancelled_start_event
cdef _dummy_event _start_completed_event

cpdef _kill(Greenlet glet, object exception, object waiter)

@cython.locals(diehards=list)
cdef _killall3(list greenlets, object exception, object waiter)
cdef _killall(list greenlets, object exception)

@cython.locals(done=list)
cpdef joinall(greenlets, timeout=*, raise_error=*, count=*)

cdef set _spawn_callbacks
cdef void _call_spawn_callbacks(Greenlet gr) except *
