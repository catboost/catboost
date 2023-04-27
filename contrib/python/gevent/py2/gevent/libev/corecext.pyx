# Copyright (c) 2009-2012 Denis Bilenko. See LICENSE for details.

# This first directive, supported in Cython 0.24+, causes sources
# files to be *much* smaller when it's false (139,027 LOC vs 35,000
# LOC) and thus cythonpp.py (and probably the compiler; also Visual C
# has limits on source file sizes) to be faster (73s vs 46s). But it does
# make debugging more difficult. Auto-pickling was added in 0.26, and
# that's a new feature that we don't need or want to allow in a gevent
# point release.

# cython: emit_code_comments=False, auto_pickle=False, language_level=3str

# NOTE: We generally cannot use the Cython IF directive as documented
# at
# http://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#conditional-compilation
# (e.g., IF UNAME_SYSNAME == "Windows") because when Cython says
# "compilation", it means when *Cython* compiles, not when the C
# compiler compiles. We distribute an sdist with a single pre-compiled
# C file for all platforms so that end users that don't use a binary
# wheel don't have to sit through cythonpp and other steps the Makefile does.
# See https://github.com/gevent/gevent/issues/1076
# We compile in 3str mode, which should mean we get absolute import
# by default.
from __future__ import absolute_import

cimport cython
cimport libev

from cpython.ref cimport Py_INCREF
from cpython.ref cimport Py_DECREF
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Free
from libc.errno cimport errno

cdef extern from "Python.h":
    int    Py_ReprEnter(object)
    void   Py_ReprLeave(object)

import sys
import os
import traceback
import signal as signalmodule
from gevent import getswitchinterval
from gevent.exceptions import HubDestroyed


__all__ = ['get_version',
           'get_header_version',
           'supported_backends',
           'recommended_backends',
           'embeddable_backends',
           'time',
           'loop']

cdef tuple integer_types

if sys.version_info[0] >= 3:
    integer_types = int,
else:
    integer_types = (int, long)


cdef extern from "callbacks.h":
    void gevent_callback_io(libev.ev_loop, void*, int)
    void gevent_callback_timer(libev.ev_loop, void*, int)
    void gevent_callback_signal(libev.ev_loop, void*, int)
    void gevent_callback_idle(libev.ev_loop, void*, int)
    void gevent_callback_prepare(libev.ev_loop, void*, int)
    void gevent_callback_check(libev.ev_loop, void*, int)
    void gevent_callback_fork(libev.ev_loop, void*, int)
    void gevent_callback_async(libev.ev_loop, void*, int)
    void gevent_callback_child(libev.ev_loop, void*, int)
    void gevent_callback_stat(libev.ev_loop, void*, int)
    void gevent_run_callbacks(libev.ev_loop, void*, int)
    void gevent_periodic_signal_check(libev.ev_loop, void*, int)
    void gevent_call(loop, callback)
    void gevent_noop(libev.ev_loop, void*, int)
    void* gevent_realloc(void*, long size)

cdef extern from "stathelper.c":
    object _pystat_fromstructstat(void*)


UNDEF = libev.EV_UNDEF
NONE = libev.EV_NONE
READ = libev.EV_READ
WRITE = libev.EV_WRITE
TIMER = libev.EV_TIMER
PERIODIC = libev.EV_PERIODIC
SIGNAL = libev.EV_SIGNAL
CHILD = libev.EV_CHILD
STAT = libev.EV_STAT
IDLE = libev.EV_IDLE
PREPARE = libev.EV_PREPARE
CHECK = libev.EV_CHECK
EMBED = libev.EV_EMBED
FORK = libev.EV_FORK
CLEANUP = libev.EV_CLEANUP
ASYNC = libev.EV_ASYNC
CUSTOM = libev.EV_CUSTOM
ERROR = libev.EV_ERROR

READWRITE = libev.EV_READ | libev.EV_WRITE

MINPRI = libev.EV_MINPRI
MAXPRI = libev.EV_MAXPRI

BACKEND_SELECT = libev.EVBACKEND_SELECT
BACKEND_POLL = libev.EVBACKEND_POLL
BACKEND_EPOLL = libev.EVBACKEND_EPOLL
BACKEND_KQUEUE = libev.EVBACKEND_KQUEUE
BACKEND_DEVPOLL = libev.EVBACKEND_DEVPOLL
BACKEND_PORT = libev.EVBACKEND_PORT
#BACKEND_LINUXAIO = libev.EVBACKEND_LINUXAIO
#BACKEND_IOURING = libev.EVBACKEND_IOURING


FORKCHECK = libev.EVFLAG_FORKCHECK
NOINOTIFY = libev.EVFLAG_NOINOTIFY
SIGNALFD = libev.EVFLAG_SIGNALFD
NOSIGMASK = libev.EVFLAG_NOSIGMASK


@cython.internal
cdef class _EVENTSType:

    def __repr__(self):
        return 'gevent.core.EVENTS'


cdef public object GEVENT_CORE_EVENTS = _EVENTSType()
EVENTS = GEVENT_CORE_EVENTS


def get_version():
    return 'libev-%d.%02d' % (libev.ev_version_major(), libev.ev_version_minor())


def get_header_version():
    return 'libev-%d.%02d' % (libev.EV_VERSION_MAJOR, libev.EV_VERSION_MINOR)


# This list backends in the order they are actually tried by libev,
# as defined in loop_init. The names must be lower case.
_flags = [
    # IOCP
    (libev.EVBACKEND_PORT, 'port'),
    (libev.EVBACKEND_KQUEUE, 'kqueue'),
    #(libev.EVBACKEND_IOURING, 'linux_iouring'),
    #(libev.EVBACKEND_LINUXAIO, "linux_aio"),
    (libev.EVBACKEND_EPOLL, 'epoll'),
    (libev.EVBACKEND_POLL, 'poll'),
    (libev.EVBACKEND_SELECT, 'select'),

    (libev.EVFLAG_NOENV, 'noenv'),
    (libev.EVFLAG_FORKCHECK, 'forkcheck'),
    (libev.EVFLAG_NOINOTIFY, 'noinotify'),
    (libev.EVFLAG_SIGNALFD, 'signalfd'),
    (libev.EVFLAG_NOSIGMASK, 'nosigmask')
]


_flags_str2int = dict((string, flag) for (flag, string) in _flags)


_events = [(libev.EV_READ,     'READ'),
           (libev.EV_WRITE,    'WRITE'),
           (libev.EV__IOFDSET, '_IOFDSET'),
           (libev.EV_PERIODIC, 'PERIODIC'),
           (libev.EV_SIGNAL,   'SIGNAL'),
           (libev.EV_CHILD,    'CHILD'),
           (libev.EV_STAT,     'STAT'),
           (libev.EV_IDLE,     'IDLE'),
           (libev.EV_PREPARE,  'PREPARE'),
           (libev.EV_CHECK,    'CHECK'),
           (libev.EV_EMBED,    'EMBED'),
           (libev.EV_FORK,     'FORK'),
           (libev.EV_CLEANUP,  'CLEANUP'),
           (libev.EV_ASYNC,    'ASYNC'),
           (libev.EV_CUSTOM,   'CUSTOM'),
           (libev.EV_ERROR,    'ERROR')]


cpdef _flags_to_list(unsigned int flags):
    cdef list result = []
    for code, value in _flags:
        if flags & code:
            result.append(value)
        flags &= ~code
        if not flags:
            break
    if flags:
        result.append(flags)
    return result


if sys.version_info[0] >= 3:
    basestring = (bytes, str)
else:
    basestring = __builtins__.basestring


cpdef unsigned int _flags_to_int(object flags) except? -1:
    # Note, that order does not matter, libev has its own predefined order
    if not flags:
        return 0
    if isinstance(flags, integer_types):
        return flags
    cdef unsigned int result = 0
    try:
        if isinstance(flags, basestring):
            flags = flags.split(',')
        for value in flags:
            value = value.strip().lower()
            if value:
                result |= _flags_str2int[value]
    except KeyError as ex:
        raise ValueError('Invalid backend or flag: %s\nPossible values: %s' % (ex, ', '.join(sorted(_flags_str2int.keys()))))
    return result


cdef str _str_hex(object flag):
    if isinstance(flag, integer_types):
        return hex(flag)
    return str(flag)


cpdef _check_flags(unsigned int flags):
    cdef list as_list
    flags &= libev.EVBACKEND_MASK
    if not flags:
        return
    if not (flags & libev.EVBACKEND_ALL):
        raise ValueError('Invalid value for backend: 0x%x' % flags)
    if not (flags & libev.ev_supported_backends()):
        as_list = [_str_hex(x) for x in _flags_to_list(flags)]
        raise ValueError('Unsupported backend: %s' % '|'.join(as_list))


cpdef _events_to_str(int events):
    cdef list result = []
    cdef int c_flag
    for (flag, string) in _events:
        c_flag = flag
        if events & c_flag:
            result.append(string)
            events = events & (~c_flag)
        if not events:
            break
    if events:
        result.append(hex(events))
    return '|'.join(result)


def supported_backends():
    return _flags_to_list(libev.ev_supported_backends())


def recommended_backends():
    return _flags_to_list(libev.ev_recommended_backends())


def embeddable_backends():
    return _flags_to_list(libev.ev_embeddable_backends())


def time():
    return libev.ev_time()

cdef bint _check_loop(loop loop) except -1:
    if not loop._ptr:
        raise ValueError('operation on destroyed loop')
    return 1



cdef public class callback [object PyGeventCallbackObject, type PyGeventCallback_Type]:
    cdef public object callback
    cdef public tuple args
    cdef callback next

    def __init__(self, callback, args):
        self.callback = callback
        self.args = args

    def stop(self):
        self.callback = None
        self.args = None

    close = stop

    # Note, that __nonzero__ and pending are different
    # nonzero is used in contexts where we need to know whether to schedule another callback,
    # so it's true if it's pending or currently running
    # 'pending' has the same meaning as libev watchers: it is cleared before entering callback

    def __nonzero__(self):
        # it's nonzero if it's pending or currently executing
        return self.args is not None

    @property
    def pending(self):
        return self.callback is not None

    def __repr__(self):
        if Py_ReprEnter(self) != 0:
            return "<...>"
        try:
            format = self._format()
            result = "<%s at 0x%x%s" % (self.__class__.__name__, id(self), format)
            if self.pending:
                result += " pending"
            if self.callback is not None:
                result += " callback=%r" % (self.callback, )
            if self.args is not None:
                result += " args=%r" % (self.args, )
            if self.callback is None and self.args is None:
                result += " stopped"
            return result + ">"
        finally:
            Py_ReprLeave(self)

    def _format(self):
        return ''

# See comments in cares.pyx about DEF constants and when to use
# what kind.
cdef extern from *:
    """
    #define CALLBACK_CHECK_COUNT 50
    """
    int CALLBACK_CHECK_COUNT

@cython.final
@cython.internal
cdef class CallbackFIFO(object):
    cdef callback head
    cdef callback tail

    def __init__(self):
        self.head = None
        self.tail = None

    cdef inline clear(self):
        self.head = None
        self.tail = None

    cdef inline callback popleft(self):
        cdef callback head = self.head
        self.head = head.next
        if self.head is self.tail or self.head is None:
            self.tail = None
        head.next = None
        return head

    cdef inline append(self, callback new_tail):
        assert not new_tail.next
        if self.tail is None:
            if self.head is None:
                # Completely empty, so this
                # is now our head
                self.head = new_tail
                return
            self.tail = self.head

        assert self.head is not None
        old_tail = self.tail
        old_tail.next = new_tail
        self.tail = new_tail

    def __nonzero__(self):
        return self.head is not None

    def __len__(self):
        cdef Py_ssize_t count = 0
        head = self.head
        while head is not None:
            count += 1
            head = head.next
        return count

    def __iter__(self):
        cdef list objects = []
        head = self.head
        while head is not None:
            objects.append(head)
            head = head.next
        return iter(objects)

    cdef bint has_callbacks(self):
        return self.head

    def __repr__(self):
        return "<callbacks@%r len=%d head=%r tail=%r>" % (id(self), len(self), self.head, self.tail)


cdef public class loop [object PyGeventLoopObject, type PyGeventLoop_Type]:
    ## embedded struct members
    cdef libev.ev_prepare _prepare
    cdef libev.ev_timer _timer0
    cdef libev.ev_async _threadsafe_async
    # We'll only actually start this timer if we're on Windows,
    # but it doesn't hurt to compile it in on all platforms.
    cdef libev.ev_timer _periodic_signal_checker

    ## pointer members
    cdef public object error_handler
    cdef libev.ev_loop* _ptr
    cdef public CallbackFIFO _callbacks

    ## data members
    cdef bint starting_timer_may_update_loop_time
    # We must capture the 'default' state at initialiaztion
    # time. Destroying the default loop in libev sets
    # the libev internal pointer to 0, and ev_is_default_loop will
    # no longer work.
    cdef bint _default
    cdef readonly double approx_timer_resolution

    def __cinit__(self, object flags=None, object default=None, libev.intptr_t ptr=0):
        self.starting_timer_may_update_loop_time = 0
        self._default = 0
        libev.ev_prepare_init(&self._prepare,
                              <void*>gevent_run_callbacks)
        libev.ev_timer_init(&self._periodic_signal_checker,
                            <void*>gevent_periodic_signal_check,
                            0.3, 0.3)
        libev.ev_timer_init(&self._timer0,
                            <void*>gevent_noop,
                            0.0, 0.0)
        libev.ev_async_init(&self._threadsafe_async,
                            <void*>gevent_noop)

        cdef unsigned int c_flags
        if ptr:
            self._ptr = <libev.ev_loop*>ptr
            self._default = libev.ev_is_default_loop(self._ptr)
        else:
            c_flags = _flags_to_int(flags)
            _check_flags(c_flags)
            c_flags |= libev.EVFLAG_NOENV
            c_flags |= libev.EVFLAG_FORKCHECK
            if default is None:
                default = True
            if default:
                self._default = 1
                self._ptr = libev.gevent_ev_default_loop(c_flags)
                if not self._ptr:
                    raise SystemError("ev_default_loop(%s) failed" % (c_flags, ))
                if sys.platform == "win32":
                    libev.ev_timer_start(self._ptr, &self._periodic_signal_checker)
                    libev.ev_unref(self._ptr)
            else:
                self._ptr = libev.ev_loop_new(c_flags)
                if not self._ptr:
                    raise SystemError("ev_loop_new(%s) failed" % (c_flags, ))
            if default or SYSERR_CALLBACK is None:
                set_syserr_cb(self._handle_syserr)

        # Mark as not destroyed
        libev.ev_set_userdata(self._ptr, self._ptr)

        libev.ev_prepare_start(self._ptr, &self._prepare)
        libev.ev_unref(self._ptr)

        libev.ev_async_start(self._ptr, &self._threadsafe_async)
        libev.ev_unref(self._ptr)


    def __init__(self, object flags=None, object default=None, libev.intptr_t ptr=0):
        self._callbacks = CallbackFIFO()
        # See libev.corecffi for this attribute.
        self.approx_timer_resolution = 0.00001

    cdef _run_callbacks(self):
        cdef callback cb
        cdef int count = CALLBACK_CHECK_COUNT
        self.starting_timer_may_update_loop_time = True
        cdef libev.ev_tstamp now = libev.ev_now(self._ptr)
        cdef libev.ev_tstamp expiration = now + <libev.ev_tstamp>getswitchinterval()

        try:
            libev.ev_timer_stop(self._ptr, &self._timer0)
            while self._callbacks.head is not None:
                cb = self._callbacks.popleft()

                libev.ev_unref(self._ptr)
                # On entry, this will set cb.callback to None,
                # changing cb.pending from True to False; on exit,
                # this will set cb.args to None, changing bool(cb)
                # from True to False.
                # XXX: Why is this a C callback, not cython?
                gevent_call(self, cb)
                count -= 1

                if count == 0 and self._callbacks.head is not None:
                    # We still have more to run but we've reached
                    # the end of one check group
                    count = CALLBACK_CHECK_COUNT

                    libev.ev_now_update(self._ptr)
                    if libev.ev_now(self._ptr) >= expiration:
                        now = 0
                        break

            if now != 0:
                libev.ev_now_update(self._ptr)
            if self._callbacks.head is not None:
                libev.ev_timer_start(self._ptr, &self._timer0)
        finally:
            self.starting_timer_may_update_loop_time = False

    cdef _stop_watchers(self, libev.ev_loop* ptr):
        if not ptr:
            return

        if libev.ev_is_active(&self._prepare):
            libev.ev_ref(ptr)
            libev.ev_prepare_stop(ptr, &self._prepare)
        if libev.ev_is_active(&self._periodic_signal_checker):
            libev.ev_ref(ptr)
            libev.ev_timer_stop(ptr, &self._periodic_signal_checker)
        if libev.ev_is_active(&self._threadsafe_async):
            libev.ev_ref(ptr)
            libev.ev_async_stop(ptr, &self._threadsafe_async)

    def destroy(self):
        cdef libev.ev_loop* ptr = self._ptr
        self._ptr = NULL

        if ptr:
            if not libev.ev_userdata(ptr):
                # Whoops! Program error. They destroyed the loop,
                # using a different loop object. Our _ptr is still
                # valid, but the libev loop is gone. Doing anything
                # else with it will likely cause a crash.
                return
            # Mark as destroyed
            self._stop_watchers(ptr)
            libev.ev_set_userdata(ptr, NULL)
            if SYSERR_CALLBACK == self._handle_syserr:
                set_syserr_cb(None)
            libev.ev_loop_destroy(ptr)

    def __dealloc__(self):
        cdef libev.ev_loop* ptr = self._ptr
        self._ptr = NULL
        if ptr != NULL:
            if not libev.ev_userdata(ptr):
                # See destroy(). This is a bug in the caller.
                return
            self._stop_watchers(ptr)
            if not self._default:
                libev.ev_loop_destroy(ptr)
                # Mark as destroyed
                libev.ev_set_userdata(ptr, NULL)

    @property
    def ptr(self):
        return <size_t>self._ptr

    @property
    def WatcherType(self):
        return watcher

    @property
    def MAXPRI(self):
        return libev.EV_MAXPRI

    @property
    def MINPRI(self):
        return libev.EV_MINPRI

    def _handle_syserr(self, message, errno):
        if sys.version_info[0] >= 3:
            message = message.decode()
        self.handle_error(None, SystemError, SystemError(message + ': ' + os.strerror(errno)), None)

    cpdef handle_error(self, context, type, value, tb):
        cdef object handle_error
        cdef object error_handler = self.error_handler
        if type is HubDestroyed:
            self._callbacks.clear()
            self.break_()
            return

        if error_handler is not None:
            # we do want to do getattr every time so that setting Hub.handle_error property just works
            handle_error = getattr(error_handler, 'handle_error', error_handler)
            handle_error(context, type, value, tb)
        else:
            self._default_handle_error(context, type, value, tb)

    cpdef _default_handle_error(self, context, type, value, tb):
        # note: Hub sets its own error handler so this is not used by gevent
        # this is here to make core.loop usable without the rest of gevent
        traceback.print_exception(type, value, tb)
        if self._ptr:
            libev.ev_break(self._ptr, libev.EVBREAK_ONE)

    def run(self, nowait=False, once=False):
        _check_loop(self)
        cdef unsigned int flags = 0
        if nowait:
            flags |= libev.EVRUN_NOWAIT
        if once:
            flags |= libev.EVRUN_ONCE
        with nogil:
            libev.ev_run(self._ptr, flags)

    def reinit(self):
        if self._ptr:
            libev.ev_loop_fork(self._ptr)

    def ref(self):
        _check_loop(self)
        libev.ev_ref(self._ptr)

    def unref(self):
        _check_loop(self)
        libev.ev_unref(self._ptr)

    def break_(self, int how=libev.EVBREAK_ONE):
        _check_loop(self)
        libev.ev_break(self._ptr, how)

    def verify(self):
        _check_loop(self)
        libev.ev_verify(self._ptr)

    cpdef libev.ev_tstamp now(self) except *:
        _check_loop(self)
        return libev.ev_now(self._ptr)

    cpdef void update_now(self) except *:
        _check_loop(self)
        libev.ev_now_update(self._ptr)

    update = update_now # Old name, deprecated.

    def __repr__(self):
        return '<%s at 0x%x %s>' % (self.__class__.__name__, id(self), self._format())

    @property
    def default(self):
        # If we're destroyed, we are not the default loop anymore,
        # as far as Python is concerned.
        return self._default if self._ptr else False

    @property
    def iteration(self):
        _check_loop(self)
        return libev.ev_iteration(self._ptr)

    @property
    def depth(self):
        _check_loop(self)
        return libev.ev_depth(self._ptr)

    @property
    def backend_int(self):
        _check_loop(self)
        return libev.ev_backend(self._ptr)

    @property
    def backend(self):
        _check_loop(self)
        cdef unsigned int backend = libev.ev_backend(self._ptr)
        for key, value in _flags:
            if key == backend:
                return value
        return backend

    @property
    def pendingcnt(self):
        _check_loop(self)
        return libev.ev_pending_count(self._ptr)

    def io(self, libev.vfd_socket_t fd, int events, ref=True, priority=None):
        return io(self, fd, events, ref, priority)

    def closing_fd(self, libev.vfd_socket_t fd):
        _check_loop(self)
        cdef int pending_before = libev.ev_pending_count(self._ptr)
        libev.ev_feed_fd_event(self._ptr, fd, 0xFFFF)
        cdef int pending_after = libev.ev_pending_count(self._ptr)
        return pending_after > pending_before

    def timer(self, double after, double repeat=0.0, ref=True, priority=None):
        return timer(self, after, repeat, ref, priority)

    def signal(self, int signum, ref=True, priority=None):
        return signal(self, signum, ref, priority)

    def idle(self, ref=True, priority=None):
        return idle(self, ref, priority)

    def prepare(self, ref=True, priority=None):
        return prepare(self, ref, priority)

    def check(self, ref=True, priority=None):
        return check(self, ref, priority)

    def fork(self, ref=True, priority=None):
        return fork(self, ref, priority)

    def async_(self, ref=True, priority=None):
        return async_(self, ref, priority)

    # cython doesn't enforce async as a keyword
    async = async_

    def child(self, int pid, bint trace=0, ref=True):
        if sys.platform == 'win32':
            raise AttributeError("Child watchers are not supported on Windows")
        return child(self, pid, trace, ref)

    def install_sigchld(self):
        libev.gevent_install_sigchld_handler()

    def reset_sigchld(self):
        libev.gevent_reset_sigchld_handler()

    def stat(self, str path, float interval=0.0, ref=True, priority=None):
        return stat(self, path, interval, ref, priority)

    def run_callback(self, func, *args):
        _check_loop(self)
        cdef callback cb = callback(func, args)
        self._callbacks.append(cb)
        libev.ev_ref(self._ptr)
        return cb

    def run_callback_threadsafe(self, func, *args):
        # We rely on the GIL to make this threadsafe.
        cb = self.run_callback(func, *args)
        libev.ev_async_send(self._ptr, &self._threadsafe_async)
        return cb

    def _format(self):
        if not self._ptr:
            return 'destroyed'
        cdef object msg = self.backend
        if self._default:
            msg += ' default'
        msg += ' pending=%s' % self.pendingcnt
        msg += self._format_details()
        return msg

    def _format_details(self):
        cdef str msg = ''
        cdef object fileno = self.fileno()
        cdef object activecnt = None
        try:
            activecnt = self.activecnt
        except AttributeError:
            pass
        if activecnt is not None:
            msg += ' ref=' + repr(activecnt)
        if fileno is not None:
            msg += ' fileno=' + repr(fileno)
        return msg

    def fileno(self):
        cdef int fd
        if self._ptr:
            fd = libev.gevent_ev_loop_backend_fd(self._ptr)
            if fd >= 0:
                return fd

    @property
    def activecnt(self):
        _check_loop(self)
        return libev.gevent_ev_loop_activecnt(self._ptr)

    @property
    def sig_pending(self):
        _check_loop(self)
        return libev.gevent_ev_loop_sig_pending(self._ptr)

    @property
    def origflags(self):
        return _flags_to_list(self.origflags_int)

    @property
    def origflags_int(self):
        _check_loop(self)
        return libev.gevent_ev_loop_origflags(self._ptr)

    @property
    def sigfd(self):
        _check_loop(self)
        fd = libev.gevent_ev_loop_sigfd(self._ptr)
        if fd >= 0:
            return fd

        # Explicitly not EV_USE_SIGNALFD
        raise AttributeError("sigfd")


from zope.interface import classImplements

# XXX: This invokes the side-table lookup, we would
# prefer to have it stored directly on the class. That means we
# need a class variable ``__implemented__``, but that's hard in
# Cython
from gevent._interfaces import ILoop
from gevent._interfaces import ICallback
classImplements(loop, ILoop)
classImplements(callback, ICallback)


cdef extern from *:
    """
    #define FLAG_WATCHER_OWNS_PYREF  (1 << 0) /* 0x1 */
    #define FLAG_WATCHER_NEEDS_EVREF (1 << 1) /* 0x2 */
    #define FLAG_WATCHER_UNREF_BEFORE_START (1 << 2) /* 0x4 */
    #define FLAG_WATCHER_MASK_UNREF_NEEDS_REF 0x6
    """
    # about readonly _flags attribute:
    # bit #1 set if object owns Python reference to itself (Py_INCREF was
    # called and we must call Py_DECREF later)
    unsigned int FLAG_WATCHER_OWNS_PYREF
    # bit #2 set if ev_unref() was called and we must call ev_ref() later
    unsigned int FLAG_WATCHER_NEEDS_EVREF
    # bit #3 set if user wants to call ev_unref() before start()
    unsigned int FLAG_WATCHER_UNREF_BEFORE_START
    # bits 2 and 3 are *both* set when we are active, but the user
    # request us not to be ref'd anymore. We unref us (because going active will
    # ref us) and then make a note of this in the future
    unsigned int FLAG_WATCHER_MASK_UNREF_NEEDS_REF


cdef void _python_incref(watcher self):
    if not self._flags & FLAG_WATCHER_OWNS_PYREF:
        Py_INCREF(self)
        self._flags |= FLAG_WATCHER_OWNS_PYREF

cdef void _python_decref(watcher self):
    if self._flags & FLAG_WATCHER_OWNS_PYREF:
        Py_DECREF(self)
        self._flags &= ~FLAG_WATCHER_OWNS_PYREF

cdef void _libev_ref(watcher self):
    if self._flags & FLAG_WATCHER_NEEDS_EVREF:
        libev.ev_ref(self.loop._ptr)
        self._flags &= ~FLAG_WATCHER_NEEDS_EVREF

cdef void _libev_unref(watcher self):
    if self._flags & FLAG_WATCHER_MASK_UNREF_NEEDS_REF == FLAG_WATCHER_UNREF_BEFORE_START:
        libev.ev_unref(self.loop._ptr)
        self._flags |= FLAG_WATCHER_NEEDS_EVREF


ctypedef void (*start_stop_func)(libev.ev_loop*, void*) nogil

cdef struct start_and_stop:
    start_stop_func start
    start_stop_func stop

cdef start_and_stop make_ss(void* start, void* stop):
    cdef start_and_stop result = start_and_stop(<start_stop_func>start, <start_stop_func>stop)
    return result

cdef bint _watcher_start(watcher self, object callback, tuple args) except -1:
    # This method should be called by subclasses of watcher, if they
    # override the python-level `start` function: they've already paid
    # for argument unpacking, and `start` cannot be cpdef since it
    # uses varargs.

    # We keep this as a function, not a cdef method of watcher.
    # If it's a cdef method, it could potentially be overridden
    # by a subclass, which means that the watcher gains a pointer to a
    # function table (vtable), making each object 8 bytes larger.

    _check_loop(self.loop)
    if callback is None or not callable(callback):
        raise TypeError("Expected callable, not %r" % (callback, ))
    self._callback = callback
    self.args = args
    _libev_unref(self)
    _python_incref(self)
    self._w_ss.start(self.loop._ptr, self._w_watcher)
    return 1

cdef public class watcher [object PyGeventWatcherObject, type PyGeventWatcher_Type]:
    """Abstract base class for all the watchers"""
    ## pointer members
    cdef public loop loop
    cdef object _callback
    cdef public tuple args

    # By keeping a _w_watcher cached, the size of the io and timer
    # structs becomes 152 bytes and child is 160 and stat is 512 (when
    # the start_and_stop is inlined). On 64-bit macOS CPython 2.7. I
    # hoped that using libev's data pointer and allocating the
    # watchers directly and not as inline members would result in
    # overall savings thanks to better padding, but it didn't. And it
    # added lots of casts, making the code ugly.

    # Table:
    # gevent ver   | 1.2 | This | +data
    # Watcher Kind |     |      |
    # Timer        | 120 | 152  | 160
    # IO           | 120 | 152  | 160
    # Child        | 128 | 160  | 168
    # Stat         | 480 | 512  | 512
    cdef libev.ev_watcher* _w_watcher

    # By inlining the start_and_stop struct, instead of taking the address
    # of a static struct or using the watcher's data pointer, we
    # use an additional pointer of memory and incur an additional pointer copy
    # on creation.
    # But we use fewer pointer accesses for start/stop, and they have
    # better cache locality. (Then again, we're bigger).
    # Right now we're going for size, so we use the pointer. IO/Timer objects
    # are then 144 bytes.
    cdef start_and_stop* _w_ss

    ## Int members

    # Our subclasses will declare the ev_X struct
    # as an inline member. This is good for locality, but
    # probably bad for alignment, as it will get tacked on
    # immediately after our data.

    # But all ev_watchers start with some ints, so maybe we can help that
    # out by putting our ints here.
    cdef readonly unsigned int _flags

    def __init__(self, loop loop, ref=True, priority=None):
        if not self._w_watcher or not self._w_ss.start or not self._w_ss.stop:
            raise ValueError("Cannot construct a bare watcher")
        self.loop = loop
        self._flags = 0 if ref else FLAG_WATCHER_UNREF_BEFORE_START
        if priority is not None:
            libev.ev_set_priority(self._w_watcher, priority)

    @property
    def ref(self):
        return False if self._flags & 4 else True

    @ref.setter
    def ref(self, object value):
        _check_loop(self.loop)
        if value:
            # self.ref should be true after this.
            if self.ref:
               return  # ref is already True

            if self._flags & FLAG_WATCHER_NEEDS_EVREF:  # ev_unref was called, undo
               libev.ev_ref(self.loop._ptr)
			# do not want unref, no outstanding unref
            self._flags &= ~FLAG_WATCHER_MASK_UNREF_NEEDS_REF
        else:
			# self.ref must be false after this
            if not self.ref:
               return  # ref is already False
            self._flags |= FLAG_WATCHER_UNREF_BEFORE_START
            if not self._flags & FLAG_WATCHER_NEEDS_EVREF and libev.ev_is_active(self._w_watcher):
               libev.ev_unref(self.loop._ptr)
               self._flags |= FLAG_WATCHER_NEEDS_EVREF

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, object callback):
        if callback is not None and not callable(callback):
           raise TypeError("Expected callable, not %r" % (callback, ))
        self._callback = callback

    @property
    def priority(self):
        return libev.ev_priority(self._w_watcher)

    @priority.setter
    def priority(self, int priority):
        cdef libev.ev_watcher* w = self._w_watcher
        if libev.ev_is_active(w):
           raise AttributeError("Cannot set priority of an active watcher")
        libev.ev_set_priority(w, priority)

    @property
    def active(self):
        return True if libev.ev_is_active(self._w_watcher) else False

    @property
    def pending(self):
        return True if libev.ev_is_pending(self._w_watcher) else False

    def start(self, object callback, *args):
        _watcher_start(self, callback, args)

    def stop(self):
        _check_loop(self.loop)
        _libev_ref(self)
        # The callback cannot possibly fire while we are executing,
        # so this is safe.
        self._callback = None
        self.args = None
        self._w_ss.stop(self.loop._ptr, self._w_watcher)
        _python_decref(self)

    def feed(self, int revents, object callback, *args):
        _check_loop(self.loop)
        self.callback = callback
        self.args = args
        _libev_unref(self)
        libev.ev_feed_event(self.loop._ptr, self._w_watcher, revents)
        _python_incref(self)

    def __repr__(self):
        if Py_ReprEnter(self) != 0:
            return "<...>"
        try:
            format = self._format()
            result = "<%s at 0x%x native=0x%x%s" % (
                self.__class__.__name__,
                id(self),
                <unsigned long>self._w_watcher,
                format
            )
            if self.active:
                result += " active"
            if self.pending:
                result += " pending"
            if self.callback is not None:
                result += " callback=%r" % (self.callback, )
            if self.args is not None:
                result += " args=%r" % (self.args, )
            return result + ">"
        finally:
            Py_ReprLeave(self)

    def _format(self):
        return ''

    def close(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        self.close()
        return

cdef start_and_stop io_ss = make_ss(<void*>libev.ev_io_start, <void*>libev.ev_io_stop)

cdef public class io(watcher) [object PyGeventIOObject, type PyGeventIO_Type]:

    cdef libev.ev_io _watcher

    def start(self, object callback, *args, pass_events=False):
        if pass_events:
            args = (GEVENT_CORE_EVENTS, ) + args
        _watcher_start(self, callback, args)

    def __init__(self, loop loop, libev.vfd_socket_t fd, int events, ref=True, priority=None):
        watcher.__init__(self, loop, ref, priority)

    def __cinit__(self, loop loop, libev.vfd_socket_t fd, int events, ref=True, priority=None):
        if fd < 0:
            raise ValueError('fd must be non-negative: %r' % fd)
        if events & ~(libev.EV__IOFDSET | libev.EV_READ | libev.EV_WRITE):
            raise ValueError('illegal event mask: %r' % events)
        # All the vfd_functions are no-ops on POSIX
        cdef int vfd = libev.vfd_open(fd)
        libev.ev_io_init(&self._watcher, <void *>gevent_callback_io, vfd, events)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &io_ss

    def __dealloc__(self):
        libev.vfd_free(self._watcher.fd)

    @property
    def fd(self):
        return libev.vfd_get(self._watcher.fd)

    @fd.setter
    def fd(self, long fd):
        if libev.ev_is_active(&self._watcher):
            raise AttributeError("'io' watcher attribute 'fd' is read-only while watcher is active")
        cdef int vfd = libev.vfd_open(fd)
        libev.vfd_free(self._watcher.fd)
        libev.ev_io_init(&self._watcher, <void *>gevent_callback_io, vfd, self._watcher.events)

    @property
    def events(self):
        return self._watcher.events

    @events.setter
    def events(self, int events):
        if libev.ev_is_active(&self._watcher):
            raise AttributeError("'io' watcher attribute 'events' is read-only while watcher is active")
        libev.ev_io_init(&self._watcher, <void *>gevent_callback_io, self._watcher.fd, events)

    @property
    def events_str(self):
        return _events_to_str(self._watcher.events)

    def _format(self):
        return ' fd=%s events=%s' % (self.fd, self.events_str)

cdef start_and_stop timer_ss = make_ss(<void*>libev.ev_timer_start, <void*>libev.ev_timer_stop)

cdef public class timer(watcher) [object PyGeventTimerObject, type PyGeventTimer_Type]:

    cdef libev.ev_timer _watcher

    def __cinit__(self, loop loop, double after=0.0, double repeat=0.0, ref=True, priority=None):
        if repeat < 0.0:
            raise ValueError("repeat must be positive or zero: %r" % repeat)
        libev.ev_timer_init(&self._watcher, <void *>gevent_callback_timer, after, repeat)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &timer_ss

    def __init__(self, loop loop, double after=0.0, double repeat=0.0, ref=True, priority=None):
        watcher.__init__(self, loop, ref, priority)

    def start(self, object callback, *args, update=None):
        update = update if update is not None else self.loop.starting_timer_may_update_loop_time
        if update:
            self.loop.update_now()
        _watcher_start(self, callback, args)

    @property
    def at(self):
        return self._watcher.at

    # QQQ: add 'after' and 'repeat' properties?

    def again(self, object callback, *args, update=True):
        _check_loop(self.loop)
        self.callback = callback
        self.args = args
        _libev_unref(self)
        if update:
            libev.ev_now_update(self.loop._ptr)
        libev.ev_timer_again(self.loop._ptr, &self._watcher)
        _python_incref(self)



cdef start_and_stop signal_ss = make_ss(<void*>libev.ev_signal_start, <void*>libev.ev_signal_stop)

cdef public class signal(watcher) [object PyGeventSignalObject, type PyGeventSignal_Type]:

    cdef libev.ev_signal _watcher

    def __cinit__(self, loop loop, int signalnum, ref=True, priority=None):
        if signalnum < 1 or signalnum >= signalmodule.NSIG:
            raise ValueError('illegal signal number: %r' % signalnum)
        # still possible to crash on one of libev's asserts:
        # 1) "libev: ev_signal_start called with illegal signal number"
        #    EV_NSIG might be different from signal.NSIG on some platforms
        # 2) "libev: a signal must not be attached to two different loops"
        #    we probably could check that in LIBEV_EMBED mode, but not in general
        libev.ev_signal_init(&self._watcher, <void *>gevent_callback_signal, signalnum)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &signal_ss

    def __init__(self, loop loop, int signalnum, ref=True, priority=None):
        watcher.__init__(self, loop, ref, priority)



cdef start_and_stop idle_ss = make_ss(<void*>libev.ev_idle_start, <void*>libev.ev_idle_stop)

cdef public class idle(watcher) [object PyGeventIdleObject, type PyGeventIdle_Type]:

    cdef libev.ev_idle _watcher

    def __cinit__(self, loop loop, ref=True, priority=None):
        libev.ev_idle_init(&self._watcher, <void*>gevent_callback_idle)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &idle_ss



cdef start_and_stop prepare_ss = make_ss(<void*>libev.ev_prepare_start, <void*>libev.ev_prepare_stop)

cdef public class prepare(watcher) [object PyGeventPrepareObject, type PyGeventPrepare_Type]:

    cdef libev.ev_prepare _watcher

    def __cinit__(self, loop loop, ref=True, priority=None):
        libev.ev_prepare_init(&self._watcher, <void*>gevent_callback_prepare)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &prepare_ss



cdef start_and_stop check_ss = make_ss(<void*>libev.ev_check_start, <void*>libev.ev_check_stop)

cdef public class check(watcher) [object PyGeventCheckObject, type PyGeventCheck_Type]:

    cdef libev.ev_check _watcher

    def __cinit__(self, loop loop, ref=True, priority=None):
        libev.ev_check_init(&self._watcher, <void*>gevent_callback_check)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &check_ss



cdef start_and_stop fork_ss = make_ss(<void*>libev.ev_fork_start, <void*>libev.ev_fork_stop)

cdef public class fork(watcher) [object PyGeventForkObject, type PyGeventFork_Type]:

    cdef libev.ev_fork _watcher

    def __cinit__(self, loop loop, ref=True, priority=None):
        libev.ev_fork_init(&self._watcher, <void*>gevent_callback_fork)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &fork_ss


cdef start_and_stop async_ss = make_ss(<void*>libev.ev_async_start, <void*>libev.ev_async_stop)

cdef public class async_(watcher) [object PyGeventAsyncObject, type PyGeventAsync_Type]:

    cdef libev.ev_async _watcher

    @property
    def pending(self):
        # Note the use of ev_async_pending instead of ev_is_pending
        return True if libev.ev_async_pending(&self._watcher) else False

    def __cinit__(self, loop loop, ref=True, priority=None):
        libev.ev_async_init(&self._watcher, <void*>gevent_callback_async)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &async_ss


    def send(self):
        _check_loop(self.loop)
        libev.ev_async_send(self.loop._ptr, &self._watcher)

    def send_ignoring_arg(self, _ignored):
        return self.send()

async = async_

cdef start_and_stop child_ss = make_ss(<void*>libev.ev_child_start, <void*>libev.ev_child_stop)

cdef public class child(watcher) [object PyGeventChildObject, type PyGeventChild_Type]:

    cdef libev.ev_child _watcher

    def __cinit__(self, loop loop, int pid, bint trace=0, ref=True):
        if sys.platform == 'win32':
            raise AttributeError("Child watchers are not supported on Windows")
        if not loop.default:
            raise TypeError('child watchers are only available on the default loop')
        libev.gevent_install_sigchld_handler()
        libev.ev_child_init(&self._watcher, <void *>gevent_callback_child, pid, trace)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &child_ss

    def __init__(self, loop loop, int pid, bint trace=0, ref=True):
        watcher.__init__(self, loop, ref, None)


    def _format(self):
        return ' pid=%r rstatus=%r' % (self.pid, self.rstatus)

    @property
    def pid(self):
        return self._watcher.pid

    @property
    def rpid(self):
        return self._watcher.rpid

    @rpid.setter
    def rpid(self, int value):
        self._watcher.rpid = value

    @property
    def rstatus(self):
        return self._watcher.rstatus

    @rstatus.setter
    def rstatus(self, int value):
        self._watcher.rstatus = value

cdef start_and_stop stat_ss = make_ss(<void*>libev.ev_stat_start, <void*>libev.ev_stat_stop)

cdef public class stat(watcher) [object PyGeventStatObject, type PyGeventStat_Type]:

    cdef libev.ev_stat _watcher
    cdef readonly str path
    cdef readonly bytes _paths

    def __cinit__(self, loop loop, str path, float interval=0.0, ref=True, priority=None):
        self.path = path
        cdef bytes paths
        if isinstance(path, unicode):
            # the famous Python3 filesystem encoding debacle hits us here. Can we do better?
            # We must keep a reference to the encoded string so that its bytes don't get freed
            # and overwritten, leading to strange errors from libev ("no such file or directory")
            paths = (<unicode>path).encode(sys.getfilesystemencoding())
            self._paths = paths
        else:
            paths = <bytes>path
            self._paths = paths
        libev.ev_stat_init(&self._watcher, <void *>gevent_callback_stat, <char*>paths, interval)
        self._w_watcher = <libev.ev_watcher*>&self._watcher
        self._w_ss = &stat_ss

    def __init__(self, loop loop, str path, float interval=0.0, ref=True, priority=None):
        watcher.__init__(self, loop, ref, priority)


    @property
    def attr(self):
        if not self._watcher.attr.st_nlink:
            return
        return _pystat_fromstructstat(&self._watcher.attr)

    @property
    def prev(self):
        if not self._watcher.prev.st_nlink:
            return
        return _pystat_fromstructstat(&self._watcher.prev)

    @property
    def interval(self):
        return self._watcher.interval



cdef object SYSERR_CALLBACK = None


cdef void _syserr_cb(char* msg) with gil:
    try:
        SYSERR_CALLBACK(msg, errno)
    except:
        set_syserr_cb(None)
        print_exc = getattr(traceback, 'print_exc', None)
        if print_exc is not None:
            print_exc()


cpdef set_syserr_cb(callback):
    global SYSERR_CALLBACK
    if callback is None:
        libev.ev_set_syserr_cb(NULL)
        SYSERR_CALLBACK = None
    elif callable(callback):
        libev.ev_set_syserr_cb(<void*>_syserr_cb)
        SYSERR_CALLBACK = callback
    else:
        raise TypeError('Expected callable or None, got %r' % (callback, ))

libev.ev_set_allocator(<void*>gevent_realloc)

LIBEV_EMBED = bool(libev.LIBEV_EMBED)
EV_USE_FLOOR = libev.EV_USE_FLOOR
EV_USE_CLOCK_SYSCALL = libev.EV_USE_CLOCK_SYSCALL
EV_USE_REALTIME = libev.EV_USE_REALTIME
EV_USE_MONOTONIC = libev.EV_USE_MONOTONIC
EV_USE_NANOSLEEP = libev.EV_USE_NANOSLEEP
EV_USE_INOTIFY = libev.EV_USE_INOTIFY
EV_USE_SIGNALFD = libev.EV_USE_SIGNALFD
EV_USE_EVENTFD = libev.EV_USE_EVENTFD
EV_USE_4HEAP = libev.EV_USE_4HEAP

# Things used in callbacks.c

from cpython cimport PyErr_Fetch
from cpython cimport PyObject

cdef public void gevent_handle_error(loop loop, object context):
    cdef PyObject* typep
    cdef PyObject* valuep
    cdef PyObject* tracebackp

    cdef object type
    cdef object value = None
    cdef object traceback = None

    # If it was set, this will clear it, and we will own
    # the references.
    PyErr_Fetch(&typep, &valuep, &tracebackp)
    # TODO: Should we call PyErr_Normalize? There's code in
    # Hub.handle_error that works around what looks like an
    # unnormalized exception.

    if not typep:
        return
    # This assignment will do a Py_INCREF
    # on the value. We already own the reference
    # returned from PyErr_Fetch,
    # so we must decref immediately
    type = <object>typep
    Py_DECREF(type)

    if valuep:
        value = <object>valuep
        Py_DECREF(value)
    if tracebackp:
        traceback = <object>tracebackp
        Py_DECREF(traceback)

    # If this method fails by raising an exception,
    # cython will print it for us because we don't return a
    # Python object and we don't declare an `except` clause.
    loop.handle_error(context, type, value, traceback)

cdef public tuple _empty_tuple = ()

cdef public object gevent_loop_run_callbacks(loop loop):
    return loop._run_callbacks()
