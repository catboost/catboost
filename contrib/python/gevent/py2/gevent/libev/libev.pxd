# From cython/includes/libc/stdint.pxd
#   Longness only used for type promotion.
#   Actual compile time size used for conversions.
# We don't have stdint.h on visual studio 9.0 (2008) on windows, sigh,
# so go with Py_ssize_t
# ssize_t -> intptr_t

cdef extern from "libev_vfd.h":
# cython doesn't process pre-processor directives, so they
# don't matter in this file. It just takes the last definition it sees.
    ctypedef Py_ssize_t intptr_t
    ctypedef intptr_t vfd_socket_t

    vfd_socket_t vfd_get(int)
    int vfd_open(long) except -1
    void vfd_free(int)

cdef extern from "libev.h" nogil:
    int LIBEV_EMBED
    int EV_MINPRI
    int EV_MAXPRI

    int EV_VERSION_MAJOR
    int EV_VERSION_MINOR

    int EV_USE_FLOOR
    int EV_USE_CLOCK_SYSCALL
    int EV_USE_REALTIME
    int EV_USE_MONOTONIC
    int EV_USE_NANOSLEEP
    int EV_USE_SELECT
    int EV_USE_POLL
    int EV_USE_EPOLL
    int EV_USE_KQUEUE
    int EV_USE_PORT
    int EV_USE_INOTIFY
    int EV_USE_SIGNALFD
    int EV_USE_EVENTFD
    int EV_USE_4HEAP
    int EV_USE_IOCP
    int EV_SELECT_IS_WINSOCKET

    int EV_UNDEF
    int EV_NONE
    int EV_READ
    int EV_WRITE
    int EV__IOFDSET
    int EV_TIMER
    int EV_PERIODIC
    int EV_SIGNAL
    int EV_CHILD
    int EV_STAT
    int EV_IDLE
    int EV_PREPARE
    int EV_CHECK
    int EV_EMBED
    int EV_FORK
    int EV_CLEANUP
    int EV_ASYNC
    int EV_CUSTOM
    int EV_ERROR

    int EVFLAG_AUTO
    int EVFLAG_NOENV
    int EVFLAG_FORKCHECK
    int EVFLAG_NOINOTIFY
    int EVFLAG_SIGNALFD
    int EVFLAG_NOSIGMASK

    int EVBACKEND_SELECT
    int EVBACKEND_POLL
    int EVBACKEND_EPOLL
    int EVBACKEND_KQUEUE
    int EVBACKEND_DEVPOLL
    int EVBACKEND_PORT
    int EVBACKEND_IOCP
    #int EVBACKEND_IOURING
    #int EVBACKEND_LINUXAIO
    int EVBACKEND_ALL
    int EVBACKEND_MASK

    int EVRUN_NOWAIT
    int EVRUN_ONCE

    int EVBREAK_CANCEL
    int EVBREAK_ONE
    int EVBREAK_ALL

    struct ev_loop:
        int activecnt
        int sig_pending
        int backend_fd
        int sigfd
        unsigned int origflags

    struct ev_watcher:
        void* data;

    struct ev_io:
        int fd
        int events

    struct ev_timer:
        double at

    struct ev_signal:
        pass

    struct ev_idle:
        pass

    struct ev_prepare:
        pass

    struct ev_check:
        pass

    struct ev_fork:
        pass

    struct ev_async:
        pass

    struct ev_child:
        int pid
        int rpid
        int rstatus

    struct stat:
        int st_nlink

    struct ev_stat:
        stat attr
        stat prev
        double interval

    union ev_any_watcher:
        ev_watcher w
        ev_io io
        ev_timer timer
        ev_signal signal
        ev_idle idle

    int ev_version_major()
    int ev_version_minor()

    unsigned int ev_supported_backends()
    unsigned int ev_recommended_backends()
    unsigned int ev_embeddable_backends()

    ctypedef double ev_tstamp

    ev_tstamp ev_time()
    void ev_set_syserr_cb(void*)
    void ev_set_allocator(void*)

    int ev_priority(void*)
    void ev_set_priority(void*, int)

    int ev_is_pending(void*)
    int ev_is_active(void*)
    void ev_io_init(ev_io*, void* callback, int fd, int events)
    void ev_io_start(ev_loop*, ev_io*)
    void ev_io_stop(ev_loop*, ev_io*)
    void ev_feed_event(ev_loop*, void*, int)
    void ev_feed_fd_event(ev_loop*, vfd_socket_t, int)

    void ev_timer_init(ev_timer*, void* callback, double, double)
    void ev_timer_start(ev_loop*, ev_timer*)
    void ev_timer_stop(ev_loop*, ev_timer*)
    void ev_timer_again(ev_loop*, ev_timer*)

    void ev_signal_init(ev_signal*, void* callback, int)
    void ev_signal_start(ev_loop*, ev_signal*)
    void ev_signal_stop(ev_loop*, ev_signal*)

    void ev_idle_init(ev_idle*, void* callback)
    void ev_idle_start(ev_loop*, ev_idle*)
    void ev_idle_stop(ev_loop*, ev_idle*)

    void ev_prepare_init(ev_prepare*, void* callback)
    void ev_prepare_start(ev_loop*, ev_prepare*)
    void ev_prepare_stop(ev_loop*, ev_prepare*)

    void ev_check_init(ev_check*, void* callback)
    void ev_check_start(ev_loop*, ev_check*)
    void ev_check_stop(ev_loop*, ev_check*)

    void ev_fork_init(ev_fork*, void* callback)
    void ev_fork_start(ev_loop*, ev_fork*)
    void ev_fork_stop(ev_loop*, ev_fork*)

    void ev_async_init(ev_async*, void* callback)
    void ev_async_start(ev_loop*, ev_async*)
    void ev_async_stop(ev_loop*, ev_async*)
    void ev_async_send(ev_loop*, ev_async*)
    int ev_async_pending(ev_async*)

    void ev_child_init(ev_child*, void* callback, int, int)
    void ev_child_start(ev_loop*, ev_child*)
    void ev_child_stop(ev_loop*, ev_child*)

    void ev_stat_init(ev_stat*, void* callback, char*, double)
    void ev_stat_start(ev_loop*, ev_stat*)
    void ev_stat_stop(ev_loop*, ev_stat*)

    ev_loop* ev_default_loop(unsigned int flags)
    ev_loop* ev_loop_new(unsigned int flags)
    void* ev_userdata(ev_loop*)
    void ev_set_userdata(ev_loop*, void*)
    void ev_loop_destroy(ev_loop*)
    void ev_loop_fork(ev_loop*)
    int ev_is_default_loop(ev_loop*)
    unsigned int ev_iteration(ev_loop*)
    unsigned int ev_depth(ev_loop*)
    unsigned int ev_backend(ev_loop*)
    void ev_verify(ev_loop*)
    void ev_run(ev_loop*, int flags) nogil

    ev_tstamp ev_now(ev_loop*)
    void ev_now_update(ev_loop*)

    void ev_ref(ev_loop*)
    void ev_unref(ev_loop*)
    void ev_break(ev_loop*, int)
    unsigned int ev_pending_count(ev_loop*)

    # gevent extra functions. These are defined in libev.h.
    ev_loop* gevent_ev_default_loop(unsigned int flags)
    void gevent_install_sigchld_handler()
    void gevent_reset_sigchld_handler()

    # These compensate for lack of access to ev_loop struct definition
    # when LIBEV_EMBED is false.
    unsigned int gevent_ev_loop_origflags(ev_loop*);
    int gevent_ev_loop_sig_pending(ev_loop*);
    int gevent_ev_loop_backend_fd(ev_loop*);
    int gevent_ev_loop_activecnt(ev_loop*);
    int gevent_ev_loop_sigfd(ev_loop*);
