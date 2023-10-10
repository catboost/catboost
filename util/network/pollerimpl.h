#pragma once

#include "socket.h"

#include <util/system/error.h>
#include <util/system/mutex.h>
#include <util/system/defaults.h>
#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/datetime/base.h>

#if defined(_freebsd_) || defined(_darwin_)
    #define HAVE_KQUEUE_POLLER
#endif

#if (defined(_linux_) && !defined(_bionic_)) || (__ANDROID_API__ >= 21)
    #define HAVE_EPOLL_POLLER
#endif

//now we always have it
#define HAVE_SELECT_POLLER

#if defined(HAVE_KQUEUE_POLLER)
    #include <sys/event.h>
#endif

#if defined(HAVE_EPOLL_POLLER)
    #include <sys/epoll.h>
#endif

enum EContPoll {
    CONT_POLL_READ = 1,
    CONT_POLL_WRITE = 2,
    CONT_POLL_RDHUP = 4,
    CONT_POLL_ONE_SHOT = 8,        // Disable after first event
    CONT_POLL_MODIFY = 16,         // Modify already added event
    CONT_POLL_EDGE_TRIGGERED = 32, // Notify only about new events
    CONT_POLL_BACKLOG_EMPTY = 64,  // Backlog is empty (seen end of request, EAGAIN or truncated read)
};

static inline bool IsSocket(SOCKET fd) noexcept {
    int val = 0;
    socklen_t len = sizeof(val);

    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, (char*)&val, &len) == 0) {
        return true;
    }

    return LastSystemError() != ENOTSOCK;
}

static inline int MicroToMilli(int timeout) noexcept {
    if (timeout) {
        /*
         * 1. API of epoll syscall allows to specify timeout with millisecond
         * accuracy only
         * 2. It is quite complicated to guarantee time resolution of blocking
         * syscall less than kernel 1/HZ
         *
         * Without this rounding we just waste cpu time and do a lot of
         * fast epoll_wait(..., 0) syscalls.
         */
        return Max(timeout / 1000, 1);
    }

    return 0;
}

struct TWithoutLocking {
    using TMyMutex = TFakeMutex;
};

#if defined(HAVE_KQUEUE_POLLER)
static inline int Kevent(int kq, struct kevent* changelist, int nchanges,
                         struct kevent* eventlist, int nevents, const struct timespec* timeout) noexcept {
    int ret;

    do {
        ret = kevent(kq, changelist, nchanges, eventlist, nevents, timeout);
    } while (ret == -1 && errno == EINTR);

    return ret;
}

template <class TLockPolicy>
class TKqueuePoller {
public:
    typedef struct ::kevent TEvent;

    inline TKqueuePoller()
        : Fd_(kqueue())
    {
        if (Fd_ == -1) {
            ythrow TSystemError() << "kqueue failed";
        }
    }

    inline ~TKqueuePoller() {
        close(Fd_);
    }

    inline int Fd() const noexcept {
        return Fd_;
    }

    inline void SetImpl(void* data, int fd, int what) {
        TEvent e[2];
        int flags = EV_ADD;

        if (what & CONT_POLL_EDGE_TRIGGERED) {
            if (what & CONT_POLL_BACKLOG_EMPTY) {
                // When backlog is empty, edge-triggered does not need restart.
                return;
            }
            flags |= EV_CLEAR;
        }

        if (what & CONT_POLL_ONE_SHOT) {
            flags |= EV_ONESHOT;
        }

        Zero(e);

        EV_SET(e + 0, fd, EVFILT_READ, flags | ((what & CONT_POLL_READ) ? EV_ENABLE : EV_DISABLE), 0, 0, data);
        EV_SET(e + 1, fd, EVFILT_WRITE, flags | ((what & CONT_POLL_WRITE) ? EV_ENABLE : EV_DISABLE), 0, 0, data);

        if (Kevent(Fd_, e, 2, nullptr, 0, nullptr) == -1) {
            ythrow TSystemError() << "kevent add failed";
        }
    }

    inline void Remove(int fd) noexcept {
        TEvent e[2];

        Zero(e);

        EV_SET(e + 0, fd, EVFILT_READ, EV_DELETE, 0, 0, 0);
        EV_SET(e + 1, fd, EVFILT_WRITE, EV_DELETE, 0, 0, 0);

        Y_ABORT_UNLESS(!(Kevent(Fd_, e, 2, nullptr, 0, nullptr) == -1 && errno != ENOENT), "kevent remove failed: %s", LastSystemErrorText());
    }

    inline size_t Wait(TEvent* events, size_t len, int timeout) noexcept {
        struct timespec ts;

        ts.tv_sec = timeout / 1000000;
        ts.tv_nsec = (timeout % 1000000) * 1000;

        const int ret = Kevent(Fd_, nullptr, 0, events, len, &ts);

        Y_ABORT_UNLESS(ret >= 0, "kevent failed: %s", LastSystemErrorText());

        return (size_t)ret;
    }

    static inline void* ExtractEvent(const TEvent* event) noexcept {
        return event->udata;
    }

    static inline int ExtractStatus(const TEvent* event) noexcept {
        if (event->flags & EV_ERROR) {
            return EIO;
        }

        return event->fflags;
    }

    static inline int ExtractFilterImpl(const TEvent* event) noexcept {
        if (event->filter == EVFILT_READ) {
            return CONT_POLL_READ;
        }

        if (event->filter == EVFILT_WRITE) {
            return CONT_POLL_WRITE;
        }

        if (event->flags & EV_EOF) {
            return CONT_POLL_READ | CONT_POLL_WRITE;
        }

        return 0;
    }

private:
    int Fd_;
};
#endif

#if defined(HAVE_EPOLL_POLLER)
static inline int ContEpollWait(int epfd, struct epoll_event* events, int maxevents, int timeout) noexcept {
    int ret;

    do {
        ret = epoll_wait(epfd, events, maxevents, Min<int>(timeout, 35 * 60 * 1000));
    } while (ret == -1 && errno == EINTR);

    return ret;
}

template <class TLockPolicy>
class TEpollPoller {
public:
    typedef struct ::epoll_event TEvent;

    inline TEpollPoller(bool closeOnExec = false)
        : Fd_(epoll_create1(closeOnExec ? EPOLL_CLOEXEC : 0))
    {
        if (Fd_ == -1) {
            ythrow TSystemError() << "epoll_create failed";
        }
    }

    inline ~TEpollPoller() {
        close(Fd_);
    }

    inline int Fd() const noexcept {
        return Fd_;
    }

    inline void SetImpl(void* data, int fd, int what) {
        TEvent e;

        Zero(e);

        if (what & CONT_POLL_EDGE_TRIGGERED) {
            if (what & CONT_POLL_BACKLOG_EMPTY) {
                // When backlog is empty, edge-triggered does not need restart.
                return;
            }
            e.events |= EPOLLET;
        }

        if (what & CONT_POLL_ONE_SHOT) {
            e.events |= EPOLLONESHOT;
        }

        if (what & CONT_POLL_READ) {
            e.events |= EPOLLIN;
        }

        if (what & CONT_POLL_WRITE) {
            e.events |= EPOLLOUT;
        }

        if (what & CONT_POLL_RDHUP) {
            e.events |= EPOLLRDHUP;
        }

        e.data.ptr = data;

        if (what & CONT_POLL_MODIFY) {
            if (epoll_ctl(Fd_, EPOLL_CTL_MOD, fd, &e) == -1) {
                ythrow TSystemError() << "epoll modify failed (fd=" << fd << ", what=" << what << ")";
            }
        } else if (epoll_ctl(Fd_, EPOLL_CTL_ADD, fd, &e) == -1) {
            if (LastSystemError() != EEXIST) {
                ythrow TSystemError() << "epoll add failed (fd=" << fd << ", what=" << what << ")";
            }

            if (epoll_ctl(Fd_, EPOLL_CTL_MOD, fd, &e) == -1) {
                ythrow TSystemError() << "epoll modify failed (fd=" << fd << ", what=" << what << ")";
            }
        }
    }

    inline void Remove(int fd) noexcept {
        TEvent e;

        Zero(e);

        epoll_ctl(Fd_, EPOLL_CTL_DEL, fd, &e);
    }

    inline size_t Wait(TEvent* events, size_t len, int timeout) noexcept {
        const int ret = ContEpollWait(Fd_, events, len, MicroToMilli(timeout));

        Y_ABORT_UNLESS(ret >= 0, "epoll wait error: %s", LastSystemErrorText());

        return (size_t)ret;
    }

    static inline void* ExtractEvent(const TEvent* event) noexcept {
        return event->data.ptr;
    }

    static inline int ExtractStatus(const TEvent* event) noexcept {
        if (event->events & (EPOLLERR | EPOLLHUP)) {
            return EIO;
        }

        return 0;
    }

    static inline int ExtractFilterImpl(const TEvent* event) noexcept {
        int ret = 0;

        if (event->events & EPOLLIN) {
            ret |= CONT_POLL_READ;
        }

        if (event->events & EPOLLOUT) {
            ret |= CONT_POLL_WRITE;
        }

        if (event->events & EPOLLRDHUP) {
            ret |= CONT_POLL_RDHUP;
        }

        return ret;
    }

private:
    int Fd_;
};
#endif

#if defined(HAVE_SELECT_POLLER)
    #include <util/memory/tempbuf.h>
    #include <util/generic/hash.h>

    #include "pair.h"

static inline int ContSelect(int n, fd_set* r, fd_set* w, fd_set* e, struct timeval* t) noexcept {
    int ret;

    do {
        ret = select(n, r, w, e, t);
    } while (ret == -1 && errno == EINTR);

    return ret;
}

struct TSelectPollerNoTemplate {
    struct THandle {
        void* Data_;
        int Filter_;

        inline THandle()
            : Data_(nullptr)
            , Filter_(0)
        {
        }

        inline void* Data() const noexcept {
            return Data_;
        }

        inline void Set(void* d, int s) noexcept {
            Data_ = d;
            Filter_ = s;
        }

        inline void Clear(int c) noexcept {
            Filter_ &= ~c;
        }

        inline int Filter() const noexcept {
            return Filter_;
        }
    };

    class TFds: public THashMap<SOCKET, THandle> {
    public:
        inline void Set(SOCKET fd, void* data, int filter) {
            (*this)[fd].Set(data, filter);
        }

        inline void Remove(SOCKET fd) {
            erase(fd);
        }

        inline SOCKET Build(fd_set* r, fd_set* w, fd_set* e) const noexcept {
            SOCKET ret = 0;

            for (const auto& it : *this) {
                const SOCKET fd = it.first;
                const THandle& handle = it.second;

                FD_SET(fd, e);

                if (handle.Filter() & CONT_POLL_READ) {
                    FD_SET(fd, r);
                }

                if (handle.Filter() & CONT_POLL_WRITE) {
                    FD_SET(fd, w);
                }

                if (fd > ret) {
                    ret = fd;
                }
            }

            return ret;
        }
    };

    struct TEvent: public THandle {
        inline int Status() const noexcept {
            return -Min(Filter(), 0);
        }

        inline void Error(void* d, int err) noexcept {
            Set(d, -err);
        }

        inline void Success(void* d, int what) noexcept {
            Set(d, what);
        }
    };
};

template <class TLockPolicy>
class TSelectPoller: public TSelectPollerNoTemplate {
    using TMyMutex = typename TLockPolicy::TMyMutex;

public:
    inline TSelectPoller()
        : Begin_(nullptr)
        , End_(nullptr)
    {
        SocketPair(Signal_);
        SetNonBlock(WaitSock());
        SetNonBlock(SigSock());
    }

    inline ~TSelectPoller() {
        closesocket(Signal_[0]);
        closesocket(Signal_[1]);
    }

    inline void SetImpl(void* data, SOCKET fd, int what) {
        with_lock (CommandLock_) {
            Commands_.push_back(TCommand(fd, what, data));
        }

        Signal();
    }

    inline void Remove(SOCKET fd) noexcept {
        with_lock (CommandLock_) {
            Commands_.push_back(TCommand(fd, 0));
        }

        Signal();
    }

    inline size_t Wait(TEvent* events, size_t len, int timeout) noexcept {
        auto guard = Guard(Lock_);

        do {
            if (Begin_ != End_) {
                const size_t ret = Min<size_t>(End_ - Begin_, len);

                memcpy(events, Begin_, sizeof(*events) * ret);
                Begin_ += ret;

                return ret;
            }

            if (len >= EventNumberHint()) {
                return WaitBase(events, len, timeout);
            }

            Begin_ = SavedEvents();
            End_ = Begin_ + WaitBase(Begin_, EventNumberHint(), timeout);
        } while (Begin_ != End_);

        return 0;
    }

    inline TEvent* SavedEvents() {
        if (!SavedEvents_) {
            SavedEvents_.Reset(new TEvent[EventNumberHint()]);
        }

        return SavedEvents_.Get();
    }

    inline size_t WaitBase(TEvent* events, size_t len, int timeout) noexcept {
        with_lock (CommandLock_) {
            for (auto command = Commands_.begin(); command != Commands_.end(); ++command) {
                if (command->Filter_ != 0) {
                    Fds_.Set(command->Fd_, command->Cookie_, command->Filter_);
                } else {
                    Fds_.Remove(command->Fd_);
                }
            }

            Commands_.clear();
        }

        TTempBuf tmpBuf(3 * sizeof(fd_set) + Fds_.size() * sizeof(SOCKET));

        fd_set* in = (fd_set*)tmpBuf.Data();
        fd_set* out = &in[1];
        fd_set* errFds = &in[2];

        SOCKET* keysToDeleteBegin = (SOCKET*)&in[3];
        SOCKET* keysToDeleteEnd = keysToDeleteBegin;

    #if defined(_msan_enabled_) // msan doesn't handle FD_ZERO and cause false positive BALANCER-1347
        memset(in, 0, sizeof(*in));
        memset(out, 0, sizeof(*out));
        memset(errFds, 0, sizeof(*errFds));
    #endif

        FD_ZERO(in);
        FD_ZERO(out);
        FD_ZERO(errFds);

        FD_SET(WaitSock(), in);

        const SOCKET maxFdNum = Max(Fds_.Build(in, out, errFds), WaitSock());
        struct timeval tout;

        tout.tv_sec = timeout / 1000000;
        tout.tv_usec = timeout % 1000000;

        int ret = ContSelect(int(maxFdNum + 1), in, out, errFds, &tout);

        if (ret > 0 && FD_ISSET(WaitSock(), in)) {
            --ret;
            TryWait();
        }

        Y_ABORT_UNLESS(ret >= 0 && (size_t)ret <= len, "select error: %s", LastSystemErrorText());

        TEvent* eventsStart = events;

        for (typename TFds::iterator it = Fds_.begin(); it != Fds_.end(); ++it) {
            const SOCKET fd = it->first;
            THandle& handle = it->second;

            if (FD_ISSET(fd, errFds)) {
                (events++)->Error(handle.Data(), EIO);

                if (handle.Filter() & CONT_POLL_ONE_SHOT) {
                    *keysToDeleteEnd = fd;
                    ++keysToDeleteEnd;
                }

            } else {
                int what = 0;

                if (FD_ISSET(fd, in)) {
                    what |= CONT_POLL_READ;
                }

                if (FD_ISSET(fd, out)) {
                    what |= CONT_POLL_WRITE;
                }

                if (what) {
                    (events++)->Success(handle.Data(), what);

                    if (handle.Filter() & CONT_POLL_ONE_SHOT) {
                        *keysToDeleteEnd = fd;
                        ++keysToDeleteEnd;
                    }

                    if (handle.Filter() & CONT_POLL_EDGE_TRIGGERED) {
                        // Emulate edge-triggered for level-triggered select().
                        // User must restart waiting this event when needed.
                        handle.Clear(what);
                    }
                }
            }
        }

        while (keysToDeleteBegin != keysToDeleteEnd) {
            Fds_.erase(*keysToDeleteBegin);
            ++keysToDeleteBegin;
        }

        return events - eventsStart;
    }

    inline size_t EventNumberHint() const noexcept {
        return sizeof(fd_set) * 8 * 2;
    }

    static inline void* ExtractEvent(const TEvent* event) noexcept {
        return event->Data();
    }

    static inline int ExtractStatus(const TEvent* event) noexcept {
        return event->Status();
    }

    static inline int ExtractFilterImpl(const TEvent* event) noexcept {
        return event->Filter();
    }

private:
    inline void Signal() noexcept {
        char ch = 13;

        send(SigSock(), &ch, 1, 0);
    }

    inline void TryWait() {
        char ch[32];

        while (recv(WaitSock(), ch, sizeof(ch), 0) > 0) {
            Y_ASSERT(ch[0] == 13);
        }
    }

    inline SOCKET WaitSock() const noexcept {
        return Signal_[1];
    }

    inline SOCKET SigSock() const noexcept {
        return Signal_[0];
    }

private:
    struct TCommand {
        SOCKET Fd_;
        int Filter_; // 0 to remove
        void* Cookie_;

        TCommand(SOCKET fd, int filter, void* cookie)
            : Fd_(fd)
            , Filter_(filter)
            , Cookie_(cookie)
        {
        }

        TCommand(SOCKET fd, int filter)
            : Fd_(fd)
            , Filter_(filter)
            , Cookie_(nullptr)
        {
        }
    };

    TFds Fds_;

    TMyMutex Lock_;
    TArrayHolder<TEvent> SavedEvents_;
    TEvent* Begin_;
    TEvent* End_;

    TMyMutex CommandLock_;
    TVector<TCommand> Commands_;

    SOCKET Signal_[2];
};
#endif

static inline TDuration PollStep(const TInstant& deadLine, const TInstant& now) noexcept {
    if (deadLine < now) {
        return TDuration::Zero();
    }

    return Min(deadLine - now, TDuration::Seconds(1000));
}

template <class TBase>
class TGenericPoller: public TBase {
public:
    using TBase::TBase;

    using TEvent = typename TBase::TEvent;

    inline void Set(void* data, SOCKET fd, int what) {
        if (what) {
            this->SetImpl(data, fd, what);
        } else {
            this->Remove(fd);
        }
    }

    static inline int ExtractFilter(const TEvent* event) noexcept {
        if (TBase::ExtractStatus(event)) {
            return CONT_POLL_READ | CONT_POLL_WRITE | CONT_POLL_RDHUP;
        }

        return TBase::ExtractFilterImpl(event);
    }

    inline size_t WaitD(TEvent* events, size_t len, TInstant deadLine, TInstant now = TInstant::Now()) noexcept {
        if (!len) {
            return 0;
        }

        size_t ret;

        do {
            ret = this->Wait(events, len, (int)PollStep(deadLine, now).MicroSeconds());
        } while (!ret && ((now = TInstant::Now()) < deadLine));

        return ret;
    }
};

#if defined(HAVE_KQUEUE_POLLER)
    #define TPollerImplBase TKqueuePoller
#elif defined(HAVE_EPOLL_POLLER)
    #define TPollerImplBase TEpollPoller
#elif defined(HAVE_SELECT_POLLER)
    #define TPollerImplBase TSelectPoller
#else
    #error "unsupported platform"
#endif

template <class TLockPolicy>
using TPollerImpl = TGenericPoller<TPollerImplBase<TLockPolicy>>;

#undef TPollerImplBase
