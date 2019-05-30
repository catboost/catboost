#include "impl.h"
#include "schedule_callback.h"

#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>

template <>
void Out<TCont>(IOutputStream& out, const TCont& c) {
    c.PrintMe(out);
}

template <>
void Out<TContRep>(IOutputStream& out, const TContRep& c) {
    c.ContPtr()->PrintMe(out);
}

TContRep::TContRep(TContStackAllocator* alloc)
    : real(alloc->Allocate())
    , full((char*)real->Data(), real->Length())
    , stack(full.data(), EffectiveStackLength(full.size()))
    , cont(stack.end(), Align(sizeof(TCont)))
    , machine(cont.end(), Align(sizeof(TExceptionSafeContext)))
{}

void TContRep::DoRun() {
    try {
        Y_CORO_DBGOUT(Y_CORO_PRINT(ContPtr()) << " execute");
        ContPtr()->Execute();
    } catch (...) {
        try {
            Y_CORO_DBGOUT(CurrentExceptionMessage());
        } catch (...) {
        }

        Y_VERIFY(!ContPtr()->Executor()->FailOnError(), "uncaught exception");
    }

    ContPtr()->WakeAllWaiters();
    ContPtr()->Executor()->Exit(this);
}

void TContRep::Construct(TContExecutor* executor, TContFunc func, void* arg, const char* name) {
    TContClosure closure = {
        this,
        stack,
    };

    THolder<TExceptionSafeContext, TDestructor> mc(new (MachinePtr()) TExceptionSafeContext(closure));

    new (ContPtr()) TCont(executor, this, func, arg, name);
    Y_UNUSED(mc.Release());
}

void TContRep::Destruct() noexcept {
    ContPtr()->~TCont();
    MachinePtr()->~TExceptionSafeContext();
}

void TContExecutor::WaitForIO() {
    Y_CORO_DBGOUT("scheduler: WaitForIO R,RN,WQ=" << Ready_.Size() << "," << ReadyNext_.Size() << "," << !WaitQueue_.Empty());

    while (Ready_.Empty() && !WaitQueue_.Empty()) {
        const auto now = TInstant::Now();

        // Waking a coroutine puts it into ReadyNext_ list
        const auto next = WaitQueue_.WakeTimedout(now);

        // Polling will return as soon as there is an event to process or a timeout.
        // If there are woken coroutines we do not want to sleep in the poller
        //      yet still we want to check for new io
        //      to prevent ourselves from locking out of io by constantly waking coroutines.
        Poller_.Wait(Events_, ReadyNext_.Empty() ? next : now);

        // Waking a coroutine puts it into ReadyNext_ list
        ProcessEvents();

        Ready_.Append(ReadyNext_);
    }

    Y_CORO_DBGOUT("scheduler: Done WaitForIO R,RN,WQ="
           << Ready_.Size() << "," << ReadyNext_.Size() << "," << !WaitQueue_.Empty());
}

void TContExecutor::ProcessEvents() {
    for (auto event : Events_) {
        auto* lst = (NCoro::TPollEventList*)event.Data;
        const int status = event.Status;

        if (status) {
            for (auto it = lst->Begin(); it != lst->End();) {
                (it++)->OnPollEvent(status);
            }
        } else {
            const ui16 filter = event.Filter;

            for (auto it = lst->Begin(); it != lst->End();) {
                if (it->What() & filter) {
                    (it++)->OnPollEvent(0);
                } else {
                    ++it;
                }
            }
        }
    }
}

void TCont::PrintMe(IOutputStream& out) const noexcept {
    out << "cont("
        << "func = " << (size_t)(void*)Func_ << ", "
        << "arg = " << (size_t)(void*)Arg_ << ", "
        << "name = " << Name_
        << ")";
}

int TCont::SelectD(SOCKET fds[], int what[], size_t nfds, SOCKET* outfd, TInstant deadline) {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " prepare select");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    if (Cancelled()) {
        return ECANCELED;
    }

    if (nfds == 0) {
        return 0;
    }

    TTempArray<TFdEvent> events(nfds);

    for (auto i : xrange(nfds)) {
        new (events.Data() + i) TFdEvent(this, fds[i], (ui16)what[i], deadline);
    }

    Y_DEFER {
        for (auto i : xrange(nfds)) {
            (events.Data() + i)->~TFdEvent();
        }
    };

    ExecuteEvents(events.Data(), events.Data() + nfds);

    if (Cancelled()) {
        return ECANCELED;
    }

    TFdEvent* ret = nullptr;
    int status = EINPROGRESS;

    for (auto i : xrange(nfds)) {
        auto& ev = *(events.Data() + i);
        switch (ev.Status()) {
        case EINPROGRESS:
            break;
        case ETIMEDOUT:
            if (status != EINPROGRESS) {
                break;
            } // else fallthrough
        default:
            status = ev.Status();
            ret = &ev;
        }
    }

    if (ret) {
        if (outfd) {
            *outfd = ret->Fd();
        }
        return ret->Status();
    }

    return EINPROGRESS;
}

TContIOStatus TCont::ReadVectorD(SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do readv");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    while (true) {
        ssize_t res = DoReadVector(fd, vec);

        if (res >= 0) {
            return TContIOStatus::Success((size_t)res);
        }

        {
            const int err = LastSystemError();

            if (!IsBlocked(err)) {
                return TContIOStatus::Error(err);
            }
        }

        if ((res = PollD(fd, CONT_POLL_READ, deadline)) != 0) {
            return TContIOStatus::Error((int)res);
        }
    }
}

TContIOStatus TCont::ReadD(SOCKET fd, void* buf, size_t len, TInstant deadline) noexcept {
    IOutputStream::TPart part(buf, len);
    TContIOVector vec(&part, 1);
    return ReadVectorD(fd, &vec, deadline);
}

TContIOStatus TCont::WriteVectorD(SOCKET fd, TContIOVector* vec, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do writev");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    size_t written = 0;

    while (!vec->Complete()) {
        ssize_t res = DoWriteVector(fd, vec);

        if (res >= 0) {
            written += res;

            vec->Proceed((size_t)res);
        } else {
            {
                const int err = LastSystemError();

                if (!IsBlocked(err)) {
                    return TContIOStatus(written, err);
                }
            }

            if ((res = PollD(fd, CONT_POLL_WRITE, deadline)) != 0) {
                return TContIOStatus(written, (int)res);
            }
        }
    }

    return TContIOStatus::Success(written);
}

TContIOStatus TCont::WriteD(SOCKET fd, const void* buf, size_t len, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do write");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    size_t written = 0;

    while (len) {
        ssize_t res = DoWrite(fd, (const char*)buf, len);

        if (res >= 0) {
            written += res;
            buf = (const char*)buf + res;
            len -= res;
        } else {
            {
                const int err = LastSystemError();

                if (!IsBlocked(err)) {
                    return TContIOStatus(written, err);
                }
            }

            if ((res = PollD(fd, CONT_POLL_WRITE, deadline)) != 0) {
                return TContIOStatus(written, (int)res);
            }
        }
    }

    return TContIOStatus::Success(written);
}

int TCont::ConnectD(SOCKET s, const struct sockaddr* name, socklen_t namelen, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do connect");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    if (connect(s, name, namelen)) {
        const int err = LastSystemError();

        if (!IsBlocked(err) && err != EINPROGRESS) {
            return err;
        }

        int ret = PollD(s, CONT_POLL_WRITE, deadline);

        if (ret) {
            return ret;
        }

        // check if we really connected
        // FIXME: Unportable ??
        int serr = 0;
        socklen_t slen = sizeof(serr);

        ret = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&serr, &slen);

        if (ret) {
            return LastSystemError();
        }

        if (serr) {
            return serr;
        }
    }

    return 0;
}

int TCont::AcceptD(SOCKET s, struct sockaddr* addr, socklen_t* addrlen, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do accept");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    SOCKET ret;

    while ((ret = Accept4(s, addr, addrlen)) == INVALID_SOCKET) {
        int err = LastSystemError();

        if (!IsBlocked(err)) {
            return -err;
        }

        err = PollD(s, CONT_POLL_READ, deadline);

        if (err) {
            return -err;
        }
    }

    return (int)ret;
}

ssize_t TCont::DoRead(SOCKET fd, char* buf, size_t len) noexcept {
#if defined(_win_)
    if (IsSocket(fd)) {
        return recv(fd, buf, (int)len, 0);
    }

    return _read((int)fd, buf, (int)len);
#else
    return read(fd, buf, len);
#endif
}

ssize_t TCont::DoReadVector(SOCKET fd, TContIOVector* vec) noexcept {
    return readv(fd, (const iovec*)vec->Parts(), Min(IOV_MAX, (int)vec->Count()));
}

ssize_t TCont::DoWrite(SOCKET fd, const char* buf, size_t len) noexcept {
#if defined(_win_)
    if (IsSocket(fd)) {
        return send(fd, buf, (int)len, 0);
    }

    return _write((int)fd, buf, (int)len);
#else
    return write(fd, buf, len);
#endif
}

ssize_t TCont::DoWriteVector(SOCKET fd, TContIOVector* vec) noexcept {
    return writev(fd, (const iovec*)vec->Parts(), Min(IOV_MAX, (int)vec->Count()));
}

int TCont::Connect(TSocketHolder& s, const struct addrinfo& ai, TInstant deadLine) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do connect addrinfo");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    TSocketHolder res(Socket(ai));

    if (res.Closed()) {
        return LastSystemError();
    }

    const int ret = ConnectD(res, ai.ai_addr, (socklen_t)ai.ai_addrlen, deadLine);

    if (!ret) {
        s.Swap(res);
    }

    return ret;
}

int TCont::Connect(TSocketHolder& s, const TNetworkAddress& addr, TInstant deadLine) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do connect netaddr");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    int ret = EHOSTUNREACH;

    for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
        ret = Connect(s, *it, deadLine);

        if (ret == 0 || ret == ETIMEDOUT) {
            return ret;
        }
    }

    return ret;
}

void TCont::Die() noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " die");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));
    Dead_ = true;
}

bool TCont::Join(TCont* c, TInstant deadLine) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " join " << Y_CORO_PRINT(c));
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));
    Y_VERIFY(!c->Dead_, "%s -> %s", Y_CORO_PRINTF(this), Y_CORO_PRINTF(c));
    TJoinWait ev(this);

    c->Waiters_.PushBack(&ev);

    do {
        if (SleepD(deadLine) == ETIMEDOUT || Cancelled()) {
            if (!ev.Empty()) {
                c->Cancel();

                do {
                    SwitchToScheduler();
                } while (!ev.Empty());
            }

            return false;
        }
    } while (!ev.Empty());

    return true;
}

void TCont::WakeAllWaiters() noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " wake all waiters");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));
    while (!Waiters_.Empty()) {
        Waiters_.PopFront()->Wake();
    }
}

int TCont::MsgPeek(SOCKET s) noexcept {
    char c;
    return recv(s, &c, 1, MSG_PEEK);
}

int TCont::SleepD(TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do sleep");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    TTimerEvent event(this, deadline);

    return ExecuteEvent(&event);
}

int TCont::PollD(SOCKET fd, int what, TInstant deadline) noexcept {
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " prepare poll");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    TFdEvent event(this, fd, (ui16)what, deadline);

    return ExecuteEvent(&event);
}

void TCont::Yield() noexcept {
    if (SleepD(TInstant::Zero())) {
        ReScheduleAndSwitch();
    }
}

TContExecutor::TContExecutor(size_t stackSize, THolder<IPollerFace> poller, NCoro::IScheduleCallback* callback)
    : Poller_(std::move(poller))
    , MyPool_(new TContRepPool(stackSize))
    , Pool_(*MyPool_)
    , Current_(nullptr)
    , CallbackPtr_(callback)
    , FailOnError_(false)
{
}

TContExecutor::TContExecutor(TContRepPool* pool, THolder<IPollerFace> poller, NCoro::IScheduleCallback* callback)
    : Poller_(std::move(poller))
    , Pool_(*pool)
    , Current_(nullptr)
    , CallbackPtr_(callback)
    , FailOnError_(false)
{
}

TContExecutor::~TContExecutor() {
}

void TContExecutor::RunScheduler() noexcept {
    Y_CORO_DBGOUT("scheduler: started");

    try {
        while (true) {
            Ready_.Append(ReadyNext_);

            if (Ready_.Empty()) {
                break;
            }

            TContRep* cont = Ready_.PopFront();

            Y_CORO_DBGOUT(Y_CORO_PRINT(cont->ContPtr()) << " prepare for activate");
            if (CallbackPtr_) {
                CallbackPtr_->OnSchedule(*this, *cont->ContPtr());
            }
            Activate(cont);
            if (CallbackPtr_) {
                CallbackPtr_->OnUnschedule(*this);
            }

            WaitForIO();
            DeleteScheduled();
        }
    } catch (...) {
        Y_FAIL("Uncaught exception in scheduler: %s", CurrentExceptionMessage().c_str());
    }

    Y_CORO_DBGOUT("scheduler: stopped");
}
