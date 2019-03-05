#define FROM_IMPL_CPP
#include "impl.h"

#include <util/stream/output.h>
#include <util/generic/yexception.h>
#include <util/system/yassert.h>

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
#if defined(STACK_GROW_DOWN)
    , stack(full.Data(), EffectiveStackLength(full.Size()))
    , cont(stack.End(), Align(sizeof(TCont)))
    , machine(cont.End(), Align(sizeof(TExceptionSafeContext)))
#else
#error todo
#endif
{
}

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

#if defined(_unix_)
#include <sys/mman.h>
#endif

void TProtectedContStackAllocator::Protect(void* ptr, size_t len) noexcept {
#if defined(_unix_) && !defined(_cygwin_)
    if (mprotect(ptr, len, PROT_NONE)) {
        Y_FAIL("failed to mprotect (protect): %s", LastSystemErrorText());
    }
#else
    Y_UNUSED(ptr);
    Y_UNUSED(len);
#endif
}

void TProtectedContStackAllocator::UnProtect(void* ptr, size_t len) noexcept {
#if defined(_unix_) && !defined(_cygwin_)
    if (mprotect(ptr, len, PROT_READ | PROT_WRITE)) {
        Y_FAIL("failed to mprotect (unprotect): %s", LastSystemErrorText());
    }
#else
    Y_UNUSED(ptr);
    Y_UNUSED(len);
#endif
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

void TContExecutor::ProcessEvents(){
    for (auto event : Events_) {
        TPollEventList* lst = (TPollEventList*)event.Data;
        const int status = event.Status;

        if (status) {
            for (TPollEventList::TIterator it = lst->Begin(); it != lst->End();) {
                (it++)->OnPollEvent(status);
            }
        } else {
            const ui16 filter = event.Filter;

            for (TPollEventList::TIterator it = lst->Begin(); it != lst->End();) {
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

    TTempBuf memoryBuf(nfds * sizeof(TFdEvent));
    void* memory = memoryBuf.Data();
    TContPollEventHolder holder(memory, this, fds, what, nfds, deadline);
    holder.ScheduleIoWait(Executor());

    SwitchToScheduler();

    if (Cancelled()) {
        return ECANCELED;
    }

    TFdEvent* ev = holder.TriggeredEvent();

    if (ev) {
        if (outfd) {
            *outfd = ev->Fd();
        }

        return ev->Status();
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
    Y_CORO_DBGOUT(Y_CORO_PRINT(this) << " do read");
    Y_VERIFY(!Dead_, "%s", Y_CORO_PRINTF(this));

    while (true) {
        ssize_t res = DoRead(fd, (char*)buf, len);

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

TContExecutor::TContExecutor(size_t stackSize, THolder<IPollerFace> poller)
    : Poller_(std::move(poller))
    , MyPool_(new TContRepPool(stackSize))
    , Pool_(*MyPool_)
    , Current_(nullptr)
    , FailOnError_(false)
{
}

TContExecutor::TContExecutor(TContRepPool* pool, THolder<IPollerFace> poller)
    : Poller_(std::move(poller))
    , Pool_(*pool)
    , Current_(nullptr)
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
            Activate(cont);

            WaitForIO();
            DeleteScheduled();
        }
    } catch (...) {
        Y_FAIL("Uncaught exception in scheduler: %s", CurrentExceptionMessage().c_str());
    }

    Y_CORO_DBGOUT("scheduler: stopped");
}

TContPollEventHolder::TContPollEventHolder(void* memory, TCont* rep, SOCKET fds[], int what[], size_t nfds, TInstant deadline)
    : Events_((TFdEvent*)memory)
    , Count_(nfds)
{
    for (size_t i = 0; i < Count_; ++i) {
        new (&(Events_[i])) TFdEvent(rep, fds[i], (ui16)what[i], deadline);
    }
}

TContPollEventHolder::~TContPollEventHolder() {
    for (size_t i = 0; i < Count_; ++i) {
        Events_[i].~TFdEvent();
    }
}

void TContPollEventHolder::ScheduleIoWait(TContExecutor* executor) {
    for (size_t i = 0; i < Count_; ++i) {
        executor->ScheduleIoWait(&(Events_[i]));
    }
}

TFdEvent* TContPollEventHolder::TriggeredEvent() noexcept {
    TFdEvent* ret = nullptr;
    int status = EINPROGRESS;

    for (size_t i = 0; i < Count_; ++i) {
        TFdEvent& ev = Events_[i];

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

    return ret;
}

void TInterruptibleEvent::Interrupt() noexcept {
    if (!Interrupted_) {
        Interrupted_ = true;
        if (!Cont_->Scheduled()) {
            Cont_->ReSchedule();
        }
    }
}
