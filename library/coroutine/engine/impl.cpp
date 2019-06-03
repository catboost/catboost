#include "impl.h"
#include "schedule_callback.h"

#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>


TContRep::TContRep(TContStackAllocator* alloc)
    : real(alloc->Allocate())
    , full((char*)real->Data(), real->Length())
    , stack(full.data(), EffectiveStackLength(full.size()))
    , cont(stack.end(), Align(sizeof(TCont)))
    , machine(cont.end(), Align(sizeof(TExceptionSafeContext)))
{}

void TContRep::DoRun() {
    try {
        ContPtr()->Execute();
    } catch (...) {
        Y_VERIFY(
            !ContPtr()->Executor()->FailOnError(),
            "uncaught exception %s", CurrentExceptionMessage().c_str()
        );
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
        << "name = " << Name_ << ", "
        << "addr = " << Hex((size_t)this)
        << ")";
}

bool TCont::Join(TCont* c, TInstant deadLine) noexcept {
    TJoinWait ev(this);
    c->Waiters_.PushBack(&ev);

    do {
        if (SleepD(deadLine) == ETIMEDOUT || Cancelled()) {
            if (!ev.Empty()) {
                c->Cancel();

                do {
                    _SwitchToScheduler();
                } while (!ev.Empty());
            }

            return false;
        }
    } while (!ev.Empty());

    return true;
}

void TCont::WakeAllWaiters() noexcept {
    while (!Waiters_.Empty()) {
        Waiters_.PopFront()->Wake();
    }
}

int TCont::SleepD(TInstant deadline) noexcept {
    TTimerEvent event(this, deadline);

    return ExecuteEvent(&event);
}

void TCont::Yield() noexcept {
    if (SleepD(TInstant::Zero())) {
        ReScheduleAndSwitch();
    }
}

void TCont::ReScheduleAndSwitch() noexcept {
    ReSchedule();
    _SwitchToScheduler();
}

void TCont::Exit() {
    Executor()->Exit(Rep());
}

bool TCont::IAmRunning() const noexcept {
    return Rep() == Executor()->Running();
}

void TCont::Cancel() noexcept {
    if (Cancelled()) {
        return;
    }

    Cancelled_ = true;

    if (!IAmRunning()) {
        ReSchedule();
    }
}

void TCont::ReSchedule() noexcept {
    if (Cancelled()) {
        // Legacy code may expect a Cancelled coroutine to be scheduled without delay.
        Executor()->ScheduleExecutionNow(Rep());
    } else {
        Executor()->ScheduleExecution(Rep());
    }
}

void TCont::_SwitchToScheduler() noexcept {
    Context()->SwitchTo(Executor()->SchedCont());
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
    try {
        while (true) {
            Ready_.Append(ReadyNext_);

            if (Ready_.Empty()) {
                break;
            }

            TContRep* cont = Ready_.PopFront();

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
        Y_FAIL("Uncaught exception in the scheduler: %s", CurrentExceptionMessage().c_str());
    }
}

template <>
void Out<TCont>(IOutputStream& out, const TCont& c) {
    c.PrintMe(out);
}

template <>
void Out<TContRep>(IOutputStream& out, const TContRep& c) {
    c.ContPtr()->PrintMe(out);
}
