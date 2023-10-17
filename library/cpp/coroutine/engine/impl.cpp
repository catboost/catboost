#include "impl.h"

#include "stack/stack_allocator.h"
#include "stack/stack_guards.h"

#include <util/generic/scope.h>
#include <util/thread/singleton.h>
#include <util/stream/format.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>


TCont::TJoinWait::TJoinWait(TCont& c) noexcept
    : Cont_(c)
{}

void TCont::TJoinWait::Wake() noexcept {
    Cont_.ReSchedule();
}

TCont::TCont(NCoro::NStack::IAllocator& allocator,
             uint32_t stackSize,
             TContExecutor& executor,
             NCoro::TTrampoline::TFunc func,
             const char* name) noexcept
    : Executor_(executor)
    , Name_(name)
    , Trampoline_(
        allocator,
        stackSize,
        std::move(func),
        this
    )
{}


void TCont::PrintMe(IOutputStream& out) const noexcept {
    out << "cont("
        << "name = " << Name_ << ", "
        << "addr = " << Hex((size_t)this)
        << ")";
}

bool TCont::Join(TCont* c, TInstant deadLine, std::function<void(TJoinWait&, TCont*)> forceStop) noexcept {
    TJoinWait ev(*this);
    c->Waiters_.PushBack(&ev);

    do {
        if (SleepD(deadLine) == ETIMEDOUT || Cancelled()) {
            if (!ev.Empty()) {
                if (forceStop) {
                    forceStop(ev, c);
                } else {
                    c->Cancel();
                }

                do {
                    Switch();
                } while (!ev.Empty());
            }

            return false;
        }
    } while (!ev.Empty());

    return true;
}

int TCont::SleepD(TInstant deadline) noexcept {
    TTimerEvent event(this, deadline);

    return ExecuteEvent(&event);
}

void TCont::Switch() noexcept {
    Executor()->RunScheduler();
}

void TCont::Yield() noexcept {
    if (SleepD(TInstant::Zero())) {
        ReScheduleAndSwitch();
    }
}

void TCont::ReScheduleAndSwitch() noexcept {
    ReSchedule();
    Switch();
}

void TCont::Terminate() {
    while (!Waiters_.Empty()) {
        Waiters_.PopFront()->Wake();
    }
    Executor()->Exit(this);
}

bool TCont::IAmRunning() const noexcept {
    return this == Executor()->Running();
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

void TCont::Cancel(THolder<std::exception> exception) noexcept {
    if (!Cancelled()) {
        SetException(std::move(exception));
        Cancel();
    }
}

void TCont::ReSchedule() noexcept {
    if (Cancelled()) {
        // Legacy code may expect a Cancelled coroutine to be scheduled without delay.
        Executor()->ScheduleExecutionNow(this);
    } else {
        Executor()->ScheduleExecution(this);
    }
}


TContExecutor::TContExecutor(
    uint32_t defaultStackSize,
    THolder<IPollerFace> poller,
    NCoro::IScheduleCallback* scheduleCallback,
    NCoro::IEnterPollerCallback* enterPollerCallback,
    NCoro::NStack::EGuard defaultGuard,
    TMaybe<NCoro::NStack::TPoolAllocatorSettings> poolSettings,
    NCoro::ITime* time
)
    : ScheduleCallback_(scheduleCallback)
    , EnterPollerCallback_(enterPollerCallback)
    , DefaultStackSize_(defaultStackSize)
    , Poller_(std::move(poller))
    , Time_(time)
{
    StackAllocator_ = NCoro::NStack::GetAllocator(poolSettings, defaultGuard);
}

TContExecutor::~TContExecutor() {
    Y_ABORT_UNLESS(Allocated_ == 0, "leaked %u coroutines", (ui32)Allocated_);
}

void TContExecutor::Execute() noexcept {
    auto nop = [](void*){};
    Execute(nop);
}

void TContExecutor::Execute(TContFunc func, void* arg) noexcept {
    CreateOwned([=](TCont* cont) {
        func(cont, arg);
    }, "sys_main");
    RunScheduler();
}

void TContExecutor::WaitForIO() {
    while (Ready_.Empty() && !WaitQueue_.Empty()) {
        const auto now = Now();

        // Waking a coroutine puts it into ReadyNext_ list
        const auto next = WaitQueue_.WakeTimedout(now);

        if (!UserEvents_.Empty()) {
            TIntrusiveList<IUserEvent> userEvents;
            userEvents.Swap(UserEvents_);
            do {
                userEvents.PopFront()->Execute();
            } while (!userEvents.Empty());
        }

        // Polling will return as soon as there is an event to process or a timeout.
        // If there are woken coroutines we do not want to sleep in the poller
        //      yet still we want to check for new io
        //      to prevent ourselves from locking out of io by constantly waking coroutines.

        if (ReadyNext_.Empty()) {
            if (EnterPollerCallback_) {
                EnterPollerCallback_->OnEnterPoller();
            }
            Poll(next);
            if (EnterPollerCallback_) {
                EnterPollerCallback_->OnExitPoller();
            }
        } else if (LastPoll_ + TDuration::MilliSeconds(5) < now) {
            if (EnterPollerCallback_) {
                EnterPollerCallback_->OnEnterPoller();
            }
            Poll(now);
            if (EnterPollerCallback_) {
                EnterPollerCallback_->OnExitPoller();
            }
        }

        Ready_.Append(ReadyNext_);
    }
}

void TContExecutor::Poll(TInstant deadline) {
    Poller_.Wait(PollerEvents_, deadline);
    LastPoll_ = Now();

    // Waking a coroutine puts it into ReadyNext_ list
    for (auto event : PollerEvents_) {
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

void TContExecutor::Abort() noexcept {
    WaitQueue_.Abort();
    auto visitor = [](TCont* c) {
        c->Cancel();
    };
    Ready_.ForEach(visitor);
    ReadyNext_.ForEach(visitor);
}

TCont* TContExecutor::Create(
    TContFunc func,
    void* arg,
    const char* name,
    TMaybe<ui32> customStackSize
) noexcept {
    return CreateOwned([=](TCont* cont) {
        func(cont, arg);
    }, name, customStackSize);
}

TCont* TContExecutor::CreateOwned(
    NCoro::TTrampoline::TFunc func,
    const char* name,
    TMaybe<ui32> customStackSize
) noexcept {
    Allocated_ += 1;
    if (!customStackSize) {
        customStackSize = DefaultStackSize_;
    }
    auto* cont = new TCont(*StackAllocator_, *customStackSize, *this, std::move(func), name);
    ScheduleExecution(cont);
    return cont;
}

NCoro::NStack::TAllocatorStats TContExecutor::GetAllocatorStats() const noexcept {
    return StackAllocator_->GetStackStats();
}

void TContExecutor::Release(TCont* cont) noexcept {
    delete cont;
    Allocated_ -= 1;
}

void TContExecutor::ScheduleToDelete(TCont* cont) noexcept {
    ToDelete_.PushBack(cont);
}

void TContExecutor::ScheduleExecution(TCont* cont) noexcept {
    cont->Scheduled_ = true;
    ReadyNext_.PushBack(cont);
}

void TContExecutor::ScheduleExecutionNow(TCont* cont) noexcept {
    cont->Scheduled_ = true;
    Ready_.PushBack(cont);
}

namespace {
    inline TContExecutor*& ThisThreadExecutor() {
        struct TThisThreadExecutorHolder {
            TContExecutor* Executor = nullptr;
        };
        return FastTlsSingletonWithPriority<TThisThreadExecutorHolder, 0>()->Executor;
    }
}

void TContExecutor::DeleteScheduled() noexcept {
    ToDelete_.ForEach([this](TCont* c) {
        Release(c);
    });
}

TCont* RunningCont() {
    TContExecutor* thisThreadExecutor = ThisThreadExecutor();
    return thisThreadExecutor ? thisThreadExecutor->Running() : nullptr;
}

void TContExecutor::RunScheduler() noexcept {
    try {
        TContExecutor* const prev = ThisThreadExecutor();
        ThisThreadExecutor() = this;
        TCont* caller = Current_;
        TExceptionSafeContext* context = caller ? caller->Trampoline_.Context() : &SchedContext_;
        Y_DEFER {
            ThisThreadExecutor() = prev;
        };

        while (true) {
            if (ScheduleCallback_ && Current_) {
                ScheduleCallback_->OnUnschedule(*this);
            }

            WaitForIO();
            DeleteScheduled();
            Ready_.Append(ReadyNext_);

            if (Ready_.Empty()) {
                Current_ = nullptr;
                if (caller) {
                    context->SwitchTo(&SchedContext_);
                }
                break;
            }

            TCont* cont = Ready_.PopFront();

            if (ScheduleCallback_) {
                ScheduleCallback_->OnSchedule(*this, *cont);
            }

            Current_ = cont;
            cont->Scheduled_ = false;
            if (cont == caller) {
                break;
            }
            context->SwitchTo(cont->Trampoline_.Context());
            if (Paused_) {
                Paused_ = false;
                Current_ = nullptr;
                break;
            }
            if (caller) {
                break;
            }
        }
    } catch (...) {
        TBackTrace::FromCurrentException().PrintTo(Cerr);
        Y_ABORT("Uncaught exception in the scheduler: %s", CurrentExceptionMessage().c_str());
    }
}

void TContExecutor::Pause() {
    if (auto cont = Running()) {
        Paused_ = true;
        ScheduleExecutionNow(cont);
        cont->SwitchTo(&SchedContext_);
    }
}

void TContExecutor::Exit(TCont* cont) noexcept {
    ScheduleToDelete(cont);
    cont->SwitchTo(&SchedContext_);
    Y_ABORT("can not return from exit");
}

TInstant TContExecutor::Now() {
    return Y_LIKELY(Time_ == nullptr) ? TInstant::Now() : Time_->Now();
}

template <>
void Out<TCont>(IOutputStream& out, const TCont& c) {
    c.PrintMe(out);
}
