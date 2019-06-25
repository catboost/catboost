#include "impl.h"
#include "schedule_callback.h"

#include <util/stream/format.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>
#include <util/system/valgrind.h>

#include <cstdlib>

namespace NCoro {
    ui32 RealCoroStackSize(ui32 coroStackSize) {
#if defined(_san_enabled_) || !defined(NDEBUG)
        coroStackSize *= 4;
#endif
        return coroStackSize;
    }

    TMaybe<ui32> RealCoroStackSize(TMaybe<ui32> coroStackSize) {
        if (coroStackSize) {
            return RealCoroStackSize(*coroStackSize);
        } else {
            return Nothing();
        }
    }

    namespace {
        constexpr TStringBuf CANARY = AsStringBuf(
            "4ef8f9c2f7eb6cb8af66f2e441f4250c0f819a30d07821895b53e6017f90fbcd");
    }
}

TCont::TTrampoline::TTrampoline(ui32 stackSize, TContFunc f, TCont* cont, void* arg) noexcept
    : Stack_((char*)malloc(stackSize))
    , StackSize_(stackSize)
    , Clo_{this, {
        Stack_.Get() + NCoro::CANARY.size(),
        StackSize_ - NCoro::CANARY.size()
    }}
    , Ctx_(Clo_)
    , Func_(f)
    , Cont_(cont)
    , Arg_(arg)
{
    Y_VERIFY(Stack_, "out of memory");
    StackId_ = VALGRIND_STACK_REGISTER(
        Stack_.Get() + NCoro::CANARY.size(),
        Stack_.Get() + StackSize_ - NCoro::CANARY.size()
    );
    memcpy(Stack_.Get(), NCoro::CANARY.data(), NCoro::CANARY.size());
}

TCont::TTrampoline::~TTrampoline() {
    VALGRIND_STACK_DEREGISTER(StackId_);
}

void TCont::TTrampoline::SwitchTo(TExceptionSafeContext* ctx) noexcept {
    Y_VERIFY(
        TStringBuf(Stack_.Get(), NCoro::CANARY.size()) == NCoro::CANARY,
        "Stack overflow"
    );
    Ctx_.SwitchTo(ctx);
}

void TCont::TTrampoline::DoRun() {
    try {
        Func_(Cont_, Arg_);
    } catch (...) {
        Y_VERIFY(
            !Cont_->Executor_->FailOnError(),
            "uncaught exception %s", CurrentExceptionMessage().c_str()
        );
    }

    Cont_->WakeAllWaiters();
    Cont_->Exit();
}

TCont::TJoinWait::TJoinWait(TCont* c) noexcept
    : Cont_(c)
{}

void TCont::TJoinWait::Wake() noexcept {
    Cont_->ReSchedule();
}

TCont::TCont(size_t stackSize, TContExecutor* executor, TContFunc func, void* arg, const char* name) noexcept
    : Executor_(executor)
    , Trampoline_(stackSize, func, this, arg)
    , Name_(name)
{}


void TCont::PrintMe(IOutputStream& out) const noexcept {
    out << "cont("
        << "func = " << (size_t)(void*)Trampoline_.Func_ << ", "
        << "arg = " << (size_t)(void*)Trampoline_.Arg_ << ", "
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
                    SwitchTo(Executor()->SchedContext());
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
    SwitchTo(Executor()->SchedContext());
}

void TCont::Exit() {
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

void TCont::ReSchedule() noexcept {
    if (Cancelled()) {
        // Legacy code may expect a Cancelled coroutine to be scheduled without delay.
        Executor()->ScheduleExecutionNow(this);
    } else {
        Executor()->ScheduleExecution(this);
    }
}


TContExecutor::TContExecutor(ui32 defaultStackSize, THolder<IPollerFace> poller, NCoro::IScheduleCallback* callback)
    : CallbackPtr_(callback)
    , DefaultStackSize_(defaultStackSize)
    , Poller_(std::move(poller))
{}

TContExecutor::~TContExecutor() {
    Y_VERIFY(Allocated_ == 0, "leaked %u coroutines", (ui32)Allocated_);
}

void TContExecutor::Execute() noexcept {
    auto nop = [](void*){};
    Execute(nop);
}

void TContExecutor::Execute(TContFunc func, void* arg) noexcept {
    Create(func, arg, "sys_main");
    RunScheduler();
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

void TContExecutor::Abort() noexcept {
    WaitQueue_.Abort();
    auto visitor = [](TCont* c) {
        c->Cancel();
    };
    Ready_.ForEach(visitor);
    ReadyNext_.ForEach(visitor);
}

TCont* TContExecutor::Create(TContFunc func, void* arg, const char* name, TMaybe<ui32> stackSize) noexcept {
    Allocated_ += 1;
    if (!stackSize) {
        stackSize = DefaultStackSize_;
    }
    auto* cont = new TCont(*stackSize, this, func, arg, name);
    ScheduleExecution(cont);
    return cont;
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

void TContExecutor::Activate(TCont* cont) noexcept {
    Current_ = cont;
    cont->Scheduled_ = false;
    SchedContext_.SwitchTo(cont->Context());
}

void TContExecutor::DeleteScheduled() noexcept {
    ToDelete_.ForEach([this](TCont* c) {
        Release(c);
    });
}

void TContExecutor::RunScheduler() noexcept {
    try {
        while (true) {
            Ready_.Append(ReadyNext_);

            if (Ready_.Empty()) {
                break;
            }

            TCont* cont = Ready_.PopFront();

            if (CallbackPtr_) {
                CallbackPtr_->OnSchedule(*this, *cont);
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

void TContExecutor::Exit(TCont* cont) noexcept {
    ScheduleToDelete(cont);
    cont->SwitchTo(&SchedContext_);
    Y_FAIL("can not return from exit");
}

template <>
void Out<TCont>(IOutputStream& out, const TCont& c) {
    c.PrintMe(out);
}
