#pragma once

#include "callbacks.h"
#include "cont_poller.h"
#include "iostatus.h"
#include "poller.h"
#include "stack/stack_common.h"
#include "trampoline.h"
#include "custom_time.h"

#include <library/cpp/containers/intrusive_rb_tree/rb_tree.h>

#include <util/system/error.h>
#include <util/generic/ptr.h>
#include <util/generic/intrlist.h>
#include <util/datetime/base.h>
#include <util/generic/maybe.h>
#include <util/generic/function.h>


#define EWAKEDUP 34567

class TCont;
struct TContRep;
class TContExecutor;
class TContPollEvent;

namespace NCoro::NStack {
    class IAllocator;
}

class TCont : private TIntrusiveListItem<TCont> {
    friend class TContExecutor;
    friend class TIntrusiveListItem<TCont>;
    friend class NCoro::TEventWaitQueue;
    friend class NCoro::TTrampoline;

public:
    struct TJoinWait: public TIntrusiveListItem<TJoinWait> {
        TJoinWait(TCont& c) noexcept;

        void Wake() noexcept;

    public:
        TCont& Cont_;
    };

private:
    TCont(
        NCoro::NStack::IAllocator& allocator,
        uint32_t stackSize,
        TContExecutor& executor,
        NCoro::TTrampoline::TFunc func,
        const char* name
    ) noexcept;

public:
    TContExecutor* Executor() noexcept {
        return &Executor_;
    }

    const TContExecutor* Executor() const noexcept {
        return &Executor_;
    }

    const char* Name() const noexcept {
        return Name_;
    }

    void PrintMe(IOutputStream& out) const noexcept;

    void Yield() noexcept;

    void ReScheduleAndSwitch() noexcept;

    /// @return ETIMEDOUT on success
    int SleepD(TInstant deadline) noexcept;

    int SleepT(TDuration timeout) noexcept {
        return SleepD(timeout.ToDeadLine());
    }

    int SleepI() noexcept {
        return SleepD(TInstant::Max());
    }

    bool IAmRunning() const noexcept;

    void Cancel() noexcept;
    void Cancel(THolder<std::exception> exception) noexcept;

    bool Cancelled() const noexcept {
        return Cancelled_;
    }

    bool Scheduled() const noexcept {
        return Scheduled_;
    }

    /// \param this корутина, которая будет ждать
    /// \param c корутина, которую будем ждать
    /// \param deadLine максимальное время ожидания
    /// \param forceStop кастомный обработчик ситуации, когда завершается время ожидания или отменяется ожидающая корутина (this)
    /// дефолтное поведение - отменить ожидаемую корутину (c->Cancel())
    bool Join(TCont* c, TInstant deadLine = TInstant::Max(), std::function<void(TJoinWait&, TCont*)> forceStop = {}) noexcept;

    void ReSchedule() noexcept;

    void Switch() noexcept;

    void SwitchTo(TExceptionSafeContext* ctx) {
        Trampoline_.SwitchTo(ctx);
    }

    THolder<std::exception> TakeException() noexcept {
        return std::move(Exception_);
    }

    void SetException(THolder<std::exception> exception) noexcept {
        Exception_ = std::move(exception);
    }

private:
    void Terminate();

private:
    TContExecutor& Executor_;

    // TODO(velavokr): allow name storage owning (for generated names backed by TString)
    const char* Name_ = nullptr;

    NCoro::TTrampoline Trampoline_;

    TIntrusiveList<TJoinWait> Waiters_;
    bool Cancelled_ = false;
    bool Scheduled_ = false;

    THolder<std::exception> Exception_;
};

TCont* RunningCont();


template <class Functor>
static void ContHelperFunc(TCont* cont, void* arg) {
    (*((Functor*)(arg)))(cont);
}

template <typename T, void (T::*M)(TCont*)>
static void ContHelperMemberFunc(TCont* c, void* arg) {
    ((reinterpret_cast<T*>(arg))->*M)(c);
}

class IUserEvent
    : public TIntrusiveListItem<IUserEvent>
{
public:
    virtual ~IUserEvent() = default;

    virtual void Execute() = 0;
};

/// Central coroutine class.
/// Note, coroutines are single-threaded, and all methods must be called from the single thread
class TContExecutor {
    friend class TCont;
    using TContList = TIntrusiveList<TCont>;

public:
    TContExecutor(
        uint32_t defaultStackSize,
        THolder<IPollerFace> poller = IPollerFace::Default(),
        NCoro::IScheduleCallback* = nullptr,
        NCoro::IEnterPollerCallback* = nullptr,
        NCoro::NStack::EGuard stackGuard = NCoro::NStack::EGuard::Canary,
        TMaybe<NCoro::NStack::TPoolAllocatorSettings> poolSettings = Nothing(),
        NCoro::ITime* time = nullptr
    );

    ~TContExecutor();

    // if we already have a coroutine to run
    void Execute() noexcept;

    void Execute(TContFunc func, void* arg = nullptr) noexcept;

    template <class Functor>
    void Execute(Functor& f) noexcept {
        Execute((TContFunc)ContHelperFunc<Functor>, (void*)&f);
    }

    template <typename T, void (T::*M)(TCont*)>
    void Execute(T* obj) noexcept {
        Execute(ContHelperMemberFunc<T, M>, obj);
    }

    template <class Functor>
    TCont* Create(
        Functor& f,
        const char* name,
        TMaybe<ui32> customStackSize = Nothing()
    ) noexcept {
        return Create((TContFunc)ContHelperFunc<Functor>, (void*)&f, name, customStackSize);
    }

    template <typename T, void (T::*M)(TCont*)>
    TCont* Create(
        T* obj,
        const char* name,
        TMaybe<ui32> customStackSize = Nothing()
    ) noexcept {
        return Create(ContHelperMemberFunc<T, M>, obj, name, customStackSize);
    }

    TCont* Create(
        TContFunc func,
        void* arg,
        const char* name,
        TMaybe<ui32> customStackSize = Nothing()
    ) noexcept;

    TCont* CreateOwned(
        NCoro::TTrampoline::TFunc func,
        const char* name,
        TMaybe<ui32> customStackSize = Nothing()
    ) noexcept;

    NCoro::TContPoller* Poller() noexcept {
        return &Poller_;
    }

    TCont* Running() noexcept {
        return Current_;
    }

    const TCont* Running() const noexcept {
        return Current_;
    }

    size_t TotalReadyConts() const noexcept {
        return Ready_.Size() + TotalScheduledConts();
    }

    size_t TotalScheduledConts() const noexcept {
        return ReadyNext_.Size();
    }

    size_t TotalConts() const noexcept {
        return Allocated_;
    }

    size_t TotalWaitingConts() const noexcept {
        return TotalConts() - TotalReadyConts();
    }

    NCoro::NStack::TAllocatorStats GetAllocatorStats() const noexcept;

    // TODO(velavokr): rename, it is just CancelAll actually
    void Abort() noexcept;

    void SetFailOnError(bool fail) noexcept {
        FailOnError_ = fail;
    }

    bool FailOnError() const noexcept {
        return FailOnError_;
    }

    void RegisterInWaitQueue(NCoro::TContPollEvent* event) {
        WaitQueue_.Register(event);
    }

    void ScheduleIoWait(TFdEvent* event) {
        RegisterInWaitQueue(event);
        Poller_.Schedule(event);
    }

    void ScheduleIoWait(TTimerEvent* event) noexcept {
        RegisterInWaitQueue(event);
    }

    void ScheduleUserEvent(IUserEvent* event) {
        UserEvents_.PushBack(event);
    }

    void Pause();
    TInstant Now();
private:
    void Release(TCont* cont) noexcept;

    void Exit(TCont* cont) noexcept;

    void RunScheduler() noexcept;

    void ScheduleToDelete(TCont* cont) noexcept;

    void ScheduleExecution(TCont* cont) noexcept;

    void ScheduleExecutionNow(TCont* cont) noexcept;

    void DeleteScheduled() noexcept;

    void WaitForIO();

    void Poll(TInstant deadline);

private:
    NCoro::IScheduleCallback* const ScheduleCallback_ = nullptr;
    NCoro::IEnterPollerCallback* const EnterPollerCallback_ = nullptr;
    const uint32_t DefaultStackSize_;
    THolder<NCoro::NStack::IAllocator> StackAllocator_;

    TExceptionSafeContext SchedContext_;

    TContList ToDelete_;
    TContList Ready_;
    TContList ReadyNext_;
    NCoro::TEventWaitQueue WaitQueue_;
    NCoro::TContPoller Poller_;
    NCoro::TContPoller::TEvents PollerEvents_;
    TInstant LastPoll_;

    TIntrusiveList<IUserEvent> UserEvents_;

    size_t Allocated_ = 0;
    TCont* Current_ = nullptr;
    bool FailOnError_ = false;
    bool Paused_ = false;
    NCoro::ITime* Time_ = nullptr;
};
