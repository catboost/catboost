#pragma once

#include "cont_poller.h"
#include "iostatus.h"
#include "poller.h"
#include "schedule_callback.h"

#include <library/containers/intrusive_rb_tree/rb_tree.h>

#include <util/system/error.h>
#include <util/system/context.h>
#include <util/system/defaults.h>
#include <util/generic/ptr.h>
#include <util/generic/intrlist.h>
#include <util/datetime/base.h>
#include <util/generic/maybe.h>

#if !defined(STACK_GROW_DOWN)
#   error "unsupported"
#endif

#define EWAKEDUP 34567

class TCont;
struct TContRep;
class TContExecutor;
class TContPollEvent;

/* TODO(velavokr): some minor improvements:
 * 1) allow any std::function objects, not only TContFunc
 * 2) allow name storage owning (for generated names backed by TString)
 */

namespace NCoro {
    class IScheduleCallback;

    // accounts for asan stack space overhead
    ui32 RealCoroStackSize(ui32 coroStackSize);
    TMaybe<ui32> RealCoroStackSize(TMaybe<ui32> coroStackSize);
}

typedef void (*TContFunc)(TCont*, void*);


class TCont : private TIntrusiveListItem<TCont>, private ITrampoLine {
    struct TTrampoline : public ITrampoLine, TNonCopyable {
        TTrampoline(
            ui32 stackSize,
            TContFunc f,
            TCont* cont,
            void* arg
        ) noexcept;

        ~TTrampoline();

        void SwitchTo(TExceptionSafeContext* ctx) noexcept;

        void DoRun();

    public:
        const THolder<char, TFree> Stack_;
        const ui32 StackSize_;
        const TContClosure Clo_;
        TExceptionSafeContext Ctx_;
        TContFunc const Func_ = nullptr;
        TCont* const Cont_;
        size_t StackId_ = 0;
        void* const Arg_;
    };

    struct TJoinWait: public TIntrusiveListItem<TJoinWait> {
        TJoinWait(TCont* c) noexcept;

        void Wake() noexcept;

    public:
        TCont* Cont_;
    };

    friend class TContExecutor;
    friend class TIntrusiveListItem<TCont>;
    friend class NCoro::TEventWaitQueue;

private:
    TCont(
        size_t stackSize,
        TContExecutor* executor,
        TContFunc func,
        void* arg,
        const char* name
    ) noexcept;

public:
    TContExecutor* Executor() noexcept {
        return Executor_;
    }

    const TContExecutor* Executor() const noexcept {
        return Executor_;
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

    bool Cancelled() const noexcept {
        return Cancelled_;
    }

    bool Scheduled() const noexcept {
        return Scheduled_;
    }

    void WakeAllWaiters() noexcept;

    bool Join(TCont* c, TInstant deadLine = TInstant::Max()) noexcept;

    void ReSchedule() noexcept;

    void SwitchTo(TExceptionSafeContext* ctx) {
        Trampoline_.SwitchTo(ctx);
    }

private:
    void Exit();

    TExceptionSafeContext* Context() noexcept {
        return &Trampoline_.Ctx_;
    }

private:
    TContExecutor* Executor_ = nullptr;
    TTrampoline Trampoline_;

    const char* Name_ = nullptr;
    TIntrusiveList<TJoinWait> Waiters_;
    bool Cancelled_ = false;
    bool Scheduled_ = false;
};


template <class Functor>
static void ContHelperFunc(TCont* cont, void* arg) {
    (*((Functor*)(arg)))(cont);
}

template <typename T, void (T::*M)(TCont*)>
static void ContHelperMemberFunc(TCont* c, void* arg) {
    ((reinterpret_cast<T*>(arg))->*M)(c);
}

/// Central coroutine class.
/// Note, coroutines are single-threaded, and all methods must be called from the single thread
class TContExecutor {
    friend class TCont;
    using TContList = TIntrusiveList<TCont>;

public:
    TContExecutor(
        ui32 defaultStackSize,
        THolder<IPollerFace> poller = IPollerFace::Default(),
        NCoro::IScheduleCallback* = nullptr
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

    TExceptionSafeContext* SchedContext() noexcept {
        return &SchedContext_;
    }

    template <class Functor>
    TCont* Create(Functor& f, const char* name, TMaybe<ui32> stackSize = Nothing()) noexcept {
        return Create((TContFunc)ContHelperFunc<Functor>, (void*)&f, name, stackSize);
    }

    template <typename T, void (T::*M)(TCont*)>
    TCont* Create(T* obj, const char* name, TMaybe<ui32> stackSize = Nothing()) noexcept {
        return Create(ContHelperMemberFunc<T, M>, obj, name, stackSize);
    }

    TCont* Create(TContFunc func, void* arg, const char* name, TMaybe<ui32> stackSize = Nothing()) noexcept;

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
        return Ready_.Size() + ReadyNext_.Size();
    }

    size_t TotalConts() const noexcept {
        return Allocated_;
    }

    size_t TotalWaitingConts() const noexcept {
        return TotalConts() - TotalReadyConts();
    }

    void Abort() noexcept;

    void SetFailOnError(bool fail) noexcept {
        FailOnError_ = fail;
    }

    bool FailOnError() const noexcept {
        return FailOnError_;
    }

    void ScheduleIoWait(TFdEvent* event) {
        WaitQueue_.Register(event);
        Poller_.Schedule(event);
    }

    void ScheduleIoWait(TTimerEvent* event) noexcept {
        WaitQueue_.Register(event);
    }

private:
    void Release(TCont* cont) noexcept;

    void Exit(TCont* cont) noexcept;

    void RunScheduler() noexcept;

    void ScheduleToDelete(TCont* cont) noexcept;

    void ScheduleExecution(TCont* cont) noexcept;

    void ScheduleExecutionNow(TCont* cont) noexcept;

    void Activate(TCont* cont) noexcept;

    void DeleteScheduled() noexcept;

    void WaitForIO();

    void ProcessEvents();

private:
    NCoro::IScheduleCallback* const CallbackPtr_ = nullptr;
    const ui32 DefaultStackSize_;

    TExceptionSafeContext SchedContext_;

    TContList ToDelete_;
    TContList Ready_;
    TContList ReadyNext_;
    NCoro::TEventWaitQueue WaitQueue_;
    NCoro::TContPoller Poller_;
    NCoro::TContPoller::TEvents Events_;

    size_t Allocated_ = 0;
    TCont* Current_ = nullptr;
    bool FailOnError_ = false;
};
