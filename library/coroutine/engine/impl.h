#pragma once

#include "schedule_callback.h"
#include "iostatus.h"
#include "poller.h"
#include "cont_poller.h"
#include "stack.h"

#include <library/containers/intrusive_rb_tree/rb_tree.h>

#include <util/system/mutex.h>
#include <util/system/error.h>
#include <util/system/context.h>
#include <util/system/defaults.h>
#include <util/system/valgrind.h>
#include <util/network/iovec.h>
#include <util/memory/tempbuf.h>
#include <util/memory/smallobj.h>
#include <util/memory/addstorage.h>
#include <util/network/socket.h>
#include <util/network/nonblock.h>
#include <util/generic/ptr.h>
#include <util/generic/buffer.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/generic/intrlist.h>
#include <util/generic/yexception.h>
#include <util/datetime/base.h>
#include <util/stream/format.h>
#include <util/string/builder.h>

#if !defined(STACK_GROW_DOWN)
#   error "unsupported"
#endif

#define EWAKEDUP 34567

class TCont;
struct TContRep;
class TContEvent;
class TContExecutor;
class TContPollEvent;

namespace NCoro {
    class IScheduleCallback;
}

typedef void (*TContFunc)(TCont*, void*);


class TCont {
    struct TJoinWait: public TIntrusiveListItem<TJoinWait> {
        TJoinWait(TCont* c) noexcept
            : C(c)
        {
        }

        void Wake() noexcept {
            C->ReSchedule();
        }

        TCont* C;
    };

    friend struct TContRep;
    friend class TContExecutor;
    friend class TContPollEvent;

public:
    TCont(TContExecutor* executor, TContRep* rep, TContFunc func, void* arg, const char* name)
        : Executor_(executor)
        , Rep_(rep)
        , Func_(func)
        , Arg_(arg)
        , Name_(name)
    {
    }

    ~TCont() {
        Executor_ = nullptr;
        Rep_ = nullptr;
    }

    TExceptionSafeContext* Context() noexcept {
        return (TExceptionSafeContext*)(((char*)(this)) + Align(sizeof(TCont)));
    }

    const TExceptionSafeContext* Context() const noexcept {
        return const_cast<TCont*>(this)->Context();
    }

    TContExecutor* Executor() noexcept {
        return Executor_;
    }

    const TContExecutor* Executor() const noexcept {
        return Executor_;
    }

    TContRep* Rep() noexcept {
        return Rep_;
    }

    const TContRep* Rep() const noexcept {
        return Rep_;
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

    void Exit();

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

    void _SwitchToScheduler() noexcept;

private:

    void Execute() {
        Y_ASSERT(Func_);

        Func_(this, Arg_);
    }

private:
    TContExecutor* Executor_ = nullptr;
    TContRep* Rep_ = nullptr;
    TContFunc Func_ = nullptr;
    void* Arg_ = nullptr;
    const char* Name_ = nullptr;
    TIntrusiveList<TJoinWait> Waiters_;
    bool Cancelled_ = false;
    bool Scheduled_ = false;
};


struct TContRep : public TIntrusiveListItem<TContRep>, public ITrampoLine {
    TContRep(TContStackAllocator* alloc);

    void DoRun() override;

    void Construct(TContExecutor* executor, TContFunc func, void* arg, const char* name);
    void Destruct() noexcept;

    TCont* ContPtr() noexcept {
        return (TCont*)cont.data();
    }

    const TCont* ContPtr() const noexcept {
        return (const TCont*)cont.data();
    }

    TExceptionSafeContext* MachinePtr() noexcept {
        return (TExceptionSafeContext*)machine.data();
    }

    static size_t Overhead() noexcept {
        return Align(sizeof(TCont)) + Align(sizeof(TExceptionSafeContext));
    }

    static size_t EffectiveStackLength(size_t alloced) noexcept {
        return alloced - Overhead();
    }

    static size_t ToAllocate(size_t stackLen) noexcept {
        return Align(stackLen) + Overhead();
    }

    bool IAmRuning() const noexcept {
        return ContPtr()->IAmRunning();
    }

    TContStackAllocator::TStackPtr real;

    TArrayRef<char> full;
    TArrayRef<char> stack;
    TArrayRef<char> cont;
    TArrayRef<char> machine;
};


class TEventWaitQueue {
    struct TCancel {
        void operator()(NCoro::TContPollEvent* e) noexcept {
            e->Cont()->Cancel();
        }

        void operator()(NCoro::TContPollEvent& e) noexcept {
            operator()(&e);
        }
    };

    typedef TRbTree<NCoro::TContPollEvent, NCoro::TContPollEventCompare> TIoWait;

public:
    void Register(NCoro::TContPollEvent* event) {
        IoWait_.Insert(event);
        event->Cont()->Rep()->Unlink();
    }

    bool Empty() const noexcept {
        return IoWait_.Empty();
    }

    void Abort() noexcept {
        TCancel visitor;

        IoWait_.ForEach(visitor);
    }

    TInstant WakeTimedout(TInstant now) noexcept {
        TIoWait::TIterator it = IoWait_.Begin();

        if (it != IoWait_.End()) {
            if (it->DeadLine() > now) {
                return it->DeadLine();
            }

            do {
                (it++)->Wake(ETIMEDOUT);
            } while (it != IoWait_.End() && it->DeadLine() <= now);
        }

        return now;
    }

private:
    TIoWait IoWait_;
};


class TContRepPool {
    using TFreeReps = TIntrusiveListWithAutoDelete<TContRep, TDelete>;

public:
    TContRepPool(TContStackAllocator* alloc)
        : Alloc_(alloc)
    {
    }

    TContRepPool(size_t stackLen)
        : MyAlloc_(new TDefaultStackAllocator(TContRep::ToAllocate(AlignUp<size_t>(stackLen, 2 * STACK_ALIGN))))
        , Alloc_(MyAlloc_.Get())
    {
    }

    ~TContRepPool() {
        unsigned long long all = Allocated_;
        Y_VERIFY(Allocated_ == 0, "leaked coroutines: %llu", all);
    }

    TContRep* Allocate() {
        Allocated_ += 1;

        if (Free_.Empty()) {
            return new TContRep(Alloc_);
        }

        return Free_.PopFront();
    }

    void Release(TContRep* cont) noexcept {
        Allocated_ -= 1;
        Free_.PushFront(cont);
    }

    [[nodiscard]] size_t Allocated() const {
        return Allocated_;
    }

private:
    THolder<TContStackAllocator> MyAlloc_;
    TContStackAllocator* const Alloc_;
    TFreeReps Free_;
    ui64 Allocated_ = 0;
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
    friend struct TContRep;
    friend class TContEvent;
    friend class TContPollEvent;
    friend class TContPollEventHolder;
    using TContList = TIntrusiveList<TContRep>;

public:
    TContExecutor(
        size_t stackSize,
        THolder<IPollerFace> poller = IPollerFace::Default(),
        NCoro::IScheduleCallback* = nullptr
    );

    TContExecutor(
        TContRepPool* pool,
        THolder<IPollerFace> poller = IPollerFace::Default(),
        NCoro::IScheduleCallback* = nullptr
    );

    ~TContExecutor();

    /*
     * assume we already create all necessary coroutines
     */
    void Execute() {
        auto nop = [](void*){};
        Execute(nop);
    }

    void Execute(TContFunc func, void* arg = nullptr) {
        Create(func, arg, "sys_main");
        RunScheduler();
    }

    template <class Functor>
    void Execute(Functor& f) {
        Execute((TContFunc)ContHelperFunc<Functor>, (void*)&f);
    }

    template <typename T, void (T::*M)(TCont*)>
    void Execute(T* obj) {
        Execute(ContHelperMemberFunc<T, M>, obj);
    }

    TExceptionSafeContext* SchedCont() noexcept {
        return &SchedContext_;
    }

    template <class Functor>
    TContRep* Create(Functor& f, const char* name) {
        return Create((TContFunc)ContHelperFunc<Functor>, (void*)&f, name);
    }

    template <typename T, void (T::*M)(TCont*)>
    TContRep* Create(T* obj, const char* name) {
        return Create(ContHelperMemberFunc<T, M>, obj, name);
    }

    TContRep* Create(TContFunc func, void* arg, const char* name) {
        TContRep* cont = CreateImpl(func, arg, name);

        ScheduleExecution(cont);

        return cont;
    }

    NCoro::TContPoller* Poller() noexcept {
        return &Poller_;
    }

    TEventWaitQueue* WaitQueue() noexcept {
        return &WaitQueue_;
    }

    TContRep* Running() noexcept {
        return Current_;
    }

    const TContRep* Running() const noexcept {
        return Current_;
    }

    size_t TotalReadyConts() const noexcept {
        return Ready_.Size() + ReadyNext_.Size();
    }

    size_t TotalConts() const noexcept {
        return Pool_.Allocated();
    }

    size_t TotalWaitingConts() const noexcept {
        return TotalConts() - TotalReadyConts();
    }

    void Abort() noexcept {
        WaitQueue_.Abort();
        auto visitor = [](TContRep* c) {
            c->ContPtr()->Cancel();
        };
        Ready_.ForEach(visitor);
        ReadyNext_.ForEach(visitor);
    }

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
    TContRep* CreateImpl(TContFunc func, void* arg, const char* name) {
        TContRep* cont = Pool_.Allocate();

        cont->Construct(this, func, arg, name);
        cont->Unlink();

        return cont;
    }

    void Release(TContRep* cont) noexcept {
        cont->Unlink();
        cont->Destruct();
        Pool_.Release(cont);
    }

    void Exit(TContRep* cont) noexcept {
        ScheduleToDelete(cont);
        cont->ContPtr()->_SwitchToScheduler();

        Y_FAIL("can not return from exit");
    }

    void RunScheduler() noexcept;

    void ScheduleToDelete(TContRep* cont) noexcept {
        ToDelete_.PushBack(cont);
    }

    void ScheduleExecution(TContRep* cont) noexcept {
        cont->ContPtr()->Scheduled_ = true;
        ReadyNext_.PushBack(cont);
    }

    void ScheduleExecutionNow(TContRep* cont) noexcept {
        cont->ContPtr()->Scheduled_ = true;
        Ready_.PushBack(cont);
    }

    void Activate(TContRep* cont) noexcept {
        Current_ = cont;
        TCont* contPtr = cont->ContPtr();
        contPtr->Scheduled_ = false;
        SchedContext_.SwitchTo(contPtr->Context());
    }

    void DeleteScheduled() noexcept {
        ToDelete_.ForEach([](TContRep* c) {
            c->ContPtr()->Executor()->Release(c);
        });
    }

    void WaitForIO();

    void ProcessEvents();

private:
    TContList ToDelete_;
    TContList Ready_;
    TContList ReadyNext_;
    TEventWaitQueue WaitQueue_;
    NCoro::TContPoller Poller_;
    THolder<TContRepPool> MyPool_;
    TContRepPool& Pool_;
    TExceptionSafeContext SchedContext_;
    TContRep* Current_ = nullptr;
    NCoro::IScheduleCallback* const CallbackPtr_ = nullptr;
    using TEvents = NCoro::TContPoller::TEvents;
    TEvents Events_;
    bool FailOnError_;
};
