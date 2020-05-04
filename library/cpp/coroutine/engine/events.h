#pragma once

#include "impl.h"

#include <util/datetime/base.h>

class TContEvent {
public:
    TContEvent(TCont* current) noexcept
        : Cont_(current)
        , Status_(0)
    {
    }

    ~TContEvent() {
    }

    int WaitD(TInstant deadline) {
        Status_ = 0;
        const int ret = Cont_->SleepD(deadline);

        return Status_ ? Status_ : ret;
    }

    int WaitT(TDuration timeout) {
        return WaitD(timeout.ToDeadLine());
    }

    int WaitI() {
        return WaitD(TInstant::Max());
    }

    void Wake() noexcept {
        SetStatus(EWAKEDUP);
        Cont_->ReSchedule();
    }

    TCont* Cont() noexcept {
        return Cont_;
    }

    int Status() const noexcept {
        return Status_;
    }

    void SetStatus(int status) noexcept {
        Status_ = status;
    }

private:
    TCont* Cont_;
    int Status_;
};

class TContWaitQueue {
    class TWaiter: public TContEvent, public TIntrusiveListItem<TWaiter> {
    public:
        TWaiter(TCont* current) noexcept
            : TContEvent(current)
        {
        }

        ~TWaiter() {
        }
    };

public:
    TContWaitQueue() noexcept {
    }

    ~TContWaitQueue() {
        Y_ASSERT(Waiters_.Empty());
    }

    int WaitD(TCont* current, TInstant deadline) {
        TWaiter waiter(current);

        Waiters_.PushBack(&waiter);

        return waiter.WaitD(deadline);
    }

    int WaitT(TCont* current, TDuration timeout) {
        return WaitD(current, timeout.ToDeadLine());
    }

    int WaitI(TCont* current) {
        return WaitD(current, TInstant::Max());
    }

    void Signal() noexcept {
        if (!Waiters_.Empty()) {
            Waiters_.PopFront()->Wake();
        }
    }

    void BroadCast() noexcept {
        while (!Waiters_.Empty()) {
            Waiters_.PopFront()->Wake();
        }
    }

    void BroadCast(size_t number) noexcept {
        for (size_t i = 0; i < number && !Waiters_.Empty(); ++i) {
            Waiters_.PopFront()->Wake();
        }
    }

private:
    TIntrusiveList<TWaiter> Waiters_;
};


class TContSimpleEvent {
public:
    TContSimpleEvent(TContExecutor* e)
        : E_(e)
    {
    }

    TContExecutor* Executor() const noexcept {
        return E_;
    }

    void Signal() noexcept {
        Q_.Signal();
    }

    void BroadCast() noexcept {
        Q_.BroadCast();
    }

    int WaitD(TInstant deadLine) noexcept {
        return Q_.WaitD(E_->Running(), deadLine);
    }

    int WaitT(TDuration timeout) noexcept {
        return WaitD(timeout.ToDeadLine());
    }

    int WaitI() noexcept {
        return WaitD(TInstant::Max());
    }

private:
    TContWaitQueue Q_;
    TContExecutor* E_;
};
