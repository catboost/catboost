#pragma once

class TContEvent {
public:
    inline TContEvent(TCont* current) noexcept
        : Cont_(current)
        , Status_(0)
    {
    }

    inline ~TContEvent() {
    }

    inline int WaitD(TInstant deadline) {
        Status_ = 0;
        const int ret = Cont_->SleepD(deadline);

        return Status_ ? Status_ : ret;
    }

    inline int WaitT(TDuration timeout) {
        return WaitD(timeout.ToDeadLine());
    }

    inline int WaitI() {
        return WaitD(TInstant::Max());
    }

    inline void Wake() noexcept {
        SetStatus(EWAKEDUP);
        Cont_->ReSchedule();
    }

    inline TCont* Cont() noexcept {
        return Cont_;
    }

    inline int Status() const noexcept {
        return Status_;
    }

    inline void SetStatus(int status) noexcept {
        Status_ = status;
    }

private:
    TCont* Cont_;
    int Status_;
};

class TContWaitQueue {
    class TWaiter: public TContEvent, public TIntrusiveListItem<TWaiter> {
    public:
        inline TWaiter(TCont* current) noexcept
            : TContEvent(current)
        {
        }

        inline ~TWaiter() {
        }
    };

public:
    inline TContWaitQueue() noexcept {
    }

    inline ~TContWaitQueue() {
        Y_ASSERT(Waiters_.Empty());
    }

    inline int WaitD(TCont* current, TInstant deadline) {
        TWaiter waiter(current);

        Waiters_.PushBack(&waiter);

        return waiter.WaitD(deadline);
    }

    inline int WaitT(TCont* current, TDuration timeout) {
        return WaitD(current, timeout.ToDeadLine());
    }

    inline int WaitI(TCont* current) {
        return WaitD(current, TInstant::Max());
    }

    inline void Signal() noexcept {
        if (!Waiters_.Empty()) {
            Waiters_.PopFront()->Wake();
        }
    }

    inline void BroadCast() noexcept {
        while (!Waiters_.Empty()) {
            Waiters_.PopFront()->Wake();
        }
    }

    inline void BroadCast(size_t number) noexcept {
        for (size_t i = 0; i < number && !Waiters_.Empty(); ++i) {
            Waiters_.PopFront()->Wake();
        }
    }

private:
    TIntrusiveList<TWaiter> Waiters_;
};

class TContSimpleEvent {
public:
    inline TContSimpleEvent(TContExecutor* e)
        : E_(e)
    {
    }

    inline TContExecutor* Executor() const noexcept {
        return E_;
    }

    inline void Signal() noexcept {
        Q_.Signal();
    }

    inline void BroadCast() noexcept {
        Q_.BroadCast();
    }

    inline int WaitD(TInstant deadLine) noexcept {
        return Q_.WaitD(E_->Running()->ContPtr(), deadLine);
    }

    inline int WaitT(TDuration timeout) noexcept {
        return WaitD(timeout.ToDeadLine());
    }

    inline int WaitI() noexcept {
        return WaitD(TInstant::Max());
    }

private:
    TContWaitQueue Q_;
    TContExecutor* E_;
};
