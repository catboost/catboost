#pragma once

#include "io_service_impl.h"

namespace NAsio {
    class TTimerOperation: public TOperation {
    public:
        TTimerOperation(TIOService::TImpl::TTimer* t, TInstant deadline)
            : TOperation(deadline)
            , T_(t)
        {
        }

        void AddOp(TIOService::TImpl&) override {
            Y_ASSERT(0);
        }

        void Finalize() override {
            DBGOUT("TTimerDeadlineOperation::Finalize()");
            T_->DelOp(this);
        }

    protected:
        TIOService::TImpl::TTimer* T_;
    };

    class TRegisterTimerOperation: public TTimerOperation {
    public:
        TRegisterTimerOperation(TIOService::TImpl::TTimer* t, TInstant deadline = TInstant::Max())
            : TTimerOperation(t, deadline)
        {
            Speculative_ = true;
        }

        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            T_->GetIOServiceImpl().SyncRegisterTimer(T_);
            return true;
        }
    };

    class TTimerDeadlineOperation: public TTimerOperation {
    public:
        TTimerDeadlineOperation(TIOService::TImpl::TTimer* t, TDeadlineTimer::THandler h, TInstant deadline)
            : TTimerOperation(t, deadline)
            , H_(h)
        {
        }

        void AddOp(TIOService::TImpl&) override {
            T_->AddOp(this);
        }

        bool Execute(int errorCode) override {
            DBGOUT("TTimerDeadlineOperation::Execute(" << errorCode << ")");
            H_(errorCode == ETIMEDOUT ? 0 : errorCode, *this);
            return true;
        }

    private:
        TDeadlineTimer::THandler H_;
    };

    class TCancelTimerOperation: public TTimerOperation {
    public:
        TCancelTimerOperation(TIOService::TImpl::TTimer* t)
            : TTimerOperation(t, TInstant::Max())
        {
            Speculative_ = true;
        }

        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            T_->FailOperations(ECANCELED);
            return true;
        }
    };

    class TUnregisterTimerOperation: public TTimerOperation {
    public:
        TUnregisterTimerOperation(TIOService::TImpl::TTimer* t, TInstant deadline = TInstant::Max())
            : TTimerOperation(t, deadline)
        {
            Speculative_ = true;
        }

        bool Execute(int errorCode) override {
            Y_UNUSED(errorCode);
            DBGOUT("TUnregisterTimerOperation::Execute(" << errorCode << ")");
            T_->GetIOServiceImpl().SyncUnregisterAndDestroyTimer(T_);
            return true;
        }
    };

    class TDeadlineTimer::TImpl: public TIOService::TImpl::TTimer {
    public:
        TImpl(TIOService::TImpl& srv)
            : TIOService::TImpl::TTimer(srv)
        {
        }

        void AsyncWaitExpireAt(TDeadline d, TDeadlineTimer::THandler h) {
            Srv_.ScheduleOp(new TTimerDeadlineOperation(this, h, d));
        }

        void Cancel() {
            Srv_.ScheduleOp(new TCancelTimerOperation(this));
        }
    };
}
