#pragma once

#include "neh.h"
#include <library/neh/utils.h>

#include <library/http/io/headers.h>

#include <util/generic/singleton.h>
#include <util/system/atomic.h>

namespace NNeh {
    class TNotifyHandle: public THandle {
    public:
        inline TNotifyHandle(IOnRecv* r, const TMessage& msg, TStatCollector* s = nullptr) noexcept
            : THandle(r, s)
            , Msg_(msg)
            , StartTime_(TInstant::Now())
        {
        }

        void NotifyResponse(const TString& resp, const THttpHeaders& headers = Default<THttpHeaders>()) {
            Notify(new TResponse(Msg_, resp, ExecDuration(), headers));
        }

        void NotifyError(const TString& errorText) {
            Notify(TResponse::FromError(Msg_, new TError(errorText), ExecDuration()));
        }

        void NotifyError(const TString& errorText, const TString& data) {
            Notify(TResponse::FromError(Msg_, new TError(errorText), data, ExecDuration()));
        }

        void NotifyError(TErrorRef error) {
            Notify(TResponse::FromError(Msg_, error, ExecDuration()));
        }

        void NotifyError(TErrorRef error, const TString& data) {
            Notify(TResponse::FromError(Msg_, error, data, ExecDuration()));
        }

        const TMessage& Message() const noexcept {
            return Msg_;
        }

    private:
        inline TDuration ExecDuration() const {
            TInstant now = TInstant::Now();
            if (now > StartTime_) {
                return now - StartTime_;
            }

            return TDuration::Zero();
        }

        const TMessage Msg_;
        const TInstant StartTime_;
    };

    typedef TIntrusivePtr<TNotifyHandle> TNotifyHandleRef;

    class TSimpleHandle: public TNotifyHandle {
    public:
        inline TSimpleHandle(IOnRecv* r, const TMessage& msg, TStatCollector* s = nullptr) noexcept
            : TNotifyHandle(r, msg, s)
            , SendComplete_(false)
            , Canceled_(false)
        {
        }

        bool MessageSendedCompletely() const noexcept override {
            return SendComplete_;
        }

        void Cancel() noexcept override {
            Canceled_ = true;
            THandle::Cancel();
        }

        inline void SetSendComplete() noexcept {
            SendComplete_ = true;
        }

        inline bool Canceled() const noexcept {
            return Canceled_;
        }

        inline const TAtomicBool* CanceledPtr() const noexcept {
            return &Canceled_;
        }

        void ResetOnRecv() noexcept {
            F_ = nullptr;
        }

    private:
        volatile bool SendComplete_;
        TAtomicBool Canceled_;
    };

    typedef TIntrusivePtr<TSimpleHandle> TSimpleHandleRef;
}
