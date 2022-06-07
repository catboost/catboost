#pragma once

#include "neh.h"
#include <library/cpp/neh/utils.h>

#include <library/cpp/http/io/headers.h>

#include <util/generic/singleton.h>
#include <library/cpp/deprecated/atomic/atomic.h>

namespace NNeh {
    class TNotifyHandle: public THandle {
    public:
        inline TNotifyHandle(IOnRecv* r, const TMessage& msg, TStatCollector* s = nullptr) noexcept
            : THandle(r, s)
            , Msg_(msg)
            , StartTime_(TInstant::Now())
        {
        }

        void NotifyResponse(const TString& resp, const TString& firstLine = {}, const THttpHeaders& headers = Default<THttpHeaders>()) {
            Notify(new TResponse(Msg_, resp, ExecDuration(), firstLine, headers));
        }

        void NotifyError(const TString& errorText) {
            Notify(TResponse::FromError(Msg_, new TError(errorText), ExecDuration()));
        }

        void NotifyError(TErrorRef error) {
            Notify(TResponse::FromError(Msg_, error, ExecDuration()));
        }

        /** Calls when asnwer is received and reponse has headers and first line.
         */
        void NotifyError(TErrorRef error, const TString& data, const TString& firstLine, const THttpHeaders& headers) {
            Notify(TResponse::FromError(Msg_, error, data, ExecDuration(), firstLine, headers));
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
        TAtomicBool SendComplete_;
        TAtomicBool Canceled_;
    };

    typedef TIntrusivePtr<TSimpleHandle> TSimpleHandleRef;
}
