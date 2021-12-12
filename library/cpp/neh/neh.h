#pragma once

#include "wfmo.h"
#include "stat.h"

#include <library/cpp/http/io/headers.h>

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/datetime/base.h>

#include <utility>

namespace NNeh {
    struct TMessage {
        TMessage() = default;

        inline TMessage(TString addr, TString data)
            : Addr(std::move(addr))
            , Data(std::move(data))
        {
        }

        static TMessage FromString(TStringBuf request);

        TString Addr;
        TString Data;
    };

    using TMessageRef = TAutoPtr<TMessage>;

    struct TError {
    public:
        enum TType {
            UnknownType,
            Cancelled,
            ProtocolSpecific
        };

        TError(TString text, TType type = UnknownType, i32 code = 0, i32 systemCode = 0)
            : Type(std::move(type))
            , Code(code)
            , Text(text)
            , SystemCode(systemCode)
        {
        }

        TType Type = UnknownType;
        i32 Code = 0; // protocol specific code (example(http): 404)
        TString Text;
        i32 SystemCode = 0; // system error code
    };

    using TErrorRef = TAutoPtr<TError>;

    struct TResponse;
    using TResponseRef = TAutoPtr<TResponse>;

    struct TResponse {
        inline TResponse(TMessage req,
                         TString data,
                         const TDuration duration)
            : TResponse(std::move(req), std::move(data), duration, {} /* firstLine */, {} /* headers */, {} /* error */)
        {
        }

        inline TResponse(TMessage req,
                         TString data,
                         const TDuration duration,
                         TString firstLine,
                         THttpHeaders headers)
            : TResponse(std::move(req), std::move(data), duration, std::move(firstLine), std::move(headers), {} /* error */)
        {
        }

        inline TResponse(TMessage req,
                         TString data,
                         const TDuration duration,
                         TString firstLine,
                         THttpHeaders headers,
                         TErrorRef error)
            : Request(std::move(req))
            , Data(std::move(data))
            , Duration(duration)
            , FirstLine(std::move(firstLine))
            , Headers(std::move(headers))
            , Error_(std::move(error))
        {
        }

        inline static TResponseRef FromErrorText(TMessage msg, TString error, const TDuration duration) {
            return new TResponse(std::move(msg), {} /* data */, duration, {} /* firstLine */, {} /* headers */, new TError(std::move(error)));
        }

        inline static TResponseRef FromError(TMessage msg, TErrorRef error, const TDuration duration) {
            return new TResponse(std::move(msg), {} /* data */, duration, {} /* firstLine */, {} /* headers */, error);
        }

        inline static TResponseRef FromError(TMessage msg, TErrorRef error, const TDuration duration,
                                             TString data, TString firstLine, THttpHeaders headers)
        {
            return new TResponse(std::move(msg), std::move(data), duration, std::move(firstLine), std::move(headers), error);
        }

        inline static TResponseRef FromError(
            TMessage msg,
            TErrorRef error,
            TString data,
            const TDuration duration,
            TString firstLine,
            THttpHeaders headers)
        {
            return new TResponse(std::move(msg), std::move(data), duration, std::move(firstLine), std::move(headers), error);
        }

        inline bool IsError() const {
            return Error_.Get();
        }

        inline TError::TType GetErrorType() const {
            return Error_.Get() ? Error_->Type : TError::UnknownType;
        }

        inline i32 GetErrorCode() const {
            return Error_.Get() ? Error_->Code : 0;
        }

        inline i32 GetSystemErrorCode() const {
            return Error_.Get() ? Error_->SystemCode : 0;
        }

        inline TString GetErrorText() const {
            return Error_.Get() ? Error_->Text : TString();
        }

        const TMessage Request;
        const TString Data;
        const TDuration Duration;
        const TString FirstLine;
        THttpHeaders Headers;

    private:
        THolder<TError> Error_;
    };

    class THandle;

    class IOnRecv {
    public:
        virtual ~IOnRecv() = default;

        virtual void OnNotify(THandle&) {
        } //callback on receive response
        virtual void OnEnd() {
        }                                       //response was extracted by Wait() method, - OnRecv() will not be called
        virtual void OnRecv(THandle& resp) = 0; //callback on destroy handler
    };

    class THandle: public TThrRefBase, public TWaitHandle {
    public:
        inline THandle(IOnRecv* f, TStatCollector* s = nullptr) noexcept
            : F_(f)
            , Stat_(s)
        {
        }

        ~THandle() override {
            if (F_) {
                try {
                    F_->OnRecv(*this);
                } catch (...) {
                }
            }
        }

        virtual bool MessageSendedCompletely() const noexcept {
            //TODO
            return true;
        }

        virtual void Cancel() noexcept {
            //TODO
            if (!!Stat_)
                Stat_->OnCancel();
        }

        inline const TResponse* Response() const noexcept {
            return R_.Get();
        }

        //method MUST be called only after success Wait() for this handle or from callback IOnRecv::OnRecv()
        //else exist chance for memory leak (race between Get()/Notify())
        inline TResponseRef Get() noexcept {
            return R_;
        }

        inline bool Wait(TResponseRef& msg, const TInstant deadLine) {
            if (WaitForOne(*this, deadLine)) {
                if (F_) {
                    F_->OnEnd();
                    F_ = nullptr;
                }
                msg = Get();

                return true;
            }

            return false;
        }

        inline bool Wait(TResponseRef& msg, const TDuration timeOut) {
            return Wait(msg, timeOut.ToDeadLine());
        }

        inline bool Wait(TResponseRef& msg) {
            return Wait(msg, TInstant::Max());
        }

        inline TResponseRef Wait(const TInstant  deadLine) {
            TResponseRef ret;

            Wait(ret, deadLine);

            return ret;
        }

        inline TResponseRef Wait(const TDuration  timeOut) {
            return Wait(timeOut.ToDeadLine());
        }

        inline TResponseRef Wait() {
            return Wait(TInstant::Max());
        }

    protected:
        inline void Notify(TResponseRef resp) {
            if (!!Stat_) {
                if (!resp || resp->IsError()) {
                    Stat_->OnFail();
                } else {
                    Stat_->OnSuccess();
                }
            }
            R_.Swap(resp);
            if (F_) {
                try {
                    F_->OnNotify(*this);
                } catch (...) {
                }
            }
            Signal();
        }

        IOnRecv* F_;

    private:
        TResponseRef R_;
        THolder<TStatCollector> Stat_;
    };

    using THandleRef = TIntrusivePtr<THandle>;

    THandleRef Request(const TMessage& msg, IOnRecv* fallback);

    inline THandleRef Request(const TMessage& msg) {
        return Request(msg, nullptr);
    }

    THandleRef Request(const TString& req, IOnRecv* fallback);

    inline THandleRef Request(const TString& req) {
        return Request(req, nullptr);
    }

    class IMultiRequester {
    public:
        virtual ~IMultiRequester() = default;

        virtual void Add(const THandleRef& req) = 0;
        virtual void Del(const THandleRef& req) = 0;
        virtual bool Wait(THandleRef& req, TInstant deadLine) = 0;
        virtual bool IsEmpty() const = 0;

        inline void Schedule(const TString& req) {
            Add(Request(req));
        }

        inline bool Wait(THandleRef& req, TDuration timeOut) {
            return Wait(req, timeOut.ToDeadLine());
        }

        inline bool Wait(THandleRef& req) {
            return Wait(req, TInstant::Max());
        }

        inline bool Wait(TResponseRef& resp, TInstant deadLine) {
            THandleRef req;

            while (Wait(req, deadLine)) {
                resp = req->Get();

                if (!!resp) {
                    return true;
                }
            }

            return false;
        }

        inline bool Wait(TResponseRef& resp) {
            return Wait(resp, TInstant::Max());
        }
    };

    using IMultiRequesterRef = TAutoPtr<IMultiRequester>;

    IMultiRequesterRef CreateRequester();

    bool SetProtocolOption(TStringBuf protoOption, TStringBuf value);
}
