#pragma once

#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/maybe.h>
#include <util/stream/output.h>
#include <util/datetime/base.h>
#include <functional>

namespace NNeh {
    using TData = TVector<char>;

    class TDataSaver: public TData, public IOutputStream {
    public:
        TDataSaver() = default;
        ~TDataSaver() override = default;
        TDataSaver(TDataSaver&&) noexcept = default;
        TDataSaver& operator=(TDataSaver&&) noexcept = default;

        void DoWrite(const void* buf, size_t len) override {
            insert(end(), (const char*)buf, (const char*)buf + len);
        }
    };

    class IRequest {
    public:
        IRequest()
            : ArrivalTime_(TInstant::Now())
        {
        }

        virtual ~IRequest() = default;

        virtual TStringBuf Scheme() const = 0;
        virtual TString RemoteHost() const = 0; //IP-literal / IPv4address / reg-name()
        virtual TStringBuf Service() const = 0;
        virtual TStringBuf Data() const = 0;
        virtual TStringBuf RequestId() const = 0;
        virtual bool Canceled() const = 0;
        virtual void SendReply(TData& data) = 0;
        enum TResponseError {
            BadRequest,             // bad request data - http_code 400
            Forbidden,              // forbidden request - http_code 403
            NotExistService,        // not found request handler - http_code 404
            TooManyRequests,        // too many requests for the handler - http_code 429
            InternalError,          // s...amthing happen - http_code 500
            NotImplemented,         // not implemented - http_code 501
            BadGateway,             // remote backend not available - http_code 502
            ServiceUnavailable,     // overload - http_code 503
            BandwidthLimitExceeded, // 5xx version of 429
            MaxResponseError        // count error types
        };
        virtual void SendError(TResponseError err, const TString& details = TString()) = 0;
        virtual TInstant ArrivalTime() const {
            return ArrivalTime_;
        }

    private:
        TInstant ArrivalTime_;
    };

    using IRequestRef = TAutoPtr<IRequest>;

    struct IOnRequest {
        virtual void OnRequest(IRequestRef req) = 0;
    };

    class TRequestOut: public TDataSaver {
    public:
        inline TRequestOut(IRequest* req)
            : Req_(req)
        {
        }

        ~TRequestOut() override {
            try {
                Finish();
            } catch (...) {
            }
        }

        void DoFinish() override {
            if (Req_) {
                Req_->SendReply(*this);
                Req_ = nullptr;
            }
        }

    private:
        IRequest* Req_;
    };

    struct IRequester {
        virtual ~IRequester() = default;
    };

    using IRequesterRef = TAtomicSharedPtr<IRequester>;

    struct IService: public TThrRefBase {
        virtual void ServeRequest(const IRequestRef& request) = 0;
    };

    using IServiceRef = TIntrusivePtr<IService>;
    using TServiceFunction = std::function<void(const IRequestRef&)>;

    IServiceRef Wrap(const TServiceFunction& func);

    class IServices {
    public:
        virtual ~IServices() = default;

        /// use current thread and run #threads-1 in addition
        virtual void Loop(size_t threads) = 0;
        /// run #threads and return control
        virtual void ForkLoop(size_t threads) = 0;
        /// send stopping request and wait stopping all services
        virtual void SyncStopFork() = 0;
        /// send stopping request and return control (async call)
        virtual void Stop() = 0;
        /// just listen, don't start any threads
        virtual void Listen() = 0;

        inline IServices& Add(const TString& service, IServiceRef srv) {
            DoAdd(service, srv);

            return *this;
        }

        inline IServices& Add(const TString& service, const TServiceFunction& func) {
            return Add(service, Wrap(func));
        }

        template <class T>
        inline IServices& Add(const TString& service, T& t) {
            return this->Add(service, std::bind(&T::ServeRequest, std::ref(t), std::placeholders::_1));
        }

        template <class T, void (T::*M)(const IRequestRef&)>
        inline IServices& Add(const TString& service, T& t) {
            return this->Add(service, std::bind(M, std::ref(t), std::placeholders::_1));
        }

    private:
        virtual void DoAdd(const TString& service, IServiceRef srv) = 0;
    };

    using IServicesRef = TAutoPtr<IServices>;
    using TCheck = std::function<TMaybe<IRequest::TResponseError>(const IRequestRef&)>;

    IServicesRef CreateLoop();
    // if request fails check it will be cancelled
    IServicesRef CreateLoop(TCheck check);
}
