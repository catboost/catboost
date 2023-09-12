#pragma once

#include "conn.h"
#include "options.h"

#include <util/thread/pool.h>
#include <library/cpp/http/io/stream.h>
#include <util/memory/blob.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <library/cpp/deprecated/atomic/atomic.h>

class IThreadFactory;
class TClientRequest;
class TClientConnection;

class THttpServer {
    friend class TClientRequest;
    friend class TClientConnection;

public:
    class ICallBack {
    public:
        struct TFailLogData {
            int failstate;
            TString url;
        };

        virtual ~ICallBack() {
        }

        virtual void OnFailRequest(int /*failstate*/) {
        }

        virtual void OnFailRequestEx(const TFailLogData& d) {
            OnFailRequest(d.failstate);
        }

        virtual void OnException() {
        }

        virtual void OnMaxConn() {
        }

        virtual TClientRequest* CreateClient() = 0;

        virtual void OnListenStart() {
        }

        virtual void OnListenStop() {
        }

        virtual void OnWait() {
        }

        virtual void* CreateThreadSpecificResource() {
            return nullptr;
        }

        virtual void DestroyThreadSpecificResource(void*) {
        }
    };

    typedef THttpServerOptions TOptions;
    typedef TSimpleSharedPtr<IThreadPool> TMtpQueueRef;

    THttpServer(ICallBack* cb, const TOptions& options = TOptions(), IThreadFactory* pool = nullptr);
    THttpServer(ICallBack* cb, TMtpQueueRef mainWorkers, TMtpQueueRef failWorkers, const TOptions& options = TOptions());
    virtual ~THttpServer();

    bool Start();

    // shutdown a.s.a.p.
    void Stop();

    // graceful shutdown with serving all already open connections
    void Shutdown();

    void Wait();
    int GetErrorCode();
    const char* GetError();
    void RestartRequestThreads(ui32 nTh, ui32 maxQS);
    const TOptions& Options() const noexcept;
    i64 GetClientCount() const;

    class TImpl;
    size_t GetRequestQueueSize() const;
    size_t GetFailQueueSize() const;

    const IThreadPool& GetRequestQueue() const;
    const IThreadPool& GetFailQueue() const;

    static TAtomicBase AcceptReturnsInvalidSocketCounter();

private:
    bool MaxRequestsReached() const;

private:
    THolder<TImpl> Impl_;
};

/**
 * @deprecated Use TRequestReplier instead
 */
class TClientRequest: public IObjectInQueue {
    friend class THttpServer::TImpl;

public:
    TClientRequest();
    ~TClientRequest() override;

    inline THttpInput& Input() noexcept {
        return *HttpConn_->Input();
    }

    inline THttpOutput& Output() noexcept {
        return *HttpConn_->Output();
    }

    THttpServer* HttpServ() const noexcept;
    const TSocket& Socket() const noexcept;
    NAddr::IRemoteAddrRef GetListenerSockAddrRef() const noexcept;
    TInstant AcceptMoment() const noexcept;

    bool IsLocal() const;
    bool CheckLoopback();
    void ProcessFailRequest(int failstate);

    void ReleaseConnection();

    void ResetConnection();

private:
    /*
     * Processes the request after 'connection' been created and 'Headers' been read
     * Returns 'false' if the processing must be continued by the next handler,
     * 'true' otherwise ('this' will be deleted)
     */
    virtual bool Reply(void* ThreadSpecificResource);
    virtual bool BeforeParseRequestOk(void* ThreadSpecificResource) {
        Y_UNUSED(ThreadSpecificResource);
        return true;
    }
    void Process(void* ThreadSpecificResource) override;

public:
    TVector<std::pair<TString, TString>> ParsedHeaders;
    TString RequestString;

private:
    THolder<TClientConnection> Conn_;
    THolder<THttpServerConn> HttpConn_;
};

class TRequestReplier: public TClientRequest {
public:
    TRequestReplier();
    ~TRequestReplier() override;

    struct TReplyParams {
        void* ThreadSpecificResource;
        THttpInput& Input;
        THttpOutput& Output;
    };

    /*
     * Processes the request after 'connection' been created and 'Headers' been read
     * Returns 'false' if the processing must be continued by the next handler,
     * 'true' otherwise ('this' will be deleted)
     */
    virtual bool DoReply(const TReplyParams& params) = 0;

private:
    bool Reply(void* threadSpecificResource) final;

    using TClientRequest::Input;
    using TClientRequest::Output;
};

bool TryToBindAddresses(const THttpServerOptions& options, const std::function<void(TSocket)>* callbackOnBoundAddress = nullptr);
