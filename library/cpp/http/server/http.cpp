#include "http.h"
#include "http_ex.h"

#include <library/cpp/threading/equeue/equeue.h>
#include <library/cpp/threading/equeue/fast.h>

#include <util/generic/buffer.h>
#include <util/generic/intrlist.h>
#include <util/generic/yexception.h>
#include <util/network/address.h>
#include <util/network/socket.h>
#include <util/network/poller.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/compat.h> // stricmp, strnicmp, strlwr, strupr, stpcpy
#include <util/system/defaults.h>
#include <util/system/event.h>
#include <util/system/mutex.h>
#include <util/system/pipe.h>
#include <util/system/thread.h>
#include <util/thread/factory.h>

#include <cerrno>
#include <cstring>

using namespace NAddr;

namespace {
    class IPollAble {
    public:
        inline IPollAble() noexcept {
        }

        virtual ~IPollAble() {
        }

        virtual void OnPollEvent(TInstant now) = 0;
    };

    struct TShouldStop {
    };

    struct TWakeupPollAble: public IPollAble {
        void OnPollEvent(TInstant) override {
            throw TShouldStop();
        }
    };
}

class TClientConnection: public IPollAble, public TIntrusiveListItem<TClientConnection> {
public:
    TClientConnection(const TSocket& s, THttpServer::TImpl* serv, NAddr::IRemoteAddrRef listenerSockAddrRef);
    ~TClientConnection() override;

    void OnPollEvent(TInstant now) override;

    inline void Activate(TInstant now) noexcept;
    inline void DeActivate();
    inline void Reject();

public:
    TSocket Socket_;
    NAddr::IRemoteAddrRef ListenerSockAddrRef_;
    THttpServer::TImpl* HttpServ_ = nullptr;
    bool Reject_ = false;
    TInstant LastUsed;
    TInstant AcceptMoment;
    size_t ReceivedRequests = 0;
};

class THttpServer::TImpl {
public:
    class TConnections {
    public:
        inline TConnections(TSocketPoller* poller, const THttpServerOptions& options)
            : Poller_(poller)
            , Options(options)
        {
        }

        inline ~TConnections() {
        }

        inline void Add(TClientConnection* c) noexcept {
            TGuard<TMutex> g(Mutex_);

            Conns_.PushBack(c);
            Poller_->WaitRead(c->Socket_, (void*)static_cast<const IPollAble*>(c));
        }

        inline void Erase(TClientConnection* c, TInstant now) noexcept {
            TGuard<TMutex> g(Mutex_);
            EraseUnsafe(c);
            if (Options.ExpirationTimeout > TDuration::Zero()) {
                TryRemovingUnsafe(now - Options.ExpirationTimeout);
            }
        }

        inline void Clear() noexcept {
            TGuard<TMutex> g(Mutex_);

            Conns_.Clear();
        }

        inline bool RemoveOld(TInstant border) noexcept {
            TGuard<TMutex> g(Mutex_);
            return TryRemovingUnsafe(border);
        }

        bool TryRemovingUnsafe(TInstant border) noexcept {
            if (Conns_.Empty()) {
                return false;
            }
            TClientConnection* c = &*(Conns_.Begin());
            if (c->LastUsed > border) {
                return false;
            }
            EraseUnsafe(c);
            delete c;
            return true;
        }

        void EraseUnsafe(TClientConnection* c) noexcept {
            Poller_->Unwait(c->Socket_);
            c->Unlink();
        }

    public:
        TMutex Mutex_;
        TIntrusiveListWithAutoDelete<TClientConnection, TDelete> Conns_;
        TSocketPoller* Poller_ = nullptr;
        const THttpServerOptions& Options;
    };

    static void* ListenSocketFunction(void* param) {
        try {
            ((TImpl*)param)->ListenSocket();
        } catch (...) {

        }

        return nullptr;
    }

    TAutoPtr<TClientRequest> CreateRequest(TAutoPtr<TClientConnection> c) {
        THolder<TClientRequest> obj(Cb_->CreateClient());

        obj->Conn_.Reset(c.Release());

        return obj;
    }

    void AddRequestFromSocket(const TSocket& s, TInstant now, NAddr::IRemoteAddrRef listenerSockAddrRef) {
        if (MaxRequestsReached()) {
            Cb_->OnMaxConn();
            bool wasRemoved = Connections->RemoveOld(TInstant::Max());
            if (!wasRemoved && Options_.RejectExcessConnections) {
                (new TClientConnection(s, this, listenerSockAddrRef))->Reject();
                return;
            }
        }

        auto connection = new TClientConnection(s, this, listenerSockAddrRef);
        connection->LastUsed = now;
        connection->DeActivate();
    }

    void SaveErrorCode() {
        ErrorCode = WSAGetLastError();
    }

    int GetErrorCode() const {
        return ErrorCode;
    }

    const char* GetError() const {
        return LastSystemErrorText(ErrorCode);
    }

    bool Start() {
        Poller.Reset(new TSocketPoller());
        Connections.Reset(new TConnections(Poller.Get(), Options_));

        // Start the listener thread
        ListenerRunningOK = false;

        // throws on error
        TPipeHandle::Pipe(ListenWakeupReadFd, ListenWakeupWriteFd);

        SetNonBlock(ListenWakeupWriteFd, true);
        SetNonBlock(ListenWakeupReadFd, true);

        Poller->WaitRead(ListenWakeupReadFd, &WakeupPollAble);

        ListenStartEvent.Reset();
        try {
            ListenThread.Reset(new TThread(ListenSocketFunction, this));
            ListenThread->Start();
        } catch (const yexception&) {
            SaveErrorCode();
            return false;
        }

        // Wait until the thread has completely started and return the success indicator
        ListenStartEvent.Wait();

        return ListenerRunningOK;
    }

    void Wait() {
        Cb_->OnWait();
        TGuard<TMutex> g(StopMutex);
        if (ListenThread) {
            ListenThread->Join();
            ListenThread.Reset(nullptr);
        }
    }

    void Stop() {
        Shutdown();

        TGuard<TMutex> g(StopMutex);
        if (ListenThread) {
            ListenThread->Join();
            ListenThread.Reset(nullptr);
        }

        while (ConnectionCount) {
            usleep(10000);
            Connections->Clear();
        }

        Connections.Destroy();
        Poller.Destroy();
    }

    void Shutdown() {
        ListenWakeupWriteFd.Write("", 1);
        // ignore result
    }

    void AddRequest(TAutoPtr<TClientRequest> req, bool fail) {
        struct TFailRequest: public THttpClientRequestEx {
            inline TFailRequest(TAutoPtr<TClientRequest> parent) {
                Conn_.Reset(parent->Conn_.Release());
                HttpConn_.Reset(parent->HttpConn_.Release());
            }

            bool Reply(void*) override {
                if (!ProcessHeaders()) {
                    return true;
                }

                ProcessFailRequest(0);
                return true;
            }
        };

        if (!fail && Requests->Add(req.Get())) {
            Y_UNUSED(req.Release());
        } else {
            req = new TFailRequest(req);

            if (FailRequests->Add(req.Get())) {
                Y_UNUSED(req.Release());
            } else {
                Cb_->OnFailRequest(-1);
            }
        }
    }

    size_t GetRequestQueueSize() const {
        return Requests->Size();
    }

    size_t GetFailQueueSize() const {
        return FailRequests->Size();
    }

    const IThreadPool& GetRequestQueue() const {
        return *Requests;
    }

    const IThreadPool& GetFailQueue() const {
        return *FailRequests;
    }

    class TListenSocket: public IPollAble, public TIntrusiveListItem<TListenSocket> {
    public:
        inline TListenSocket(const TSocket& s, TImpl* parent)
            : S_(s)
            , Server_(parent)
            , SockAddrRef_(GetSockAddr(S_))
        {
        }

        ~TListenSocket() override {
        }

        void OnPollEvent(TInstant) override {
            SOCKET s = ::accept(S_, nullptr, nullptr);

            if (s == INVALID_SOCKET) {
                ythrow yexception() << "accept: " << LastSystemErrorText();
            }

            Server_->AddRequestFromSocket(s, TInstant::Now(), SockAddrRef_);
        }

        SOCKET GetSocket() const noexcept {
            return S_;
        }

    private:
        TSocket S_;
        TImpl* Server_ = nullptr;
        NAddr::IRemoteAddrRef SockAddrRef_;
    };

    void ListenSocket() {
        TThread::SetCurrentThreadName(Options_.ListenThreadName.c_str());

        ErrorCode = 0;
        TIntrusiveListWithAutoDelete<TListenSocket, TDelete> Reqs;

        std::function<void(TSocket)> callback = [&](TSocket socket) {
            THolder<TListenSocket> ls(new TListenSocket(socket, this));
            Poller->WaitRead(socket, static_cast<IPollAble*>(ls.Get()));
            Reqs.PushBack(ls.Release());
        };
        bool addressesBound = TryToBindAddresses(Options_, &callback);
        if (!addressesBound) {
            SaveErrorCode();
            ListenStartEvent.Signal();

            return;
        }

        Requests->Start(Options_.nThreads, Options_.MaxQueueSize);
        FailRequests->Start(Options_.nFThreads, Options_.MaxFQueueSize);
        Cb_->OnListenStart();
        ListenerRunningOK = true;
        ListenStartEvent.Signal();

        TVector<void*> events;
        events.resize(Options_.EpollMaxEvents);

        TInstant now = TInstant::Now();
        for (;;) {
            try {
                const TInstant deadline = Options_.PollTimeout == TDuration::Zero() ? TInstant::Max() : now + Options_.PollTimeout;
                const size_t ret = Poller->WaitD(events.data(), events.size(), deadline);

                now = TInstant::Now();
                for (size_t i = 0; i < ret; ++i) {
                    ((IPollAble*)events[i])->OnPollEvent(now);
                }

                if (ret == 0 && Options_.ExpirationTimeout > TDuration::Zero()) {
                    Connections->RemoveOld(now - Options_.ExpirationTimeout);
                }

                // When MaxConnections is limited or ExpirationTimeout is set, OnPollEvent can call
                // RemoveOld and destroy other IPollAble* objects in the
                // poller. Thus in this case we can safely process only one
                // event from the poller at a time.
                if (!Options_.MaxConnections && Options_.ExpirationTimeout == TDuration::Zero()) {
                    if (ret >= events.size()) {
                        events.resize(ret * 2);
                    }
                }
            } catch (const TShouldStop&) {
                break;
            } catch (...) {
                Cb_->OnException();
            }
        }

        while (!Reqs.Empty()) {
            THolder<TListenSocket> ls(Reqs.PopFront());

            Poller->Unwait(ls->GetSocket());
        }

        Requests->Stop();
        FailRequests->Stop();
        Cb_->OnListenStop();
    }

    void RestartRequestThreads(ui32 nTh, ui32 maxQS) {
        Requests->Stop();
        Options_.nThreads = nTh;
        Options_.MaxQueueSize = maxQS;
        Requests->Start(Options_.nThreads, Options_.MaxQueueSize);
    }

    TImpl(THttpServer* parent, ICallBack* cb, TMtpQueueRef mainWorkers, TMtpQueueRef failWorkers, const TOptions& options_)
        : Requests(mainWorkers)
        , FailRequests(failWorkers)
        , Options_(options_)
        , Cb_(cb)
        , Parent_(parent)
    {
    }

    TImpl(THttpServer* parent, ICallBack* cb, const TOptions& options, IThreadFactory* factory)
        : TImpl(
              parent,
              cb,
              MakeThreadPool<TSimpleThreadPool>(factory, options, cb, options.RequestsThreadName),
              MakeThreadPool<TThreadPool>(factory, options, nullptr, options.FailRequestsThreadName),
              options) {
    }

    ~TImpl() {
        try {
            Stop();
        } catch (...) {
        }
    }

    inline const TOptions& Options() const noexcept {
        return Options_;
    }

    inline void DecreaseConnections() noexcept {
        AtomicDecrement(ConnectionCount);
    }

    inline void IncreaseConnections() noexcept {
        AtomicIncrement(ConnectionCount);
    }

    inline i64 GetClientCount() const {
        return AtomicGet(ConnectionCount);
    }

    inline bool MaxRequestsReached() const {
        return Options_.MaxConnections && ((size_t)GetClientCount() >= Options_.MaxConnections);
    }

    THolder<TThread> ListenThread;
    TPipeHandle ListenWakeupReadFd;
    TPipeHandle ListenWakeupWriteFd;
    TSystemEvent ListenStartEvent;
    TMtpQueueRef Requests;
    TMtpQueueRef FailRequests;
    TAtomic ConnectionCount = 0;
    THolder<TSocketPoller> Poller;
    THolder<TConnections> Connections;
    bool ListenerRunningOK = false;
    int ErrorCode = 0;
    TOptions Options_;
    ICallBack* Cb_ = nullptr;
    THttpServer* Parent_ = nullptr;
    TWakeupPollAble WakeupPollAble;
    TMutex StopMutex;

private:
    template <class TThreadPool_>
    static THolder<IThreadPool> MakeThreadPool(ICallBack* callback, const IThreadPool::TParams& params) {
        if (callback) {
            return MakeHolder<TThreadPoolBinder<TThreadPool_, THttpServer::ICallBack>>(callback, params);
        } else {
            return  MakeHolder<TThreadPool_>(params);
        }
    }

    template <class TThreadPool_>
    static THolder<IThreadPool> MakeThreadPool(IThreadFactory* factory, const TOptions& options, ICallBack* callback = nullptr, const TString& threadName = {}) {
        if (!factory) {
            factory = SystemThreadFactory();
        }

        THolder<IThreadPool> pool;
        const auto params = IThreadPool::TParams().SetFactory(factory).SetThreadName(threadName);

        if (options.UseFastElasticQueues) {
            pool = MakeThreadPool<TFastElasticQueue>(callback, params);
        } else {
            pool = MakeThreadPool<TThreadPool_>(callback, params);
            if (options.UseElasticQueues) {
                pool = MakeHolder<TElasticQueue>(std::move(pool));
            }
        }

        return pool;
    }
};

THttpServer::THttpServer(ICallBack* cb, const TOptions& options, IThreadFactory* pool)
    : Impl_(new TImpl(this, cb, options, pool))
{
}

THttpServer::THttpServer(ICallBack* cb, TMtpQueueRef mainWorkers, TMtpQueueRef failWorkers, const TOptions& options)
    : Impl_(new TImpl(this, cb, mainWorkers, failWorkers, options))
{
}

THttpServer::~THttpServer() {
}

i64 THttpServer::GetClientCount() const {
    return Impl_->GetClientCount();
}

bool THttpServer::Start() {
    return Impl_->Start();
}

void THttpServer::Stop() {
    Impl_->Stop();
}

void THttpServer::Shutdown() {
    Impl_->Shutdown();
}

void THttpServer::Wait() {
    Impl_->Wait();
}

int THttpServer::GetErrorCode() {
    return Impl_->GetErrorCode();
}

const char* THttpServer::GetError() {
    return Impl_->GetError();
}

void THttpServer::RestartRequestThreads(ui32 n, ui32 queue) {
    Impl_->RestartRequestThreads(n, queue);
}

const THttpServer::TOptions& THttpServer::Options() const noexcept {
    return Impl_->Options();
}

size_t THttpServer::GetRequestQueueSize() const {
    return Impl_->GetRequestQueueSize();
}

size_t THttpServer::GetFailQueueSize() const {
    return Impl_->GetFailQueueSize();
}

const IThreadPool& THttpServer::GetRequestQueue() const {
    return Impl_->GetRequestQueue();
}

const IThreadPool& THttpServer::GetFailQueue() const {
    return Impl_->GetFailQueue();
}

bool THttpServer::MaxRequestsReached() const {
    return Impl_->MaxRequestsReached();
}

TClientConnection::TClientConnection(const TSocket& s, THttpServer::TImpl* serv, NAddr::IRemoteAddrRef listenerSockAddrRef)
    : Socket_(s)
    , ListenerSockAddrRef_(listenerSockAddrRef)
    , HttpServ_(serv)
{
    SetNoDelay(Socket_, true);

    const TDuration& clientTimeout = HttpServ_->Options().ClientTimeout;
    if (clientTimeout != TDuration::Zero()) {
        SetSocketTimeout(Socket_, (long)clientTimeout.Seconds(), clientTimeout.MilliSecondsOfSecond());
    }

    HttpServ_->IncreaseConnections();
}

TClientConnection::~TClientConnection() {
    HttpServ_->DecreaseConnections();
}

void TClientConnection::OnPollEvent(TInstant now) {
    THolder<TClientConnection> this_(this);
    Activate(now);

    {
        char tmp[1];

        if (::recv(Socket_, tmp, 1, MSG_PEEK) < 1) {
            /*
             * We can received a FIN so our socket was moved to
             * TCP_CLOSE_WAIT state. Check it before adding work
             * for this socket.
             */

            return;
        }
    }

    THolder<TClientRequest> obj(HttpServ_->CreateRequest(this_));
    AcceptMoment = now;

    HttpServ_->AddRequest(obj, Reject_);
}

void TClientConnection::Activate(TInstant now) noexcept {
    HttpServ_->Connections->Erase(this, now);
    LastUsed = now;
    ++ReceivedRequests;
}

void TClientConnection::DeActivate() {
    HttpServ_->Connections->Add(this);
}

void TClientConnection::Reject() {
    Reject_ = true;

    HttpServ_->Connections->Add(this);
}

TClientRequest::TClientRequest() {
}

TClientRequest::~TClientRequest() {
}

bool TClientRequest::Reply(void* /*ThreadSpecificResource*/) {
    if (strnicmp(RequestString.data(), "GET ", 4)) {
        Output() << "HTTP/1.0 501 Not Implemented\r\n\r\n";
    } else {
        Output() << "HTTP/1.0 200 OK\r\n"
                    "Content-Type: text/html\r\n"
                    "\r\n"
                    "Hello World!\r\n";
    }

    return true;
}

bool TClientRequest::IsLocal() const {
    return HasLocalAddress(Socket());
}

bool TClientRequest::CheckLoopback() {
    bool isLocal = false;

    try {
        isLocal = IsLocal();
    } catch (const yexception& e) {
        Output() << "HTTP/1.0 500 Oops\r\n\r\n"
                 << e.what() << "\r\n";
        return false;
    }

    if (!isLocal) {
        Output() << "HTTP/1.0 403 Permission denied\r\n"
                    "Content-Type: text/html; charset=windows-1251\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                    "<html><head><title>Permission denied</title></head>"
                    "<body><h1>Permission denied</h1>"
                    "<p>This request must be sent from the localhost.</p>"
                    "</body></html>\r\n";

        return false;
    }

    return true;
}

void TClientRequest::ReleaseConnection() {
    if (Conn_ && HttpConn_ && HttpServ()->Options().KeepAliveEnabled && HttpConn_->CanBeKeepAlive() && (!HttpServ()->Options().RejectExcessConnections || !HttpServ()->MaxRequestsReached())) {
        Output().Finish();
        Conn_->DeActivate();
        Y_UNUSED(Conn_.Release());
    }
}

void TClientRequest::ResetConnection() {
    if (HttpConn_) {
        // send RST packet to client
        HttpConn_->Reset();
        HttpConn_.Destroy();
    }
}

void TClientRequest::Process(void* ThreadSpecificResource) {
    THolder<TClientRequest> this_(this);

    auto* serverImpl = Conn_->HttpServ_;

    try {
        if (!HttpConn_) {
            const size_t outputBufferSize = HttpServ()->Options().OutputBufferSize;
            if (outputBufferSize) {
                HttpConn_.Reset(new THttpServerConn(Socket(), outputBufferSize));
            } else {
                HttpConn_.Reset(new THttpServerConn(Socket()));
            }

            auto maxRequestsPerConnection = HttpServ()->Options().MaxRequestsPerConnection;
            HttpConn_->Output()->EnableKeepAlive(HttpServ()->Options().KeepAliveEnabled && (!maxRequestsPerConnection || Conn_->ReceivedRequests < maxRequestsPerConnection));
            HttpConn_->Output()->EnableCompression(HttpServ()->Options().CompressionEnabled);
        }

        if (ParsedHeaders.empty()) {
            RequestString = Input().FirstLine();

            const THttpHeaders& h = Input().Headers();
            ParsedHeaders.reserve(h.Count());
            for (THttpHeaders::TConstIterator it = h.Begin(); it != h.End(); ++it) {
                ParsedHeaders.emplace_back(it->Name(), it->Value());
            }
        }

        if (Reply(ThreadSpecificResource)) {
            ReleaseConnection();

            /*
             * *this will be destroyed...
             */

            return;
        }
    } catch (...) {
        serverImpl->Cb_->OnException();

        throw;
    }

    Y_UNUSED(this_.Release());
}

void TClientRequest::ProcessFailRequest(int failstate) {
    Output() << "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 21\r\n"
                "\r\n"
                "Service Unavailable\r\n";

    TString url;

    if (!strnicmp(RequestString.data(), "GET ", 4)) {
        // Trying to extract url...
        const char* str = RequestString.data();

        // Skipping spaces before url...
        size_t start = 3;
        while (str[start] == ' ') {
            ++start;
        }

        if (str[start]) {
            // Traversing url...
            size_t idx = start;

            while (str[idx] != ' ' && str[idx]) {
                ++idx;
            }

            url = RequestString.substr(start, idx - start);
        }
    }

    const THttpServer::ICallBack::TFailLogData d = {
        failstate,
        url,
    };

    // Handling failure...
    Conn_->HttpServ_->Cb_->OnFailRequestEx(d);
    Output().Flush();
}

THttpServer* TClientRequest::HttpServ() const noexcept {
    return Conn_->HttpServ_->Parent_;
}

const TSocket& TClientRequest::Socket() const noexcept {
    return Conn_->Socket_;
}

NAddr::IRemoteAddrRef TClientRequest::GetListenerSockAddrRef() const noexcept {
    return Conn_->ListenerSockAddrRef_;
}

TInstant TClientRequest::AcceptMoment() const noexcept {
    return Conn_->AcceptMoment;
}

/*
 * TRequestReplier
 */
TRequestReplier::TRequestReplier() {
}

TRequestReplier::~TRequestReplier() {
}

bool TRequestReplier::Reply(void* threadSpecificResource) {
    const TReplyParams params = {
        threadSpecificResource, Input(), Output()};

    return DoReply(params);
}

bool TryToBindAddresses(const THttpServerOptions& options, const std::function<void(TSocket)>* callbackOnBoundAddress) {
    THttpServerOptions::TBindAddresses addrs;
    try {
        options.BindAddresses(addrs);
    } catch (const std::exception&) {
        return false;
    }

    for (const auto& na : addrs) {
        for (TNetworkAddress::TIterator ai = na.Begin(); ai != na.End(); ++ai) {
            NAddr::TAddrInfo addr(&*ai);

            TSocket socket(::socket(addr.Addr()->sa_family, SOCK_STREAM, 0));

            if (socket == INVALID_SOCKET) {
                return false;
            }

            FixIPv6ListenSocket(socket);

            if (options.ReuseAddress) {
                int yes = 1;
                ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
            }

            if (options.ReusePort) {
                SetReusePort(socket, true);
            }

            if (::bind(socket, addr.Addr(), addr.Len()) == SOCKET_ERROR) {
                return false;
            }

            if (::listen(socket, options.ListenBacklog) == SOCKET_ERROR) {
                return false;
            }

            if (callbackOnBoundAddress != nullptr) {
                (*callbackOnBoundAddress)(socket);
            }
        }
    }

    return true;
}
