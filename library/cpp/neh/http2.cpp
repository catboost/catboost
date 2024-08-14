#include "http2.h"

#include "conn_cache.h"
#include "details.h"
#include "factory.h"
#include "http_common.h"
#include "smart_ptr.h"
#include "utils.h"

#include <library/cpp/http/push_parser/http_parser.h>
#include <library/cpp/http/misc/httpcodes.h>
#include <library/cpp/http/misc/parsed_request.h>
#include <library/cpp/neh/asio/executor.h>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/network/iovec.h>
#include <util/stream/output.h>
#include <util/stream/zlib.h>
#include <util/system/condvar.h>
#include <util/system/mutex.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>
#include <util/thread/factory.h>
#include <util/thread/singleton.h>
#include <util/system/sanitizers.h>
#include <util/system/thread.h>

#include <atomic>

#if defined(_unix_)
#include <sys/ioctl.h>
#endif

#if defined(_linux_)
#undef SIOCGSTAMP
#undef SIOCGSTAMPNS
#include <linux/sockios.h>
#define FIONWRITE SIOCOUTQ
#endif

//#define DEBUG_HTTP2

#ifdef DEBUG_HTTP2
#define DBGOUT(args) Cout << args << Endl;
#else
#define DBGOUT(args)
#endif

using namespace NDns;
using namespace NAsio;
using namespace NNeh;
using namespace NNeh::NHttp;
using namespace NNeh::NHttp2;
using namespace std::placeholders;

//
// has complex keep-alive references between entities in multi-thread enviroment,
// this create risks for races/memory leak, etc..
// so connecting/disconnecting entities must be doing carefully
//
// handler <=-> request <==> connection(socket) <= handlers, stored in io_service
//                           ^
//                           +== connections_cache
// '=>' -- shared/intrusive ptr
// '->' -- weak_ptr
//

static TDuration FixTimeoutForSanitizer(const TDuration timeout) {
    ui64 multiplier = 1;
    if (NSan::ASanIsOn()) {
        // https://github.com/google/sanitizers/wiki/AddressSanitizer
        multiplier = 4;
    } else if (NSan::MSanIsOn()) {
        // via https://github.com/google/sanitizers/wiki/MemorySanitizer
        multiplier = 3;
    } else if (NSan::TSanIsOn()) {
        // via https://clang.llvm.org/docs/ThreadSanitizer.html
        multiplier = 15;
    }

    return TDuration::FromValue(timeout.GetValue() * multiplier);
}

TDuration THttp2Options::ConnectTimeout = FixTimeoutForSanitizer(TDuration::MilliSeconds(1000));
TDuration THttp2Options::InputDeadline = TDuration::Max();
TDuration THttp2Options::OutputDeadline = TDuration::Max();
TDuration THttp2Options::SymptomSlowConnect = FixTimeoutForSanitizer(TDuration::MilliSeconds(10));
size_t THttp2Options::InputBufferSize = 16 * 1024;
bool THttp2Options::KeepInputBufferForCachedConnections = false;
size_t THttp2Options::AsioThreads = 4;
size_t THttp2Options::AsioServerThreads = 4;
bool THttp2Options::EnsureSendingCompleteByAck = false;
int THttp2Options::Backlog = 100;
TDuration THttp2Options::ServerInputDeadline = FixTimeoutForSanitizer(TDuration::MilliSeconds(500));
TDuration THttp2Options::ServerOutputDeadline = TDuration::Max();
TDuration THttp2Options::ServerInputDeadlineKeepAliveMax = FixTimeoutForSanitizer(TDuration::Seconds(120));
TDuration THttp2Options::ServerInputDeadlineKeepAliveMin = FixTimeoutForSanitizer(TDuration::Seconds(10));
bool THttp2Options::ServerUseDirectWrite = false;
bool THttp2Options::UseResponseAsErrorMessage = false;
bool THttp2Options::FullHeadersAsErrorMessage = false;
bool THttp2Options::ErrorDetailsAsResponseBody = false;
bool THttp2Options::RedirectionNotError = false;
bool THttp2Options::AnyResponseIsNotError = false;
bool THttp2Options::TcpKeepAlive = false;
i32 THttp2Options::LimitRequestsPerConnection = -1;
bool THttp2Options::QuickAck = false;
bool THttp2Options::UseAsyncSendRequest = false;
bool THttp2Options::RespectHostInHttpServerNetworkAddress = false;

bool THttp2Options::Set(TStringBuf name, TStringBuf value) {
#define HTTP2_TRY_SET(optType, optName)       \
    if (name == TStringBuf(#optName)) {      \
        optName = FromString<optType>(value); \
    }

    HTTP2_TRY_SET(TDuration, ConnectTimeout)
    else HTTP2_TRY_SET(TDuration, InputDeadline)
    else HTTP2_TRY_SET(TDuration, OutputDeadline)
    else HTTP2_TRY_SET(TDuration, SymptomSlowConnect) else HTTP2_TRY_SET(size_t, InputBufferSize) else HTTP2_TRY_SET(bool, KeepInputBufferForCachedConnections) else HTTP2_TRY_SET(size_t, AsioThreads) else HTTP2_TRY_SET(size_t, AsioServerThreads) else HTTP2_TRY_SET(bool, EnsureSendingCompleteByAck) else HTTP2_TRY_SET(int, Backlog) else HTTP2_TRY_SET(TDuration, ServerInputDeadline) else HTTP2_TRY_SET(TDuration, ServerOutputDeadline) else HTTP2_TRY_SET(TDuration, ServerInputDeadlineKeepAliveMax) else HTTP2_TRY_SET(TDuration, ServerInputDeadlineKeepAliveMin) else HTTP2_TRY_SET(bool, ServerUseDirectWrite) else HTTP2_TRY_SET(bool, UseResponseAsErrorMessage) else HTTP2_TRY_SET(bool, FullHeadersAsErrorMessage) else HTTP2_TRY_SET(bool, ErrorDetailsAsResponseBody) else HTTP2_TRY_SET(bool, RedirectionNotError) else HTTP2_TRY_SET(bool, AnyResponseIsNotError) else HTTP2_TRY_SET(bool, TcpKeepAlive) else HTTP2_TRY_SET(i32, LimitRequestsPerConnection) else HTTP2_TRY_SET(bool, QuickAck)
    else HTTP2_TRY_SET(bool, UseAsyncSendRequest) else {
        return false;
    }
    return true;
}

namespace NNeh {
    const NDns::TResolvedHost* Resolve(const TStringBuf host, ui16 port, NHttp::EResolverType resolverType);
}

namespace {
//#define DEBUG_STAT

#ifdef DEBUG_STAT
    struct TDebugStat {
        static std::atomic<size_t> ConnTotal;
        static std::atomic<size_t> ConnActive;
        static std::atomic<size_t> ConnCached;
        static std::atomic<size_t> ConnDestroyed;
        static std::atomic<size_t> ConnFailed;
        static std::atomic<size_t> ConnConnCanceled;
        static std::atomic<size_t> ConnSlow;
        static std::atomic<size_t> Conn2Success;
        static std::atomic<size_t> ConnPurgedInCache;
        static std::atomic<size_t> ConnDestroyedInCache;
        static std::atomic<size_t> RequestTotal;
        static std::atomic<size_t> RequestSuccessed;
        static std::atomic<size_t> RequestFailed;
        static void Print() {
            Cout << "ct=" << ConnTotal.load(std::memory_order_acquire)
                 << " ca=" << ConnActive.load(std::memory_order_acquire)
                 << " cch=" << ConnCached.load(std::memory_order_acquire)
                 << " cd=" << ConnDestroyed.load(std::memory_order_acquire)
                 << " cf=" << ConnFailed.load(std::memory_order_acquire)
                 << " ccc=" << ConnConnCanceled.load(std::memory_order_acquire)
                 << " csl=" << ConnSlow.load(std::memory_order_acquire)
                 << " c2s=" << Conn2Success.load(std::memory_order_acquire)
                 << " cpc=" << ConnPurgedInCache.load(std::memory_order_acquire)
                 << " cdc=" << ConnDestroyedInCache.load(std::memory_order_acquire)
                 << " rt=" << RequestTotal.load(std::memory_order_acquire)
                 << " rs=" << RequestSuccessed.load(std::memory_order_acquire)
                 << " rf=" << RequestFailed.load(std::memory_order_acquire)
                 << Endl;
        }
    };
    std::atomic<size_t> TDebugStat::ConnTotal = 0;
    std::atomic<size_t> TDebugStat::ConnActive = 0;
    std::atomic<size_t> TDebugStat::ConnCached = 0;
    std::atomic<size_t> TDebugStat::ConnDestroyed = 0;
    std::atomic<size_t> TDebugStat::ConnFailed = 0;
    std::atomic<size_t> TDebugStat::ConnConnCanceled = 0;
    std::atomic<size_t> TDebugStat::ConnSlow = 0;
    std::atomic<size_t> TDebugStat::Conn2Success = 0;
    std::atomic<size_t> TDebugStat::ConnPurgedInCache = 0;
    std::atomic<size_t> TDebugStat::ConnDestroyedInCache = 0;
    std::atomic<size_t> TDebugStat::RequestTotal = 0;
    std::atomic<size_t> TDebugStat::RequestSuccessed = 0;
    std::atomic<size_t> TDebugStat::RequestFailed = 0;
#endif

    inline void PrepareSocket(SOCKET s, const TRequestSettings& requestSettings = TRequestSettings()) {
        if (requestSettings.NoDelay) {
            SetNoDelay(s, true);
        }
    }

    bool Compress(TData& data, const TString& compressionScheme) {
        if (compressionScheme == "gzip" && data.size() > 23) {  // there is no string less than 24 bytes long that might be compressed with gzip
            try {
                TData gzipped(data.size());
                TMemoryOutput out(gzipped.data(), gzipped.size());
                TZLibCompress c(&out, ZLib::GZip);
                c.Write(data.data(), data.size());
                c.Finish();
                gzipped.resize(out.Buf() - gzipped.data());
                data.swap(gzipped);
                return true;
            } catch (yexception&) {
                // gzipped data occupies more space than original data
            }
        }
        return false;
    }

    class THttpRequestBuffers: public NAsio::TTcpSocket::IBuffers {
    public:
        THttpRequestBuffers(TRequestData::TPtr rd)
            : Req_(rd)
            , Parts_(Req_->Parts())
            , IOvec_(Parts_.data(), Parts_.size())
        {
        }

        TContIOVector* GetIOvec() override {
            return &IOvec_;
        }

    private:
        TRequestData::TPtr Req_;
        TVector<IOutputStream::TPart> Parts_;
        TContIOVector IOvec_;
    };

    struct TRequestGet1: public TRequestGet {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("http");
        }
    };

    struct TRequestPost1: public TRequestPost {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("post");
        }
    };

    struct TRequestFull1: public TRequestFull {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("full");
        }
    };

    struct TRequestGet2: public TRequestGet {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("http2");
        }
    };

    struct TRequestPost2: public TRequestPost {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("post2");
        }
    };

    struct TRequestFull2: public TRequestFull {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("full2");
        }
    };

    struct TRequestUnixSocketGet: public TRequestGet {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("http+unix");
        }

        static TRequestSettings RequestSettings() {
            return TRequestSettings()
                .SetNoDelay(false)
                .SetResolverType(EResolverType::EUNIXSOCKET);
        }
    };

    struct TRequestUnixSocketPost: public TRequestPost {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("post+unix");
        }

        static TRequestSettings RequestSettings() {
            return TRequestSettings()
                .SetNoDelay(false)
                .SetResolverType(EResolverType::EUNIXSOCKET);
        }
    };

    struct TRequestUnixSocketFull: public TRequestFull {
        static inline TStringBuf Name() noexcept {
            return TStringBuf("full+unix");
        }

        static TRequestSettings RequestSettings() {
            return TRequestSettings()
                .SetNoDelay(false)
                .SetResolverType(EResolverType::EUNIXSOCKET);
        }
    };

    typedef TAutoPtr<THttpRequestBuffers> THttpRequestBuffersPtr;

    class THttpRequest;
    typedef TSharedPtrB<THttpRequest> THttpRequestRef;

    class THttpConn;
    typedef TIntrusivePtr<THttpConn> THttpConnRef;

    typedef std::function<TRequestData::TPtr(const TMessage&, const TParsedLocation&)> TRequestBuilder;

    class THttpRequest {
    public:
        class THandle: public TSimpleHandle {
        public:
            THandle(IOnRecv* f, const TMessage& msg, TStatCollector* s) noexcept
                : TSimpleHandle(f, msg, s)
            {
            }

            bool MessageSendedCompletely() const noexcept override {
                if (TSimpleHandle::MessageSendedCompletely()) {
                    return true;
                }

                THttpRequestRef req(GetRequest());
                if (!!req && req->RequestSendedCompletely()) {
                    const_cast<THandle*>(this)->SetSendComplete();
                }

                return TSimpleHandle::MessageSendedCompletely();
            }

            void Cancel() noexcept override {
                if (TSimpleHandle::Canceled()) {
                    return;
                }

                THttpRequestRef req(GetRequest());
                if (!!req) {
                    TSimpleHandle::Cancel();
                    req->Cancel();
                }
            }

            void NotifyError(TErrorRef error, const THttpParser* rsp = nullptr) {
#ifdef DEBUG_STAT
                ++TDebugStat::RequestFailed;
#endif
                if (rsp) {
                    TSimpleHandle::NotifyError(error, rsp->DecodedContent(), rsp->FirstLine(), rsp->Headers());
                } else {
                    TSimpleHandle::NotifyError(error);
                }

                ReleaseRequest();
            }

            //not thread safe!
            void SetRequest(const TWeakPtrB<THttpRequest>& r) noexcept {
                Req_ = r;
            }

        private:
            THttpRequestRef GetRequest() const noexcept {
                TGuard<TSpinLock> g(SP_);
                return Req_;
            }

            void ReleaseRequest() noexcept {
                TWeakPtrB<THttpRequest> tmp;
                TGuard<TSpinLock> g(SP_);
                tmp.Swap(Req_);
            }

            mutable TSpinLock SP_;
            TWeakPtrB<THttpRequest> Req_;
        };

        typedef TIntrusivePtr<THandle> THandleRef;

        static void Run(THandleRef& h, const TMessage& msg, TRequestBuilder f, const TRequestSettings& s) {
            THttpRequestRef req(new THttpRequest(h, msg, f, s));
            req->WeakThis_ = req;
            h->SetRequest(req->WeakThis_);
            req->Run(req);
        }

        ~THttpRequest() {
            DBGOUT("~THttpRequest()");
        }

    private:
        THttpRequest(THandleRef& h, TMessage msg, TRequestBuilder f, const TRequestSettings& s)
            : Hndl_(h)
            , RequestBuilder_(f)
            , RequestSettings_(s)
            , Msg_(std::move(msg))
            , Loc_(Msg_.Addr)
            , Addr_(Resolve(Loc_.Host, Loc_.GetPort(), RequestSettings_.ResolverType))
            , AddrIter_(Addr_->Addr.Begin())
            , Canceled_(false)
            , RequestSendedCompletely_(false)
        {
        }

        void Run(THttpRequestRef& req);

    public:
        THttpRequestBuffersPtr BuildRequest() {
            return new THttpRequestBuffers(RequestBuilder_(Msg_, Loc_));
        }

        TRequestSettings RequestSettings() {
            return RequestSettings_;
        }

        //can create a spare socket in an attempt to decrease connecting time
        void OnDetectSlowConnecting();

        //remove extra connection on success connec
        void OnConnect(THttpConn* c);

        //have some response input
        void OnBeginRead() noexcept {
            RequestSendedCompletely_ = true;
        }

        void OnResponse(TAutoPtr<THttpParser>& rsp);

        void OnConnectFailed(THttpConn* c, const TErrorCode& ec);
        void OnSystemError(THttpConn* c, const TErrorCode& ec);
        void OnError(THttpConn* c, const TString& errorText);

        bool RequestSendedCompletely() noexcept;

        void Cancel() noexcept;

    private:
        void NotifyResponse(const TString& resp, const TString& firstLine, const THttpHeaders& headers) {
            THandleRef h(ReleaseHandler());
            if (!!h) {
                h->NotifyResponse(resp, firstLine, headers);
            }
        }

        void NotifyError(
            const TString& errorText,
            TError::TType errorType = TError::UnknownType,
            i32 errorCode = 0, i32 systemErrorCode = 0) {
            NotifyError(new TError(errorText, errorType, errorCode, systemErrorCode));
        }

        void NotifyError(TErrorRef error, const THttpParser* rsp = nullptr) {
            THandleRef h(ReleaseHandler());
            if (!!h) {
                h->NotifyError(error, rsp);
            }
        }

        void Finalize(THttpConn* skipConn = nullptr) noexcept;

        inline THandleRef ReleaseHandler() noexcept {
            THandleRef h;
            {
                TGuard<TSpinLock> g(SL_);
                h.Swap(Hndl_);
            }
            return h;
        }

        inline THttpConnRef GetConn() noexcept {
            TGuard<TSpinLock> g(SL_);
            return Conn_;
        }

        inline THttpConnRef ReleaseConn() noexcept {
            THttpConnRef c;
            {
                TGuard<TSpinLock> g(SL_);
                c.Swap(Conn_);
            }
            return c;
        }

        inline THttpConnRef ReleaseConn2() noexcept {
            THttpConnRef c;
            {
                TGuard<TSpinLock> g(SL_);
                c.Swap(Conn2_);
            }
            return c;
        }

        TSpinLock SL_; //guaranted calling notify() only once (prevent race between asio thread and current)
        THandleRef Hndl_;
        TRequestBuilder RequestBuilder_;
        TRequestSettings RequestSettings_;
        const TMessage Msg_;
        const TParsedLocation Loc_;
        const TResolvedHost* Addr_;
        TNetworkAddress::TIterator AddrIter_;
        THttpConnRef Conn_;
        THttpConnRef Conn2_; //concurrent connection used, if detected slow connecting on first connection
        TWeakPtrB<THttpRequest> WeakThis_;
        TAtomicBool Canceled_;
        TAtomicBool RequestSendedCompletely_;
    };

    TAtomicCounter* HttpOutConnCounter();

    class THttpConn: public TThrRefBase {
    public:
        static THttpConnRef Create(TIOService& srv);

        ~THttpConn() override {
            DBGOUT("~THttpConn()");
            Req_.Reset();
            HttpOutConnCounter()->Dec();
#ifdef DEBUG_STAT
            ++TDebugStat::ConnDestroyed;
#endif
        }

        void StartRequest(THttpRequestRef req, const TEndpoint& ep, size_t addrId, TDuration slowConn) {
            {
                //thread safe linking connection->request
                TGuard<TSpinLock> g(SL_);
                Req_ = req;
            }
            AddrId_ = addrId;
            try {
                TDuration connectDeadline(THttp2Options::ConnectTimeout);
                if (THttp2Options::ConnectTimeout > slowConn) {
                    //use append non fatal connect deadline, so on first timedout
                    //report about slow connecting to THttpRequest, and continue wait ConnectDeadline_ period
                    connectDeadline = slowConn;
                    ConnectDeadline_ = THttp2Options::ConnectTimeout - slowConn;
                }
                DBGOUT("AsyncConnect to " << ep.IpToString());
                AS_.AsyncConnect(ep, std::bind(&THttpConn::OnConnect, THttpConnRef(this), _1, _2), connectDeadline);
            } catch (...) {
                ReleaseRequest();
                throw;
            }
        }

        //start next request on keep-alive connection
        bool StartNextRequest(THttpRequestRef& req) {
            if (Finalized_) {
                return false;
            }

            {
                //thread safe linking connection->request
                TGuard<TSpinLock> g(SL_);
                Req_ = req;
            }

            RequestWritten_ = false;
            BeginReadResponse_ = false;

            try {
                TErrorCode ec;
                SendRequest(req->BuildRequest(), ec); //throw std::bad_alloc
                if (ec.Value() == ECANCELED) {
                    OnCancel();
                } else if (ec) {
                    OnError(ec);
                }
            } catch (...) {
                OnError(CurrentExceptionMessage());
                throw;
            }
            return true;
        }

        //connection received from cache must be validated before using
        //(process removing closed conection from cache consume some time)
        inline bool IsValid() const noexcept {
            return !Finalized_;
        }

        void SetCached(bool v) noexcept {
            Cached_ = v;
        }

        void Close() noexcept {
            try {
                Cancel();
            } catch (...) {
            }
        }

        void DetachRequest() noexcept {
            ReleaseRequest();
        }

        void Cancel() { //throw std::bad_alloc
            if (!Canceled_) {
                Canceled_ = true;
                Finalized_ = true;
                OnCancel();
                AS_.AsyncCancel();
            }
        }

        void OnCancel() {
            THttpRequestRef r(ReleaseRequest());
            if (!!r) {
                static const TString reqCanceled("request canceled");
                r->OnError(this, reqCanceled);
            }
        }

        bool RequestSendedCompletely() const noexcept {
            DBGOUT("RequestSendedCompletely()");
            if (!Connected_ || !RequestWritten_) {
                return false;
            }
            if (BeginReadResponse_) {
                return true;
            }
#if defined(FIONWRITE)
            if (THttp2Options::EnsureSendingCompleteByAck) {
                int nbytes = Max<int>();
                int err = ioctl(AS_.Native(), FIONWRITE, &nbytes);
                return err ? false : nbytes == 0;
            }
#endif
            return true;
        }

        TIOService& GetIOService() const noexcept {
            return AS_.GetIOService();
        }

    private:
        THttpConn(TIOService& srv)
            : AddrId_(0)
            , AS_(srv)
            , BuffSize_(THttp2Options::InputBufferSize)
            , Connected_(false)
            , Cached_(false)
            , Canceled_(false)
            , Finalized_(false)
            , InAsyncRead_(false)
            , RequestWritten_(false)
            , BeginReadResponse_(false)
        {
            HttpOutConnCounter()->Inc();
        }

        //can be called only from asio
        void OnConnect(const TErrorCode& ec, IHandlingContext& ctx) {
            DBGOUT("THttpConn::OnConnect: " << ec.Value());
            if (Y_UNLIKELY(ec)) {
                if (ec.Value() == ETIMEDOUT && ConnectDeadline_.GetValue()) {
                    //detect slow connecting (yet not reached final timeout)
                    DBGOUT("OnConnectTimingCheck");
                    THttpRequestRef req(GetRequest());
                    if (!req) {
                        return; //cancel from client thread can ahead us
                    }
                    TDuration newDeadline(ConnectDeadline_);
                    ConnectDeadline_ = TDuration::Zero(); //next timeout is final

                    req->OnDetectSlowConnecting();
                    //continue wait connect
                    ctx.ContinueUseHandler(newDeadline);

                    return;
                }
#ifdef DEBUG_STAT
                if (ec.Value() != ECANCELED) {
                    ++TDebugStat::ConnFailed;
                } else {
                    ++TDebugStat::ConnConnCanceled;
                }
#endif
                if (ec.Value() == EIO) {
                    //try get more detail error info
                    char buf[1];
                    TErrorCode errConnect;
                    AS_.ReadSome(buf, 1, errConnect);
                    OnConnectFailed(errConnect.Value() ? errConnect : ec);
                } else if (ec.Value() == ECANCELED) {
                    // not try connecting to next host ip addr, simple fail
                    OnError(ec);
                } else {
                    OnConnectFailed(ec);
                }
            } else {
                Connected_ = true;

                THttpRequestRef req(GetRequest());
                if (!req || Canceled_) {
                    return;
                }

                try {
                    PrepareSocket(AS_.Native(), req->RequestSettings());
                    if (THttp2Options::TcpKeepAlive) {
                        SetKeepAlive(AS_.Native(), true);
                    }
                } catch (TSystemError& err) {
                    TErrorCode ec2(err.Status());
                    OnError(ec2);
                    return;
                }

                req->OnConnect(this);

                THttpRequestBuffersPtr ptr(req->BuildRequest());
                PrepareParser();

                TErrorCode ec3;
                SendRequest(ptr, ec3);
                if (ec3) {
                    OnError(ec3);
                }
            }
        }

        void PrepareParser() {
            Prs_ = new THttpParser();
            Prs_->Prepare();
        }

        void SendRequest(const THttpRequestBuffersPtr& bfs, TErrorCode& ec) { //throw std::bad_alloc
            if (!THttp2Options::UseAsyncSendRequest) {
                size_t amount = AS_.WriteSome(*bfs->GetIOvec(), ec);

                if (ec && ec.Value() != EAGAIN && ec.Value() != EWOULDBLOCK && ec.Value() != EINPROGRESS) {
                    return;
                }
                ec.Assign(0);

                bfs->GetIOvec()->Proceed(amount);

                if (bfs->GetIOvec()->Complete()) {
                    RequestWritten_ = true;
                    StartRead();
                } else {
                    SendRequestAsync(bfs);
                }
            } else {
                SendRequestAsync(bfs);
            }
        }

        void SendRequestAsync(const THttpRequestBuffersPtr& bfs) {
            NAsio::TTcpSocket::TSendedData sd(bfs.Release());
            AS_.AsyncWrite(sd, std::bind(&THttpConn::OnWrite, THttpConnRef(this), _1, _2, _3), THttp2Options::OutputDeadline);
        }

        void OnWrite(const TErrorCode& err, size_t amount, IHandlingContext& ctx) {
            Y_UNUSED(amount);
            Y_UNUSED(ctx);
            if (err) {
                OnError(err);
            } else {
                DBGOUT("OnWrite()");
                RequestWritten_ = true;
                StartRead();
            }
        }

        inline void StartRead() {
            if (!InAsyncRead_ && !Canceled_) {
                InAsyncRead_ = true;
                AS_.AsyncPollRead(std::bind(&THttpConn::OnCanRead, THttpConnRef(this), _1, _2), THttp2Options::InputDeadline);
            }
        }

        //can be called only from asio
        void OnReadSome(const TErrorCode& err, size_t bytes, IHandlingContext& ctx) {
            if (Y_UNLIKELY(err)) {
                OnError(err);
                return;
            }
            if (!BeginReadResponse_) {
                //used in MessageSendedCompletely()
                BeginReadResponse_ = true;
                THttpRequestRef r(GetRequest());
                if (!!r) {
                    r->OnBeginRead();
                }
            }
            DBGOUT("receive:" << TStringBuf(Buff_.Get(), bytes));
            try {
                if (!Prs_) {
                    throw yexception() << TStringBuf("receive some data while not in request");
                }

#if defined(_linux_)
                if (THttp2Options::QuickAck) {
                    SetSockOpt(AS_.Native(), SOL_TCP, TCP_QUICKACK, (int)1);
                }
#endif

                DBGOUT("parse:");
                while (!Prs_->Parse(Buff_.Get(), bytes)) {
                    if (BuffSize_ == bytes) {
                        TErrorCode ec;
                        bytes = AS_.ReadSome(Buff_.Get(), BuffSize_, ec);

                        if (!ec) {
                            continue;
                        }

                        if (ec.Value() != EAGAIN && ec.Value() != EWOULDBLOCK) {
                            OnError(ec);

                            return;
                        }
                    }
                    //continue async. read from socket
                    ctx.ContinueUseHandler(THttp2Options::InputDeadline);

                    return;
                }

                //succesfully reach end of http response
                THttpRequestRef r(ReleaseRequest());
                if (!r) {
                    //lost race to req. canceling
                    DBGOUT("connection failed");
                    return;
                }

                DBGOUT("response:");
                bool keepALive = Prs_->IsKeepAlive();

                r->OnResponse(Prs_);

                if (!keepALive) {
                    return;
                }

                //continue use connection (keep-alive mode)
                PrepareParser();

                if (!THttp2Options::KeepInputBufferForCachedConnections) {
                    Buff_.Destroy();
                }
                //continue async. read from socket
                ctx.ContinueUseHandler(THttp2Options::InputDeadline);

                PutSelfToCache();
            } catch (...) {
                OnError(CurrentExceptionMessage());
            }
        }

        void PutSelfToCache();

        //method for reaction on input data for re-used keep-alive connection,
        //which free/release buffer when was placed in cache
        void OnCanRead(const TErrorCode& err, IHandlingContext& ctx) {
            if (Y_UNLIKELY(err)) {
                OnError(err);
            } else {
                if (!Buff_) {
                    Buff_.Reset(new char[BuffSize_]);
                }
                TErrorCode ec;
                OnReadSome(ec, AS_.ReadSome(Buff_.Get(), BuffSize_, ec), ctx);
            }
        }

        //unlink connection and request, thread-safe mark connection as non valid
        inline THttpRequestRef GetRequest() noexcept {
            TGuard<TSpinLock> g(SL_);
            return Req_;
        }

        inline THttpRequestRef ReleaseRequest() noexcept {
            THttpRequestRef r;
            {
                TGuard<TSpinLock> g(SL_);
                r.Swap(Req_);
            }
            return r;
        }

        void OnConnectFailed(const TErrorCode& ec);

        inline void OnError(const TErrorCode& ec) {
            OnError(ec.Text());
        }

        inline void OnError(const TString& errText);

        size_t AddrId_;
        NAsio::TTcpSocket AS_;
        TArrayHolder<char> Buff_; //input buffer
        const size_t BuffSize_;

        TAutoPtr<THttpParser> Prs_; //input data parser & parsed info storage

        TSpinLock SL_;
        THttpRequestRef Req_; //current request
        TDuration ConnectDeadline_;
        TAtomicBool Connected_;
        TAtomicBool Cached_;
        TAtomicBool Canceled_;
        TAtomicBool Finalized_;

        bool InAsyncRead_;
        TAtomicBool RequestWritten_;
        TAtomicBool BeginReadResponse_;
    };

    //conn limits monitoring, cache clean, contain used in http clients asio threads/executors
    class THttpConnManager: public IThreadFactory::IThreadAble {
    public:
        THttpConnManager()
            : TotalConn(0)
            , EP_(THttp2Options::AsioThreads)
            , InPurging_(0)
            , MaxConnId_(0)
            , Shutdown_(false)
        {
            T_ = SystemThreadFactory()->Run(this);
            Limits.SetSoft(40000);
            Limits.SetHard(50000);
        }

        ~THttpConnManager() override {
            {
                TGuard<TMutex> g(PurgeMutex_);

                Shutdown_ = true;
                CondPurge_.Signal();
            }

            EP_.SyncShutdown();

            T_->Join();
        }

        inline void SetLimits(size_t softLimit, size_t hardLimit) noexcept {
            Limits.SetSoft(softLimit);
            Limits.SetHard(hardLimit);
        }

        inline std::pair<size_t, size_t> GetLimits() const noexcept {
            return {Limits.Soft(), Limits.Hard()};
        }

        inline void CheckLimits() {
            if (ExceedSoftLimit()) {
                SuggestPurgeCache();

                if (ExceedHardLimit()) {
                    Y_ABORT("neh::http2 output connections limit reached");
                    //ythrow yexception() << "neh::http2 output connections limit reached";
                }
            }
        }

        inline bool Get(THttpConnRef& conn, size_t addrId) {
#ifdef DEBUG_STAT
            TDebugStat::ConnTotal.store(TotalConn.Val(), std::memory_order_release);
            TDebugStat::ConnActive.store(Active(), std::memory_order_release);
            TDebugStat::ConnCached.store(Cache_.Size(), std::memory_order_release);
#endif
            return Cache_.Get(conn, addrId);
        }

        inline void Put(THttpConnRef& conn, size_t addrId) {
            if (Y_LIKELY(!Shutdown_ && !ExceedHardLimit() && !CacheDisabled())) {
                if (Y_UNLIKELY(addrId > (size_t)AtomicGet(MaxConnId_))) {
                    AtomicSet(MaxConnId_, addrId);
                }
                Cache_.Put(conn, addrId);
            } else {
                conn->Close();
                conn.Drop();
            }
        }

        inline size_t OnConnError(size_t addrId) {
            return Cache_.Validate(addrId);
        }

        TIOService& GetIOService() {
            return EP_.GetExecutor().GetIOService();
        }

        bool CacheDisabled() const {
            return Limits.Soft() == 0;
        }

        bool IsShutdown() const noexcept {
            return Shutdown_;
        }

        TAtomicCounter TotalConn;

    private:
        inline size_t Total() const noexcept {
            return TotalConn.Val();
        }

        inline size_t Active() const noexcept {
            return TFdLimits::ExceedLimit(Total(), Cache_.Size());
        }

        inline size_t ExceedSoftLimit() const noexcept {
            return TFdLimits::ExceedLimit(Total(), Limits.Soft());
        }

        inline size_t ExceedHardLimit() const noexcept {
            return TFdLimits::ExceedLimit(Total(), Limits.Hard());
        }

        void SuggestPurgeCache() {
            if (AtomicTryLock(&InPurging_)) {
                //evaluate the usefulness of purging the cache
                //если в кеше мало соединений (< MaxConnId_/16 или 64), не чистим кеш
                if (Cache_.Size() > (Min((size_t)AtomicGet(MaxConnId_), (size_t)1024U) >> 4)) {
                    //по мере приближения к hardlimit нужда в чистке cache приближается к 100%
                    size_t closenessToHardLimit256 = ((Active() + 1) << 8) / (Limits.Delta() + 1);
                    //чем больше соединений в кеше, а не в работе, тем менее нужен кеш (можно его почистить)
                    size_t cacheUselessness256 = ((Cache_.Size() + 1) << 8) / (Active() + 1);

                    //итого, - пороги срабатывания:
                    //при достижении soft-limit, если соединения в кеше, а не в работе
                    //на полпути от soft-limit к hard-limit, если в кеше больше половины соединений
                    //при приближении к hardlimit пытаться почистить кеш почти постоянно
                    if ((closenessToHardLimit256 + cacheUselessness256) >= 256U) {
                        TGuard<TMutex> g(PurgeMutex_);

                        CondPurge_.Signal();
                        return; //memo: thread MUST unlock InPurging_ (see DoExecute())
                    }
                }
                AtomicUnlock(&InPurging_);
            }
        }

        void DoExecute() override {
            TThread::SetCurrentThreadName("NehHttpConnMngr");
            while (true) {
                {
                    TGuard<TMutex> g(PurgeMutex_);

                    if (Shutdown_)
                        return;

                    CondPurge_.WaitI(PurgeMutex_);
                }

                PurgeCache();

                AtomicUnlock(&InPurging_);
            }
        }

        void PurgeCache() noexcept {
            //try remove at least ExceedSoftLimit() oldest connections from cache
            //вычисляем долю кеша, которую нужно почистить (в 256 долях) (но не менее 1/32 кеша)
            size_t frac256 = Min(size_t(Max(size_t(256U / 32U), (ExceedSoftLimit() << 8) / (Cache_.Size() + 1))), (size_t)256U);

            size_t processed = 0;
            size_t maxConnId = AtomicGet(MaxConnId_);
            for (size_t i = 0; i <= maxConnId && !Shutdown_; ++i) {
                processed += Cache_.Purge(i, frac256);
                if (processed > 32) {
#ifdef DEBUG_STAT
                    TDebugStat::ConnPurgedInCache += processed;
#endif
                    processed = 0;
                    Sleep(TDuration::MilliSeconds(10)); //prevent big spike cpu/system usage
                }
            }
        }

        TFdLimits Limits;
        TExecutorsPool EP_;

        TConnCache<THttpConn> Cache_;
        TAtomic InPurging_;
        TAtomic MaxConnId_;

        TAutoPtr<IThreadFactory::IThread> T_;
        TCondVar CondPurge_;
        TMutex PurgeMutex_;
        TAtomicBool Shutdown_;
    };

    THttpConnManager* HttpConnManager() {
        return Singleton<THttpConnManager>();
    }

    TAtomicCounter* HttpOutConnCounter() {
        return &HttpConnManager()->TotalConn;
    }

    THttpConnRef THttpConn::Create(TIOService& srv) {
        if (HttpConnManager()->IsShutdown()) {
            throw yexception() << "can't create connection with shutdowned service";
        }

        return new THttpConn(srv);
    }

    void THttpConn::PutSelfToCache() {
        THttpConnRef c(this);
        HttpConnManager()->Put(c, AddrId_);
    }

    void THttpConn::OnConnectFailed(const TErrorCode& ec) {
        THttpRequestRef r(GetRequest());
        if (!!r) {
            r->OnConnectFailed(this, ec);
        }
        OnError(ec);
    }

    void THttpConn::OnError(const TString& errText) {
        Finalized_ = true;
        if (Connected_) {
            Connected_ = false;
            TErrorCode ec;
            AS_.Shutdown(NAsio::TTcpSocket::ShutdownBoth, ec);
        } else {
            if (AS_.IsOpen()) {
                AS_.AsyncCancel();
            }
        }
        THttpRequestRef r(ReleaseRequest());
        if (!!r) {
            r->OnError(this, errText);
        } else {
            if (Cached_) {
                size_t res = HttpConnManager()->OnConnError(AddrId_);
                Y_UNUSED(res);
#ifdef DEBUG_STAT
                TDebugStat::ConnDestroyedInCache += res;
#endif
            }
        }
    }

    void THttpRequest::Run(THttpRequestRef& req) {
#ifdef DEBUG_STAT
        if ((++TDebugStat::RequestTotal & 0xFFF) == 0) {
            TDebugStat::Print();
        }
#endif
        try {
            while (!Canceled_) {
                THttpConnRef conn;
                if (HttpConnManager()->Get(conn, Addr_->Id)) {
                    DBGOUT("Use connection from cache");
                    Conn_ = conn; //thread magic
                    if (!conn->StartNextRequest(req)) {
                        continue; //if use connection from cache, ignore write error and try another conn
                    }
                } else {
                    HttpConnManager()->CheckLimits(); //here throw exception if reach hard limit (or atexit() state)
                    Conn_ = THttpConn::Create(HttpConnManager()->GetIOService());
                    TEndpoint ep(new NAddr::TAddrInfo(&*AddrIter_));
                    Conn_->StartRequest(req, ep, Addr_->Id, THttp2Options::SymptomSlowConnect); // can throw
                }
                break;
            }
        } catch (...) {
            Conn_.Reset();
            throw;
        }
    }

    //it seems we have lost TCP SYN packet, create extra connection for decrease response time
    void THttpRequest::OnDetectSlowConnecting() {
#ifdef DEBUG_STAT
        ++TDebugStat::ConnSlow;
#endif
        //use some io_service (Run() thread-executor), from first conn. for more thread safety
        THttpConnRef conn = GetConn();

        if (!conn) {
            return;
        }

        THttpConnRef conn2;
        try {
            conn2 = THttpConn::Create(conn->GetIOService());
        } catch (...) {
            return; // cant create spare connection, simple continue use only main
        }

        {
            TGuard<TSpinLock> g(SL_);
            Conn2_ = conn2;
        }

        if (Y_UNLIKELY(Canceled_)) {
            ReleaseConn2();
        } else {
            //use connect timeout for disable detecting slow connecting on second conn.
            TEndpoint ep(new NAddr::TAddrInfo(&*Addr_->Addr.Begin()));
            try {
                conn2->StartRequest(WeakThis_, ep, Addr_->Id, THttp2Options::ConnectTimeout);
            } catch (...) {
                // ignore errors on spare connection
                ReleaseConn2();
            }
        }
    }

    void THttpRequest::OnConnect(THttpConn* c) {
        THttpConnRef extraConn;
        {
            TGuard<TSpinLock> g(SL_);
            if (Y_UNLIKELY(!!Conn2_)) {
                //has pair concurrent conn, 'should stay only one'
                if (Conn2_.Get() == c) {
#ifdef DEBUG_STAT
                    ++TDebugStat::Conn2Success;
#endif
                    Conn2_.Swap(Conn_);
                }
                extraConn.Swap(Conn2_);
            }
        }
        if (!!extraConn) {
            extraConn->DetachRequest(); //prevent call OnError()
            extraConn->Close();
        }
    }

    void THttpRequest::OnResponse(TAutoPtr<THttpParser>& rsp) {
        DBGOUT("THttpRequest::OnResponse()");
        ReleaseConn();
        if (Y_LIKELY(((rsp->RetCode() >= 200u && rsp->RetCode() < (!THttp2Options::RedirectionNotError ? 300u : 400u)) || THttp2Options::AnyResponseIsNotError))) {
            NotifyResponse(rsp->DecodedContent(), rsp->FirstLine(), rsp->Headers());
        } else {
            TString message;

            if (THttp2Options::FullHeadersAsErrorMessage) {
                TStringStream err;
                err << rsp->FirstLine();

                THttpHeaders hdrs = rsp->Headers();
                for (auto h = hdrs.begin(); h < hdrs.end(); h++) {
                    err << h->ToString() << TStringBuf("\r\n");
                }

                message = err.Str();
            } else if (THttp2Options::UseResponseAsErrorMessage) {
                message = rsp->DecodedContent();
            } else {
                TStringStream err;
                err << TStringBuf("request failed(") << rsp->FirstLine() << TStringBuf(")");
                message = err.Str();
            }

            NotifyError(new TError(message, TError::ProtocolSpecific, rsp->RetCode()), rsp.Get());
        }
    }

    void THttpRequest::OnConnectFailed(THttpConn* c, const TErrorCode& ec) {
        DBGOUT("THttpRequest::OnConnectFailed()");
        //detach/discard failed conn, try connect to next ip addr (if can)
        THttpConnRef cc(GetConn());
        if (c != cc.Get() || AddrIter_ == Addr_->Addr.End() || ++AddrIter_ == Addr_->Addr.End() || Canceled_) {
            return OnSystemError(c, ec);
        }
        // can try next host addr
        c->DetachRequest();
        c->Close();
        THttpConnRef nextConn;
        try {
            nextConn = THttpConn::Create(HttpConnManager()->GetIOService());
        } catch (...) {
            OnSystemError(nullptr, ec);
            return;
        }
        {
            THttpConnRef nc = nextConn;
            TGuard<TSpinLock> g(SL_);
            Conn_.Swap(nc);
        }
        TEndpoint ep(new NAddr::TAddrInfo(&*AddrIter_));
        try {
            nextConn->StartRequest(WeakThis_, ep, Addr_->Id, THttp2Options::SymptomSlowConnect);
        } catch (...) {
            OnError(nullptr, CurrentExceptionMessage());
            return;
        }

        if (Canceled_) {
            OnError(nullptr, "canceled");
        }
    }

    void THttpRequest::OnSystemError(THttpConn* c, const TErrorCode& ec) {
        DBGOUT("THttpRequest::OnSystemError()");
        NotifyError(ec.Text(), TError::TType::UnknownType, 0, ec.Value());
        Finalize(c);
    }

    void THttpRequest::OnError(THttpConn* c, const TString& errorText) {
        DBGOUT("THttpRequest::OnError()");
        NotifyError(errorText);
        Finalize(c);
    }

    bool THttpRequest::RequestSendedCompletely() noexcept {
        if (RequestSendedCompletely_) {
            return true;
        }

        THttpConnRef c(GetConn());
        return !!c ? c->RequestSendedCompletely() : false;
    }

    void THttpRequest::Cancel() noexcept {
        if (!Canceled_) {
            Canceled_ = true;
            try {
                static const TString canceled("Canceled");
                NotifyError(canceled, TError::Cancelled);
                Finalize();
            } catch (...) {
            }
        }
    }

    inline void FinalizeConn(THttpConnRef& c, THttpConn* skipConn) noexcept {
        if (!!c && c.Get() != skipConn) {
            c->DetachRequest();
            c->Close();
        }
    }

    void THttpRequest::Finalize(THttpConn* skipConn) noexcept {
        THttpConnRef c1(ReleaseConn());
        FinalizeConn(c1, skipConn);
        THttpConnRef c2(ReleaseConn2());
        FinalizeConn(c2, skipConn);
    }

    /////////////////////////////////// server side ////////////////////////////////////

    TAtomicCounter* HttpInConnCounter() {
        return Singleton<TAtomicCounter>();
    }

    TFdLimits* HttpInConnLimits() {
        return Singleton<TFdLimits>();
    }

    class THttpServer: public IRequester {
        typedef TAutoPtr<TTcpAcceptor> TTcpAcceptorPtr;
        typedef TAtomicSharedPtr<TTcpSocket> TTcpSocketRef;
        class TConn;
        typedef TSharedPtrB<TConn> TConnRef;

        class TRequest: public IHttpRequest {
        public:
            TRequest(TWeakPtrB<TConn>& c, TAutoPtr<THttpParser>& p)
                : C_(c)
                , P_(p)
                , RemoteHost_(C_->RemoteHost())
                , CompressionScheme_(P_->GetBestCompressionScheme())
                , H_(TStringBuf(P_->FirstLine()))
            {
            }

            ~TRequest() override {
                if (!!C_) {
                    try {
                        C_->SendError(Id(), 503, "service unavailable (request ignored)", P_->HttpVersion(), {});
                    } catch (...) {
                        DBGOUT("~TRequest()::SendFail() exception");
                    }
                }
            }

            TAtomicBase Id() const {
                return Id_;
            }

        protected:
            TStringBuf Scheme() const override {
                return TStringBuf("http");
            }

            TString RemoteHost() const override {
                return RemoteHost_;
            }

            TStringBuf Service() const override {
                return TStringBuf(H_.Path).Skip(1);
            }

            const THttpHeaders& Headers() const override {
                return P_->Headers();
            }

            TStringBuf Method() const override {
                return H_.Method;
            }

            TStringBuf Body() const override {
                return P_->DecodedContent();
            }

            TStringBuf Cgi() const override {
                return H_.Cgi;
            }

            TStringBuf RequestId() const override {
                return TStringBuf();
            }

            bool Canceled() const override {
                if (!C_) {
                    return false;
                }
                return C_->IsCanceled();
            }

            void SendReply(TData& data) override {
                SendReply(data, TString(), HttpCodes::HTTP_OK);
            }

            void SendReply(TData& data, const TString& headers, int httpCode) override {
                if (!!C_) {
                    C_->Send(Id(), data, CompressionScheme_, P_->HttpVersion(), headers, httpCode);
                    C_.Reset();
                }
            }

            void SendError(TResponseError err, const THttpErrorDetails& details) override {
                static const unsigned errorToHttpCode[IRequest::MaxResponseError] =
                    {
                        400,
                        403,
                        404,
                        429,
                        500,
                        501,
                        502,
                        503,
                        509};

                if (!!C_) {
                    C_->SendError(Id(), errorToHttpCode[err], details.Details, P_->HttpVersion(), details.Headers);
                    C_.Reset();
                }
            }

            static TAtomicBase NextId() {
                static TAtomic idGenerator = 0;
                TAtomicBase id = 0;
                do {
                    id = AtomicIncrement(idGenerator);
                } while (!id);
                return id;
            }

            TSharedPtrB<TConn> C_;
            TAutoPtr<THttpParser> P_;
            TString RemoteHost_;
            TString CompressionScheme_;
            TParsedHttpFull H_;
            TAtomicBase Id_ = NextId();
        };

        class TRequestGet: public TRequest {
        public:
            TRequestGet(TWeakPtrB<TConn>& c, TAutoPtr<THttpParser> p)
                : TRequest(c, p)
            {
            }

            TStringBuf Data() const override {
                return H_.Cgi;
            }
        };

        class TRequestPost: public TRequest {
        public:
            TRequestPost(TWeakPtrB<TConn>& c, TAutoPtr<THttpParser> p)
                : TRequest(c, p)
            {
            }

            TStringBuf Data() const override {
                return P_->DecodedContent();
            }
        };

        class TConn {
        private:
            TConn(THttpServer& hs, const TTcpSocketRef& s)
                : HS_(hs)
                , AS_(s)
                , RemoteHost_(NNeh::PrintHostByRfc(*AS_->RemoteEndpoint().Addr()))
                , BuffSize_(THttp2Options::InputBufferSize)
                , Buff_(new char[BuffSize_])
                , Canceled_(false)
                , LeftRequestsToDisconnect_(hs.LimitRequestsPerConnection)
            {
                DBGOUT("THttpServer::TConn()");
                HS_.OnCreateConn();
            }

            inline TConnRef SelfRef() noexcept {
                return WeakThis_;
            }

        public:
            static void Create(THttpServer& hs, const TTcpSocketRef& s) {
                TSharedPtrB<TConn> conn(new TConn(hs, s));
                conn->WeakThis_ = conn;
                conn->ExpectNewRequest();
                conn->AS_->AsyncPollRead(std::bind(&TConn::OnCanRead, conn, _1, _2), THttp2Options::ServerInputDeadline);
            }

            ~TConn() {
                DBGOUT("~THttpServer::TConn(" << (!AS_ ? -666 : AS_->Native()) << ")");
                HS_.OnDestroyConn();
            }

        private:
            void ExpectNewRequest() {
                P_.Reset(new THttpParser(THttpParser::Request));
                P_->Prepare();
            }

            void OnCanRead(const TErrorCode& ec, IHandlingContext& ctx) {
                if (ec) {
                    OnError();
                } else {
                    TErrorCode ec2;
                    OnReadSome(ec2, AS_->ReadSome(Buff_.Get(), BuffSize_, ec2), ctx);
                }
            }

            void OnError() {
                DBGOUT("Srv OnError(" << (!AS_ ? -666 : AS_->Native()) << ")");
                Canceled_ = true;
                AS_->AsyncCancel();
            }

            void OnReadSome(const TErrorCode& ec, size_t amount, IHandlingContext& ctx) {
                if (ec || !amount) {
                    OnError();

                    return;
                }

                DBGOUT("ReadSome(" << (!AS_ ? -666 : AS_->Native()) << "): " << amount);
                try {
                    size_t buffPos = 0;
                    //DBGOUT("receive and parse: " << TStringBuf(Buff_.Get(), amount));
                    while (P_->Parse(Buff_.Get() + buffPos, amount - buffPos)) {
                        if (!P_->IsKeepAlive() || LeftRequestsToDisconnect_ == 1) {
                            SeenMessageWithoutKeepalive_ = true;
                        }

                        char rt = *P_->FirstLine().data();
                        const size_t extraDataSize = P_->GetExtraDataSize();
                        if (rt == 'P' || rt == 'p') {
                            OnRequest(new TRequestPost(WeakThis_, P_));
                        } else {
                            OnRequest(new TRequestGet(WeakThis_, P_));
                        }
                        if (extraDataSize) {
                            // has http pipelining
                            buffPos = amount - extraDataSize;
                            ExpectNewRequest();
                        } else {
                            ExpectNewRequest();
                            ctx.ContinueUseHandler(HS_.GetKeepAliveTimeout());
                            return;
                        }
                    }
                    ctx.ContinueUseHandler(THttp2Options::ServerInputDeadline);
                } catch (...) {
                    OnError();
                }
            }

            void OnRequest(TRequest* r) {
                DBGOUT("OnRequest()");
                if (AtomicGet(PrimaryResponse_)) {
                    // has pipelining
                    PipelineOrder_.Enqueue(r->Id());
                } else {
                    AtomicSet(PrimaryResponse_, r->Id());
                }
                HS_.OnRequest(r);
                OnRequestDone();
            }

            void OnRequestDone() {
                DBGOUT("OnRequestDone()");
                if (LeftRequestsToDisconnect_ > 0) {
                    --LeftRequestsToDisconnect_;
                }
            }

            static void PrintHttpVersion(IOutputStream& out, const THttpVersion& ver) {
                out << TStringBuf("HTTP/") << ver.Major << TStringBuf(".") << ver.Minor;
            }

            struct TResponseData : TThrRefBase {
                TResponseData(size_t reqId, TTcpSocket::TSendedData data)
                    : RequestId_(reqId)
                    , Data_(data)
                {
                }

                size_t RequestId_;
                TTcpSocket::TSendedData Data_;
            };
            typedef TIntrusivePtr<TResponseData> TResponseDataRef;

        public:
            //called non thread-safe (from outside thread)
            void Send(TAtomicBase requestId, TData& data, const TString& compressionScheme, const THttpVersion& ver, const TString& headers, int httpCode) {
                class THttpResponseFormatter {
                public:
                    THttpResponseFormatter(TData& theData, const TString& contentEncoding, const THttpVersion& theVer, const TString& theHeaders, int theHttpCode, bool closeConnection) {
                        Header.Reserve(128 + contentEncoding.size() + theHeaders.size());
                        PrintHttpVersion(Header, theVer);
                        Header << TStringBuf(" ") << theHttpCode << ' ' << HttpCodeStr(theHttpCode);
                        if (Compress(theData, contentEncoding)) {
                            Header << TStringBuf("\r\nContent-Encoding: ") << contentEncoding;
                        }
                        Header << TStringBuf("\r\nContent-Length: ") << theData.size();
                        if (closeConnection) {
                            Header << TStringBuf("\r\nConnection: close");
                        } else if (Y_LIKELY(theVer.Major > 1 || theVer.Minor > 0)) {
                            // since HTTP/1.1 Keep-Alive is default behaviour
                            Header << TStringBuf("\r\nConnection: Keep-Alive");
                        }
                        if (theHeaders) {
                            Header << theHeaders;
                        }
                        Header << TStringBuf("\r\n\r\n");

                        Body.swap(theData);

                        Parts[0].buf = Header.Data();
                        Parts[0].len = Header.Size();
                        Parts[1].buf = Body.data();
                        Parts[1].len = Body.size();
                    }

                    TStringStream Header;
                    TData Body;
                    IOutputStream::TPart Parts[2];
                };

                class TBuffers: public THttpResponseFormatter, public TTcpSocket::IBuffers {
                public:
                    TBuffers(TData& theData, const TString& contentEncoding, const THttpVersion& theVer, const TString& theHeaders, int theHttpCode, bool closeConnection)
                        : THttpResponseFormatter(theData, contentEncoding, theVer, theHeaders, theHttpCode, closeConnection)
                        , IOVec(Parts, 2)
                    {
                    }

                    TContIOVector* GetIOvec() override {
                        return &IOVec;
                    }

                    TContIOVector IOVec;
                };

                TTcpSocket::TSendedData sd(new TBuffers(data, compressionScheme, ver, headers, httpCode, SeenMessageWithoutKeepalive_));
                SendData(requestId, sd);
            }

            //called non thread-safe (from outside thread)
            void SendError(TAtomicBase requestId, unsigned httpCode, const TString& descr, const THttpVersion& ver, const TString& headers) {
                if (Canceled_) {
                    return;
                }

                class THttpErrorResponseFormatter {
                public:
                    THttpErrorResponseFormatter(unsigned theHttpCode, const TString& theDescr, const THttpVersion& theVer, bool closeConnection, const TString& headers) {
                        PrintHttpVersion(Answer, theVer);
                        Answer << TStringBuf(" ") << theHttpCode << TStringBuf(" ");
                        if (theDescr.size() && !THttp2Options::ErrorDetailsAsResponseBody) {
                            // Reason-Phrase  = *<TEXT, excluding CR, LF>
                            // replace bad chars to '.'
                            TString reasonPhrase = theDescr;
                            for (TString::iterator it = reasonPhrase.begin(); it != reasonPhrase.end(); ++it) {
                                char& ch = *it;
                                if (ch == ' ') {
                                    continue;
                                }
                                if (((ch & 31) == ch) || static_cast<unsigned>(ch) == 127 || (static_cast<unsigned>(ch) & 0x80)) {
                                    //CTLs || DEL(127) || non ascii
                                    // (ch <= 32) || (ch >= 127)
                                    ch = '.';
                                }
                            }
                            Answer << reasonPhrase;
                        } else {
                            Answer << HttpCodeStr(static_cast<int>(theHttpCode));
                        }

                        if (closeConnection) {
                            Answer << TStringBuf("\r\nConnection: close");
                        }

                        if (headers) {
                            Answer << "\r\n" << headers;
                        }

                        if (THttp2Options::ErrorDetailsAsResponseBody) {
                            Answer << TStringBuf("\r\nContent-Length:") << theDescr.size() << "\r\n\r\n" << theDescr;
                        } else {
                            Answer << "\r\n"
                                      "Content-Length:0\r\n\r\n"sv;
                        }

                        Parts[0].buf = Answer.Data();
                        Parts[0].len = Answer.Size();
                    }

                    TStringStream Answer;
                    IOutputStream::TPart Parts[1];
                };

                class TBuffers: public THttpErrorResponseFormatter, public TTcpSocket::IBuffers {
                public:
                    TBuffers(
                        unsigned theHttpCode,
                        const TString& theDescr,
                        const THttpVersion& theVer,
                        bool closeConnection,
                        const TString& headers
                    )
                        : THttpErrorResponseFormatter(theHttpCode, theDescr, theVer, closeConnection, headers)
                        , IOVec(Parts, 1)
                    {
                    }

                    TContIOVector* GetIOvec() override {
                        return &IOVec;
                    }

                    TContIOVector IOVec;
                };

                TTcpSocket::TSendedData sd(new TBuffers(httpCode, descr, ver, SeenMessageWithoutKeepalive_, headers));
                SendData(requestId, sd);
            }

            void ProcessPipeline() {
                // on successfull response to current (PrimaryResponse_) request
                TAtomicBase requestId;
                if (PipelineOrder_.Dequeue(&requestId)) {
                    TAtomicBase oldReqId;
                    do {
                        oldReqId = AtomicGet(PrimaryResponse_);
                        Y_ABORT_UNLESS(oldReqId, "race inside http pipelining");
                    } while (!AtomicCas(&PrimaryResponse_, requestId, oldReqId));

                    ProcessResponsesData();
                } else {
                    TAtomicBase oldReqId = AtomicGet(PrimaryResponse_);
                    if (oldReqId) {
                        while (!AtomicCas(&PrimaryResponse_, 0, oldReqId)) {
                            Y_ABORT_UNLESS(oldReqId == AtomicGet(PrimaryResponse_), "race inside http pipelining [2]");
                        }
                    }
                }
            }

            void ProcessResponsesData() {
                // process responses data queue, send response (if already have next PrimaryResponse_)
                TResponseDataRef rd;
                while (ResponsesDataQueue_.Dequeue(&rd)) {
                    ResponsesData_[rd->RequestId_] = rd;
                }
                TAtomicBase requestId = AtomicGet(PrimaryResponse_);
                if (requestId) {
                    THashMap<TAtomicBase, TResponseDataRef>::iterator it = ResponsesData_.find(requestId);
                    if (it != ResponsesData_.end()) {
                        // has next primary response
                        rd = it->second;
                        ResponsesData_.erase(it);
                        AS_->AsyncWrite(rd->Data_, std::bind(&TConn::OnSend, SelfRef(), _1, _2, _3), THttp2Options::ServerOutputDeadline);
                    }
                }
            }

        private:
            void SendData(TAtomicBase requestId, TTcpSocket::TSendedData sd) {
                TContIOVector& vec = *sd->GetIOvec();

                if (requestId != AtomicGet(PrimaryResponse_)) {
                    // already has another request for response first, so push this to queue
                    // + enqueue event for safe checking queue (at local/transport thread)
                    TResponseDataRef rdr = new TResponseData(requestId, sd);
                    ResponsesDataQueue_.Enqueue(rdr);
                    AS_->GetIOService().Post(std::bind(&TConn::ProcessResponsesData, SelfRef()));
                    return;
                }
                if (THttp2Options::ServerUseDirectWrite) {
                    vec.Proceed(AS_->WriteSome(vec));
                }
                if (!vec.Complete()) {
                    DBGOUT("AsyncWrite()");
                    AS_->AsyncWrite(sd, std::bind(&TConn::OnSend, SelfRef(), _1, _2, _3), THttp2Options::ServerOutputDeadline);
                } else {
                    // run ProcessPipeline at safe thread
                    AS_->GetIOService().Post(std::bind(&TConn::ProcessPipeline, SelfRef()));
                }
            }

            void OnSend(const TErrorCode& ec, size_t amount, IHandlingContext&) {
                Y_UNUSED(amount);
                if (ec) {
                    OnError();
                } else {
                    ProcessPipeline();
                }

                if (SeenMessageWithoutKeepalive_) {
                    TErrorCode shutdown_ec;
                    AS_->Shutdown(TTcpSocket::ShutdownBoth, shutdown_ec);
                }
            }

        public:
            bool IsCanceled() const noexcept {
                return Canceled_;
            }

            const TString& RemoteHost() const noexcept {
                return RemoteHost_;
            }

        private:
            TWeakPtrB<TConn> WeakThis_;
            THttpServer& HS_;
            TTcpSocketRef AS_;
            TString RemoteHost_;
            size_t BuffSize_;
            TArrayHolder<char> Buff_;
            TAutoPtr<THttpParser> P_;
            // pipeline supporting
            TAtomic PrimaryResponse_ = 0;
            TLockFreeQueue<TAtomicBase> PipelineOrder_;
            TLockFreeQueue<TResponseDataRef> ResponsesDataQueue_;
            THashMap<TAtomicBase, TResponseDataRef> ResponsesData_;

            TAtomicBool Canceled_;
            TAtomicBool SeenMessageWithoutKeepalive_ = false;

            i32 LeftRequestsToDisconnect_ = -1;
        };

        ///////////////////////////////////////////////////////////

    public:
        THttpServer(IOnRequest* cb, const TParsedLocation& loc)
            : E_(THttp2Options::AsioServerThreads)
            , CB_(cb)
            , LimitRequestsPerConnection(THttp2Options::LimitRequestsPerConnection)
        {
            TNetworkAddress addr = THttp2Options::RespectHostInHttpServerNetworkAddress ?
                                    TNetworkAddress(TString(loc.Host), loc.GetPort())
                                    : TNetworkAddress(loc.GetPort());

            for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
                TEndpoint ep(new NAddr::TAddrInfo(&*it));
                TTcpAcceptorPtr a(new TTcpAcceptor(AcceptExecutor_.GetIOService()));
                DBGOUT("bind:" << ep.IpToString() << ":" << ep.Port());
                a->Bind(ep);
                a->Listen(THttp2Options::Backlog);
                StartAccept(a.Get());
                A_.push_back(a);
            }
        }

        ~THttpServer() override {
            AcceptExecutor_.SyncShutdown(); //cancel operation for all current sockets (include acceptors)
            A_.clear();                     //stop listening
            E_.SyncShutdown();
        }

        void OnAccept(TTcpAcceptor* a, TAtomicSharedPtr<TTcpSocket> s, const TErrorCode& ec, IHandlingContext&) {
            if (Y_UNLIKELY(ec)) {
                if (ec.Value() == ECANCELED) {
                    return;
                } else if (ec.Value() == EMFILE || ec.Value() == ENFILE || ec.Value() == ENOMEM || ec.Value() == ENOBUFS) {
                    //reach some os limit, suspend accepting
                    TAtomicSharedPtr<TDeadlineTimer> dt(new TDeadlineTimer(a->GetIOService()));
                    dt->AsyncWaitExpireAt(TDuration::Seconds(30), std::bind(&THttpServer::OnTimeoutSuspendAccept, this, a, dt, _1, _2));
                    return;
                } else {
                    Cdbg << "acc: " << ec.Text() << Endl;
                }
            } else {
                if (static_cast<size_t>(HttpInConnCounter()->Val()) < HttpInConnLimits()->Hard()) {
                    try {
                        SetNonBlock(s->Native());
                        PrepareSocket(s->Native());
                        TConn::Create(*this, s);
                    } catch (TSystemError& err) {
                        TErrorCode ec2(err.Status());
                        Cdbg << "acc: " << ec2.Text() << Endl;
                    }
                } //else accepted socket will be closed
            }
            StartAccept(a); //continue accepting
        }

        void OnTimeoutSuspendAccept(TTcpAcceptor* a, TAtomicSharedPtr<TDeadlineTimer>, const TErrorCode& ec, IHandlingContext&) {
            if (!ec) {
                DBGOUT("resume acceptor")
                StartAccept(a);
            }
        }

        void OnRequest(IRequest* r) {
            try {
                CB_->OnRequest(r);
            } catch (...) {
                Cdbg << CurrentExceptionMessage() << Endl;
            }
        }

    protected:
        void OnCreateConn() noexcept {
            HttpInConnCounter()->Inc();
        }

        void OnDestroyConn() noexcept {
            HttpInConnCounter()->Dec();
        }

        TDuration GetKeepAliveTimeout() const noexcept {
            size_t cc = HttpInConnCounter()->Val();
            TFdLimits lim(*HttpInConnLimits());

            if (!TFdLimits::ExceedLimit(cc, lim.Soft())) {
                return THttp2Options::ServerInputDeadlineKeepAliveMax;
            }

            if (cc > lim.Hard()) {
                cc = lim.Hard();
            }
            TDuration::TValue softTuneRange = THttp2Options::ServerInputDeadlineKeepAliveMax.Seconds() - THttp2Options::ServerInputDeadlineKeepAliveMin.Seconds();

            return TDuration::Seconds((softTuneRange * (cc - lim.Soft())) / (lim.Hard() - lim.Soft() + 1)) + THttp2Options::ServerInputDeadlineKeepAliveMin;
        }

    private:
        void StartAccept(TTcpAcceptor* a) {
            TAtomicSharedPtr<TTcpSocket> s(new TTcpSocket(E_.Size() ? E_.GetExecutor().GetIOService() : AcceptExecutor_.GetIOService()));
            a->AsyncAccept(*s, std::bind(&THttpServer::OnAccept, this, a, s, _1, _2));
        }

        TIOServiceExecutor AcceptExecutor_;
        TVector<TTcpAcceptorPtr> A_;
        TExecutorsPool E_;
        IOnRequest* CB_;

    public:
        const i32 LimitRequestsPerConnection;
    };

    template <class T>
    class THttp2Protocol: public IProtocol {
    public:
        IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
            return new THttpServer(cb, loc);
        }

        THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
            THttpRequest::THandleRef ret(new THttpRequest::THandle(fallback, msg, !ss ? nullptr : new TStatCollector(ss)));
            try {
                THttpRequest::Run(ret, msg, &T::Build, T::RequestSettings());
            } catch (...) {
                ret->ResetOnRecv();
                throw;
            }
            return ret.Get();
        }

        TStringBuf Scheme() const noexcept override {
            return T::Name();
        }

        bool SetOption(TStringBuf name, TStringBuf value) override {
            return THttp2Options::Set(name, value);
        }
    };
}

namespace NNeh {
    IProtocol* Http1Protocol() {
        return Singleton<THttp2Protocol<TRequestGet1>>();
    }
    IProtocol* Post1Protocol() {
        return Singleton<THttp2Protocol<TRequestPost1>>();
    }
    IProtocol* Full1Protocol() {
        return Singleton<THttp2Protocol<TRequestFull1>>();
    }
    IProtocol* Http2Protocol() {
        return Singleton<THttp2Protocol<TRequestGet2>>();
    }
    IProtocol* Post2Protocol() {
        return Singleton<THttp2Protocol<TRequestPost2>>();
    }
    IProtocol* Full2Protocol() {
        return Singleton<THttp2Protocol<TRequestFull2>>();
    }
    IProtocol* UnixSocketGetProtocol() {
        return Singleton<THttp2Protocol<TRequestUnixSocketGet>>();
    }
    IProtocol* UnixSocketPostProtocol() {
        return Singleton<THttp2Protocol<TRequestUnixSocketPost>>();
    }
    IProtocol* UnixSocketFullProtocol() {
        return Singleton<THttp2Protocol<TRequestUnixSocketFull>>();
    }

    void SetHttp2OutputConnectionsLimits(size_t softLimit, size_t hardLimit) {
        HttpConnManager()->SetLimits(softLimit, hardLimit);
    }

    void SetHttp2InputConnectionsLimits(size_t softLimit, size_t hardLimit) {
        HttpInConnLimits()->SetSoft(softLimit);
        HttpInConnLimits()->SetHard(hardLimit);
    }

    TAtomicBase GetHttpOutputConnectionCount() {
        return HttpOutConnCounter()->Val();
    }

    std::pair<size_t, size_t> GetHttpOutputConnectionLimits() {
        return HttpConnManager()->GetLimits();
    }

    TAtomicBase GetHttpInputConnectionCount() {
        return HttpInConnCounter()->Val();
    }

    void SetHttp2InputConnectionsTimeouts(unsigned minSeconds, unsigned maxSeconds) {
        THttp2Options::ServerInputDeadlineKeepAliveMin = TDuration::Seconds(minSeconds);
        THttp2Options::ServerInputDeadlineKeepAliveMax = TDuration::Seconds(maxSeconds);
    }

    class TUnixSocketResolver {
    public:
        NDns::TResolvedHost* Resolve(const TString& path) {
            TString unixSocketPath = path;
            if (path.size() > 2 && path[0] == '[' && path[path.size() - 1] == ']') {
                unixSocketPath = path.substr(1, path.size() - 2);
            }

            if (auto resolvedUnixSocket = ResolvedUnixSockets_.FindPtr(unixSocketPath)) {
                return resolvedUnixSocket->Get();
            }

            TNetworkAddress na{TUnixSocketPath(unixSocketPath)};
            ResolvedUnixSockets_[unixSocketPath] = MakeHolder<NDns::TResolvedHost>(unixSocketPath, na);

            return ResolvedUnixSockets_[unixSocketPath].Get();
        }

    private:
        THashMap<TString, THolder<NDns::TResolvedHost>> ResolvedUnixSockets_;
    };

    TUnixSocketResolver* UnixSocketResolver() {
        return FastTlsSingleton<TUnixSocketResolver>();
    }

    const NDns::TResolvedHost* Resolve(const TStringBuf host, ui16 port, NHttp::EResolverType resolverType) {
        if (resolverType == EResolverType::EUNIXSOCKET) {
            return UnixSocketResolver()->Resolve(TString(host));
        }
        return NDns::CachedResolve(NDns::TResolveInfo(host, port));

    }
}
