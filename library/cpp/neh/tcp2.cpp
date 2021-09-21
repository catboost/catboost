#include "tcp2.h"

#include "details.h"
#include "factory.h"
#include "http_common.h"
#include "neh.h"
#include "utils.h"

#include <library/cpp/dns/cache.h>
#include <library/cpp/neh/asio/executor.h>
#include <library/cpp/threading/atomic/bool.h>

#include <util/generic/buffer.h>
#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/network/endpoint.h>
#include <util/network/init.h>
#include <util/network/iovec.h>
#include <util/network/socket.h>
#include <util/string/cast.h>

#include <atomic>

//#define DEBUG_TCP2 1
#ifdef DEBUG_TCP2
TSpinLock OUT_LOCK;
#define DBGOUT(args)                                               \
    {                                                              \
        TGuard<TSpinLock> m(OUT_LOCK);                             \
        Cout << TInstant::Now().GetValue() << " " << args << Endl; \
    }
#else
#define DBGOUT(args)
#endif

using namespace std::placeholders;

namespace NNeh {
    TDuration TTcp2Options::ConnectTimeout = TDuration::MilliSeconds(300);
    size_t TTcp2Options::InputBufferSize = 16000;
    size_t TTcp2Options::AsioClientThreads = 4;
    size_t TTcp2Options::AsioServerThreads = 4;
    int TTcp2Options::Backlog = 100;
    bool TTcp2Options::ClientUseDirectWrite = true;
    bool TTcp2Options::ServerUseDirectWrite = true;
    TDuration TTcp2Options::ServerInputDeadline = TDuration::Seconds(3600);
    TDuration TTcp2Options::ServerOutputDeadline = TDuration::Seconds(10);

    bool TTcp2Options::Set(TStringBuf name, TStringBuf value) {
#define TCP2_TRY_SET(optType, optName)        \
    if (name == TStringBuf(#optName)) {      \
        optName = FromString<optType>(value); \
    }

        TCP2_TRY_SET(TDuration, ConnectTimeout)
        else TCP2_TRY_SET(size_t, InputBufferSize) else TCP2_TRY_SET(size_t, AsioClientThreads) else TCP2_TRY_SET(size_t, AsioServerThreads) else TCP2_TRY_SET(int, Backlog) else TCP2_TRY_SET(bool, ClientUseDirectWrite) else TCP2_TRY_SET(bool, ServerUseDirectWrite) else TCP2_TRY_SET(TDuration, ServerInputDeadline) else TCP2_TRY_SET(TDuration, ServerOutputDeadline) else {
            return false;
        }
        return true;
    }
}

namespace {
    namespace NNehTcp2 {
        using namespace NAsio;
        using namespace NDns;
        using namespace NNeh;

        const TString canceled = "canceled";
        const TString emptyReply = "empty reply";

        inline void PrepareSocket(SOCKET s) {
            SetNoDelay(s, true);
        }

        typedef ui64 TRequestId;

#pragma pack(push, 1) //disable align struct members (structs mapped to data transmitted other network)
        struct TBaseHeader {
            enum TMessageType {
                Request = 1,
                Response = 2,
                Cancel = 3,
                MaxMessageType
            };

            TBaseHeader(TRequestId id, ui32 headerLength, ui8 version, ui8 mType)
                : Id(id)
                , HeaderLength(headerLength)
                , Version(version)
                , Type(mType)
            {
            }

            TRequestId Id; //message id, - monotonic inc. sequence (skip nil value)
            ui32 HeaderLength;
            ui8 Version; //current version: 1
            ui8 Type;    //<- TMessageType (+ in future possible ForceResponse,etc)
        };

        struct TRequestHeader: public TBaseHeader {
            TRequestHeader(TRequestId reqId, size_t servicePathLength, size_t dataSize)
                : TBaseHeader(reqId, sizeof(TRequestHeader) + servicePathLength, 1, (ui8)Request)
                , ContentLength(dataSize)
            {
            }

            ui32 ContentLength;
        };

        struct TResponseHeader: public TBaseHeader {
            enum TErrorCode {
                Success = 0,
                EmptyReply = 1 //not found such service or service not sent response
                ,
                MaxErrorCode
            };

            TResponseHeader(TRequestId reqId, TErrorCode code, size_t dataSize)
                : TBaseHeader(reqId, sizeof(TResponseHeader), 1, (ui8)Response)
                , ErrorCode((ui16)code)
                , ContentLength(dataSize)
            {
            }

            TString ErrorDescription() const {
                if (ErrorCode == (ui16)EmptyReply) {
                    return emptyReply;
                }

                TStringStream ss;
                ss << TStringBuf("tcp2 err_code=") << ErrorCode;
                return ss.Str();
            }

            ui16 ErrorCode;
            ui32 ContentLength;
        };

        struct TCancelHeader: public TBaseHeader {
            TCancelHeader(TRequestId reqId)
                : TBaseHeader(reqId, sizeof(TCancelHeader), 1, (ui8)Cancel)
            {
            }
        };
#pragma pack(pop)

        static const size_t maxHeaderSize = sizeof(TResponseHeader);

        //buffer for read input data, - header + message data
        struct TTcp2Message {
            TTcp2Message()
                : Loader_(&TTcp2Message::LoadBaseHeader)
                , RequireBytesForComplete_(sizeof(TBaseHeader))
                , Header_(sizeof(TBaseHeader))
            {
            }

            void Clear() {
                Loader_ = &TTcp2Message::LoadBaseHeader;
                RequireBytesForComplete_ = sizeof(TBaseHeader);
                Header_.Clear();
                Content_.clear();
            }

            TBuffer& Header() noexcept {
                return Header_;
            }

            const TString& Content() const noexcept {
                return Content_;
            }

            bool IsComplete() const noexcept {
                return RequireBytesForComplete_ == 0;
            }

            size_t LoadFrom(const char* buf, size_t len) {
                return (this->*Loader_)(buf, len);
            }

            const TBaseHeader& BaseHeader() const {
                return *reinterpret_cast<const TBaseHeader*>(Header_.Data());
            }

            const TRequestHeader& RequestHeader() const {
                return *reinterpret_cast<const TRequestHeader*>(Header_.Data());
            }

            const TResponseHeader& ResponseHeader() const {
                return *reinterpret_cast<const TResponseHeader*>(Header_.Data());
            }

        private:
            size_t LoadBaseHeader(const char* buf, size_t len) {
                size_t useBytes = Min<size_t>(sizeof(TBaseHeader) - Header_.Size(), len);
                Header_.Append(buf, useBytes);
                if (Y_UNLIKELY(sizeof(TBaseHeader) > Header_.Size())) {
                    //base header yet not complete
                    return useBytes;
                }
                {
                    const TBaseHeader& hdr = BaseHeader();
                    if (BaseHeader().HeaderLength > 32000) { //some heuristic header size limit
                        throw yexception() << TStringBuf("to large neh/tcp2 header size: ") << BaseHeader().HeaderLength;
                    }
                    //header completed
                    Header_.Reserve(hdr.HeaderLength);
                }
                const TBaseHeader& hdr = BaseHeader(); //reallocation can move Header_ data to another place, so use fresh 'hdr'
                if (Y_UNLIKELY(hdr.Version != 1)) {
                    throw yexception() << TStringBuf("unsupported protocol version: ") << static_cast<unsigned>(hdr.Version);
                }
                RequireBytesForComplete_ = hdr.HeaderLength - sizeof(TBaseHeader);
                return useBytes + LoadHeader(buf + useBytes, len - useBytes);
            }

            size_t LoadHeader(const char* buf, size_t len) {
                size_t useBytes = Min<size_t>(RequireBytesForComplete_, len);
                Header_.Append(buf, useBytes);
                RequireBytesForComplete_ -= useBytes;
                if (RequireBytesForComplete_) {
                    //continue load header
                    Loader_ = &TTcp2Message::LoadHeader;
                    return useBytes;
                }

                const TBaseHeader& hdr = *reinterpret_cast<const TBaseHeader*>(Header_.Data());

                if (hdr.Type == TBaseHeader::Request) {
                    if (Header_.Size() < sizeof(TRequestHeader)) {
                        throw yexception() << TStringBuf("invalid request header size");
                    }
                    InitContentLoading(RequestHeader().ContentLength);
                } else if (hdr.Type == TBaseHeader::Response) {
                    if (Header_.Size() < sizeof(TResponseHeader)) {
                        throw yexception() << TStringBuf("invalid response header size");
                    }
                    InitContentLoading(ResponseHeader().ContentLength);
                } else if (hdr.Type == TBaseHeader::Cancel) {
                    if (Header_.Size() < sizeof(TCancelHeader)) {
                        throw yexception() << TStringBuf("invalid cancel header size");
                    }
                    return useBytes;
                } else {
                    throw yexception() << TStringBuf("unsupported request type: ") << static_cast<unsigned>(hdr.Type);
                }
                return useBytes + (this->*Loader_)(buf + useBytes, len - useBytes);
            }

            void InitContentLoading(size_t contentLength) {
                RequireBytesForComplete_ = contentLength;
                Content_.ReserveAndResize(contentLength);
                Loader_ = &TTcp2Message::LoadContent;
            }

            size_t LoadContent(const char* buf, size_t len) {
                size_t curContentSize = Content_.size() - RequireBytesForComplete_;
                size_t useBytes = Min<size_t>(RequireBytesForComplete_, len);
                memcpy(Content_.begin() + curContentSize, buf, useBytes);
                RequireBytesForComplete_ -= useBytes;
                return useBytes;
            }

        private:
            typedef size_t (TTcp2Message::*TLoader)(const char*, size_t);

            TLoader Loader_; //current loader (stages - base-header/header/content)
            size_t RequireBytesForComplete_;
            TBuffer Header_;
            TString Content_;
        };

        //base storage for output data
        class TMultiBuffers {
        public:
            TMultiBuffers()
                : IOVec_(nullptr, 0)
                , DataSize_(0)
                , PoolBytes_(0)
            {
            }

            void Clear() noexcept {
                Parts_.clear();
                DataSize_ = 0;
                PoolBytes_ = 0;
            }

            bool HasFreeSpace() const noexcept {
                return DataSize_ < 64000 && (PoolBytes_ < (MemPoolSize_ - maxHeaderSize));
            }

            bool HasData() const noexcept {
                return Parts_.size();
            }

            TContIOVector* GetIOvec() noexcept {
                return &IOVec_;
            }

        protected:
            void AddPart(const void* buf, size_t len) {
                Parts_.push_back(IOutputStream::TPart(buf, len));
                DataSize_ += len;
            }

            //used for allocate header (MUST be POD type)
            template <typename T>
            inline T* Allocate() noexcept {
                size_t poolBytes = PoolBytes_;
                PoolBytes_ += sizeof(T);
                return (T*)(MemPool_ + poolBytes);
            }

            //used for allocate header (MUST be POD type) + some tail
            template <typename T>
            inline T* AllocatePlus(size_t tailSize) noexcept {
                Y_ASSERT(tailSize <= MemPoolReserve_);
                size_t poolBytes = PoolBytes_;
                PoolBytes_ += sizeof(T) + tailSize;
                return (T*)(MemPool_ + poolBytes);
            }

        protected:
            TContIOVector IOVec_;
            TVector<IOutputStream::TPart> Parts_;
            static const size_t MemPoolSize_ = maxHeaderSize * 100;
            static const size_t MemPoolReserve_ = 32;
            size_t DataSize_;
            size_t PoolBytes_;
            char MemPool_[MemPoolSize_ + MemPoolReserve_];
        };

        //protector for limit usage tcp connection output (and used data) only from one thread at same time
        class TOutputLock {
        public:
            TOutputLock() noexcept
                : Lock_(0)
            {
            }

            bool TryAquire() noexcept {
                do {
                    if (AtomicTryLock(&Lock_)) {
                        return true;
                    }
                } while (!AtomicGet(Lock_)); //without magic loop atomic lock some unreliable
                return false;
            }

            void Release() noexcept {
                AtomicUnlock(&Lock_);
            }

            bool IsFree() const noexcept {
                return !AtomicGet(Lock_);
            }

        private:
            TAtomic Lock_;
        };

        class TClient {
            class TRequest;
            class TConnection;
            typedef TIntrusivePtr<TRequest> TRequestRef;
            typedef TIntrusivePtr<TConnection> TConnectionRef;

            class TRequest: public TThrRefBase, public TNonCopyable {
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

                        TRequestRef req = GetRequest();
                        if (!!req && req->RequestSendedCompletely()) {
                            const_cast<THandle*>(this)->SetSendComplete();
                        }

                        return TSimpleHandle::MessageSendedCompletely();
                    }

                    void Cancel() noexcept override {
                        if (TSimpleHandle::Canceled()) {
                            return;
                        }

                        TRequestRef req = GetRequest();
                        if (!!req) {
                            req->Cancel();
                            TSimpleHandle::Cancel();
                        }
                    }

                    void NotifyResponse(const TString& resp) {
                        TNotifyHandle::NotifyResponse(resp);

                        ReleaseRequest();
                    }

                    void NotifyError(const TString& error) {
                        TNotifyHandle::NotifyError(error);

                        ReleaseRequest();
                    }

                    void NotifyError(TErrorRef error) {
                        TNotifyHandle::NotifyError(error);

                        ReleaseRequest();
                    }

                    //not thread safe!
                    void SetRequest(const TRequestRef& r) noexcept {
                        Req_ = r;
                    }

                    void ReleaseRequest() noexcept {
                        TRequestRef tmp;
                        TGuard<TSpinLock> g(SP_);
                        tmp.Swap(Req_);
                    }

                private:
                    TRequestRef GetRequest() const noexcept {
                        TGuard<TSpinLock> g(SP_);
                        return Req_;
                    }

                    mutable TSpinLock SP_;
                    TRequestRef Req_;
                };

                typedef TIntrusivePtr<THandle> THandleRef;

                static void Run(THandleRef& h, const TMessage& msg, TClient& clnt) {
                    TRequestRef req(new TRequest(h, msg, clnt));
                    h->SetRequest(req);
                    req->Run(req);
                }

                ~TRequest() override {
                    DBGOUT("TClient::~TRequest()");
                }

            private:
                TRequest(THandleRef& h, TMessage msg, TClient& clnt)
                    : Hndl_(h)
                    , Clnt_(clnt)
                    , Msg_(std::move(msg))
                    , Loc_(Msg_.Addr)
                    , Addr_(CachedResolve(TResolveInfo(Loc_.Host, Loc_.GetPort())))
                    , Canceled_(false)
                    , Id_(0)
                {
                    DBGOUT("TClient::TRequest()");
                }

                void Run(TRequestRef& req) {
                    TDestination& dest = Clnt_.Dest_.Get(Addr_->Id);
                    dest.Run(req);
                }

            public:
                void OnResponse(TTcp2Message& msg) {
                    DBGOUT("TRequest::OnResponse: " << msg.ResponseHeader().Id);
                    THandleRef h = ReleaseHandler();
                    if (!h) {
                        return;
                    }

                    const TResponseHeader& respHdr = msg.ResponseHeader();
                    if (Y_LIKELY(!respHdr.ErrorCode)) {
                        h->NotifyResponse(msg.Content());
                    } else {
                        h->NotifyError(new TError(respHdr.ErrorDescription(), TError::ProtocolSpecific, respHdr.ErrorCode));
                    }
                    ReleaseConn();
                }

                void OnError(const TString& err, const i32 systemCode = 0) {
                    DBGOUT("TRequest::OnError: " << Id_.load(std::memory_order_acquire));
                    THandleRef h = ReleaseHandler();
                    if (!h) {
                        return;
                    }

                    h->NotifyError(new TError(err, TError::UnknownType, 0, systemCode));
                    ReleaseConn();
                }

                void SetConnection(TConnection* conn) noexcept {
                    auto g = Guard(AL_);
                    Conn_ = conn;
                }

                bool Canceled() const noexcept {
                    return Canceled_;
                }

                const TResolvedHost* Addr() const noexcept {
                    return Addr_;
                }

                TStringBuf Service() const noexcept {
                    return Loc_.Service;
                }

                const TString& Data() const noexcept {
                    return Msg_.Data;
                }

                TClient& Client() noexcept {
                    return Clnt_;
                }

                bool RequestSendedCompletely() const noexcept {
                    if (Id_.load(std::memory_order_acquire) == 0) {
                        return false;
                    }

                    TConnectionRef conn = GetConn();
                    if (!conn) {
                        return false;
                    }

                    TRequestId lastSendedReqId = conn->LastSendedRequestId();
                    if (lastSendedReqId >= Id_.load(std::memory_order_acquire)) {
                        return true;
                    } else if (Y_UNLIKELY((Id_.load(std::memory_order_acquire) - lastSendedReqId) > (Max<TRequestId>() - Max<ui32>()))) {
                        //overflow req-id value
                        return true;
                    }
                    return false;
                }

                void Cancel() noexcept {
                    Canceled_ = true;
                    THandleRef h = ReleaseHandler();
                    if (!h) {
                        return;
                    }

                    TConnectionRef conn = ReleaseConn();
                    if (!!conn && Id_.load(std::memory_order_acquire)) {
                        conn->Cancel(Id_.load(std::memory_order_acquire));
                    }
                    h->NotifyError(new TError(canceled, TError::Cancelled));
                }

                void SetReqId(TRequestId reqId) noexcept {
                    auto guard = Guard(IdLock_);
                    Id_.store(reqId, std::memory_order_release);
                }

                TRequestId ReqId() const noexcept {
                    return Id_.load(std::memory_order_acquire);
                }

            private:
                inline THandleRef ReleaseHandler() noexcept {
                    THandleRef h;
                    {
                        auto g = Guard(AL_);
                        h.Swap(Hndl_);
                    }
                    return h;
                }

                inline TConnectionRef GetConn() const noexcept {
                    auto g = Guard(AL_);
                    return Conn_;
                }

                inline TConnectionRef ReleaseConn() noexcept {
                    TConnectionRef c;
                    {
                        auto g = Guard(AL_);
                        c.Swap(Conn_);
                    }
                    return c;
                }

                mutable TAdaptiveLock AL_; //guaranted calling notify() only once (prevent race between asio thread and current)
                THandleRef Hndl_;
                TClient& Clnt_;
                const TMessage Msg_;
                const TParsedLocation Loc_;
                const TResolvedHost* Addr_;
                TConnectionRef Conn_;
                NAtomic::TBool Canceled_;
                TSpinLock IdLock_;
                std::atomic<TRequestId> Id_;
            };

            class TConnection: public TThrRefBase {
                enum TState {
                    Init,
                    Connecting,
                    Connected,
                    Closed,
                    MaxState
                };
                typedef THashMap<TRequestId, TRequestRef> TReqsInFly;

            public:
                class TOutputBuffers: public TMultiBuffers {
                public:
                    void AddRequest(const TRequestRef& req) {
                        Requests_.push_back(req);
                        if (req->Service().size() > MemPoolReserve_) {
                            TRequestHeader* hdr = new (Allocate<TRequestHeader>()) TRequestHeader(req->ReqId(), req->Service().size(), req->Data().size());
                            AddPart(hdr, sizeof(TRequestHeader));
                            AddPart(req->Service().data(), req->Service().size());
                        } else {
                            TRequestHeader* hdr = new (AllocatePlus<TRequestHeader>(req->Service().size())) TRequestHeader(req->ReqId(), req->Service().size(), req->Data().size());
                            AddPart(hdr, sizeof(TRequestHeader) + req->Service().size());
                            memmove(++hdr, req->Service().data(), req->Service().size());
                        }
                        AddPart(req->Data().data(), req->Data().size());
                        IOVec_ = TContIOVector(Parts_.data(), Parts_.size());
                    }

                    void AddCancelRequest(TRequestId reqId) {
                        TCancelHeader* hdr = new (Allocate<TCancelHeader>()) TCancelHeader(reqId);
                        AddPart(hdr, sizeof(TCancelHeader));
                        IOVec_ = TContIOVector(Parts_.data(), Parts_.size());
                    }

                    void Clear() {
                        TMultiBuffers::Clear();
                        Requests_.clear();
                    }

                private:
                    TVector<TRequestRef> Requests_;
                };

                TConnection(TIOService& srv)
                    : AS_(srv)
                    , State_(Init)
                    , BuffSize_(TTcp2Options::InputBufferSize)
                    , Buff_(new char[BuffSize_])
                    , NeedCheckReqsQueue_(0)
                    , NeedCheckCancelsQueue_(0)
                    , GenReqId_(0)
                    , LastSendedReqId_(0)
                {
                }

                ~TConnection() override {
                    try {
                        DBGOUT("TClient::~TConnection()");
                        OnError("~");
                    } catch (...) {
                        Cdbg << "tcp2::~cln_conn: " << CurrentExceptionMessage() << Endl;
                    }
                }

                //called from client thread
                bool Run(TRequestRef& req) {
                    if (Y_UNLIKELY(AtomicGet(State_) == Closed)) {
                        return false;
                    }

                    req->Ref();
                    try {
                        Reqs_.Enqueue(req.Get());
                    } catch (...) {
                        req->UnRef();
                        throw;
                    }

                    AtomicSet(NeedCheckReqsQueue_, 1);
                    req->SetConnection(this);
                    TAtomicBase state = AtomicGet(State_);
                    if (Y_LIKELY(state == Connected)) {
                        ProcessOutputReqsQueue();
                        return true;
                    }

                    if (state == Init) {
                        if (AtomicCas(&State_, Connecting, Init)) {
                            try {
                                TEndpoint addr(new NAddr::TAddrInfo(&*req->Addr()->Addr.Begin()));
                                AS_.AsyncConnect(addr, std::bind(&TConnection::OnConnect, TConnectionRef(this), _1, _2), TTcp2Options::ConnectTimeout);
                            } catch (...) {
                                AS_.GetIOService().Post(std::bind(&TConnection::OnErrorCallback, TConnectionRef(this), CurrentExceptionMessage()));
                            }
                            return true;
                        }
                    }
                    state = AtomicGet(State_);
                    if (state == Connected) {
                        ProcessOutputReqsQueue();
                    } else if (state == Closed) {
                        SafeOnError();
                    }
                    return true;
                }

                //called from client thread
                void Cancel(TRequestId id) {
                    Cancels_.Enqueue(id);
                    AtomicSet(NeedCheckCancelsQueue_, 1);
                    if (Y_LIKELY(AtomicGet(State_) == Connected)) {
                        ProcessOutputCancelsQueue();
                    }
                }

                void ProcessOutputReqsQueue() {
                    if (OutputLock_.TryAquire()) {
                        SendMessages(false);
                    }
                }

                void ProcessOutputCancelsQueue() {
                    if (OutputLock_.TryAquire()) {
                        AS_.GetIOService().Post(std::bind(&TConnection::SendMessages, TConnectionRef(this), true));
                        return;
                    }
                }

                //must be called only from asio thread
                void ProcessReqsInFlyQueue() {
                    if (AtomicGet(State_) == Closed) {
                        return;
                    }

                    TRequest* reqPtr;

                    while (ReqsInFlyQueue_.Dequeue(&reqPtr)) {
                        TRequestRef reqTmp(reqPtr);
                        reqPtr->UnRef();
                        ReqsInFly_[reqPtr->ReqId()].Swap(reqTmp);
                    }
                }

                //must be called only from asio thread
                void OnConnect(const TErrorCode& ec, IHandlingContext&) {
                    DBGOUT("TConnect::OnConnect: " << ec.Value());
                    if (Y_UNLIKELY(ec)) {
                        if (ec.Value() == EIO) {
                            //try get more detail error info
                            char buf[1];
                            TErrorCode errConnect;
                            AS_.ReadSome(buf, 1, errConnect);
                            OnErrorCode(errConnect.Value() ? errConnect : ec);
                        } else {
                            OnErrorCode(ec);
                        }
                    } else {
                        try {
                            PrepareSocket(AS_.Native());
                            AtomicSet(State_, Connected);
                            AS_.AsyncPollRead(std::bind(&TConnection::OnCanRead, TConnectionRef(this), _1, _2));
                            if (OutputLock_.TryAquire()) {
                                SendMessages(true);
                                return;
                            }
                        } catch (...) {
                            OnError(CurrentExceptionMessage());
                        }
                    }
                }

                //must be called only after succes aquiring output
                void SendMessages(bool asioThread) {
                    //DBGOUT("SendMessages");
                    if (Y_UNLIKELY(AtomicGet(State_) == Closed)) {
                        if (asioThread) {
                            OnError(Error_);
                        } else {
                            SafeOnError();
                        }
                        return;
                    }

                    do {
                        if (asioThread) {
                            AtomicSet(NeedCheckCancelsQueue_, 0);
                            TRequestId reqId;

                            ProcessReqsInFlyQueue();
                            while (Cancels_.Dequeue(&reqId)) {
                                TReqsInFly::iterator it = ReqsInFly_.find(reqId);
                                if (it == ReqsInFly_.end()) {
                                    continue;
                                }

                                ReqsInFly_.erase(it);
                                OutputBuffers_.AddCancelRequest(reqId);
                                if (Y_UNLIKELY(!OutputBuffers_.HasFreeSpace())) {
                                    if (!FlushOutputBuffers(asioThread, 0)) {
                                        return;
                                    }
                                }
                            }
                        } else if (AtomicGet(NeedCheckCancelsQueue_)) {
                            AS_.GetIOService().Post(std::bind(&TConnection::SendMessages, TConnectionRef(this), true));
                            return;
                        }

                        TRequestId lastReqId = 0;
                        {
                            AtomicSet(NeedCheckReqsQueue_, 0);
                            TRequest* reqPtr;

                            while (Reqs_.Dequeue(&reqPtr)) {
                                TRequestRef reqTmp(reqPtr);
                                reqPtr->UnRef();
                                reqPtr->SetReqId(GenerateReqId());
                                if (reqPtr->Canceled()) {
                                    continue;
                                }
                                lastReqId = reqPtr->ReqId();
                                if (asioThread) {
                                    TRequestRef& req = ReqsInFly_[(TRequestId)reqPtr->ReqId()];
                                    req.Swap(reqTmp);
                                    OutputBuffers_.AddRequest(req);
                                } else { //can access to ReqsInFly_ only from asio thread, so enqueue req to update ReqsInFly_ queue
                                    try {
                                        reqTmp->Ref();
                                        ReqsInFlyQueue_.Enqueue(reqPtr);
                                    } catch (...) {
                                        reqTmp->UnRef();
                                        throw;
                                    }
                                    OutputBuffers_.AddRequest(reqTmp);
                                }
                                if (Y_UNLIKELY(!OutputBuffers_.HasFreeSpace())) {
                                    if (!FlushOutputBuffers(asioThread, lastReqId)) {
                                        return;
                                    }
                                }
                            }
                        }

                        if (OutputBuffers_.HasData()) {
                            if (!FlushOutputBuffers(asioThread, lastReqId)) {
                                return;
                            }
                        }

                        OutputLock_.Release();

                        if (!AtomicGet(NeedCheckReqsQueue_) && !AtomicGet(NeedCheckCancelsQueue_)) {
                            DBGOUT("TClient::SendMessages(exit2)");
                            return;
                        }
                    } while (OutputLock_.TryAquire());
                    DBGOUT("TClient::SendMessages(exit1)");
                }

                TRequestId GenerateReqId() noexcept {
                    TRequestId reqId;
                    {
                        auto guard = Guard(GenReqIdLock_);
                        reqId = ++GenReqId_;
                    }
                    return Y_LIKELY(reqId) ? reqId : GenerateReqId();
                }

                //called non thread-safe (from outside thread)
                bool FlushOutputBuffers(bool asioThread, TRequestId reqId) {
                    if (asioThread || TTcp2Options::ClientUseDirectWrite) {
                        TContIOVector& vec = *OutputBuffers_.GetIOvec();
                        TErrorCode err;
                        vec.Proceed(AS_.WriteSome(vec, err));

                        if (Y_UNLIKELY(err)) {
                            if (asioThread) {
                                OnErrorCode(err);
                            } else {
                                AS_.GetIOService().Post(std::bind(&TConnection::OnErrorCode, TConnectionRef(this), err));
                            }
                            return false;
                        }

                        if (vec.Complete()) {
                            LastSendedReqId_.store(reqId, std::memory_order_release);
                            DBGOUT("Client::FlushOutputBuffers(" << reqId << ")");
                            OutputBuffers_.Clear();
                            return true;
                        }
                    }

                    DBGOUT("Client::AsyncWrite(" << reqId << ")");
                    AS_.AsyncWrite(OutputBuffers_.GetIOvec(), std::bind(&TConnection::OnSend, TConnectionRef(this), reqId, _1, _2, _3), TTcp2Options::ServerOutputDeadline);
                    return false;
                }

                //must be called only from asio thread
                void OnSend(TRequestId reqId, const TErrorCode& ec, size_t amount, IHandlingContext&) {
                    Y_UNUSED(amount);
                    if (Y_UNLIKELY(ec)) {
                        OnErrorCode(ec);
                    } else {
                        if (Y_LIKELY(reqId)) {
                            DBGOUT("Client::OnSend(" << reqId << ")");
                            LastSendedReqId_.store(reqId, std::memory_order_release);
                        }
                        //output already aquired, used asio thread
                        OutputBuffers_.Clear();
                        SendMessages(true);
                    }
                }

                //must be called only from asio thread
                void OnCanRead(const TErrorCode& ec, IHandlingContext& ctx) {
                    //DBGOUT("OnCanRead(" << ec.Value() << ")");
                    if (Y_UNLIKELY(ec)) {
                        OnErrorCode(ec);
                    } else {
                        TErrorCode ec2;
                        OnReadSome(ec2, AS_.ReadSome(Buff_.Get(), BuffSize_, ec2), ctx);
                    }
                }

                //must be called only from asio thread
                void OnReadSome(const TErrorCode& ec, size_t amount, IHandlingContext& ctx) {
                    //DBGOUT("OnReadSome(" << ec.Value() << ", " <<  amount << ")");
                    if (Y_UNLIKELY(ec)) {
                        OnErrorCode(ec);

                        return;
                    }

                    while (1) {
                        if (Y_UNLIKELY(!amount)) {
                            OnError("tcp conn. closed");

                            return;
                        }

                        try {
                            const char* buff = Buff_.Get();
                            size_t leftBytes = amount;
                            do {
                                size_t useBytes = Msg_.LoadFrom(buff, leftBytes);
                                leftBytes -= useBytes;
                                buff += useBytes;
                                if (Msg_.IsComplete()) {
                                    //DBGOUT("OnReceiveMessage(" << Msg_.BaseHeader().Id << "): " << leftBytes);
                                    OnReceiveMessage();
                                    Msg_.Clear();
                                }
                            } while (leftBytes);

                            if (amount == BuffSize_) {
                                //try decrease system calls, - re-run ReadSome if has full filled buffer
                                TErrorCode ecR;
                                amount = AS_.ReadSome(Buff_.Get(), BuffSize_, ecR);
                                if (!ecR) {
                                    continue; //process next input data
                                }
                                if (ecR.Value() == EAGAIN || ecR.Value() == EWOULDBLOCK) {
                                    ctx.ContinueUseHandler();
                                } else {
                                    OnErrorCode(ec);
                                }
                            } else {
                                ctx.ContinueUseHandler();
                            }
                        } catch (...) {
                            OnError(CurrentExceptionMessage());
                        }

                        return;
                    }
                }

                //must be called only from asio thread
                void OnErrorCode(TErrorCode ec) {
                    OnError(ec.Text(), ec.Value());
                }

                //must be called only from asio thread
                void OnErrorCallback(TString err) {
                    OnError(err);
                }

                //must be called only from asio thread
                void OnError(const TString& err, const i32 systemCode = 0) {
                    if (AtomicGet(State_) != Closed) {
                        Error_ = err;
                        SystemCode_ = systemCode;
                        AtomicSet(State_, Closed);
                        AS_.AsyncCancel();
                    }
                    SafeOnError();
                    for (auto& it : ReqsInFly_) {
                        it.second->OnError(err);
                    }
                    ReqsInFly_.clear();
                }

                void SafeOnError() {
                    TRequest* reqPtr;

                    while (Reqs_.Dequeue(&reqPtr)) {
                        TRequestRef req(reqPtr);
                        reqPtr->UnRef();
                        //DBGOUT("err queue(" << AS_.Native() << "):" << size_t(reqPtr));
                        req->OnError(Error_, SystemCode_);
                    }

                    while (ReqsInFlyQueue_.Dequeue(&reqPtr)) {
                        TRequestRef req(reqPtr);
                        reqPtr->UnRef();
                        //DBGOUT("err fly queue(" << AS_.Native() << "):" << size_t(reqPtr));
                        req->OnError(Error_, SystemCode_);
                    }
                }

                //must be called only from asio thread
                void OnReceiveMessage() {
                    //DBGOUT("OnReceiveMessage");
                    const TBaseHeader& hdr = Msg_.BaseHeader();

                    if (hdr.Type == TBaseHeader::Response) {
                        ProcessReqsInFlyQueue();
                        TReqsInFly::iterator it = ReqsInFly_.find(hdr.Id);
                        if (it == ReqsInFly_.end()) {
                            DBGOUT("ignore response: " << hdr.Id);
                            return;
                        }

                        it->second->OnResponse(Msg_);
                        ReqsInFly_.erase(it);
                    } else {
                        throw yexception() << TStringBuf("unsupported message type: ") << hdr.Type;
                    }
                }

                TRequestId LastSendedRequestId() const noexcept {
                    return LastSendedReqId_.load(std::memory_order_acquire);
                }

            private:
                NAsio::TTcpSocket AS_;
                TAtomic State_; //state machine status (TState)
                TString Error_;
                i32 SystemCode_ = 0;

                //input
                size_t BuffSize_;
                TArrayHolder<char> Buff_;
                TTcp2Message Msg_;

                //output
                TOutputLock OutputLock_;
                TAtomic NeedCheckReqsQueue_;
                TLockFreeQueue<TRequest*> Reqs_;
                TAtomic NeedCheckCancelsQueue_;
                TLockFreeQueue<TRequestId> Cancels_;
                TAdaptiveLock GenReqIdLock_;
                std::atomic<TRequestId> GenReqId_;
                std::atomic<TRequestId> LastSendedReqId_;
                TLockFreeQueue<TRequest*> ReqsInFlyQueue_;
                TReqsInFly ReqsInFly_;
                TOutputBuffers OutputBuffers_;
            };

            class TDestination {
            public:
                void Run(TRequestRef& req) {
                    while (1) {
                        TConnectionRef conn = GetConnection();
                        if (!!conn && conn->Run(req)) {
                            return;
                        }

                        DBGOUT("TDestination CreateConnection");
                        CreateConnection(conn, req->Client().ExecutorsPool().GetExecutor().GetIOService());
                    }
                }

            private:
                TConnectionRef GetConnection() {
                    TGuard<TSpinLock> g(L_);
                    return Conn_;
                }

                void CreateConnection(TConnectionRef& oldConn, TIOService& srv) {
                    TConnectionRef conn(new TConnection(srv));
                    TGuard<TSpinLock> g(L_);
                    if (Conn_ == oldConn) {
                        Conn_.Swap(conn);
                    }
                }

                TSpinLock L_;
                TConnectionRef Conn_;
            };

            //////////// TClient /////////

        public:
            TClient()
                : EP_(TTcp2Options::AsioClientThreads)
            {
            }

            ~TClient() {
                EP_.SyncShutdown();
            }

            THandleRef Schedule(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) {
                //find exist connection or create new
                TRequest::THandleRef hndl(new TRequest::THandle(fallback, msg, !ss ? nullptr : new TStatCollector(ss)));
                try {
                    TRequest::Run(hndl, msg, *this);
                } catch (...) {
                    hndl->ResetOnRecv();
                    hndl->ReleaseRequest();
                    throw;
                }
                return hndl.Get();
            }

            TExecutorsPool& ExecutorsPool() {
                return EP_;
            }

        private:
            NNeh::NHttp::TLockFreeSequence<TDestination> Dest_;
            TExecutorsPool EP_;
        };

        ////////// server side ////////////////////////////////////////////////////////////////////////////////////////////

        class TServer: public IRequester {
            typedef TAutoPtr<TTcpAcceptor> TTcpAcceptorPtr;
            typedef TAtomicSharedPtr<TTcpSocket> TTcpSocketRef;
            class TConnection;
            typedef TIntrusivePtr<TConnection> TConnectionRef;

            struct TRequest: public IRequest {
                struct TState: public TThrRefBase {
                    TState()
                        : Canceled(false)
                    {
                    }

                    TAtomicBool Canceled;
                };
                typedef TIntrusivePtr<TState> TStateRef;

                TRequest(const TConnectionRef& conn, TBuffer& buf, const TString& content);
                ~TRequest() override;

                TStringBuf Scheme() const override {
                    return TStringBuf("tcp2");
                }

                TString RemoteHost() const override;

                TStringBuf Service() const override {
                    return TStringBuf(Buf.Data() + sizeof(TRequestHeader), Buf.End());
                }

                TStringBuf Data() const override {
                    return TStringBuf(Content_);
                }

                TStringBuf RequestId() const override {
                    return TStringBuf();
                }

                bool Canceled() const override {
                    return State->Canceled;
                }

                void SendReply(TData& data) override;

                void SendError(TResponseError, const TString&) override {
                    // TODO
                }

                const TRequestHeader& RequestHeader() const noexcept {
                    return *reinterpret_cast<const TRequestHeader*>(Buf.Data());
                }

            private:
                TConnectionRef Conn;
                TBuffer Buf; //service-name + message-data
                TString Content_;
                TAtomic Replied_;

            public:
                TIntrusivePtr<TState> State;
            };

            class TConnection: public TThrRefBase {
            private:
                TConnection(TServer& srv, const TTcpSocketRef& sock)
                    : Srv_(srv)
                    , AS_(sock)
                    , Canceled_(false)
                    , RemoteHost_(NNeh::PrintHostByRfc(*AS_->RemoteEndpoint().Addr()))
                    , BuffSize_(TTcp2Options::InputBufferSize)
                    , Buff_(new char[BuffSize_])
                    , NeedCheckOutputQueue_(0)
                {
                    DBGOUT("TServer::TConnection()");
                }

            public:
                class TOutputBuffers: public TMultiBuffers {
                public:
                    void AddResponse(TRequestId reqId, TData& data) {
                        TResponseHeader* hdr = new (Allocate<TResponseHeader>()) TResponseHeader(reqId, TResponseHeader::Success, data.size());
                        ResponseData_.push_back(TAutoPtr<TData>(new TData()));
                        TData& movedData = *ResponseData_.back();
                        movedData.swap(data);
                        AddPart(hdr, sizeof(TResponseHeader));
                        AddPart(movedData.data(), movedData.size());
                        IOVec_ = TContIOVector(Parts_.data(), Parts_.size());
                    }

                    void AddError(TRequestId reqId, TResponseHeader::TErrorCode errCode) {
                        TResponseHeader* hdr = new (Allocate<TResponseHeader>()) TResponseHeader(reqId, errCode, 0);
                        AddPart(hdr, sizeof(TResponseHeader));
                        IOVec_ = TContIOVector(Parts_.data(), Parts_.size());
                    }

                    void Clear() {
                        TMultiBuffers::Clear();
                        ResponseData_.clear();
                    }

                private:
                    TVector<TAutoPtr<TData>> ResponseData_;
                };

                static void Create(TServer& srv, const TTcpSocketRef& sock) {
                    TConnectionRef conn(new TConnection(srv, sock));
                    conn->AS_->AsyncPollRead(std::bind(&TConnection::OnCanRead, conn, _1, _2), TTcp2Options::ServerInputDeadline);
                }

                ~TConnection() override {
                    DBGOUT("~TServer::TConnection(" << (!AS_ ? -666 : AS_->Native()) << ")");
                }

            private:
                void OnCanRead(const TErrorCode& ec, IHandlingContext& ctx) {
                    if (ec) {
                        OnError();
                    } else {
                        TErrorCode ec2;
                        OnReadSome(ec2, AS_->ReadSome(Buff_.Get(), BuffSize_, ec2), ctx);
                    }
                }

                void OnError() {
                    DBGOUT("Srv OnError(" << (!AS_ ? -666 : AS_->Native()) << ")"
                                          << " c=" << (size_t)this);
                    Canceled_ = true;
                    AS_->AsyncCancel();
                }

                void OnReadSome(const TErrorCode& ec, size_t amount, IHandlingContext& ctx) {
                    while (1) {
                        if (ec || !amount) {
                            OnError();
                            return;
                        }

                        try {
                            const char* buff = Buff_.Get();
                            size_t leftBytes = amount;
                            do {
                                size_t useBytes = Msg_.LoadFrom(buff, leftBytes);
                                leftBytes -= useBytes;
                                buff += useBytes;
                                if (Msg_.IsComplete()) {
                                    OnReceiveMessage();
                                }
                            } while (leftBytes);

                            if (amount == BuffSize_) {
                                //try decrease system calls, - re-run ReadSome if has full filled buffer
                                TErrorCode ecR;
                                amount = AS_->ReadSome(Buff_.Get(), BuffSize_, ecR);
                                if (!ecR) {
                                    continue;
                                }
                                if (ecR.Value() == EAGAIN || ecR.Value() == EWOULDBLOCK) {
                                    ctx.ContinueUseHandler();
                                } else {
                                    OnError();
                                }
                            } else {
                                ctx.ContinueUseHandler();
                            }
                        } catch (...) {
                            DBGOUT("exc. " << CurrentExceptionMessage());
                            OnError();
                        }
                        return;
                    }
                }

                void OnReceiveMessage() {
                    DBGOUT("OnReceiveMessage()");
                    const TBaseHeader& hdr = Msg_.BaseHeader();

                    if (hdr.Type == TBaseHeader::Request) {
                        TRequest* reqPtr = new TRequest(TConnectionRef(this), Msg_.Header(), Msg_.Content());
                        IRequestRef req(reqPtr);
                        ReqsState_[reqPtr->RequestHeader().Id] = reqPtr->State;
                        OnRequest(req);
                    } else if (hdr.Type == TBaseHeader::Cancel) {
                        OnCancelRequest(hdr.Id);
                    } else {
                        throw yexception() << "unsupported message type: " << (ui32)hdr.Type;
                    }
                    Msg_.Clear();
                    {
                        TRequestId reqId;
                        while (FinReqs_.Dequeue(&reqId)) {
                            ReqsState_.erase(reqId);
                        }
                    }
                }

                void OnRequest(IRequestRef& r) {
                    DBGOUT("OnRequest()");
                    Srv_.OnRequest(r);
                }

                void OnCancelRequest(TRequestId reqId) {
                    THashMap<TRequestId, TRequest::TStateRef>::iterator it = ReqsState_.find(reqId);
                    if (it == ReqsState_.end()) {
                        return;
                    }

                    it->second->Canceled = true;
                }

            public:
                class TOutputData {
                public:
                    TOutputData(TRequestId reqId)
                        : ReqId(reqId)
                    {
                    }

                    virtual ~TOutputData() {
                    }

                    virtual void MoveTo(TOutputBuffers& bufs) = 0;

                    TRequestId ReqId;
                };

                class TResponseData: public TOutputData {
                public:
                    TResponseData(TRequestId reqId, TData& data)
                        : TOutputData(reqId)
                    {
                        Data.swap(data);
                    }

                    void MoveTo(TOutputBuffers& bufs) override {
                        bufs.AddResponse(ReqId, Data);
                    }

                    TData Data;
                };

                class TResponseErrorData: public TOutputData {
                public:
                    TResponseErrorData(TRequestId reqId, TResponseHeader::TErrorCode errorCode)
                        : TOutputData(reqId)
                        , ErrorCode(errorCode)
                    {
                    }

                    void MoveTo(TOutputBuffers& bufs) override {
                        bufs.AddError(ReqId, ErrorCode);
                    }

                    TResponseHeader::TErrorCode ErrorCode;
                };

                //called non thread-safe (from client thread)
                void SendResponse(TRequestId reqId, TData& data) {
                    DBGOUT("SendResponse: " << reqId << " " << (size_t)~data << " c=" << (size_t)this);
                    TAutoPtr<TOutputData> od(new TResponseData(reqId, data));
                    OutputData_.Enqueue(od);
                    ProcessOutputQueue();
                }

                //called non thread-safe (from outside thread)
                void SendError(TRequestId reqId, TResponseHeader::TErrorCode err) {
                    DBGOUT("SendResponseError: " << reqId << " c=" << (size_t)this);
                    TAutoPtr<TOutputData> od(new TResponseErrorData(reqId, err));
                    OutputData_.Enqueue(od);
                    ProcessOutputQueue();
                }

                void ProcessOutputQueue() {
                    AtomicSet(NeedCheckOutputQueue_, 1);
                    if (OutputLock_.TryAquire()) {
                        SendMessages(false);
                        return;
                    }
                    DBGOUT("ProcessOutputQueue: !AquireOutputOwnership: " << (int)OutputLock_.IsFree());
                }

                //must be called only after success aquiring output
                void SendMessages(bool asioThread) {
                    DBGOUT("TServer::SendMessages(enter)");
                    try {
                        do {
                            AtomicUnlock(&NeedCheckOutputQueue_);
                            TAutoPtr<TOutputData> d;
                            while (OutputData_.Dequeue(&d)) {
                                d->MoveTo(OutputBuffers_);
                                if (!OutputBuffers_.HasFreeSpace()) {
                                    if (!FlushOutputBuffers(asioThread)) {
                                        return;
                                    }
                                }
                            }

                            if (OutputBuffers_.HasData()) {
                                if (!FlushOutputBuffers(asioThread)) {
                                    return;
                                }
                            }

                            OutputLock_.Release();

                            if (!AtomicGet(NeedCheckOutputQueue_)) {
                                DBGOUT("Server::SendMessages(exit2): " << (int)OutputLock_.IsFree());
                                return;
                            }
                        } while (OutputLock_.TryAquire());
                        DBGOUT("Server::SendMessages(exit1)");
                    } catch (...) {
                        OnError();
                    }
                }

                bool FlushOutputBuffers(bool asioThread) {
                    DBGOUT("FlushOutputBuffers: cnt=" << OutputBuffers_.GetIOvec()->Count() << " c=" << (size_t)this);
                    //TODO:reseach direct write efficiency
                    if (asioThread || TTcp2Options::ServerUseDirectWrite) {
                        TContIOVector& vec = *OutputBuffers_.GetIOvec();

                        vec.Proceed(AS_->WriteSome(vec));

                        if (vec.Complete()) {
                            OutputBuffers_.Clear();
                            //DBGOUT("WriteResponse: " << " c=" << (size_t)this);
                            return true;
                        }
                    }

                    //socket buffer filled - use async write for sending left data
                    DBGOUT("AsyncWriteResponse: "
                           << " [" << OutputBuffers_.GetIOvec()->Bytes() << "]"
                           << " c=" << (size_t)this);
                    AS_->AsyncWrite(OutputBuffers_.GetIOvec(), std::bind(&TConnection::OnSend, TConnectionRef(this), _1, _2, _3), TTcp2Options::ServerOutputDeadline);
                    return false;
                }

                void OnFinishRequest(TRequestId reqId) {
                    if (Y_LIKELY(!Canceled_)) {
                        FinReqs_.Enqueue(reqId);
                    }
                }

            private:
                void OnSend(const TErrorCode& ec, size_t amount, IHandlingContext&) {
                    Y_UNUSED(amount);
                    DBGOUT("TServer::OnSend(" << ec.Value() << ", " << amount << ")");
                    if (ec) {
                        OnError();
                    } else {
                        OutputBuffers_.Clear();
                        SendMessages(true);
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
                TServer& Srv_;
                TTcpSocketRef AS_;
                NAtomic::TBool Canceled_;
                TString RemoteHost_;

                //input
                size_t BuffSize_;
                TArrayHolder<char> Buff_;
                TTcp2Message Msg_;
                THashMap<TRequestId, TRequest::TStateRef> ReqsState_;
                TLockFreeQueue<TRequestId> FinReqs_;

                //output
                TOutputLock OutputLock_; //protect socket/buffers from simultaneous access from few threads
                TAtomic NeedCheckOutputQueue_;
                NNeh::TAutoLockFreeQueue<TOutputData> OutputData_;
                TOutputBuffers OutputBuffers_;
            };

            //////////// TServer /////////
        public:
            TServer(IOnRequest* cb, ui16 port)
                : EP_(TTcp2Options::AsioServerThreads)
                , CB_(cb)
            {
                TNetworkAddress addr(port);

                for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
                    TEndpoint ep(new NAddr::TAddrInfo(&*it));
                    TTcpAcceptorPtr a(new TTcpAcceptor(EA_.GetIOService()));
                    //DBGOUT("bind:" << ep.IpToString() << ":" << ep.Port());
                    a->Bind(ep);
                    a->Listen(TTcp2Options::Backlog);
                    StartAccept(a.Get());
                    A_.push_back(a);
                }
            }

            ~TServer() override {
                EA_.SyncShutdown(); //cancel accepting connections
                A_.clear();         //stop listening
                EP_.SyncShutdown(); //close all exist connections
            }

            void StartAccept(TTcpAcceptor* a) {
                const auto s = MakeAtomicShared<TTcpSocket>(EP_.Size() ? EP_.GetExecutor().GetIOService() : EA_.GetIOService());
                a->AsyncAccept(*s, std::bind(&TServer::OnAccept, this, a, s, _1, _2));
            }

            void OnAccept(TTcpAcceptor* a, TTcpSocketRef s, const TErrorCode& ec, IHandlingContext&) {
                if (Y_UNLIKELY(ec)) {
                    if (ec.Value() == ECANCELED) {
                        return;
                    } else if (ec.Value() == EMFILE || ec.Value() == ENFILE || ec.Value() == ENOMEM || ec.Value() == ENOBUFS) {
                        //reach some os limit, suspend accepting for preventing busyloop (100% cpu usage)
                        TSimpleSharedPtr<TDeadlineTimer> dt(new TDeadlineTimer(a->GetIOService()));
                        dt->AsyncWaitExpireAt(TDuration::Seconds(30), std::bind(&TServer::OnTimeoutSuspendAccept, this, a, dt, _1, _2));
                    } else {
                        Cdbg << "acc: " << ec.Text() << Endl;
                    }
                } else {
                    SetNonBlock(s->Native());
                    PrepareSocket(s->Native());
                    TConnection::Create(*this, s);
                }
                StartAccept(a); //continue accepting
            }

            void OnTimeoutSuspendAccept(TTcpAcceptor* a, TSimpleSharedPtr<TDeadlineTimer>, const TErrorCode& ec, IHandlingContext&) {
                if (!ec) {
                    DBGOUT("resume acceptor");
                    StartAccept(a);
                }
            }

            void OnRequest(IRequestRef& r) {
                try {
                    CB_->OnRequest(r);
                } catch (...) {
                    Cdbg << CurrentExceptionMessage() << Endl;
                }
            }

        private:
            TVector<TTcpAcceptorPtr> A_;
            TIOServiceExecutor EA_; //thread, where accepted incoming tcp connections
            TExecutorsPool EP_;     //threads, for process write/read data to/from tcp connections (if empty, use EA_ for r/w)
            IOnRequest* CB_;
        };

        TServer::TRequest::TRequest(const TConnectionRef& conn, TBuffer& buf, const TString& content)
            : Conn(conn)
            , Content_(content)
            , Replied_(0)
            , State(new TState())
        {
            DBGOUT("TServer::TRequest()");
            Buf.Swap(buf);
        }

        TServer::TRequest::~TRequest() {
            DBGOUT("TServer::~TRequest()");
            if (!AtomicGet(Replied_)) {
                Conn->SendError(RequestHeader().Id, TResponseHeader::EmptyReply);
            }
            Conn->OnFinishRequest(RequestHeader().Id);
        }

        TString TServer::TRequest::RemoteHost() const {
            return Conn->RemoteHost();
        }

        void TServer::TRequest::SendReply(TData& data) {
            do {
                if (AtomicCas(&Replied_, 1, 0)) {
                    Conn->SendResponse(RequestHeader().Id, data);
                    return;
                }
            } while (AtomicGet(Replied_) == 0);
        }

        class TProtocol: public IProtocol {
        public:
            inline TProtocol() {
                InitNetworkSubSystem();
            }

            IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
                return new TServer(cb, loc.GetPort());
            }

            THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
                return Singleton<TClient>()->Schedule(msg, fallback, ss);
            }

            TStringBuf Scheme() const noexcept override {
                return TStringBuf("tcp2");
            }

            bool SetOption(TStringBuf name, TStringBuf value) override {
                return TTcp2Options::Set(name, value);
            }
        };
    }
}

NNeh::IProtocol* NNeh::Tcp2Protocol() {
    return Singleton<NNehTcp2::TProtocol>();
}
