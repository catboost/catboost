#include "details.h"
#include "factory.h"
#include "http_common.h"
#include "location.h"
#include "multi.h"
#include "netliba.h"
#include "netliba_udp_http.h"
#include "lfqueue.h"
#include "utils.h"

#include <library/cpp/dns/cache.h>

#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>

#include <atomic>

using namespace NDns;
using namespace NNeh;
namespace NNeh {
    size_t TNetLibaOptions::ClientThreads = 4;
    TDuration TNetLibaOptions::AckTailEffect = TDuration::Seconds(30);

    bool TNetLibaOptions::Set(TStringBuf name, TStringBuf value) {
#define NETLIBA_TRY_SET(optType, optName)     \
    if (name == TStringBuf(#optName)) {      \
        optName = FromString<optType>(value); \
    }

        NETLIBA_TRY_SET(size_t, ClientThreads)
        else NETLIBA_TRY_SET(TDuration, AckTailEffect) else {
            return false;
        }
        return true;
    }
}

namespace {
    namespace NNetLiba {
        using namespace NNetliba;
        using namespace NNehNetliba;

        typedef NNehNetliba::IRequester INetLibaRequester;
        typedef TAutoPtr<TUdpHttpRequest> TUdpHttpRequestPtr;
        typedef TAutoPtr<TUdpHttpResponse> TUdpHttpResponsePtr;

        static inline const addrinfo* FindIPBase(const TNetworkAddress* addr, int family) {
            for (TNetworkAddress::TIterator it = addr->Begin(); it != addr->End(); ++it) {
                if (it->ai_family == family) {
                    return &*it;
                }
            }

            return nullptr;
        }

        static inline const sockaddr_in6& FindIP(const TNetworkAddress* addr) {
            //prefer ipv6
            const addrinfo* ret = FindIPBase(addr, AF_INET6);

            if (!ret) {
                ret = FindIPBase(addr, AF_INET);
            }

            if (!ret) {
                ythrow yexception() << "ip not supported by " << *addr;
            }

            return *(const sockaddr_in6*)(ret->ai_addr);
        }

        class TLastAckTimes {
            struct TTimeVal {
                TTimeVal()
                    : Val(0)
                {
                }

                std::atomic<TInstant::TValue> Val;
            };

        public:
            TInstant::TValue Get(size_t idAddr) {
                return Tm_.Get(idAddr).Val.load(std::memory_order_acquire);
            }

            void Set(size_t idAddr) {
                Tm_.Get(idAddr).Val.store(TInstant::Now().GetValue(), std::memory_order_release);
            }

            static TLastAckTimes& Common() {
                return *Singleton<TLastAckTimes>();
            }

        private:
            NNeh::NHttp::TLockFreeSequence<TTimeVal> Tm_;
        };

        class TRequest: public TSimpleHandle {
        public:
            inline TRequest(TIntrusivePtr<INetLibaRequester>& r, size_t idAddr, const TMessage& msg, IOnRecv* cb, TStatCollector* s)
                : TSimpleHandle(cb, msg, s)
                , R_(r)
                , IdAddr_(idAddr)
                , Notified_(false)
            {
                CreateGuid(&Guid_);
            }

            void Cancel() noexcept override {
                TSimpleHandle::Cancel();
                R_->CancelRequest(Guid_);
            }

            inline const TString& Addr() const noexcept {
                return Message().Addr;
            }

            inline const TGUID& Guid() const noexcept {
                return Guid_;
            }

            //return false if already notifie
            inline bool SetNotified() noexcept {
                bool ret = Notified_;
                Notified_ = true;
                return !ret;
            }

            void OnSend() {
                if (TNetLibaOptions::AckTailEffect.GetValue() && TLastAckTimes::Common().Get(IdAddr_) + TNetLibaOptions::AckTailEffect.GetValue() > TInstant::Now().GetValue()) {
                    //fake(predicted) completing detection
                    SetSendComplete();
                }
            }

            void OnRequestAck() {
                if (TNetLibaOptions::AckTailEffect.GetValue()) {
                    TLastAckTimes::Common().Set(IdAddr_);
                }
                SetSendComplete();
            }

        private:
            TIntrusivePtr<INetLibaRequester> R_;
            size_t IdAddr_;
            TGUID Guid_;
            bool Notified_;
        };

        typedef TIntrusivePtr<TRequest> TRequestRef;

        class TNetLibaBus {
            class TEventsHandler: public IEventsCollector {
                typedef THashMap<TGUID, TRequestRef, TGUIDHash> TInFly;

            public:
                inline void OnSend(TRequestRef& req) {
                    Q_.Enqueue(req);
                    req->OnSend();
                }

            private:
                void UpdateInFly() {
                    TRequestRef req;

                    while (Q_.Dequeue(&req)) {
                        if (!req) {
                            return;
                        }

                        InFly_[req->Guid()] = req;
                    }
                }

                void AddRequest(TUdpHttpRequest* req) override {
                    //ignore received requests in client
                    delete req;
                }

                void AddResponse(TUdpHttpResponse* resp) override {
                    TUdpHttpResponsePtr ptr(resp);

                    UpdateInFly();
                    TInFly::iterator it = InFly_.find(resp->ReqId);

                    Y_ABORT_UNLESS(it != InFly_.end(), "incorrect incoming message");

                    TRequestRef& req = it->second;

                    if (req->SetNotified()) {
                        if (resp->Ok == TUdpHttpResponse::OK) {
                            req->NotifyResponse(TString(resp->Data.data(), resp->Data.size()));
                        } else {
                            if (resp->Ok == TUdpHttpResponse::CANCELED) {
                                req->NotifyError(new TError(resp->Error, TError::Cancelled));
                            } else {
                                req->NotifyError(new TError(resp->Error));
                            }
                        }
                    }

                    InFly_.erase(it);
                }

                void AddCancel(const TGUID& guid) override {
                    UpdateInFly();
                    TInFly::iterator it = InFly_.find(guid);

                    if (it != InFly_.end() && it->second->SetNotified()) {
                        it->second->NotifyError("Canceled (before ack)");
                    }
                }

                void AddRequestAck(const TGUID& guid) override {
                    UpdateInFly();
                    TInFly::iterator it = InFly_.find(guid);

                    Y_ABORT_UNLESS(it != InFly_.end(), "incorrect complete notification");

                    it->second->OnRequestAck();
                }

            private:
                TLockFreeQueue<TRequestRef> Q_;
                TInFly InFly_;
            };

            struct TClientThread {
                TClientThread(int physicalCpu)
                    : EH_(new TEventsHandler())
                    , R_(CreateHttpUdpRequester(0, IEventsCollectorRef(EH_.Get()), physicalCpu))
                {
                    R_->EnableReportRequestAck();
                }

                ~TClientThread() {
                    R_->StopNoWait();
                }

                TIntrusivePtr<TEventsHandler> EH_;
                TIntrusivePtr<INetLibaRequester> R_;
            };

        public:
            TNetLibaBus() {
                for (size_t i = 0; i < TNetLibaOptions::ClientThreads; ++i) {
                    Clnt_.push_back(new TClientThread(i));
                }
            }

            inline THandleRef Schedule(const TMessage& msg, IOnRecv* cb, TServiceStatRef& ss) {
                TParsedLocation loc(msg.Addr);
                TUdpAddress addr;

                const TResolvedHost* resHost = CachedResolve(TResolveInfo(loc.Host, loc.GetPort()));
                GetUdpAddress(&addr, FindIP(&resHost->Addr));

                TClientThread& clnt = *Clnt_[resHost->Id % Clnt_.size()];
                TIntrusivePtr<INetLibaRequester> rr = clnt.R_;
                TRequestRef req(new TRequest(rr, resHost->Id, msg, cb, !ss ? nullptr : new TStatCollector(ss)));

                clnt.EH_->OnSend(req);
                rr->SendRequest(addr, ToString(loc.Service), msg.Data, req->Guid());

                return THandleRef(req.Get());
            }

        private:
            TVector<TAutoPtr<TClientThread>> Clnt_;
        };

        //server
        class TRequester: public TThrRefBase {
            struct TSrvRequestState: public TAtomicRefCount<TSrvRequestState> {
                TSrvRequestState()
                    : Canceled(false)
                {
                }

                TAtomicBool Canceled;
            };

            class TRequest: public IRequest {
            public:
                inline TRequest(TAutoPtr<TUdpHttpRequest> req, TIntrusivePtr<TSrvRequestState> state, TRequester* parent)
                    : R_(req)
                    , S_(state)
                    , P_(parent)
                {
                }

                ~TRequest() override {
                    if (!!P_) {
                        P_->RequestProcessed(this);
                    }
                }

                TStringBuf Scheme() const override {
                    return TStringBuf("netliba");
                }

                TString RemoteHost() const override {
                    if (!H_) {
                        TUdpAddress tmp(R_->PeerAddress);
                        tmp.Scope = 0; //discard scope from serialized addr

                        TString addr = GetAddressAsString(tmp);

                        TStringBuf host, port;

                        TStringBuf(addr).RSplit(':', host, port);
                        H_ = host;
                    }
                    return H_;
                }

                TStringBuf Service() const override {
                    return TStringBuf(R_->Url.c_str(), R_->Url.length());
                }

                TStringBuf Data() const override {
                    return TStringBuf((const char*)R_->Data.data(), R_->Data.size());
                }

                TStringBuf RequestId() const override {
                    const TGUID& g = R_->ReqId;

                    return TStringBuf((const char*)g.dw, sizeof(g.dw));
                }

                bool Canceled() const override {
                    return S_->Canceled;
                }

                void SendReply(TData& data) override {
                    TIntrusivePtr<TRequester> p;
                    p.Swap(P_);
                    if (!!p) {
                        if (!Canceled()) {
                            p->R_->SendResponse(R_->ReqId, &data);
                        }
                        p->RequestProcessed(this);
                    }
                }

                void SendError(TResponseError, const TString&) override {
                    // TODO
                }

                inline const TGUID& RequestGuid() const noexcept {
                    return R_->ReqId;
                }

            private:
                TAutoPtr<TUdpHttpRequest> R_;
                mutable TString H_;
                TIntrusivePtr<TSrvRequestState> S_;
                TIntrusivePtr<TRequester> P_;
            };

            class TEventsHandler: public IEventsCollector {
            public:
                TEventsHandler(TRequester* parent)
                {
                    P_.store(parent, std::memory_order_release);
                }

                void RequestProcessed(const TRequest* r) {
                    FinishedReqs_.Enqueue(r->RequestGuid());
                }

                //thread safe method for disable proxy callbacks to parent (OnRequest(...))
                void SyncStop() {
                    P_.store(nullptr, std::memory_order_release);
                    while (!RequesterPtrPotector_.TryAcquire()) {
                        Sleep(TDuration::MicroSeconds(100));
                    }
                    RequesterPtrPotector_.Release();
                }

            private:
                typedef THashMap<TGUID, TIntrusivePtr<TSrvRequestState>, TGUIDHash> TStatesInProcessRequests;

                void AddRequest(TUdpHttpRequest* req) override {
                    TUdpHttpRequestPtr ptr(req);

                    TSrvRequestState* state = new TSrvRequestState();

                    InProcess_[req->ReqId] = state;
                    try {
                        TGuard<TSpinLock> m(RequesterPtrPotector_);
                        if (TRequester* p = P_.load(std::memory_order_acquire)) {
                            p->OnRequest(ptr, state); //move req. owning to parent
                        }
                    } catch (...) {
                        Cdbg << "ignore exc.: " << CurrentExceptionMessage() << Endl;
                    }
                }

                void AddResponse(TUdpHttpResponse*) override {
                    Y_ABORT("unexpected response in neh netliba server");
                }

                void AddCancel(const TGUID& guid) override {
                    UpdateInProcess();
                    TStatesInProcessRequests::iterator ustate = InProcess_.find(guid);
                    if (ustate != InProcess_.end())
                        ustate->second->Canceled = true;
                }

                void AddRequestAck(const TGUID&) override {
                    Y_ABORT("unexpected acc in neh netliba server");
                }

                void UpdateInProcess() {
                    TGUID guid;

                    while (FinishedReqs_.Dequeue(&guid)) {
                        InProcess_.erase(guid);
                    }
                }

            private:
                TLockFreeStack<TGUID> FinishedReqs_; //processed requests (responded or destroyed)
                TStatesInProcessRequests InProcess_;
                TSpinLock RequesterPtrPotector_;
                std::atomic<TRequester*> P_;
            };

        public:
            inline TRequester(IOnRequest* cb, ui16 port)
                : CB_(cb)
                , EH_(new TEventsHandler(this))
                , R_(CreateHttpUdpRequester(port, EH_.Get()))
            {
                R_->EnableReportRequestCancel();
            }

            ~TRequester() override {
                Shutdown();
            }

            void Shutdown() noexcept {
                if (!Shutdown_) {
                    Shutdown_ = true;
                    R_->StopNoWait();
                    EH_->SyncStop();
                }
            }

            void OnRequest(TUdpHttpRequestPtr req, TSrvRequestState* state) {
                CB_->OnRequest(new TRequest(req, state, this));
            }

            void RequestProcessed(const TRequest* r) {
                EH_->RequestProcessed(r);
            }

        private:
            IOnRequest* CB_;
            TIntrusivePtr<TEventsHandler> EH_;
            TIntrusivePtr<INetLibaRequester> R_;
            bool Shutdown_ = false;
        };

        typedef TIntrusivePtr<TRequester> TRequesterRef;

        class TRequesterAutoShutdown: public NNeh::IRequester {
        public:
            TRequesterAutoShutdown(const TRequesterRef& r)
                : R_(r)
            {
            }

            ~TRequesterAutoShutdown() override {
                R_->Shutdown();
            }

        private:
            TRequesterRef R_;
        };

        class TProtocol: public IProtocol {
        public:
            THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
                return Singleton<TNetLibaBus>()->Schedule(msg, fallback, ss);
            }

            NNeh::IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
                TRequesterRef r(new TRequester(cb, loc.GetPort()));
                return new TRequesterAutoShutdown(r);
            }

            TStringBuf Scheme() const noexcept override {
                return TStringBuf("netliba");
            }
        };
    }
}

IProtocol* NNeh::NetLibaProtocol() {
    return Singleton<NNetLiba::TProtocol>();
}
