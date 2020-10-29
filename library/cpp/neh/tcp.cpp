#include "tcp.h"

#include "details.h"
#include "factory.h"
#include "location.h"
#include "pipequeue.h"
#include "utils.h"

#include <library/cpp/coroutine/listener/listen.h>
#include <library/cpp/coroutine/engine/events.h>
#include <library/cpp/coroutine/engine/sockpool.h>
#include <library/cpp/dns/cache.h>

#include <util/ysaveload.h>
#include <util/generic/buffer.h>
#include <util/generic/guid.h>
#include <util/generic/hash.h>
#include <util/generic/intrlist.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>
#include <util/system/unaligned_mem.h>
#include <util/stream/buffered.h>
#include <util/stream/mem.h>

using namespace NDns;
using namespace NNeh;

using TNehMessage = TMessage;

template <>
struct TSerializer<TGUID> {
    static inline void Save(IOutputStream* out, const TGUID& g) {
        out->Write(&g.dw, sizeof(g.dw));
    }

    static inline void Load(IInputStream* in, TGUID& g) {
        in->Load(&g.dw, sizeof(g.dw));
    }
};

namespace {
    namespace NNehTCP {
        typedef IOutputStream::TPart TPart;

        static inline ui64 LocalGuid(const TGUID& g) {
            return ReadUnaligned<ui64>(g.dw);
        }

        static inline TString LoadStroka(IInputStream& input, size_t len) {
            TString tmp;

            tmp.ReserveAndResize(len);
            input.Load(tmp.begin(), tmp.size());

            return tmp;
        }

        struct TParts: public TVector<TPart> {
            template <class T>
            inline void Push(const T& t) {
                Push(TPart(t));
            }

            inline void Push(const TPart& part) {
                if (part.len) {
                    push_back(part);
                }
            }

            inline void Clear() noexcept {
                clear();
            }
        };

        template <class T>
        struct TMessageQueue {
            inline TMessageQueue(TContExecutor* e)
                : Ev(e)
            {
            }

            template <class TPtr>
            inline void Enqueue(TPtr p) noexcept {
                L.PushBack(p.Release());
                Ev.Signal();
            }

            template <class TPtr>
            inline bool Dequeue(TPtr& p) noexcept {
                do {
                    if (TryDequeue(p)) {
                        return true;
                    }
                } while (Ev.WaitI() != ECANCELED);

                return false;
            }

            template <class TPtr>
            inline bool TryDequeue(TPtr& p) noexcept {
                if (L.Empty()) {
                    return false;
                }

                p.Reset(L.PopFront());

                return true;
            }

            inline TContExecutor* Executor() const noexcept {
                return Ev.Executor();
            }

            TIntrusiveListWithAutoDelete<T, TDelete> L;
            TContSimpleEvent Ev;
        };

        template <class Q, class C>
        inline bool Dequeue(Q& q, C& c, size_t len) {
            typename C::value_type t;
            size_t slen = 0;

            if (q.Dequeue(t)) {
                slen += t->Length();
                c.push_back(t);

                while (slen < len && q.TryDequeue(t)) {
                    slen += t->Length();
                    c.push_back(t);
                }

                return true;
            }

            return false;
        }

        struct TServer: public IRequester, public TContListener::ICallBack {
            struct TLink;
            typedef TIntrusivePtr<TLink> TLinkRef;

            struct TResponce: public TIntrusiveListItem<TResponce> {
                inline TResponce(const TLinkRef& link, TData& data, TStringBuf reqid)
                    : Link(link)
                {
                    Data.swap(data);

                    TMemoryOutput out(Buf, sizeof(Buf));

                    ::Save(&out, (ui32)(reqid.size() + Data.size()));
                    out.Write(reqid.data(), reqid.size());

                    Y_ASSERT(reqid.size() == 16);

                    Len = out.Buf() - Buf;
                }

                inline void Serialize(TParts& parts) {
                    parts.Push(TStringBuf(Buf, Len));
                    parts.Push(TStringBuf(Data.data(), Data.size()));
                }

                inline size_t Length() const noexcept {
                    return Len + Data.size();
                }

                TLinkRef Link;
                TData Data;
                char Buf[32];
                size_t Len;
            };

            typedef TAutoPtr<TResponce> TResponcePtr;

            struct TRequest: public IRequest {
                inline TRequest(const TLinkRef& link, IInputStream& in, size_t len)
                    : Link(link)
                {
                    Buf.Proceed(len);
                    in.Load(Buf.Data(), Buf.Size());
                    if ((ServiceBegin() - Buf.Data()) + ServiceLen() > Buf.Size()) {
                        throw yexception() << "invalid request (service len)";
                    }
                }

                TStringBuf Scheme() const override {
                    return TStringBuf("tcp");
                }

                TString RemoteHost() const override {
                    return Link->RemoteHost;
                }

                TStringBuf Service() const override {
                    return TStringBuf(ServiceBegin(), ServiceLen());
                }

                TStringBuf Data() const override {
                    return TStringBuf(Service().end(), Buf.End());
                }

                TStringBuf RequestId() const override {
                    return TStringBuf(Buf.Data(), 16);
                }

                bool Canceled() const override {
                    //TODO
                    return false;
                }

                void SendReply(TData& data) override {
                    Link->P->Schedule(new TResponce(Link, data, RequestId()));
                }

                void SendError(TResponseError, const TString&) override {
                    // TODO
                }

                size_t ServiceLen() const noexcept {
                    const char* ptr = RequestId().end();
                    return *(ui32*)ptr;
                }

                const char* ServiceBegin() const noexcept {
                    return RequestId().end() + sizeof(ui32);
                }

                TBuffer Buf;
                TLinkRef Link;
            };

            struct TLink: public TAtomicRefCount<TLink> {
                inline TLink(TServer* parent, const TAcceptFull& a)
                    : P(parent)
                    , MQ(Executor())
                {
                    S.Swap(*a.S);
                    SetNoDelay(S, true);

                    RemoteHost = PrintHostByRfc(*GetPeerAddr(S));

                    TLinkRef self(this);

                    Executor()->Create<TLink, &TLink::RecvCycle>(this, "recv");
                    Executor()->Create<TLink, &TLink::SendCycle>(this, "send");

                    Executor()->Running()->Yield();
                }

                inline void Enqueue(TResponcePtr res) {
                    MQ.Enqueue(res);
                }

                inline TContExecutor* Executor() const noexcept {
                    return P->E.Get();
                }

                void SendCycle(TCont* c) {
                    TLinkRef self(this);

                    try {
                        DoSendCycle(c);
                    } catch (...) {
                        Cdbg << "neh/tcp/1: " << CurrentExceptionMessage() << Endl;
                    }
                }

                inline void DoSendCycle(TCont* c) {
                    TVector<TResponcePtr> responses;
                    TParts parts;

                    while (Dequeue(MQ, responses, 7000)) {
                        for (size_t i = 0; i < responses.size(); ++i) {
                            responses[i]->Serialize(parts);
                        }

                        {
                            TContIOVector iovec(parts.data(), parts.size());
                            NCoro::WriteVectorI(c, S, &iovec);
                        }

                        parts.Clear();
                        responses.clear();
                    }
                }

                void RecvCycle(TCont* c) {
                    TLinkRef self(this);

                    try {
                        DoRecvCycle(c);
                    } catch (...) {
                        if (!c->Cancelled()) {
                            Cdbg << "neh/tcp/2: " << CurrentExceptionMessage() << Endl;
                        }
                    }
                }

                inline void DoRecvCycle(TCont* c) {
                    TContIO io(S, c);
                    TBufferedInput input(&io, 8192 * 4);

                    while (true) {
                        ui32 len;

                        try {
                            ::Load(&input, len);
                        } catch (TLoadEOF&) {
                            return;
                        }

                        P->CB->OnRequest(new TRequest(this, input, len));
                    }
                }

                TServer* P;
                TMessageQueue<TResponce> MQ;
                TSocketHolder S;
                TString RemoteHost;
            };

            inline TServer(IOnRequest* cb, ui16 port)
                : CB(cb)
                , Addr(port)
            {
                Thrs.push_back(Spawn<TServer, &TServer::Run>(this));
            }

            ~TServer() override {
                Schedule(nullptr);

                for (size_t i = 0; i < Thrs.size(); ++i) {
                    Thrs[i]->Join();
                }
            }

            void Run() {
                E = MakeHolder<TContExecutor>(RealStackSize(32000));
                THolder<TContListener> L(new TContListener(this, E.Get(), TContListener::TOptions().SetDeferAccept(true)));
                //SetHighestThreadPriority();
                L->Bind(Addr);
                E->Create<TServer, &TServer::RunDispatcher>(this, "dispatcher");
                L->Listen();
                E->Execute();
            }

            void OnAcceptFull(const TAcceptFull& a) override {
                //I love such code
                new TLink(this, a);
            }

            void OnError() override {
                Cerr << CurrentExceptionMessage() << Endl;
            }

            inline void Schedule(TResponcePtr res) {
                PQ.EnqueueSafe(res);
            }

            void RunDispatcher(TCont* c) {
                while (true) {
                    TResponcePtr res;

                    PQ.DequeueSafe(c, res);

                    if (!res) {
                        break;
                    }

                    TLinkRef link = res->Link;

                    link->Enqueue(res);
                }

                c->Executor()->Abort();
            }
            THolder<TContExecutor> E;
            IOnRequest* CB;
            TNetworkAddress Addr;
            TOneConsumerPipeQueue<TResponce> PQ;
            TVector<TThreadRef> Thrs;
        };

        struct TClient {
            struct TRequest: public TIntrusiveListItem<TRequest> {
                inline TRequest(const TSimpleHandleRef& hndl, const TNehMessage& msg)
                    : Hndl(hndl)
                    , Msg(msg)
                    , Loc(Msg.Addr)
                    , RI(CachedThrResolve(TResolveInfo(Loc.Host, Loc.GetPort())))
                {
                    CreateGuid(&Guid);
                }

                inline void Serialize(TParts& parts) {
                    TMemoryOutput out(Buf, sizeof(Buf));

                    ::Save(&out, (ui32)MsgLen());
                    ::Save(&out, Guid);
                    ::Save(&out, (ui32) Loc.Service.size());

                    if (Loc.Service.size() > out.Avail()) {
                        parts.Push(TStringBuf(Buf, out.Buf()));
                        parts.Push(Loc.Service);
                    } else {
                        out.Write(Loc.Service.data(), Loc.Service.size());
                        parts.Push(TStringBuf(Buf, out.Buf()));
                    }

                    parts.Push(Msg.Data);
                }

                inline size_t Length() const noexcept {
                    return sizeof(ui32) + MsgLen();
                }

                inline size_t MsgLen() const noexcept {
                    return sizeof(Guid.dw) + sizeof(ui32) + Loc.Service.size() + Msg.Data.size();
                }

                void OnError(const TString& errText) {
                    Hndl->NotifyError(errText);
                }

                TSimpleHandleRef Hndl;
                TNehMessage Msg;
                TGUID Guid;
                const TParsedLocation Loc;
                const TResolvedHost* RI;
                char Buf[128];
            };

            typedef TAutoPtr<TRequest> TRequestPtr;

            struct TChannel {
                struct TLink: public TIntrusiveListItem<TLink>, public TSimpleRefCount<TLink> {
                    inline TLink(TChannel* parent)
                        : P(parent)
                    {
                        Executor()->Create<TLink, &TLink::SendCycle>(this, "send");
                    }

                    void SendCycle(TCont* c) {
                        TIntrusivePtr<TLink> self(this);

                        try {
                            DoSendCycle(c);
                            OnError("shutdown");
                        } catch (...) {
                            OnError(CurrentExceptionMessage());
                        }

                        Unlink();
                    }

                    inline void DoSendCycle(TCont* c) {
                        if (int ret = NCoro::ConnectI(c, S, P->RI->Addr)) {
                            ythrow TSystemError(ret) << "can't connect";
                        }
                        SetNoDelay(S, true);
                        Executor()->Create<TLink, &TLink::RecvCycle>(this, "recv");

                        TVector<TRequestPtr> reqs;
                        TParts parts;

                        while (Dequeue(P->Q, reqs, 7000)) {
                            for (size_t i = 0; i < reqs.size(); ++i) {
                                TRequestPtr& req = reqs[i];

                                req->Serialize(parts);
                                InFly[LocalGuid(req->Guid)] = req;
                            }

                            {
                                TContIOVector vec(parts.data(), parts.size());
                                NCoro::WriteVectorI(c, S, &vec);
                            }

                            reqs.clear();
                            parts.Clear();
                        }
                    }

                    void RecvCycle(TCont* c) {
                        TIntrusivePtr<TLink> self(this);

                        try {
                            DoRecvCycle(c);
                            OnError("service close connection");
                        } catch (...) {
                            OnError(CurrentExceptionMessage());
                        }
                    }

                    inline void DoRecvCycle(TCont* c) {
                        TContIO io(S, c);
                        TBufferedInput input(&io, 8192 * 4);

                        while (true) {
                            ui32 len;
                            TGUID g;

                            try {
                                ::Load(&input, len);
                            } catch (TLoadEOF&) {
                                return;
                            }
                            ::Load(&input, g);
                            const TString data(LoadStroka(input, len - sizeof(g.dw)));

                            TInFly::iterator it = InFly.find(LocalGuid(g));

                            if (it == InFly.end()) {
                                continue;
                            }

                            TRequestPtr req = it->second;

                            InFly.erase(it);
                            req->Hndl->NotifyResponse(data);
                        }
                    }

                    inline TContExecutor* Executor() const noexcept {
                        return P->Q.Executor();
                    }

                    void OnError(const TString& errText) {
                        for (auto& it : InFly) {
                            it.second->OnError(errText);
                        }
                        InFly.clear();

                        TRequestPtr req;
                        while (P->Q.TryDequeue(req)) {
                            req->OnError(errText);
                        }
                    }

                    TChannel* P;
                    TSocketHolder S;
                    typedef THashMap<ui64, TRequestPtr> TInFly;
                    TInFly InFly;
                };

                inline TChannel(TContExecutor* e, const TResolvedHost* ri)
                    : Q(e)
                    , RI(ri)
                {
                }

                inline void Enqueue(TRequestPtr req) {
                    Q.Enqueue(req);

                    if (Links.Empty()) {
                        for (size_t i = 0; i < 1; ++i) {
                            SpawnLink();
                        }
                    }
                }

                inline void SpawnLink() {
                    Links.PushBack(new TLink(this));
                }

                TMessageQueue<TRequest> Q;
                TIntrusiveList<TLink> Links;
                const TResolvedHost* RI;
            };

            typedef TAutoPtr<TChannel> TChannelPtr;

            inline TClient() {
                Thr = Spawn<TClient, &TClient::RunExecutor>(this);
            }

            inline ~TClient() {
                Reqs.Enqueue(nullptr);
                Thr->Join();
            }

            inline THandleRef Schedule(const TNehMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) {
                TSimpleHandleRef ret(new TSimpleHandle(fallback, msg, !ss ? nullptr : new TStatCollector(ss)));

                Reqs.Enqueue(new TRequest(ret, msg));

                return ret.Get();
            }

            void RunExecutor() {
                //SetHighestThreadPriority();
                TContExecutor e(RealStackSize(32000));

                e.Create<TClient, &TClient::RunDispatcher>(this, "dispatcher");
                e.Execute();
            }

            void RunDispatcher(TCont* c) {
                TRequestPtr req;

                while (true) {
                    Reqs.DequeueSafe(c, req);

                    if (!req) {
                        break;
                    }

                    TChannelPtr& ch = Channels.Get(req->RI->Id);

                    if (!ch) {
                        ch.Reset(new TChannel(c->Executor(), req->RI));
                    }

                    ch->Enqueue(req);
                }

                c->Executor()->Abort();
            }

            TThreadRef Thr;
            TOneConsumerPipeQueue<TRequest> Reqs;
            TSocketMap<TChannelPtr> Channels;
        };

        struct TMultiClient {
            inline TMultiClient()
                : Next(0)
            {
                for (size_t i = 0; i < 2; ++i) {
                    Clients.push_back(new TClient());
                }
            }

            inline THandleRef Schedule(const TNehMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) {
                return Clients[AtomicIncrement(Next) % Clients.size()]->Schedule(msg, fallback, ss);
            }

            TVector<TAutoPtr<TClient>> Clients;
            TAtomic Next;
        };

#if 0
    static inline TMultiClient* Client() {
        return Singleton<NNehTCP::TMultiClient>();
    }
#else
        static inline TClient* Client() {
            return Singleton<NNehTCP::TClient>();
        }
#endif

        class TTcpProtocol: public IProtocol {
        public:
            inline TTcpProtocol() {
                InitNetworkSubSystem();
            }

            IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
                return new TServer(cb, loc.GetPort());
            }

            THandleRef ScheduleRequest(const TNehMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
                return Client()->Schedule(msg, fallback, ss);
            }

            TStringBuf Scheme() const noexcept override {
                return TStringBuf("tcp");
            }
        };
    }
}

IProtocol* NNeh::TcpProtocol() {
    return Singleton<NNehTCP::TTcpProtocol>();
}
