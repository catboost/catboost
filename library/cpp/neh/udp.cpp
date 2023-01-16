#include "udp.h"
#include "details.h"
#include "neh.h"
#include "location.h"
#include "utils.h"
#include "factory.h"

#include <library/cpp/dns/cache.h>

#include <util/network/socket.h>
#include <util/network/address.h>
#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/buffer.h>
#include <util/generic/singleton.h>
#include <util/digest/murmur.h>
#include <util/random/random.h>
#include <util/ysaveload.h>
#include <util/system/thread.h>
#include <util/system/pipe.h>
#include <util/system/error.h>
#include <util/stream/mem.h>
#include <util/stream/buffer.h>
#include <util/string/cast.h>

using namespace NNeh;
using namespace NDns;
using namespace NAddr;

namespace {
    namespace NUdp {
        enum EPacketType {
            PT_REQUEST = 1,
            PT_RESPONSE = 2,
            PT_STOP = 3,
            PT_TIMEOUT = 4
        };

        struct TUdpHandle: public TNotifyHandle {
            inline TUdpHandle(IOnRecv* r, const TMessage& msg, TStatCollector* sc) noexcept
                : TNotifyHandle(r, msg, sc)
            {
            }

            void Cancel() noexcept override {
                THandle::Cancel(); //inform stat collector
            }

            bool MessageSendedCompletely() const noexcept override {
                //TODO
                return true;
            }
        };

        static inline IRemoteAddrPtr GetSendAddr(SOCKET s) {
            IRemoteAddrPtr local = GetSockAddr(s);
            const sockaddr* addr = local->Addr();

            switch (addr->sa_family) {
                case AF_INET: {
                    const TIpAddress a = *(const sockaddr_in*)addr;

                    return MakeHolder<TIPv4Addr>(TIpAddress(InetToHost(INADDR_LOOPBACK), a.Port()));
                }

                case AF_INET6: {
                    sockaddr_in6 a = *(const sockaddr_in6*)addr;

                    a.sin6_addr = in6addr_loopback;

                    return MakeHolder<TIPv6Addr>(a);
                }
            }

            ythrow yexception() << "unsupported";
        }

        typedef ui32 TCheckSum;

        static inline TString GenerateGuid() {
            const ui64 res[2] = {
                RandomNumber<ui64>(), RandomNumber<ui64>()};

            return TString((const char*)res, sizeof(res));
        }

        static inline TCheckSum Sum(const TStringBuf& s) noexcept {
            return HostToInet(MurmurHash<TCheckSum>(s.data(), s.size()));
        }

        struct TPacket;

        template <class T>
        static inline void Serialize(TPacket& p, const T& t);

        struct TPacket {
            inline TPacket(IRemoteAddrPtr addr)
                : Addr(std::move(addr))
            {
            }

            template <class T>
            inline TPacket(const T& t, IRemoteAddrPtr addr)
                : Addr(std::move(addr))
            {
                NUdp::Serialize(*this, t);
            }

            inline TPacket(TSocketHolder& s, TBuffer& tmp) {
                TAutoPtr<TOpaqueAddr> addr(new TOpaqueAddr());

            retry_on_intr : {
                const int rv = recvfrom(s, tmp.Data(), tmp.size(), MSG_WAITALL, addr->MutableAddr(), addr->LenPtr());

                if (rv < 0) {
                    int err = LastSystemError();
                    if (err == EAGAIN || err == EWOULDBLOCK) {
                        Data.Resize(sizeof(TCheckSum) + 1);
                        *(Data.data() + sizeof(TCheckSum)) = static_cast<char>(PT_TIMEOUT);
                    } else if (err == EINTR) {
                        goto retry_on_intr;
                    } else {
                        ythrow TSystemError() << "recv failed";
                    }
                } else {
                    Data.Append(tmp.Data(), (size_t)rv);
                    Addr.Reset(addr.Release());
                    CheckSign();
                }
            }
            }

            inline void SendTo(TSocketHolder& s) {
                Sign();

                if (sendto(s, Data.data(), Data.size(), 0, Addr->Addr(), Addr->Len()) < 0) {
                    Cdbg << LastSystemErrorText() << Endl;
                }
            }

            IRemoteAddrPtr Addr;
            TBuffer Data;

            inline void Sign() {
                const TCheckSum sum = CalcSign();

                memcpy(Data.Data(), &sum, sizeof(sum));
            }

            inline char Type() const noexcept {
                return *(Data.data() + sizeof(TCheckSum));
            }

            inline void CheckSign() const {
                if (Data.size() < 16) {
                    ythrow yexception() << "small packet";
                }

                if (StoredSign() != CalcSign()) {
                    ythrow yexception() << "bad checksum";
                }
            }

            inline TCheckSum StoredSign() const noexcept {
                TCheckSum sum;

                memcpy(&sum, Data.Data(), sizeof(sum));

                return sum;
            }

            inline TCheckSum CalcSign() const noexcept {
                return Sum(Body());
            }

            inline TStringBuf Body() const noexcept {
                return TStringBuf(Data.data() + sizeof(TCheckSum), Data.End());
            }
        };

        typedef TAutoPtr<TPacket> TPacketRef;

        class TPacketInput: public TMemoryInput {
        public:
            inline TPacketInput(const TPacket& p)
                : TMemoryInput(p.Body().data(), p.Body().size())
            {
            }
        };

        class TPacketOutput: public TBufferOutput {
        public:
            inline TPacketOutput(TPacket& p)
                : TBufferOutput(p.Data)
            {
                p.Data.Proceed(sizeof(TCheckSum));
            }
        };

        template <class T>
        static inline void Serialize(TPacketOutput* out, const T& t) {
            Save(out, t.Type());
            t.Serialize(out);
        }

        template <class T>
        static inline void Serialize(TPacket& p, const T& t) {
            TPacketOutput out(p);

            NUdp::Serialize(&out, t);
        }

        namespace NPrivate {
            template <class T>
            static inline void Deserialize(TPacketInput* in, T& t) {
                char type;
                Load(in, type);

                if (type != t.Type()) {
                    ythrow yexception() << "unsupported packet";
                }

                t.Deserialize(in);
            }

            template <class T>
            static inline void Deserialize(const TPacket& p, T& t) {
                TPacketInput in(p);

                Deserialize(&in, t);
            }
        }

        struct TRequestPacket {
            TString Guid;
            TString Service;
            TString Data;

            inline TRequestPacket(const TPacket& p) {
                NPrivate::Deserialize(p, *this);
            }

            inline TRequestPacket(const TString& srv, const TString& data)
                : Guid(GenerateGuid())
                , Service(srv)
                , Data(data)
            {
            }

            inline char Type() const noexcept {
                return static_cast<char>(PT_REQUEST);
            }

            inline void Serialize(TPacketOutput* out) const {
                Save(out, Guid);
                Save(out, Service);
                Save(out, Data);
            }

            inline void Deserialize(TPacketInput* in) {
                Load(in, Guid);
                Load(in, Service);
                Load(in, Data);
            }
        };

        template <class TStore>
        struct TResponsePacket {
            TString Guid;
            TStore Data;

            inline TResponsePacket(const TString& guid, TStore& data)
                : Guid(guid)
            {
                Data.swap(data);
            }

            inline TResponsePacket(const TPacket& p) {
                NPrivate::Deserialize(p, *this);
            }

            inline char Type() const noexcept {
                return static_cast<char>(PT_RESPONSE);
            }

            inline void Serialize(TPacketOutput* out) const {
                Save(out, Guid);
                Save(out, Data);
            }

            inline void Deserialize(TPacketInput* in) {
                Load(in, Guid);
                Load(in, Data);
            }
        };

        struct TStopPacket {
            inline char Type() const noexcept {
                return static_cast<char>(PT_STOP);
            }

            inline void Serialize(TPacketOutput* out) const {
                Save(out, TString("stop packet"));
            }
        };

        struct TBindError: public TSystemError {
        };

        struct TSocketDescr {
            inline TSocketDescr(TSocketHolder& s, int family)
                : S(s.Release())
                , Family(family)
            {
            }

            TSocketHolder S;
            int Family;
        };

        typedef TAutoPtr<TSocketDescr> TSocketRef;
        typedef TVector<TSocketRef> TSockets;

        static inline void CreateSocket(TSocketHolder& s, const IRemoteAddr& addr) {
            TSocketHolder res(socket(addr.Addr()->sa_family, SOCK_DGRAM, IPPROTO_UDP));

            if (!res) {
                ythrow TSystemError() << "can not create socket";
            }

            FixIPv6ListenSocket(res);

            if (bind(res, addr.Addr(), addr.Len()) != 0) {
                ythrow TBindError() << "can not bind " << PrintHostAndPort(addr);
            }

            res.Swap(s);
        }

        static inline void CreateSockets(TSockets& s, ui16 port) {
            TNetworkAddress addr(port);

            for (TNetworkAddress::TIterator it = addr.Begin(); it != addr.End(); ++it) {
                TSocketHolder res;

                CreateSocket(res, TAddrInfo(&*it));

                s.push_back(new TSocketDescr(res, it->ai_family));
            }
        }

        static inline void CreateSocketsOnRandomPort(TSockets& s) {
            while (true) {
                try {
                    TSockets tmp;

                    CreateSockets(tmp, 5000 + (RandomNumber<ui16>() % 1000));
                    tmp.swap(s);

                    return;
                } catch (const TBindError&) {
                }
            }
        }

        typedef ui64 TTimeStamp;

        static inline TTimeStamp TimeStamp() noexcept {
            return GetCycleCount() >> 31;
        }

        struct TRequestDescr: public TIntrusiveListItem<TRequestDescr> {
            inline TRequestDescr(const TString& guid, const TNotifyHandleRef& hndl, const TMessage& msg)
                : Guid(guid)
                , Hndl(hndl)
                , Msg(msg)
                , TS(TimeStamp())
            {
            }

            TString Guid;
            TNotifyHandleRef Hndl;
            TMessage Msg;
            TTimeStamp TS;
        };

        typedef TAutoPtr<TRequestDescr> TRequestDescrRef;

        class TProto {
            class TRequest: public IRequest, public TRequestPacket {
            public:
                inline TRequest(TPacket& p, TProto* parent)
                    : TRequestPacket(p)
                    , Addr_(std::move(p.Addr))
                    , H_(PrintHostByRfc(*Addr_))
                    , P_(parent)
                {
                }

                TStringBuf Scheme() const override {
                    return TStringBuf("udp");
                }

                TString RemoteHost() const override {
                    return H_;
                }

                TStringBuf Service() const override {
                    return ((TRequestPacket&)(*this)).Service;
                }

                TStringBuf Data() const override {
                    return ((TRequestPacket&)(*this)).Data;
                }

                TStringBuf RequestId() const override {
                    return ((TRequestPacket&)(*this)).Guid;
                }

                bool Canceled() const override {
                    //TODO ?
                    return false;
                }

                void SendReply(TData& data) override {
                    P_->Schedule(new TPacket(TResponsePacket<TData>(Guid, data), std::move(Addr_)));
                }

                void SendError(TResponseError, const TString&) override {
                    // TODO
                }

            private:
                IRemoteAddrPtr Addr_;
                TString H_;
                TProto* P_;
            };

        public:
            inline TProto(IOnRequest* cb, TSocketHolder& s)
                : CB_(cb)
                , ToSendEv_(TSystemEvent::rAuto)
                , S_(s.Release())
            {
                SetSocketTimeout(S_, 10);
                Thrs_.push_back(Spawn<TProto, &TProto::ExecuteRecv>(this));
                Thrs_.push_back(Spawn<TProto, &TProto::ExecuteSend>(this));
            }

            inline ~TProto() {
                Schedule(new TPacket(TStopPacket(), GetSendAddr(S_)));

                for (size_t i = 0; i < Thrs_.size(); ++i) {
                    Thrs_[i]->Join();
                }
            }

            inline TPacketRef Recv() {
                TBuffer tmp;

                tmp.Resize(128 * 1024);

                while (true) {
                    try {
                        return new TPacket(S_, tmp);
                    } catch (...) {
                        Cdbg << CurrentExceptionMessage() << Endl;

                        continue;
                    }
                }
            }

            typedef THashMap<TString, TRequestDescrRef> TInFlyBase;

            struct TInFly: public TInFlyBase, public TIntrusiveList<TRequestDescr> {
                typedef TInFlyBase::iterator TIter;
                typedef TInFlyBase::const_iterator TContsIter;

                inline void Insert(TRequestDescrRef& d) {
                    PushBack(d.Get());
                    (*this)[d->Guid] = d;
                }

                inline void EraseStale() noexcept {
                    const TTimeStamp now = TimeStamp();

                    for (TIterator it = Begin(); (it != End()) && (it->TS < now) && ((now - it->TS) > 120);) {
                        it->Hndl->NotifyError("request timeout");
                        TString safe_key = (it++)->Guid;
                        erase(safe_key);
                    }
                }
            };

            inline void ExecuteRecv() {
                SetHighestThreadPriority();

                TInFly infly;

                while (true) {
                    TPacketRef p = Recv();

                    switch (static_cast<EPacketType>(p->Type())) {
                        case PT_REQUEST:
                            if (CB_) {
                                CB_->OnRequest(new TRequest(*p, this));
                            } else {
                                //skip request in case of client
                            }

                            break;

                        case PT_RESPONSE: {
                            CancelStaleRequests(infly);

                            TResponsePacket<TString> rp(*p);

                            TInFly::TIter it = static_cast<TInFlyBase&>(infly).find(rp.Guid);

                            if (it == static_cast<TInFlyBase&>(infly).end()) {
                                break;
                            }

                            const TRequestDescrRef& d = it->second;
                            d->Hndl->NotifyResponse(rp.Data);

                            infly.erase(it);

                            break;
                        }

                        case PT_STOP:
                            Schedule(nullptr);

                            return;

                        case PT_TIMEOUT:
                            CancelStaleRequests(infly);

                            break;
                    }
                }
            }

            inline void ExecuteSend() {
                SetHighestThreadPriority();

                while (true) {
                    TPacketRef p;

                    while (!ToSend_.Dequeue(&p)) {
                        ToSendEv_.Wait();
                    }

                    //shutdown
                    if (!p) {
                        return;
                    }

                    p->SendTo(S_);
                }
            }

            inline void Schedule(TPacketRef p) {
                ToSend_.Enqueue(p);
                ToSendEv_.Signal();
            }

            inline void Schedule(TRequestDescrRef dsc, TPacketRef p) {
                ScheduledReqs_.Enqueue(dsc);
                Schedule(p);
            }

        protected:
            void CancelStaleRequests(TInFly& infly) {
                TRequestDescrRef d;

                while (ScheduledReqs_.Dequeue(&d)) {
                    infly.Insert(d);
                }

                infly.EraseStale();
            }

            IOnRequest* CB_;
            NNeh::TAutoLockFreeQueue<TPacket> ToSend_;
            NNeh::TAutoLockFreeQueue<TRequestDescr> ScheduledReqs_;
            TSystemEvent ToSendEv_;
            TSocketHolder S_;
            TVector<TThreadRef> Thrs_;
        };

        class TProtos {
        public:
            inline TProtos() {
                TSockets s;

                CreateSocketsOnRandomPort(s);
                Init(nullptr, s);
            }

            inline TProtos(IOnRequest* cb, ui16 port) {
                TSockets s;

                CreateSockets(s, port);
                Init(cb, s);
            }

            static inline TProtos* Instance() {
                return Singleton<TProtos>();
            }

            inline void Schedule(const TMessage& msg, const TNotifyHandleRef& hndl) {
                TParsedLocation loc(msg.Addr);
                const TNetworkAddress* addr = &CachedThrResolve(TResolveInfo(loc.Host, loc.GetPort()))->Addr;

                for (TNetworkAddress::TIterator ai = addr->Begin(); ai != addr->End(); ++ai) {
                    TProto* proto = Find(ai->ai_family);

                    if (proto) {
                        TRequestPacket rp(ToString(loc.Service), msg.Data);
                        TRequestDescrRef rd(new TRequestDescr(rp.Guid, hndl, msg));
                        IRemoteAddrPtr raddr(new TAddrInfo(&*ai));
                        TPacketRef p(new TPacket(rp, std::move(raddr)));

                        proto->Schedule(rd, p);

                        return;
                    }
                }

                ythrow yexception() << "unsupported protocol family";
            }

        private:
            inline void Init(IOnRequest* cb, TSockets& s) {
                for (auto& it : s) {
                    P_[it->Family] = new TProto(cb, it->S);
                }
            }

            inline TProto* Find(int family) const {
                TProtoStorage::const_iterator it = P_.find(family);

                if (it == P_.end()) {
                    return nullptr;
                }

                return it->second.Get();
            }

        private:
            typedef TAutoPtr<TProto> TProtoRef;
            typedef THashMap<int, TProtoRef> TProtoStorage;
            TProtoStorage P_;
        };

        class TRequester: public IRequester, public TProtos {
        public:
            inline TRequester(IOnRequest* cb, ui16 port)
                : TProtos(cb, port)
            {
            }
        };

        class TProtocol: public IProtocol {
        public:
            IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
                return new TRequester(cb, loc.GetPort());
            }

            THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
                TNotifyHandleRef ret(new TUdpHandle(fallback, msg, !ss ? nullptr : new TStatCollector(ss)));

                TProtos::Instance()->Schedule(msg, ret);

                return ret.Get();
            }

            TStringBuf Scheme() const noexcept override {
                return TStringBuf("udp");
            }
        };
    }
}

IProtocol* NNeh::UdpProtocol() {
    return Singleton<NUdp::TProtocol>();
}
