#include "stdafx.h"
#include "cpu_affinity.h"
#include "settings.h"
#include "udp_host.h"
#include "udp_http.h"
#include "udp_socket.h"

#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/hp_timer.h>
#include <util/thread/lfqueue.h>
#include <util/system/thread.h>
#include <library/cpp/threading/future/future.h>
#include <util/system/spinlock.h>
#if !defined(_win_)
#include <signal.h>
#include <pthread.h>
#endif
#include "block_chain.h"
#include <util/system/shmat.h>
#include <util/generic/hash_multi_map.h>
#include <exception>

#include <atomic>

namespace NNetliba_v12 {
    const float HTTP_TIMEOUT = 15.0f;
    const float CONNECTION_TIMEOUT = 600;
    const int MIN_SHARED_MEM_PACKET = 1000;

    static TAtomic PanicAttack;
    static std::atomic<NHPTimer::STime> LastHeartbeat;
    static std::atomic<double> HeartbeatTimeout;

    static int GetTransferSize(TUdpRequest* req) {
        if (req && req->Data.Get())
            return req->Data->GetSize();
        return 0;
    }

    TUdpHttpRequest::TUdpHttpRequest()
        : ReqId(TGUID())
    {
    }

    TUdpHttpRequest::~TUdpHttpRequest() {
    }

    TUdpHttpResponse::TUdpHttpResponse()
        : Ok(FAILED)
        , IsHighPriority(false)
    {
    }

    TUdpHttpResponse::~TUdpHttpResponse() {
    }

    class TRequesterUserQueueSizes: public TThrRefBase {
    public:
        TAtomic ReqCount, RespCount;
        TAtomic ReqQueueSize, RespQueueSize;

        TRequesterUserQueueSizes()
            : ReqCount(0)
            , RespCount(0)
            , ReqQueueSize(0)
            , RespQueueSize(0)
        {
        }
    };

    template <class T>
    void EraseList(TLockFreeQueue<T*>* data) {
        T* ptr = nullptr;
        while (data->Dequeue(&ptr)) {
            delete ptr;
        }
    }

    class TRequesterUserQueues: public TThrRefBase {
        TIntrusivePtr<TRequesterUserQueueSizes> QueueSizes;
        TLockFreeQueue<TUdpHttpRequest*> RequestList, RequestListHighPiority;
        TLockFreeQueue<TUdpHttpResponse*> ResponseList, ResponseListHighPiority;
        TLockFreeStack<TGUID> CancelList, SendRequestAccList; // any order will do
        TMuxEvent AsyncEvent;

        void UpdateAsyncSignalState() {
            // not sure about this one. Idea is that AsyncEvent.Reset() is a memory barrier
            if (RequestList.IsEmpty() && RequestListHighPiority.IsEmpty() && ResponseList.IsEmpty() && CancelList.IsEmpty() && SendRequestAccList.IsEmpty()) {
                AsyncEvent.Reset();
                if (!RequestList.IsEmpty() || !RequestListHighPiority.IsEmpty() || !ResponseList.IsEmpty() || !CancelList.IsEmpty() || !SendRequestAccList.IsEmpty())
                    AsyncEvent.Signal();
            }
        }
        ~TRequesterUserQueues() override {
            EraseList(&RequestList);
            EraseList(&RequestListHighPiority);
            EraseList(&ResponseList);
            EraseList(&ResponseListHighPiority);
        }

    public:
        TRequesterUserQueues(TRequesterUserQueueSizes* queueSizes)
            : QueueSizes(queueSizes)
        {
        }
        TUdpHttpRequest* GetRequest();
        TUdpHttpResponse* GetResponse();
        bool GetRequestCancel(TGUID* req) {
            bool res = CancelList.Dequeue(req);
            UpdateAsyncSignalState();
            return res;
        }
        bool GetSendRequestAcc(TGUID* req) {
            bool res = SendRequestAccList.Dequeue(req);
            UpdateAsyncSignalState();
            return res;
        }

        void AddRequest(TUdpHttpRequest* res) {
            AtomicAdd(QueueSizes->ReqCount, 1);
            AtomicAdd(QueueSizes->ReqQueueSize, GetTransferSize(res->DataHolder.Get()));
            if (res->DataHolder->IsHighPriority) {
                RequestListHighPiority.Enqueue(res);
            } else {
                RequestList.Enqueue(res);
            }
            AsyncEvent.Signal();
        }
        void AddResponse(TUdpHttpResponse* res) {
            AtomicAdd(QueueSizes->RespCount, 1);
            AtomicAdd(QueueSizes->RespQueueSize, GetTransferSize(res->DataHolder.Get()));
            if (res->DataHolder.Get() && res->DataHolder->IsHighPriority) {
                res->IsHighPriority = true;
                ResponseListHighPiority.Enqueue(res);
            } else {
                ResponseList.Enqueue(res);
            }
            AsyncEvent.Signal();
        }
        void AddCancel(const TGUID& req) {
            CancelList.Enqueue(req);
            AsyncEvent.Signal();
        }
        void AddSendRequestAcc(const TGUID& req) {
            SendRequestAccList.Enqueue(req);
            AsyncEvent.Signal();
        }
        TMuxEvent& GetAsyncEvent() {
            return AsyncEvent;
        }
        void AsyncSignal() {
            AsyncEvent.Signal();
        }
    };

    struct TOutRequestState {
        enum EState {
            S_SENDING,
            S_WAITING,
            S_WAITING_PING_SENDING,
            S_WAITING_PING_SENT,
            S_CANCEL_AT_SENDING
        };
        EState State;
        TIntrusivePtr<IConnection> Connection;
        ui64 TransferId;

        double TimePassed;
        TTransfer PingTransfer;
        TIntrusivePtr<TRequesterUserQueues> UserQueues;

        TOutRequestState()
            : State(S_SENDING)
            , TransferId(0)
            , TimePassed(0)
        {
        }
    };

    struct TInRequestState {
        enum EState {
            S_WAITING,
            S_RESPONSE_SENDING,
            S_CANCELED,
        };
        EState State;
        TIntrusivePtr<IConnection> Connection;
        ui64 ResponseTransferId;

        TInRequestState()
            : State(S_WAITING)
            , ResponseTransferId(0)
        {
        }
        TInRequestState(const TIntrusivePtr<IConnection>& connection)
            : State(S_WAITING)
            , Connection(connection)
            , ResponseTransferId(0)
        {
        }
    };

    enum EHttpPacket {
        PKT_REQUEST,
        PKT_PING,
        PKT_PING_RESPONSE,
        PKT_RESPONSE,
        PKT_GETDEBUGINFO,
        PKT_LOCAL_REQUEST,
        PKT_LOCAL_RESPONSE,
        PKT_CANCEL,
    };

    class TUdpHttp: public IRequester {
        enum EDir {
            DIR_OUT,
            DIR_IN
        };
        struct TTransferPurpose {
            EDir Dir;
            TGUID Guid;
            TTransferPurpose()
                : Dir(DIR_OUT)
            {
            }
            TTransferPurpose(EDir dir, TGUID guid)
                : Dir(dir)
                , Guid(guid)
            {
            }
        };

        struct TSendRequest: public TWithCustomAllocator {
            TConnectionAddress Address;
            TAutoPtr<TRopeDataPacket> Data;
            TGUID ReqGuid;
            TIntrusivePtr<TWaitResponse> WR;
            TIntrusivePtr<TRequesterUserQueues> UserQueues;

            TSendRequest() {
            }

            TSendRequest(const TConnectionAddress& address, TAutoPtr<TRopeDataPacket>* data, const TGUID& reqGuid, TWaitResponse* wr, TRequesterUserQueues* userQueues)
                : Address(address)
                , Data(*data)
                , ReqGuid(reqGuid)
                , WR(wr)
                , UserQueues(userQueues)
            {
            }
        };
        struct TSendResponse: public TWithCustomAllocator {
            TVector<char> Data;
            TGUID ReqGuid;
            EPacketPriority Priority;
            TColors Colors;

            TSendResponse()
                : ReqGuid()
                , Priority(PP_NORMAL)
            {
            }

            TSendResponse(const TGUID& reqGuid, EPacketPriority prior, TVector<char>* data, const TColors& colors)
                : ReqGuid(reqGuid)
                , Priority(prior)
                , Colors(colors)
            {
                if (data && !data->empty()) {
                    data->swap(Data);
                }
            }
        };
        struct TCancelRequest {
            TGUID ReqGuid;

            TCancelRequest() {
            }
            TCancelRequest(const TGUID& reqguid)
                : ReqGuid(reqguid)
            {
            }
        };
        struct TBreakRequest {
            TGUID ReqGuid;

            TBreakRequest() {
            }
            TBreakRequest(const TGUID& reqguid)
                : ReqGuid(reqguid)
            {
            }
        };

        TThread UdpHttpThread;
        TAtomic KeepRunning, AbortTransactions;
        TSpinLock CS;
        TSystemEvent HasStarted;

        NHPTimer::STime PingsSendT;

        TIntrusivePtr<IUdpHost> Host;
        TIntrusivePtr<ISocket> Socket;
        typedef THashMap<TGUID, TOutRequestState, TGUIDHash, TEqualTo<TGUID>, TCustomAllocator<std::pair<const TGUID, TOutRequestState>>> TOutRequestHash;
        typedef THashMap<TGUID, TInRequestState, TGUIDHash, TEqualTo<TGUID>, TCustomAllocator<std::pair<const TGUID, TInRequestState>>> TInRequestHash;
        TOutRequestHash OutRequests;
        TInRequestHash InRequests;

        THashMap<TTransfer, TTransferPurpose, THash<TTransfer>, TEqualTo<TTransfer>, TCustomAllocator<std::pair<const TTransfer, TTransferPurpose>>> TransferHash;
        THashMap<TGUID, TIntrusivePtr<TWaitResponse>, TGUIDHash, TEqualTo<TGUID>, TCustomAllocator<std::pair<const TGUID, TIntrusivePtr<TWaitResponse>>>> SyncRequests;

        // hold it here to not construct on every DoSends()
        THashSet<TGUID, TGUIDHash, TEqualTo<TGUID>, TCustomAllocator<TGUID>> AnticipateCancels;

        TLockFreeQueue<TSendRequest*> SendReqList;
        TLockFreeQueue<TSendResponse*> SendRespList;
        TLockFreeQueue<TCancelRequest> CancelReqList;
        TLockFreeQueue<TBreakRequest> BreakReqList;

        TIntrusivePtr<TRequesterUserQueueSizes> QueueSizes;
        TIntrusivePtr<TRequesterUserQueues> UserQueues;

        typedef THashMultiMap<TUdpAddress, TIntrusivePtr<IConnection>> TConnectionsCache;
        mutable TConnectionsCache ConnectionsCache, OldConnectionsCache;
        mutable NHPTimer::STime ConnectionsCacheT;

        struct TStatsRequest: public TThrRefBase {
            enum EReq {
                DEBUG_INFO,
                HAS_IN_REQUEST,
                GET_PEER_ADDRESS,
                GET_PEER_QUEUE_STATS,
            };
            EReq Req;
            TRequesterPendingDataStats TotalPendingDataSize;
            TColoredRequesterPendingDataStats ColoredPendingDataSize;
            TString DebugInfo;
            TGUID RequestId;
            TUdpAddress PeerAddress;
            TIntrusivePtr<IPeerQueueStats> QueueStats;
            bool RequestFound;
            TSystemEvent Complete;

            TStatsRequest(EReq req)
                : Req(req)
                , RequestFound(false)
            {
            }
        };
        TLockFreeQueue<TIntrusivePtr<TStatsRequest>> StatsReqList;

        bool ReportRequestCancel;
        bool ReportSendRequestAcc;

        void FinishRequest(TOutRequestHash::iterator i, TUdpHttpResponse::EResult ok, TAutoPtr<TUdpRequest> data, const char* error = nullptr) {
            TOutRequestState& s = i->second;
            TUdpHttpResponse* res = new TUdpHttpResponse;
            res->DataHolder = data;
            res->ReqId = i->first;
            res->PeerAddress = s.Connection->GetAddress();
            res->Ok = ok;
            if (ok == TUdpHttpResponse::FAILED)
                res->Error = error ? error : "request failed";
            else if (ok == TUdpHttpResponse::CANCELED)
                res->Error = error ? error : "request cancelled";
            const auto k = SyncRequests.find(res->ReqId);
            if (k != SyncRequests.end()) {
                TIntrusivePtr<TWaitResponse>& wr = k->second;
                wr->SetResponse(res);
                SyncRequests.erase(k);
            } else {
                s.UserQueues->AddResponse(res);
            }

            OutRequests.erase(i);
        }
        TTransfer SendWithSystemPriority(const TIntrusivePtr<IConnection>& connection, TAutoPtr<TRopeDataPacket> data, const TTos& tos, const ui8 netlibaColor) {
            return Host->Send(connection, data.Release(), PP_SYSTEM, tos, netlibaColor);
        }
        void ProcessIncomingPackets() {
            TVector<TGUID, TCustomAllocator<TGUID>> failedRequests;
            for (;;) {
                TAutoPtr<TUdpRequest> req = Host->GetRequest();
                if (req.Get() == nullptr)
                    break;

                TBlockChainIterator reqData(req->Data->GetChain());
                char pktType;
                reqData.Read(&pktType, 1);
                switch (pktType) {
                    case PKT_REQUEST:
                    case PKT_LOCAL_REQUEST: {
                        TGUID reqId;
                        reqData.Read(&reqId, sizeof(reqId));

                        //printf("recv PKT_REQUEST or PKT_LOCAL_REQUEST\n");
                        TInRequestHash::iterator z = InRequests.find(reqId);
                        if (z != InRequests.end()) {
                            // oops, this request already exists!
                            // might happen if request can be stored in single packet
                            // and this packet had source IP broken during transmission and managed to pass crc checks
                            // since we already reported wrong source address for this request to the user
                            // the best thing we can do is to stop the program to avoid further complications
                            // but we just report the accident to stderr
                            fprintf(stderr, "Jackpot, same request %s received twice from %s and earlier from %s\n",
                                    GetGuidAsString(reqId).c_str(), GetAddressAsString(z->second.Connection->GetAddress()).c_str(),
                                    GetAddressAsString(req->Connection->GetAddress()).c_str());
                        } else {
                            InRequests[reqId] = TInRequestState(req->Connection);
                            //printf("InReq %s PKT_REQUEST recv ... -> S_WAITING\n", GetGuidAsString(res->ReqId).c_str());

                            CacheConnection(req->Connection);

                            TUdpHttpRequest* res = new TUdpHttpRequest;
                            res->ReqId = reqId;
                            res->PeerAddress = req->Connection->GetAddress();
                            res->DataHolder = req;
                            UserQueues->AddRequest(res);
                        }
                    } break;
                    case PKT_PING: {
                        //printf("recv PKT_PING\n");
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        bool ok = InRequests.find(guid) != InRequests.end();
                        TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                        ms->Write((char)PKT_PING_RESPONSE);
                        ms->Write(guid);
                        ms->Write(ok);
                        SendWithSystemPriority(req->Connection, ms.Release(), TTos(), DEFAULT_NETLIBA_COLOR);
                        //printf("InReq %s PKT_PING recv Sending PKT_PING_RESPONSE\n", GetGuidAsString(guid).c_str());
                        //printf("got PKT_PING, responding %d\n", (int)ok);
                    } break;
                    case PKT_PING_RESPONSE: {
                        //printf("recv PKT_PING_RESPONSE\n");
                        TGUID guid;
                        bool ok;
                        reqData.Read(&guid, sizeof(guid));
                        reqData.Read(&ok, sizeof(ok));
                        TOutRequestHash::iterator i = OutRequests.find(guid);
                        if (i == OutRequests.end()) {
                            ; //Y_ASSERT(0); // actually possible with some packet orders + packet may be canceled!
                        } else {
                            if (!ok) {
                                // can not delete request at this point
                                // since we can receive failed ping and response at the same moment
                                // consider sequence: client sends ping, server sends response
                                // and replies false to ping as reply is sent
                                // we can not receive failed ping_response earlier then response itself
                                // but we can receive them simultaneously
                                failedRequests.push_back(guid);
                                //printf("OutReq %s PKT_PING_RESPONSE recv no such query -> failed\n", GetGuidAsString(guid).c_str());
                            } else {
                                TOutRequestState& s = i->second;
                                switch (s.State) {
                                    case TOutRequestState::S_WAITING_PING_SENDING: {
                                        Y_ASSERT(s.PingTransfer != TTransfer());
                                        const auto k = TransferHash.find(s.PingTransfer);
                                        if (k != TransferHash.end())
                                            TransferHash.erase(k);
                                        else
                                            Y_ASSERT(0);
                                        s.PingTransfer = TTransfer();
                                        s.TimePassed = 0;
                                        s.State = TOutRequestState::S_WAITING;
                                        //printf("OutReq %s PKT_PING_RESPONSE recv S_WAITING_PING_SENDING -> S_WAITING\n", GetGuidAsString(guid).c_str());
                                    } break;
                                    case TOutRequestState::S_WAITING_PING_SENT:
                                        s.TimePassed = 0;
                                        s.State = TOutRequestState::S_WAITING;
                                        //printf("OutReq %s PKT_PING_RESPONSE recv S_WAITING_PING_SENT -> S_WAITING\n", GetGuidAsString(guid).c_str());
                                        break;
                                    default:
                                        Y_ASSERT(0);
                                        break;
                                }
                            }
                        }
                    } break;
                    case PKT_RESPONSE:
                    case PKT_LOCAL_RESPONSE: {
                        //printf("recv PKT_RESPONSE or PKT_LOCAL_RESPONSE\n");
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        TOutRequestHash::iterator i = OutRequests.find(guid);
                        if (i == OutRequests.end()) {
                            ; //Y_ASSERT(0); // does happen (for example after cancel in S_WAITING state)
                            //printf("OutReq %s PKT_RESPONSE recv for non-existing req\n", GetGuidAsString(guid).c_str());
                        } else {
                            FinishRequest(i, TUdpHttpResponse::OK, req);
                            //printf("OutReq %s PKT_RESPONSE recv ... -> ok\n", GetGuidAsString(guid).c_str());
                        }
                    } break;
                    case PKT_CANCEL: {
                        //printf("recv PKT_CANCEL\n");
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        TInRequestHash::iterator i = InRequests.find(guid);
                        if (i == InRequests.end()) {
                            ; //Y_ASSERT(0); // may happen
                            //printf("InReq %s PKT_CANCEL recv for non-existing req\n", GetGuidAsString(guid).c_str());
                        } else {
                            TInRequestState& s = i->second;

                            if (s.State != TInRequestState::S_CANCELED) {
                                if (s.State == TInRequestState::S_RESPONSE_SENDING) {
                                    Y_ASSERT(s.ResponseTransferId > 0);
                                    Host->Cancel(TTransfer(s.Connection, s.ResponseTransferId));
                                    s.ResponseTransferId = 0;
                                }

                                // it's ok that TUdpHost may have already completed transfer of response - another side will drop it.
                                if (ReportRequestCancel) {
                                    UserQueues->AddCancel(guid);
                                }
                            }
                            s.State = TInRequestState::S_CANCELED;
                            //printf("InReq %s PKT_CANCEL recv\n", GetGuidAsString(guid).c_str());
                        }
                    } break;
                    case PKT_GETDEBUGINFO: {
                        //printf("recv PKT_GETDEBUGINFO\n");
                        TString dbgInfo = GetDebugInfoLocked();
                        TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                        ms->Write(dbgInfo.c_str(), (int)dbgInfo.size());
                        SendWithSystemPriority(req->Connection, ms, TTos(), DEFAULT_NETLIBA_COLOR);
                    } break;
                    default:
                        Y_ASSERT(0);
                }
            }
            // cleanup failed requests
            for (size_t k = 0; k < failedRequests.size(); ++k) {
                const TGUID& guid = failedRequests[k];
                TOutRequestHash::iterator i = OutRequests.find(guid);
                if (i != OutRequests.end())
                    FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "failed udp ping");
            }
        }
        void AnalyzeSendResults() {
            TSendResult res;
            while (Host->GetSendResult(&res)) {
                //printf("Send result received\n");
                const auto k = TransferHash.find(res.Transfer);
                if (k != TransferHash.end()) {
                    const TTransferPurpose& tp = k->second;
                    switch (tp.Dir) {
                        case DIR_OUT: {
                            TOutRequestHash::iterator i = OutRequests.find(tp.Guid);
                            if (i != OutRequests.end()) {
                                const TGUID& reqId = i->first;
                                TOutRequestState& s = i->second;
                                switch (s.State) {
                                    case TOutRequestState::S_SENDING:
                                        if (res.Ok == TSendResult::FAILED) {
                                            FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: state S_SENDING");
                                            //printf("OutReq %s AnalyzeSendResults() S_SENDING -> failed\n", GetGuidAsString(reqId).c_str());
                                        } else if (res.Ok == TSendResult::CANCELED) {
                                            Y_ASSERT(0);
                                        } else {
                                            if (ReportSendRequestAcc) {
                                                if (s.UserQueues.Get()) {
                                                    s.UserQueues->AddSendRequestAcc(reqId);
                                                } else {
                                                    // waitable request?
                                                    const auto k2 = SyncRequests.find(reqId);
                                                    if (k2 != SyncRequests.end()) {
                                                        TIntrusivePtr<TWaitResponse>& wr = k2->second;
                                                        wr->SetRequestSent();
                                                    }
                                                }
                                            }
                                            s.State = TOutRequestState::S_WAITING;
                                            //printf("OutReq %s AnalyzeSendResults() S_SENDING -> S_WAITING\n", GetGuidAsString(reqId).c_str());
                                            s.TimePassed = 0;
                                        }
                                        break;
                                    case TOutRequestState::S_CANCEL_AT_SENDING:
                                        if (res.Ok == TSendResult::OK) {
                                            // oops, request had time to be sent
                                            DoSendCancel(s.Connection, reqId);
                                        }
                                        FinishRequest(i, TUdpHttpResponse::CANCELED, nullptr, "request failed: state S_CANCEL_AT_SENDING");
                                        break;
                                    case TOutRequestState::S_WAITING:
                                    case TOutRequestState::S_WAITING_PING_SENT:
                                        Y_ASSERT(0);
                                        break;
                                    case TOutRequestState::S_WAITING_PING_SENDING:
                                        Y_ASSERT(s.PingTransfer != TTransfer() && s.PingTransfer == res.Transfer);
                                        if (res.Ok == TSendResult::FAILED) {
                                            FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: state S_WAITING_PING_SENDING");
                                            //printf("OutReq %s AnalyzeSendResults() S_WAITING_PING_SENDING -> failed\n", GetGuidAsString(reqId).c_str());
                                        } else if (res.Ok == TSendResult::CANCELED) {
                                            Y_ASSERT(0);
                                        } else {
                                            s.PingTransfer = TTransfer();
                                            s.State = TOutRequestState::S_WAITING_PING_SENT;
                                            //printf("OutReq %s AnalyzeSendResults() S_WAITING_PING_SENDING -> S_WAITING_PING_SENT\n", GetGuidAsString(reqId).c_str());
                                            s.TimePassed = 0;
                                        }
                                        break;
                                    default:
                                        Y_ASSERT(0);
                                        break;
                                }
                            }
                        } break;
                        case DIR_IN: {
                            TInRequestHash::iterator i = InRequests.find(tp.Guid);
                            if (i != InRequests.end()) {
                                Y_ASSERT(i->second.State == TInRequestState::S_RESPONSE_SENDING || i->second.State == TInRequestState::S_CANCELED);
                                InRequests.erase(i);
                                /*
                                if (res.Ok == TSendResult::OK)
                                    printf("InReq %s AnalyzeSendResults() ... -> finished\n", GetGuidAsString(tp.Guid).c_str());
                                else if (res.Ok == TSendResult::FAILED)
                                    printf("InReq %s AnalyzeSendResults() ... -> failed response send\n", GetGuidAsString(tp.Guid).c_str());
                                else if (res.Ok == TSendResult::CANCELED)
                                    printf("InReq %s AnalyzeSendResults() ... -> canceled response send\n", GetGuidAsString(tp.Guid).c_str());
                                */
                            }
                        } break;
                        default:
                            Y_ASSERT(0);
                            break;
                    }
                    TransferHash.erase(k);
                }
            }
        }
        void SendPingsIfNeeded() {
            NHPTimer::STime tChk = PingsSendT;
            float deltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (deltaT < 0.5) {
                return;
            }
            PingsSendT = tChk;
            deltaT = ClampVal(deltaT, 0.0f, HTTP_TIMEOUT / 3);

            {
                for (TOutRequestHash::iterator i = OutRequests.begin(); i != OutRequests.end();) {
                    TOutRequestHash::iterator curIt = i++;
                    TOutRequestState& s = curIt->second;
                    const TGUID& guid = curIt->first;
                    switch (s.State) {
                        case TOutRequestState::S_WAITING:
                            s.TimePassed += deltaT;
                            if (s.TimePassed > HTTP_TIMEOUT) {
                                TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                                ms->Write((char)PKT_PING);
                                ms->Write(guid);
                                TTransfer trans = SendWithSystemPriority(s.Connection, ms.Release(), TTos(), DEFAULT_NETLIBA_COLOR);
                                TransferHash[trans] = TTransferPurpose(DIR_OUT, guid);
                                s.State = TOutRequestState::S_WAITING_PING_SENDING;
                                //printf("OutReq %s SendPingsIfNeeded() S_WAITING -> S_WAITING_PING_SENDING\n", GetGuidAsString(guid).c_str());
                                s.PingTransfer = trans;
                            }
                            break;
                        case TOutRequestState::S_WAITING_PING_SENT:
                            s.TimePassed += deltaT;
                            if (s.TimePassed > HTTP_TIMEOUT * 4) {
                                //printf("OutReq %s SendPingsIfNeeded() S_WAITING_PING_SENT -> failed\n", GetGuidAsString(guid).c_str());
                                FinishRequest(curIt, TUdpHttpResponse::FAILED, nullptr, "request failed: http timeout in state S_WAITING_PING_SENT");
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }
        void CleanCachesIfNeeded() {
            NHPTimer::STime tChk = ConnectionsCacheT;
            float deltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (deltaT > CONNECTION_TIMEOUT) {
                OldConnectionsCache.swap(ConnectionsCache);
                ConnectionsCache.clear();
                ConnectionsCacheT = tChk;
            }
        }
        static TConnectionsCache::iterator Find(TConnectionsCache& cache, const TUdpAddress& addr, const TConnectionSettings& connectionSettings) {
            std::pair<TConnectionsCache::iterator, TConnectionsCache::iterator> p = cache.equal_range(addr);
            for (TConnectionsCache::iterator i = p.first; i != p.second; ++i) {
                if (i->second->GetSettings() == connectionSettings) {
                    return i;
                }
            }
            return cache.end();
        }
        TIntrusivePtr<IConnection> GetConnection(const TUdpAddress& addr, const TConnectionSettings& connectionSettings) const {
            // actually may occur after bad dns resolving
            // Y_ASSERT(addr != TUdpAddress());

            TConnectionsCache::iterator c = Find(ConnectionsCache, addr, connectionSettings);
            if (c != ConnectionsCache.end()) {
                Y_ASSERT(Find(OldConnectionsCache, addr, connectionSettings) == OldConnectionsCache.end());
                return c->second;
            }

            TIntrusivePtr<IConnection> connection;

            c = Find(OldConnectionsCache, addr, connectionSettings);
            if (c != OldConnectionsCache.end()) {
                connection = c->second;
                OldConnectionsCache.erase(c);
            } else {
                connection = Host->Connect(addr, connectionSettings);
            }
            ConnectionsCache.insert(std::make_pair(addr, connection));

            return connection;
        }
        void CacheConnection(const TIntrusivePtr<IConnection>& connection) const {
            const TUdpAddress& addr = connection->GetAddress();
            const TConnectionSettings& connectionSettings = connection->GetSettings();

            TConnectionsCache::iterator c = Find(ConnectionsCache, addr, connectionSettings);
            if (c != ConnectionsCache.end()) {
                c->second = connection; // they are equal or we overwrite old connection
                Y_ASSERT(Find(OldConnectionsCache, addr, connectionSettings) == OldConnectionsCache.end());
                return;
            }

            c = Find(OldConnectionsCache, addr, connectionSettings);
            if (c != OldConnectionsCache.end()) {
                OldConnectionsCache.erase(c);
            }

            ConnectionsCache.insert(std::make_pair(addr, connection));
        }
        void Step() {
            {
                TGuard<TSpinLock> lock(CS);
                DoSends();
            }
            Host->Step();
            for (TIntrusivePtr<TStatsRequest> req; StatsReqList.Dequeue(&req);) {
                switch (req->Req) {
                    case TStatsRequest::DEBUG_INFO: {
                        TGuard<TSpinLock> lock(CS);
                        req->DebugInfo = GetDebugInfoLocked();
                    }
                        req->Complete.Signal();
                        break;
                    case TStatsRequest::HAS_IN_REQUEST: {
                        TGuard<TSpinLock> lock(CS);
                        req->RequestFound = (InRequests.find(req->RequestId) != InRequests.end());
                    }
                        req->Complete.Signal();
                        break;
                    case TStatsRequest::GET_PEER_ADDRESS: {
                        TGuard<TSpinLock> lock(CS);
                        TInRequestHash::const_iterator i = InRequests.find(req->RequestId);
                        if (i != InRequests.end()) {
                            req->PeerAddress = i->second.Connection->GetAddress();
                        } else {
                            TOutRequestHash::const_iterator o = OutRequests.find(req->RequestId);
                            if (o != OutRequests.end()) {
                                req->PeerAddress = o->second.Connection->GetAddress();
                            } else {
                                req->PeerAddress = TUdpAddress();
                            }
                        }
                    }
                        req->Complete.Signal();
                        break;
                    case TStatsRequest::GET_PEER_QUEUE_STATS:
                        /*
                // TODO: infinite stats
                    if (TIntrusivePtr<IConnection>* connection = ConnectionsCache.FindPtr(req->PeerAddress)) {
                        req->QueueStats = (*connection)->GetPendingDataSize();
                    }
                */
                        Y_ABORT_UNLESS(false, "NOT IMPLEMENTED");

                        break;
                    default:
                        Y_ASSERT(0);
                        req->Complete.Signal();
                        break;
                }
            }
            {
                TGuard<TSpinLock> lock(CS);
                DoSends();
                ProcessIncomingPackets();
                AnalyzeSendResults();
                SendPingsIfNeeded();
                CleanCachesIfNeeded();
            }
        }
        void Wait() {
            Host->Wait(0.1f);
        }
        void DoSendCancel(const TIntrusivePtr<IConnection>& connection, const TGUID& req) {
            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            ms->Write((char)PKT_CANCEL);
            ms->Write(req);
            SendWithSystemPriority(connection, ms, TTos(), DEFAULT_NETLIBA_COLOR);
        }
        void DoSends() {
            {
                TBreakRequest rb;
                while (BreakReqList.Dequeue(&rb)) {
                    InRequests.erase(rb.ReqGuid);
                }
            }
            {
                // cancelling requests
                TCancelRequest rc;
                while (CancelReqList.Dequeue(&rc)) {
                    TOutRequestHash::iterator i = OutRequests.find(rc.ReqGuid);
                    if (i == OutRequests.end()) {
                        AnticipateCancels.insert(rc.ReqGuid);
                        continue; // cancelling non existing request is ok
                    }
                    TOutRequestState& s = i->second;
                    if (s.State == TOutRequestState::S_SENDING) {
                        // we are in trouble - have not sent request and we already have to cancel it,
                        // we send async request to TUdpHost to cancel transfer.
                        s.State = TOutRequestState::S_CANCEL_AT_SENDING;
                        Host->Cancel(TTransfer(s.Connection, s.TransferId));
                    } else {
                        DoSendCancel(s.Connection, rc.ReqGuid);
                        FinishRequest(i, TUdpHttpResponse::CANCELED, nullptr, "request canceled: notify requested side");
                    }
                }
            }
            {
                // sending replies
                for (TSendResponse* rd = nullptr; SendRespList.Dequeue(&rd); delete rd) {
                    TInRequestHash::iterator i = InRequests.find(rd->ReqGuid);
                    if (i == InRequests.end()) {
                        Y_ASSERT(0);
                        continue;
                    }
                    TInRequestState& s = i->second;
                    if (s.State == TInRequestState::S_CANCELED) {
                        // need not send response for the canceled request
                        //printf("Do sends: request %s got canceled, erasing\n", GetGuidAsString(rd->ReqGuid).c_str());
                        InRequests.erase(i);
                        continue;
                    }

                    Y_ASSERT(s.State == TInRequestState::S_WAITING);
                    s.State = TInRequestState::S_RESPONSE_SENDING;
                    //printf("InReq %s SendResponse() ... -> S_RESPONSE_SENDING (pkt %s)\n", GetGuidAsString(reqId).c_str(), GetGuidAsString(lowPktGuid).c_str());

                    TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                    int dataSize = rd->Data.ysize();
                    if (rd->Data.ysize() > MIN_SHARED_MEM_PACKET && Host->IsLocal(s.Connection->GetAddress())) {
                        TIntrusivePtr<TPosixSharedMemory> shm = new TPosixSharedMemory;
                        if (shm->Create(dataSize)) {
                            ms->Write((char)PKT_LOCAL_RESPONSE);
                            ms->Write(rd->ReqGuid);
                            memcpy(shm->GetPtr(), &rd->Data[0], dataSize);
                            TVector<char> empty;
                            rd->Data.swap(empty);
                            ms->AttachSharedData(shm);
                        }
                    }
                    if (ms->GetSharedData() == nullptr) {
                        ms->Write((char)PKT_RESPONSE);
                        ms->Write(rd->ReqGuid);
                        ms->WriteDestructive(&rd->Data);
                    }

                    TTransfer trans = Host->Send(s.Connection, ms.Release(), rd->Priority, rd->Colors.GetResponseTos(), rd->Colors.GetNetlibaResponseColor());
                    TransferHash[trans] = TTransferPurpose(DIR_IN, rd->ReqGuid);
                    s.ResponseTransferId = trans.Id;
                }
            }
            {
                // sending requests
                for (TSendRequest* rd = nullptr; SendReqList.Dequeue(&rd); delete rd) {
                    Y_ASSERT(OutRequests.find(rd->ReqGuid) == OutRequests.end());

                    const TColors& colors = rd->Address;
                    const TConnectionSettings& connectionSettings = rd->Address;
                    const TUdpAddress addr = rd->Address.GetAddress();
                    TIntrusivePtr<IConnection> connection = GetConnection(addr, connectionSettings);

                    TOutRequestState& s = OutRequests[rd->ReqGuid];
                    s.State = TOutRequestState::S_SENDING;
                    s.Connection = connection;
                    s.UserQueues = rd->UserQueues;
                    //printf("OutReq %s SendRequest() ... -> S_SENDING\n", GetGuidAsString(guid).c_str());

                    if (rd->WR.Get())
                        SyncRequests[rd->ReqGuid] = rd->WR;

                    if (AnticipateCancels.find(rd->ReqGuid) != AnticipateCancels.end()) {
                        FinishRequest(OutRequests.find(rd->ReqGuid), TUdpHttpResponse::CANCELED, nullptr, "request canceled before transmitting");
                    } else {
                        TTransfer trans = Host->Send(connection, rd->Data.Release(), colors.GetPriority(), colors.GetRequestTos(), colors.GetNetlibaRequestColor());
                        TransferHash[trans] = TTransferPurpose(DIR_OUT, rd->ReqGuid);
                        s.TransferId = trans.Id;
                    }
                }
            }
            if (!AnticipateCancels.empty()) {
                AnticipateCancels.clear();
            }
        }

    public:
        void SendRequestImpl(const TConnectionAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId,
                             TWaitResponse* wr, TRequesterUserQueues* userQueues) {
            if (data && data->size() > MAX_PACKET_SIZE) {
                Y_ABORT_UNLESS(0, "data size is too large");
            }
            //printf("SendRequest(%s)\n", url.c_str());
            if (wr)
                wr->SetReqId(reqId);

            TVector<char> flags(4 * 2 + 2 + 1);
            (i16&)flags[0] = (i16)addr.GetResponseDataTos();
            (i16&)flags[2] = (i16)addr.GetResponseAckTos();
            (i16&)flags[4] = (i16)addr.GetRequestDataTos();
            (i16&)flags[6] = (i16)addr.GetRequestAckTos();
            (ui8&)flags[8] = addr.GetNetlibaRequestColor();
            (ui8&)flags[9] = addr.GetNetlibaResponseColor();
            if (addr.GetPriority() == PP_HIGH) {
                (ui8&)flags[10] = HF_HP_QUEUE;
            }

            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            if (data && data->ysize() > MIN_SHARED_MEM_PACKET && Host->IsLocal(addr.GetAddress())) {
                int dataSize = data->ysize();
                TIntrusivePtr<TPosixSharedMemory> shm = new TPosixSharedMemory;
                if (shm->Create(dataSize)) {
                    ms->Write((char)PKT_LOCAL_REQUEST);
                    ms->Write(reqId);
                    ms->WriteDestructive(&flags);
                    ms->WriteStroka(url);
                    memcpy(shm->GetPtr(), &(*data)[0], dataSize);
                    TVector<char> empty;
                    data->swap(empty);
                    ms->AttachSharedData(shm);
                }
            }

            if (ms->GetSharedData() == nullptr) {
                ms->Write((char)PKT_REQUEST);
                ms->Write(reqId);
                ms->WriteDestructive(&flags);
                ms->WriteStroka(url);
                ms->WriteDestructive(data);
            }

            SendReqList.Enqueue(new TSendRequest(addr, &ms, reqId, wr, userQueues));
            Host->CancelWait();
        }

        void SendRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) override {
            SendRequestImpl(addr, url, data, reqId, nullptr, UserQueues.Get());
        }
        void CancelRequest(const TGUID& reqId) override {
            CancelReqList.Enqueue(TCancelRequest(reqId));
            Host->CancelWait();
        }
        void BreakRequest(const TGUID& reqId) override {
            BreakReqList.Enqueue(TBreakRequest(reqId));
            Host->CancelWait();
        }

        void SendResponseImpl(const TGUID& reqId, EPacketPriority prior, TVector<char>* data, const TColors& colors) // non-virtual, for direct call from TRequestOps
        {
            if (data && data->size() > MAX_PACKET_SIZE) {
                Y_ABORT_UNLESS(0, "data size is too large");
            }
            SendRespList.Enqueue(new TSendResponse(reqId, prior, data, colors));
            Host->CancelWait();
        }
        void SendResponse(const TGUID& reqId, TVector<char>* data, const TColors& colors) override {
            SendResponseImpl(reqId, colors.GetPriority(), data, colors);
        }
        void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data, const TColors& colors) override {
            SendResponseImpl(reqId, PP_LOW, data, colors);
        }
        TUdpHttpRequest* GetRequest() override {
            return UserQueues->GetRequest();
        }
        TUdpHttpResponse* GetResponse() override {
            return UserQueues->GetResponse();
        }
        bool GetRequestCancel(TGUID* req) override {
            return UserQueues->GetRequestCancel(req);
        }
        bool GetSendRequestAcc(TGUID* req) override {
            return UserQueues->GetSendRequestAcc(req);
        }
        TUdpHttpResponse* Request(const TConnectionAddress& addr, const TString& url, TVector<char>* data) override {
            TIntrusivePtr<TWaitResponse> wr = WaitableRequest(addr, url, data);
            wr->Wait();
            return wr->GetResponse();
        }
        TIntrusivePtr<TWaitResponse> WaitableRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data) override {
            TIntrusivePtr<TWaitResponse> wr = new TWaitResponse;
            TGUID reqId;
            CreateGuid(&reqId);
            SendRequestImpl(addr, url, data, reqId, wr.Get(), nullptr);
            return wr;
        }
        TMuxEvent& GetAsyncEvent() override {
            return UserQueues->GetAsyncEvent();
        }
        int GetPort() override {
            return Socket.Get() ? Socket->GetPort() : 0;
        }
        void StopNoWait() override {
            AtomicSwap(&AbortTransactions, 1);
            AtomicSwap(&KeepRunning, 0);

            UserQueues->AsyncSignal();
            // calcel all outgoing requests
            TGuard<TSpinLock> lock(CS);
            while (!OutRequests.empty()) {
                // cancel without informing peer that we are cancelling the request
                FinishRequest(OutRequests.begin(), TUdpHttpResponse::CANCELED, nullptr, "request canceled: inside TUdpHttp::StopNoWait()");
            }
        }
        void ExecStatsRequest(TIntrusivePtr<TStatsRequest> req) {
            StatsReqList.Enqueue(req);
            Host->CancelWait();
            req->Complete.Wait();
        }
        TUdpAddress GetPeerAddress(const TGUID& reqId) override {
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::GET_PEER_ADDRESS);
            req->RequestId = reqId;
            ExecStatsRequest(req);
            return req->PeerAddress;
        }
        void GetPendingDataSize(TRequesterPendingDataStats* res) override {
            auto promise = NThreading::NewPromise<TRequesterPendingDataStats>();
            Host->GetAllPendingDataSize([promise](const TRequesterPendingDataStats& stat, const TColoredRequesterPendingDataStats& /*coloredStat*/) mutable {
                promise.SetValue(stat);
            });
            auto future = promise.GetFuture();
            future.Wait();
            *res = future.GetValue();
        }
        void GetAllPendingDataSize(TRequesterPendingDataStats* total, TColoredRequesterPendingDataStats* colored) override {
            auto promise = NThreading::NewPromise<std::pair<TRequesterPendingDataStats, TColoredRequesterPendingDataStats>>();
            Host->GetAllPendingDataSize([promise](const TRequesterPendingDataStats& stat, const TColoredRequesterPendingDataStats& coloredStat) mutable {
                promise.SetValue({stat, coloredStat});
            });
            auto future = promise.GetFuture();
            future.Wait();
            auto resData = future.GetValue();
            *total = resData.first;
            colored->Swap(resData.second);
        }
        bool HasRequest(const TGUID& reqId) override {
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::HAS_IN_REQUEST);
            req->RequestId = reqId;
            ExecStatsRequest(req);
            return req->RequestFound;
        }
        float GetPacketFailRate() const override {
            return Host->GetFailRate();
        }

    private:
        void FinishOutstandingTransactions() {
            // wait all pending requests, all new requests are canceled
            while ((!OutRequests.empty() || !InRequests.empty() || !SendRespList.IsEmpty() || !SendReqList.IsEmpty()) && !AtomicAdd(PanicAttack, 0)) {
                while (TUdpHttpRequest* req = GetRequest()) {
                    TInRequestHash::iterator i = InRequests.find(req->ReqId);
                    //               printf("dropping request(%s) (thread %d)\n", req->Url.c_str(), ThreadId());
                    delete req;
                    if (i == InRequests.end()) {
                        Y_ASSERT(0);
                        continue;
                    }
                    InRequests.erase(i);
                }
                Step();
                sleep(0);
            }
        }
        static void* ExecServerThread(void* param) {
            BindToSocket(0);
            SetHighestThreadPriority();
            TUdpHttp* pThis = (TUdpHttp*)param;
            pThis->Host = CreateUdpHost(pThis->Socket);
            pThis->HasStarted.Signal();
            if (!pThis->Host) {
                pThis->Socket.Drop();
                return nullptr;
            }
            NHPTimer::GetTime(&pThis->PingsSendT);
            NHPTimer::GetTime(&pThis->ConnectionsCacheT);
            while (AtomicAdd(pThis->KeepRunning, 0) && !AtomicAdd(PanicAttack, 0)) {
                if (HeartbeatTimeout.load(std::memory_order_acquire) > 0) {
                    NHPTimer::STime chk = LastHeartbeat.load(std::memory_order_acquire);
                    double passed = NHPTimer::GetTimePassed(&chk);
                    if (passed > HeartbeatTimeout.load(std::memory_order_acquire)) {
                        StopAllNetLibaThreads();
                        fprintf(stderr, "%s\tTUdpHttp\tWaiting for %0.2f, time limit %0.2f, commit a suicide!11\n", Now().ToStringUpToSeconds().c_str(), passed, HeartbeatTimeout.load(std::memory_order_acquire));
                        fflush(stderr);
#ifndef _win_
                        killpg(0, SIGKILL);
#endif
                        abort();
                        break;
                    }
                }
                pThis->Step();
                pThis->Wait();
            }
            if (!AtomicAdd(pThis->AbortTransactions, 0) && !AtomicAdd(PanicAttack, 0))
                pThis->FinishOutstandingTransactions();
            pThis->Host = nullptr;
            return nullptr;
        }
        ~TUdpHttp() override {
            if (std::uncaught_exceptions()) {
                TUdpHttp::StopNoWait();
                AtomicSwap(&AbortTransactions, 1);
            }

            if (UdpHttpThread.Running()) {
                AtomicSwap(&KeepRunning, 0);
                UdpHttpThread.Join();
            }
            for (TIntrusivePtr<TStatsRequest> req; StatsReqList.Dequeue(&req);) {
                req->Complete.Signal();
            }
        }

    public:
        TUdpHttp()
            : UdpHttpThread(TThread::TParams(ExecServerThread, (void*)this).SetName("nl12_udp_http"))
            , KeepRunning(1)
            , AbortTransactions(0)
            , PingsSendT(0)
            , ConnectionsCacheT(0)
            , ReportRequestCancel(false)
            , ReportSendRequestAcc(false)
        {
            NHPTimer::GetTime(&PingsSendT);
            NHPTimer::GetTime(&ConnectionsCacheT);
            QueueSizes = new TRequesterUserQueueSizes;
            UserQueues = new TRequesterUserQueues(QueueSizes.Get());
        }
        bool Start(const TIntrusivePtr<ISocket>& socket) {
            Y_ASSERT(Host.Get() == nullptr);
            Socket = socket;
            UdpHttpThread.Start();
            HasStarted.Wait();

            if (Host.Get()) {
                return true;
            }
            Socket.Drop();
            return false;
        }
        TString GetDebugInfoLocked() {
            TString res = KeepRunning ? "State: running\n" : "State: stopping\n";
            auto promise = NThreading::NewPromise<TString>();
            Host->GetDebugInfo([promise](const TString& str) mutable {
                promise.SetValue(str);
            });
            auto future = promise.GetFuture();
            future.Wait();
            res += future.GetValue();

            char buf[1000];
            TRequesterUserQueueSizes* qs = QueueSizes.Get();
            snprintf(buf, sizeof(buf), "\nRequest queue %d (%d bytes)\n", (int)AtomicGet(qs->ReqCount), (int)AtomicGet(qs->ReqQueueSize));
            res += buf;
            snprintf(buf, sizeof(buf), "Response queue %d (%d bytes)\n", (int)AtomicGet(qs->RespCount), (int)AtomicGet(qs->RespQueueSize));
            res += buf;

            const char* outReqStateNames[] = {
                "S_SENDING",
                "S_WAITING",
                "S_WAITING_PING_SENDING",
                "S_WAITING_PING_SENT",
                "S_CANCEL_AT_SENDING"};
            const char* inReqStateNames[] = {
                "S_WAITING",
                "S_RESPONSE_SENDING",
                "S_CANCELED"};
            res += "\nOut requests:\n";
            for (TOutRequestHash::const_iterator i = OutRequests.begin(); i != OutRequests.end(); ++i) {
                const TGUID& gg = i->first;
                const TOutRequestState& s = i->second;
                bool isSync = SyncRequests.find(gg) != SyncRequests.end();
                snprintf(buf, sizeof(buf), "%s\t%s  %s  TimePassed: %g  %s\n",
                        GetAddressAsString(s.Connection->GetAddress()).c_str(), GetGuidAsString(gg).c_str(), outReqStateNames[s.State],
                        s.TimePassed * 1000,
                        isSync ? "isSync" : "");
                res += buf;
            }
            res += "\nIn requests:\n";
            for (TInRequestHash::const_iterator i = InRequests.begin(); i != InRequests.end(); ++i) {
                const TGUID& gg = i->first;
                const TInRequestState& s = i->second;
                snprintf(buf, sizeof(buf), "%s\t%s  %s\n",
                        GetAddressAsString(s.Connection->GetAddress()).c_str(), GetGuidAsString(gg).c_str(), inReqStateNames[s.State]);
                res += buf;
            }
            return res;
        }
        TString GetDebugInfo() override {
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::DEBUG_INFO);
            ExecStatsRequest(req);
            return req->DebugInfo;
        }
        void GetRequestQueueSize(TRequesterQueueStats* res) override {
            TRequesterUserQueueSizes* qs = QueueSizes.Get();
            res->ReqCount = (int)AtomicGet(qs->ReqCount);
            res->RespCount = (int)AtomicGet(qs->RespCount);
            res->ReqQueueSize = (int)AtomicGet(qs->ReqQueueSize);
            res->RespQueueSize = (int)AtomicGet(qs->RespQueueSize);
        }
        TRequesterUserQueueSizes* GetQueueSizes() const {
            return QueueSizes.Get();
        }
        IRequestOps* CreateSubRequester() override;
        void EnableReportRequestCancel() override {
            ReportRequestCancel = true;
        }
        void EnableReportSendRequestAcc() override {
            ReportSendRequestAcc = true;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    static void ReadShm(TPosixSharedMemory* shm, TVector<char>* data) {
        Y_ASSERT(shm);
        const size_t dataSize = shm->GetSize();
        data->resize(dataSize);
        memcpy(&(*data)[0], shm->GetPtr(), dataSize);
    }

    static void LoadRequestData(TUdpHttpRequest* res) {
        if (!res)
            return;
        {
            TBlockChainIterator reqData(res->DataHolder->Data->GetChain());
            char pktType;
            reqData.Read(&pktType, 1);

            TGUID reqId;
            reqData.Read(&reqId, sizeof(reqId));
            Y_ASSERT(reqId == res->ReqId);

            TVector<char> flags;
            ReadYArr(&reqData, &flags);
            Y_ASSERT(!reqData.HasFailed());

            if (flags.size() >= 8 + 2) {
                res->Colors.SetResponseDataTos((i16&)flags[0]);
                res->Colors.SetResponseAckTos((i16&)flags[2]);
                res->Colors.SetRequestDataTos((i16&)flags[4]);
                res->Colors.SetRequestAckTos((i16&)flags[6]);
                res->Colors.SetNetlibaRequestColor((ui8&)flags[8]);
                res->Colors.SetNetlibaResponseColor((ui8&)flags[9]);
            }

            if (flags.size() >= 8 + 2 + 1) {
                if (flags[10] & HF_HP_QUEUE) {
                    res->Colors.SetPriority(PP_HIGH);
                }
            }

            ReadArr(&reqData, &res->Url);

            if (pktType == PKT_REQUEST) {
                ReadYArr(&reqData, &res->Data);
            } else if (pktType == PKT_LOCAL_REQUEST) {
                ReadShm(res->DataHolder->Data->GetSharedData(), &res->Data);
            } else
                Y_ASSERT(0);
            if (reqData.HasFailed()) {
                Y_ASSERT(0 && "wrong format, memory corruption suspected");
                res->Url = "";
                res->Data.clear();
            }
        }
        res->DataHolder.Reset(nullptr);
    }

    static void LoadResponseData(TUdpHttpResponse* res) {
        if (!res || res->DataHolder.Get() == nullptr)
            return;
        {
            TBlockChainIterator reqData(res->DataHolder->Data->GetChain());
            char pktType;
            reqData.Read(&pktType, 1);
            TGUID guid;
            reqData.Read(&guid, sizeof(guid));
            Y_ASSERT(res->ReqId == guid);
            if (pktType == PKT_RESPONSE) {
                ReadYArr(&reqData, &res->Data);
            } else if (pktType == PKT_LOCAL_RESPONSE) {
                ReadShm(res->DataHolder->Data->GetSharedData(), &res->Data);
            } else
                Y_ASSERT(0);
            if (reqData.HasFailed()) {
                Y_ASSERT(0 && "wrong format, memory corruption suspected");
                res->Ok = TUdpHttpResponse::FAILED;
                res->Data.clear();
                res->Error = "wrong response format";
            }
        }
        res->DataHolder.Reset(nullptr);
    }

    //////////////////////////////////////////////////////////////////////////
    // IRequestOps::TWaitResponse
    TUdpHttpResponse* IRequestOps::TWaitResponse::GetResponse() {
        if (!Response)
            return nullptr;
        TUdpHttpResponse* res = Response;
        Response = nullptr;
        LoadResponseData(res);
        return res;
    }

    void IRequestOps::TWaitResponse::SetResponse(TUdpHttpResponse* r) {
        Y_ASSERT(Response == nullptr || r == nullptr);
        if (r)
            Response = r;
        CompleteEvent.Signal();
    }

    //////////////////////////////////////////////////////////////////////////
    // TRequesterUserQueues
    TUdpHttpRequest* TRequesterUserQueues::GetRequest() {
        TLockFreeQueue<TUdpHttpRequest*>& rl = RequestListHighPiority.IsEmpty() ? RequestList : RequestListHighPiority;
        TUdpHttpRequest* res = nullptr;
        rl.Dequeue(&res);
        if (res) {
            AtomicAdd(QueueSizes->ReqCount, -1);
            AtomicAdd(QueueSizes->ReqQueueSize, -GetTransferSize(res->DataHolder.Get()));
        }
        UpdateAsyncSignalState();
        LoadRequestData(res);
        return res;
    }

    TUdpHttpResponse* TRequesterUserQueues::GetResponse() {
        TLockFreeQueue<TUdpHttpResponse*>& rl = ResponseListHighPiority.IsEmpty() ? ResponseList : ResponseListHighPiority;
        TUdpHttpResponse* res = nullptr;
        rl.Dequeue(&res);
        if (res) {
            AtomicAdd(QueueSizes->RespCount, -1);
            AtomicAdd(QueueSizes->RespQueueSize, -GetTransferSize(res->DataHolder.Get()));
        }
        UpdateAsyncSignalState();
        LoadResponseData(res);
        return res;
    }

    //////////////////////////////////////////////////////////////////////////
    class TRequestOps: public IRequestOps {
        TIntrusivePtr<TUdpHttp> Requester;
        TIntrusivePtr<TRequesterUserQueues> UserQueues;

    public:
        TRequestOps(TUdpHttp* req)
            : Requester(req)
        {
            UserQueues = new TRequesterUserQueues(req->GetQueueSizes());
        }
        void SendRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) override {
            Requester->SendRequestImpl(addr, url, data, reqId, nullptr, UserQueues.Get());
        }
        void CancelRequest(const TGUID& reqId) override {
            Requester->CancelRequest(reqId);
        }
        void BreakRequest(const TGUID& reqId) override {
            Requester->BreakRequest(reqId);
        }

        void SendResponse(const TGUID& reqId, TVector<char>* data, const TColors& colors) override {
            Requester->SendResponseImpl(reqId, colors.GetPriority(), data, colors);
        }
        void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data, const TColors& colors) override {
            Requester->SendResponseImpl(reqId, PP_LOW, data, colors);
        }
        TUdpHttpRequest* GetRequest() override {
            Y_ASSERT(0);
            //return UserQueues.GetRequest();
            return nullptr; // all requests are routed to the main requester
        }
        TUdpHttpResponse* GetResponse() override {
            return UserQueues->GetResponse();
        }
        bool GetRequestCancel(TGUID*) override {
            Y_ASSERT(0);
            return false; // all request cancels are routed to the main requester
        }
        bool GetSendRequestAcc(TGUID* req) override {
            return UserQueues->GetSendRequestAcc(req);
        }
        // sync mode
        TUdpHttpResponse* Request(const TConnectionAddress& addr, const TString& url, TVector<char>* data) override {
            return Requester->Request(addr, url, data);
        }
        TIntrusivePtr<TWaitResponse> WaitableRequest(const TConnectionAddress& addr, const TString& url, TVector<char>* data) override {
            return Requester->WaitableRequest(addr, url, data);
        }
        //
        TMuxEvent& GetAsyncEvent() override {
            return UserQueues->GetAsyncEvent();
        }
    };

    IRequestOps* TUdpHttp::CreateSubRequester() {
        return new TRequestOps(this);
    }

    //////////////////////////////////////////////////////////////////////////
    void AbortOnFailedRequest(TUdpHttpResponse* answer) {
        if (answer && answer->Ok == TUdpHttpResponse::FAILED) {
            printf("Failed request to host %s\n", GetAddressAsString(answer->PeerAddress).c_str());
            fflush(nullptr);
            Y_ASSERT(0);
            abort();
        }
    }

    TString GetDebugInfo(const TUdpAddress& addr, double timeout) {
        NHPTimer::STime start;
        NHPTimer::GetTime(&start);
        TIntrusivePtr<IUdpHost> host = CreateUdpHost(0);
        if (!host) {
            fprintf(stderr, "CreateUdpHost failed!\n");
            return TString();
        }

        TIntrusivePtr<IConnection> connection = host->Connect(addr, TConnectionSettings());
        {
            TAutoPtr<TRopeDataPacket> rq = new TRopeDataPacket;
            rq->Write((char)PKT_GETDEBUGINFO);
            host->Send(connection, rq.Release(), PP_HIGH, TTos(), DEFAULT_NETLIBA_COLOR);
        }
        for (;;) {
            TAutoPtr<TUdpRequest> ptr = host->GetRequest();
            if (ptr.Get()) {
                TBlockChainIterator reqData(ptr->Data->GetChain());
                int sz = reqData.GetSize();
                TString res;
                res.resize(sz);
                reqData.Read(res.begin(), sz);
                return res;
            }
            host->Step();
            host->Wait(0.1f);

            NHPTimer::STime now;
            NHPTimer::GetTime(&now);
            if (NHPTimer::GetSeconds(now - start) > timeout) {
                return TString();
            }
        }
    }

    void StopAllNetLibaThreads() {
        AtomicSwap(&PanicAttack, 1); // AAAA!!!!
    }

    void SetNetLibaHeartbeatTimeout(double timeoutSec) {
        NetLibaHeartbeat();
        HeartbeatTimeout.store(timeoutSec, std::memory_order_release);
    }

    void NetLibaHeartbeat() {
        NHPTimer::STime now;
        NHPTimer::GetTime(&now);
        LastHeartbeat.store(now, std::memory_order_release);
    }

    IRequester* CreateHttpUdpRequester(int port) {
        if (AtomicAdd(PanicAttack, 0))
            return nullptr;

        TIntrusivePtr<ISocket> socket = NNetlibaSocket::CreateBestRecvSocket();
        socket->Open(port);
        if (!socket->IsValid())
            return nullptr;

        return CreateHttpUdpRequester(socket);
    }

    IRequester* CreateHttpUdpRequester(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
        if (AtomicAdd(PanicAttack, 0))
            return nullptr;

        TIntrusivePtr<TUdpHttp> res(new TUdpHttp);
        if (!res->Start(socket))
            return nullptr;
        return res.Release();
    }

}
