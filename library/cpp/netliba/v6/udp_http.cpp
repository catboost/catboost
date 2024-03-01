#include "stdafx.h"
#include "udp_http.h"
#include "udp_client_server.h"
#include "udp_socket.h"
#include "cpu_affinity.h"

#include <library/cpp/threading/atomic/bool.h>

#include <util/system/hp_timer.h>
#include <util/thread/lfqueue.h>
#include <util/system/thread.h>
#include <util/system/spinlock.h>
#if !defined(_win_)
#include <signal.h>
#include <pthread.h>
#endif
#include "block_chain.h"
#include <util/system/shmat.h>

#include <atomic>

namespace NNetliba {
    const float HTTP_TIMEOUT = 15.0f;
    const int MIN_SHARED_MEM_PACKET = 1000;

    static ::NAtomic::TBool PanicAttack;
    static std::atomic<NHPTimer::STime> LastHeartbeat;
    static std::atomic<double> HeartbeatTimeout;

    static int GetPacketSize(TRequest* req) {
        if (req && req->Data.Get())
            return req->Data->GetSize();
        return 0;
    }

    static bool IsLocalFast(const TUdpAddress& addr) {
        if (addr.IsIPv4()) {
            return IsLocalIPv4(addr.GetIPv4());
        } else {
            return IsLocalIPv6(addr.Network, addr.Interface);
        }
    }

    bool IsLocal(const TUdpAddress& addr) {
        InitLocalIPList();
        return IsLocalFast(addr);
    }

    TUdpHttpRequest::~TUdpHttpRequest() {
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
        TLockFreeQueue<TUdpHttpRequest*> ReqList;
        TLockFreeQueue<TUdpHttpResponse*> ResponseList;
        TLockFreeStack<TGUID> CancelList, SendRequestAccList; // any order will do
        TMuxEvent AsyncEvent;

        void UpdateAsyncSignalState() {
            // not sure about this one. Idea is that AsyncEvent.Reset() is a memory barrier
            if (ReqList.IsEmpty() && ResponseList.IsEmpty() && CancelList.IsEmpty() && SendRequestAccList.IsEmpty()) {
                AsyncEvent.Reset();
                if (!ReqList.IsEmpty() || !ResponseList.IsEmpty() || !CancelList.IsEmpty() || !SendRequestAccList.IsEmpty())
                    AsyncEvent.Signal();
            }
        }
        ~TRequesterUserQueues() override {
            EraseList(&ReqList);
            EraseList(&ResponseList);
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
            AtomicAdd(QueueSizes->ReqQueueSize, GetPacketSize(res->DataHolder.Get()));
            ReqList.Enqueue(res);
            AsyncEvent.Signal();
        }
        void AddResponse(TUdpHttpResponse* res) {
            AtomicAdd(QueueSizes->RespCount, 1);
            AtomicAdd(QueueSizes->RespQueueSize, GetPacketSize(res->DataHolder.Get()));
            ResponseList.Enqueue(res);
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
            S_CANCEL_AFTER_SENDING
        };
        EState State;
        TUdpAddress Address;
        double TimePassed;
        int PingTransferId;
        TIntrusivePtr<TRequesterUserQueues> UserQueues;

        TOutRequestState()
            : State(S_SENDING)
            , TimePassed(0)
            , PingTransferId(-1)
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
        TUdpAddress Address;

        TInRequestState()
            : State(S_WAITING)
        {
        }
        TInRequestState(const TUdpAddress& address)
            : State(S_WAITING)
            , Address(address)
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

        struct TSendRequest {
            TUdpAddress Addr;
            TAutoPtr<TRopeDataPacket> Data;
            TGUID ReqGuid;
            TIntrusivePtr<TWaitResponse> WR;
            TIntrusivePtr<TRequesterUserQueues> UserQueues;
            ui32 Crc32;

            TSendRequest()
                : Crc32(0)
            {
            }
            TSendRequest(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket>* data, const TGUID& reqguid, TWaitResponse* wr, TRequesterUserQueues* userQueues)
                : Addr(addr)
                , Data(*data)
                , ReqGuid(reqguid)
                , WR(wr)
                , UserQueues(userQueues)
                , Crc32(CalcChecksum(Data->GetChain()))
            {
            }
        };
        struct TSendResponse {
            TVector<char> Data;
            TGUID ReqGuid;
            ui32 DataCrc32;
            EPacketPriority Priority;

            TSendResponse()
                : DataCrc32(0)
                , Priority(PP_NORMAL)
            {
            }
            TSendResponse(const TGUID& reqguid, EPacketPriority prior, TVector<char>* data)
                : ReqGuid(reqguid)
                , DataCrc32(0)
                , Priority(prior)
            {
                if (data && !data->empty()) {
                    data->swap(Data);
                    DataCrc32 = TIncrementalChecksumCalcer::CalcBlockSum(&Data[0], Data.ysize());
                }
            }
        };
        struct TCancelRequest {
            TGUID ReqGuid;

            TCancelRequest() = default;
            TCancelRequest(const TGUID& reqguid)
                : ReqGuid(reqguid)
            {
            }
        };
        struct TBreakRequest {
            TGUID ReqGuid;

            TBreakRequest() = default;
            TBreakRequest(const TGUID& reqguid)
                : ReqGuid(reqguid)
            {
            }
        };

        TThread myThread;
        bool KeepRunning, AbortTransactions;
        TSpinLock cs;
        TSystemEvent HasStarted;

        NHPTimer::STime PingsSendT;

        TIntrusivePtr<IUdpHost> Host;
        TIntrusivePtr<NNetlibaSocket::ISocket> Socket;
        typedef THashMap<TGUID, TOutRequestState, TGUIDHash> TOutRequestHash;
        typedef THashMap<TGUID, TInRequestState, TGUIDHash> TInRequestHash;
        TOutRequestHash OutRequests;
        TInRequestHash InRequests;

        typedef THashMap<int, TTransferPurpose> TTransferHash;
        TTransferHash TransferHash;

        typedef THashMap<TGUID, TIntrusivePtr<TWaitResponse>, TGUIDHash> TSyncRequests;
        TSyncRequests SyncRequests;

        // hold it here to not construct on every DoSends()
        typedef THashSet<TGUID, TGUIDHash> TAnticipateCancels;
        TAnticipateCancels AnticipateCancels;

        TLockFreeQueue<TSendRequest*> SendReqList;
        TLockFreeQueue<TSendResponse*> SendRespList;
        TLockFreeQueue<TCancelRequest> CancelReqList;
        TLockFreeQueue<TBreakRequest> BreakReqList;

        TIntrusivePtr<TRequesterUserQueueSizes> QueueSizes;
        TIntrusivePtr<TRequesterUserQueues> UserQueues;

        struct TStatsRequest: public TThrRefBase {
            enum EReq {
                PENDING_SIZE,
                DEBUG_INFO,
                HAS_IN_REQUEST,
                GET_PEER_ADDRESS,
                GET_PEER_QUEUE_STATS,
            };
            EReq Req;
            TRequesterPendingDataStats PendingDataSize;
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

        void FinishRequest(TOutRequestHash::iterator i, TUdpHttpResponse::EResult ok, TAutoPtr<TRequest> data, const char* error = nullptr) {
            TOutRequestState& s = i->second;
            TUdpHttpResponse* res = new TUdpHttpResponse;
            res->DataHolder = data;
            res->ReqId = i->first;
            res->PeerAddress = s.Address;
            res->Ok = ok;
            if (ok == TUdpHttpResponse::FAILED)
                res->Error = error ? error : "request failed";
            else if (ok == TUdpHttpResponse::CANCELED)
                res->Error = error ? error : "request cancelled";
            TSyncRequests::iterator k = SyncRequests.find(res->ReqId);
            if (k != SyncRequests.end()) {
                TIntrusivePtr<TWaitResponse>& wr = k->second;
                wr->SetResponse(res);
                SyncRequests.erase(k);
            } else {
                s.UserQueues->AddResponse(res);
            }

            OutRequests.erase(i);
        }
        int SendWithHighPriority(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket> data) {
            ui32 crc32 = CalcChecksum(data->GetChain());
            return Host->Send(addr, data.Release(), crc32, nullptr, PP_HIGH);
        }
        void ProcessIncomingPackets() {
            TVector<TGUID, TCustomAllocator<TGUID>> failedRequests;
            for (;;) {
                TAutoPtr<TRequest> req = Host->GetRequest();
                if (req.Get() == nullptr) {
                    if (!failedRequests.empty()) {
                        // we want to handle following sequence of events
                        // <- send ping
                        // -> send response over IB
                        // -> send ping response (no such request) over UDP
                        // Now if we are lucky enough we can get IB response waiting in the IB receive queue
                        // at the same time response sender will receive "send complete" from IB
                        // indeed, IB delivered message (but it was not parsed by ib_cs.cpp yet)
                        // so after receiving "send response complete" event resposne sender can legally response
                        // to pings with "no such request"
                        // but ping responses can be sent over UDP
                        // So we can run into situation with negative ping response in
                        // UDP receive queue and response waiting unprocessed in IB receive queue
                        // to check that there is no response in the IB queue we have to process IB queues
                        // so we call IBStep()
                        Host->IBStep();
                        req = Host->GetRequest();
                        if (req.Get() == nullptr) {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                TBlockChainIterator reqData(req->Data->GetChain());
                char pktType;
                reqData.Read(&pktType, 1);
                switch (pktType) {
                    case PKT_REQUEST:
                    case PKT_LOCAL_REQUEST: {
                        //printf("recv PKT_REQUEST or PKT_LOCAL_REQUEST\n");
                        TGUID reqId = req->Guid;
                        TInRequestHash::iterator z = InRequests.find(reqId);
                        if (z != InRequests.end()) {
                            // oops, this request already exists!
                            // might happen if request can be stored in single packet
                            // and this packet had source IP broken during transmission and managed to pass crc checks
                            // since we already reported wrong source address for this request to the user
                            // the best thing we can do is to stop the program to avoid further complications
                            // but we just report the accident to stderr
                            fprintf(stderr, "Jackpot, same request %s received twice from %s and earlier from %s\n",
                                    GetGuidAsString(reqId).c_str(), GetAddressAsString(z->second.Address).c_str(),
                                    GetAddressAsString(req->Address).c_str());
                        } else {
                            InRequests[reqId] = TInRequestState(req->Address);

                            //printf("InReq %s PKT_REQUEST recv ... -> S_WAITING\n", GetGuidAsString(reqId).c_str());

                            TUdpHttpRequest* res = new TUdpHttpRequest;
                            res->ReqId = reqId;
                            res->PeerAddress = req->Address;
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
                        SendWithHighPriority(req->Address, ms.Release());
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
                            ; //Y_ASSERT(0); // actually possible with some packet orders
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
                                        Y_ASSERT(s.PingTransferId >= 0);
                                        TTransferHash::iterator k = TransferHash.find(s.PingTransferId);
                                        if (k != TransferHash.end())
                                            TransferHash.erase(k);
                                        else
                                            Y_ASSERT(0);
                                        s.PingTransferId = -1;
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
                            ; //Y_ASSERT(0); // does happen
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
                            if (s.State != TInRequestState::S_CANCELED && ReportRequestCancel)
                                UserQueues->AddCancel(guid);
                            s.State = TInRequestState::S_CANCELED;
                            //printf("InReq %s PKT_CANCEL recv\n", GetGuidAsString(guid).c_str());
                        }
                    } break;
                    case PKT_GETDEBUGINFO: {
                        //printf("recv PKT_GETDEBUGINFO\n");
                        TString dbgInfo = GetDebugInfoLocked();
                        TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                        ms->Write(dbgInfo.c_str(), (int)dbgInfo.size());
                        SendWithHighPriority(req->Address, ms);
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
                    FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: recv no such query");
            }
        }
        void AnalyzeSendResults() {
            TSendResult res;
            while (Host->GetSendResult(&res)) {
                //printf("Send result received\n");
                TTransferHash::iterator k1 = TransferHash.find(res.TransferId);
                if (k1 != TransferHash.end()) {
                    const TTransferPurpose& tp = k1->second;
                    switch (tp.Dir) {
                        case DIR_OUT: {
                            TOutRequestHash::iterator i = OutRequests.find(tp.Guid);
                            if (i != OutRequests.end()) {
                                const TGUID& reqId = i->first;
                                TOutRequestState& s = i->second;
                                switch (s.State) {
                                    case TOutRequestState::S_SENDING:
                                        if (!res.Success) {
                                            FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: state S_SENDING");
                                            //printf("OutReq %s AnalyzeSendResults() S_SENDING -> failed\n", GetGuidAsString(reqId).c_str());
                                        } else {
                                            if (ReportSendRequestAcc) {
                                                if (s.UserQueues.Get()) {
                                                    s.UserQueues->AddSendRequestAcc(reqId);
                                                } else {
                                                    // waitable request?
                                                    TSyncRequests::iterator k2 = SyncRequests.find(reqId);
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
                                    case TOutRequestState::S_CANCEL_AFTER_SENDING:
                                        DoSendCancel(s.Address, reqId);
                                        FinishRequest(i, TUdpHttpResponse::CANCELED, nullptr, "request failed: state S_CANCEL_AFTER_SENDING");
                                        break;
                                    case TOutRequestState::S_WAITING:
                                    case TOutRequestState::S_WAITING_PING_SENT:
                                        Y_ASSERT(0);
                                        break;
                                    case TOutRequestState::S_WAITING_PING_SENDING:
                                        Y_ASSERT(s.PingTransferId >= 0 && s.PingTransferId == res.TransferId);
                                        if (!res.Success) {
                                            FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: state S_WAITING_PING_SENDING");
                                            //printf("OutReq %s AnalyzeSendResults() S_WAITING_PING_SENDING -> failed\n", GetGuidAsString(reqId).c_str());
                                        } else {
                                            s.PingTransferId = -1;
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
                                //if (res.Success)
                                //    printf("InReq %s AnalyzeSendResults() ... -> finished\n", GetGuidAsString(tp.Guid).c_str());
                                //else
                                //    printf("InReq %s AnalyzeSendResults() ... -> failed response send\n", GetGuidAsString(tp.Guid).c_str());
                            }
                        } break;
                        default:
                            Y_ASSERT(0);
                            break;
                    }
                    TransferHash.erase(k1);
                }
            }
        }
        void SendPingsIfNeeded() {
            NHPTimer::STime tChk = PingsSendT;
            float deltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (deltaT < 0.05) {
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
                                int transId = SendWithHighPriority(s.Address, ms.Release());
                                TransferHash[transId] = TTransferPurpose(DIR_OUT, guid);
                                s.State = TOutRequestState::S_WAITING_PING_SENDING;
                                //printf("OutReq %s SendPingsIfNeeded() S_WAITING -> S_WAITING_PING_SENDING\n", GetGuidAsString(guid).c_str());
                                s.PingTransferId = transId;
                            }
                            break;
                        case TOutRequestState::S_WAITING_PING_SENT:
                            s.TimePassed += deltaT;
                            if (s.TimePassed > HTTP_TIMEOUT) {
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
        void Step() {
            {
                TGuard<TSpinLock> lock(cs);
                DoSends();
            }
            Host->Step();
            for (TIntrusivePtr<TStatsRequest> req; StatsReqList.Dequeue(&req);) {
                switch (req->Req) {
                    case TStatsRequest::PENDING_SIZE:
                        Host->GetPendingDataSize(&req->PendingDataSize);
                        break;
                    case TStatsRequest::DEBUG_INFO: {
                        TGuard<TSpinLock> lock(cs);
                        req->DebugInfo = GetDebugInfoLocked();
                    } break;
                    case TStatsRequest::HAS_IN_REQUEST: {
                        TGuard<TSpinLock> lock(cs);
                        req->RequestFound = (InRequests.find(req->RequestId) != InRequests.end());
                    } break;
                    case TStatsRequest::GET_PEER_ADDRESS: {
                        TGuard<TSpinLock> lock(cs);
                        TInRequestHash::const_iterator i = InRequests.find(req->RequestId);
                        if (i != InRequests.end()) {
                            req->PeerAddress = i->second.Address;
                        } else {
                            TOutRequestHash::const_iterator o = OutRequests.find(req->RequestId);
                            if (o != OutRequests.end()) {
                                req->PeerAddress = o->second.Address;
                            } else {
                                req->PeerAddress = TUdpAddress();
                            }
                        }
                    } break;
                    case TStatsRequest::GET_PEER_QUEUE_STATS:
                        req->QueueStats = Host->GetQueueStats(req->PeerAddress);
                        break;
                    default:
                        Y_ASSERT(0);
                        break;
                }
                req->Complete.Signal();
            }
            {
                TGuard<TSpinLock> lock(cs);
                DoSends();
                ProcessIncomingPackets();
                AnalyzeSendResults();
                SendPingsIfNeeded();
            }
        }
        void Wait() {
            Host->Wait(0.1f);
        }
        void DoSendCancel(const TUdpAddress& addr, const TGUID& req) {
            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            ms->Write((char)PKT_CANCEL);
            ms->Write(req);
            SendWithHighPriority(addr, ms);
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
                        // we are in trouble - have not sent request and we already have to cancel it, wait send
                        s.State = TOutRequestState::S_CANCEL_AFTER_SENDING;
                    } else {
                        DoSendCancel(s.Address, rc.ReqGuid);
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
                        InRequests.erase(i);
                        continue;
                    }

                    Y_ASSERT(s.State == TInRequestState::S_WAITING);
                    s.State = TInRequestState::S_RESPONSE_SENDING;
                    //printf("InReq %s SendResponse() ... -> S_RESPONSE_SENDING (pkt %s)\n", GetGuidAsString(reqId).c_str(), GetGuidAsString(lowPktGuid).c_str());

                    TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                    ui32 crc32 = 0;
                    int dataSize = rd->Data.ysize();
                    if (rd->Data.ysize() > MIN_SHARED_MEM_PACKET && IsLocalFast(s.Address)) {
                        TIntrusivePtr<TSharedMemory> shm = new TSharedMemory;
                        if (shm->Create(dataSize)) {
                            ms->Write((char)PKT_LOCAL_RESPONSE);
                            ms->Write(rd->ReqGuid);
                            memcpy(shm->GetPtr(), &rd->Data[0], dataSize);
                            TVector<char> empty;
                            rd->Data.swap(empty);
                            ms->AttachSharedData(shm);
                            crc32 = CalcChecksum(ms->GetChain());
                        }
                    }
                    if (ms->GetSharedData() == nullptr) {
                        ms->Write((char)PKT_RESPONSE);
                        ms->Write(rd->ReqGuid);

                        // to offload crc calcs from inner thread, crc of data[] is calced outside and passed in DataCrc32
                        // this means that we are calculating crc when shared memory is used
                        // it is hard to avoid since in SendResponse() we don't know if shared mem will be used (peer address is not available there)
                        TIncrementalChecksumCalcer csCalcer;
                        AddChain(&csCalcer, ms->GetChain());
                        // here we are replicating the way WriteDestructive serializes data
                        csCalcer.AddBlock(&dataSize, sizeof(dataSize));
                        csCalcer.AddBlockSum(rd->DataCrc32, dataSize);
                        crc32 = csCalcer.CalcChecksum();

                        ms->WriteDestructive(&rd->Data);
                        //ui32 chkCrc = CalcChecksum(ms->GetChain()); // can not use since its slow for large responses
                        //Y_ASSERT(chkCrc == crc32);
                    }

                    int transId = Host->Send(s.Address, ms.Release(), crc32, nullptr, rd->Priority);
                    TransferHash[transId] = TTransferPurpose(DIR_IN, rd->ReqGuid);
                }
            }
            {
                // sending requests
                for (TSendRequest* rd = nullptr; SendReqList.Dequeue(&rd); delete rd) {
                    Y_ASSERT(OutRequests.find(rd->ReqGuid) == OutRequests.end());

                    {
                        TOutRequestState& s = OutRequests[rd->ReqGuid];
                        s.State = TOutRequestState::S_SENDING;
                        s.Address = rd->Addr;
                        s.UserQueues = rd->UserQueues;
                        //printf("OutReq %s SendRequest() ... -> S_SENDING\n", GetGuidAsString(guid).c_str());
                    }

                    if (rd->WR.Get())
                        SyncRequests[rd->ReqGuid] = rd->WR;

                    if (AnticipateCancels.find(rd->ReqGuid) != AnticipateCancels.end()) {
                        FinishRequest(OutRequests.find(rd->ReqGuid), TUdpHttpResponse::CANCELED, nullptr, "request canceled before transmitting");
                    } else {
                        TGUID pktGuid = rd->ReqGuid; // request packet id should match request id
                        int transId = Host->Send(rd->Addr, rd->Data.Release(), rd->Crc32, &pktGuid, PP_NORMAL);
                        TransferHash[transId] = TTransferPurpose(DIR_OUT, rd->ReqGuid);
                    }
                }
            }
            if (!AnticipateCancels.empty()) {
                AnticipateCancels.clear();
            }
        }

    public:
        void SendRequestImpl(const TUdpAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId,
                             TWaitResponse* wr, TRequesterUserQueues* userQueues) {
            if (data && data->size() > MAX_PACKET_SIZE) {
                Y_ABORT_UNLESS(0, "data size is too large");
            }
            //printf("SendRequest(%s)\n", url.c_str());
            if (wr)
                wr->SetReqId(reqId);

            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            if (data && data->ysize() > MIN_SHARED_MEM_PACKET && IsLocalFast(addr)) {
                int dataSize = data->ysize();
                TIntrusivePtr<TSharedMemory> shm = new TSharedMemory;
                if (shm->Create(dataSize)) {
                    ms->Write((char)PKT_LOCAL_REQUEST);
                    ms->WriteStroka(url);
                    memcpy(shm->GetPtr(), &(*data)[0], dataSize);
                    TVector<char> empty;
                    data->swap(empty);
                    ms->AttachSharedData(shm);
                }
            }
            if (ms->GetSharedData() == nullptr) {
                ms->Write((char)PKT_REQUEST);
                ms->WriteStroka(url);
                ms->WriteDestructive(data);
            }

            SendReqList.Enqueue(new TSendRequest(addr, &ms, reqId, wr, userQueues));
            Host->CancelWait();
        }

        void SendRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) override {
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

        void SendResponseImpl(const TGUID& reqId, EPacketPriority prior, TVector<char>* data) // non-virtual, for direct call from TRequestOps
        {
            if (data && data->size() > MAX_PACKET_SIZE) {
                Y_ABORT_UNLESS(0, "data size is too large");
            }
            SendRespList.Enqueue(new TSendResponse(reqId, prior, data));
            Host->CancelWait();
        }
        void SendResponse(const TGUID& reqId, TVector<char>* data) override {
            SendResponseImpl(reqId, PP_NORMAL, data);
        }
        void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data) override {
            SendResponseImpl(reqId, PP_LOW, data);
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
        TUdpHttpResponse* Request(const TUdpAddress& addr, const TString& url, TVector<char>* data) override {
            TIntrusivePtr<TWaitResponse> wr = WaitableRequest(addr, url, data);
            wr->Wait();
            return wr->GetResponse();
        }
        TIntrusivePtr<TWaitResponse> WaitableRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data) override {
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
            AbortTransactions = true;
            KeepRunning = false;
            UserQueues->AsyncSignal();
            // calcel all outgoing requests
            TGuard<TSpinLock> lock(cs);
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
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::PENDING_SIZE);
            ExecStatsRequest(req);
            *res = req->PendingDataSize;
        }
        bool HasRequest(const TGUID& reqId) override {
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::HAS_IN_REQUEST);
            req->RequestId = reqId;
            ExecStatsRequest(req);
            return req->RequestFound;
        }

    private:
        void FinishOutstandingTransactions() {
            // wait all pending requests, all new requests are canceled
            while ((!OutRequests.empty() || !InRequests.empty() || !SendRespList.IsEmpty() || !SendReqList.IsEmpty()) && !PanicAttack) {
                while (TUdpHttpRequest* req = GetRequest()) {
                    TInRequestHash::iterator i = InRequests.find(req->ReqId);
                    //printf("dropping request(%s) (thread %d)\n", req->Url.c_str(), ThreadId());
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
            while (pThis->KeepRunning && !PanicAttack) {
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
            if (!pThis->AbortTransactions && !PanicAttack)
                pThis->FinishOutstandingTransactions();
            pThis->Host = nullptr;
            return nullptr;
        }
        ~TUdpHttp() override {
            if (myThread.Running()) {
                KeepRunning = false;
                myThread.Join();
            }
            for (TIntrusivePtr<TStatsRequest> req; StatsReqList.Dequeue(&req);) {
                req->Complete.Signal();
            }
        }

    public:
        TUdpHttp()
            : myThread(TThread::TParams(ExecServerThread, (void*)this).SetName("nl6_udp_host"))
            , KeepRunning(true)
            , AbortTransactions(false)
            , PingsSendT(0)
            , ReportRequestCancel(false)
            , ReportSendRequestAcc(false)
        {
            NHPTimer::GetTime(&PingsSendT);
            QueueSizes = new TRequesterUserQueueSizes;
            UserQueues = new TRequesterUserQueues(QueueSizes.Get());
        }
        bool Start(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
            Y_ASSERT(Host.Get() == nullptr);
            Socket = socket;
            myThread.Start();
            HasStarted.Wait();

            if (Host.Get()) {
                return true;
            }
            Socket.Drop();
            return false;
        }
        TString GetDebugInfoLocked() {
            TString res = KeepRunning ? "State: running\n" : "State: stopping\n";
            res += Host->GetDebugInfo();

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
                "S_CANCEL_AFTER_SENDING"};
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
                        GetAddressAsString(s.Address).c_str(), GetGuidAsString(gg).c_str(), outReqStateNames[s.State],
                        s.TimePassed * 1000,
                        isSync ? "isSync" : "");
                res += buf;
            }
            res += "\nIn requests:\n";
            for (TInRequestHash::const_iterator i = InRequests.begin(); i != InRequests.end(); ++i) {
                const TGUID& gg = i->first;
                const TInRequestState& s = i->second;
                snprintf(buf, sizeof(buf), "%s\t%s  %s\n",
                        GetAddressAsString(s.Address).c_str(), GetGuidAsString(gg).c_str(), inReqStateNames[s.State]);
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
        TIntrusivePtr<IPeerQueueStats> GetQueueStats(const TUdpAddress& addr) override {
            TIntrusivePtr<TStatsRequest> req = new TStatsRequest(TStatsRequest::GET_PEER_QUEUE_STATS);
            req->PeerAddress = addr;
            ExecStatsRequest(req);
            return req->QueueStats;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    static void ReadShm(TSharedMemory* shm, TVector<char>* data) {
        Y_ASSERT(shm);
        int dataSize = shm->GetSize();
        data->yresize(dataSize);
        memcpy(&(*data)[0], shm->GetPtr(), dataSize);
    }

    static void LoadRequestData(TUdpHttpRequest* res) {
        if (!res)
            return;
        {
            TBlockChainIterator reqData(res->DataHolder->Data->GetChain());
            char pktType;
            reqData.Read(&pktType, 1);
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
        TUdpHttpRequest* res = nullptr;
        ReqList.Dequeue(&res);
        if (res) {
            AtomicAdd(QueueSizes->ReqCount, -1);
            AtomicAdd(QueueSizes->ReqQueueSize, -GetPacketSize(res->DataHolder.Get()));
        }
        UpdateAsyncSignalState();
        LoadRequestData(res);
        return res;
    }

    TUdpHttpResponse* TRequesterUserQueues::GetResponse() {
        TUdpHttpResponse* res = nullptr;
        ResponseList.Dequeue(&res);
        if (res) {
            AtomicAdd(QueueSizes->RespCount, -1);
            AtomicAdd(QueueSizes->RespQueueSize, -GetPacketSize(res->DataHolder.Get()));
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
        void SendRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data, const TGUID& reqId) override {
            Requester->SendRequestImpl(addr, url, data, reqId, nullptr, UserQueues.Get());
        }
        void CancelRequest(const TGUID& reqId) override {
            Requester->CancelRequest(reqId);
        }
        void BreakRequest(const TGUID& reqId) override {
            Requester->BreakRequest(reqId);
        }

        void SendResponse(const TGUID& reqId, TVector<char>* data) override {
            Requester->SendResponseImpl(reqId, PP_NORMAL, data);
        }
        void SendResponseLowPriority(const TGUID& reqId, TVector<char>* data) override {
            Requester->SendResponseImpl(reqId, PP_LOW, data);
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
        TUdpHttpResponse* Request(const TUdpAddress& addr, const TString& url, TVector<char>* data) override {
            return Requester->Request(addr, url, data);
        }
        TIntrusivePtr<TWaitResponse> WaitableRequest(const TUdpAddress& addr, const TString& url, TVector<char>* data) override {
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
            fprintf(stderr, "Failed request to host %s\n", GetAddressAsString(answer->PeerAddress).data());
            fprintf(stderr, "Error description: %s\n", answer->Error.data());
            fflush(nullptr);
            Y_ASSERT(0);
            abort();
        }
    }

    TString GetDebugInfo(const TUdpAddress& addr, double timeout) {
        NHPTimer::STime start;
        NHPTimer::GetTime(&start);
        TIntrusivePtr<IUdpHost> host = CreateUdpHost(0);
        {
            TAutoPtr<TRopeDataPacket> rq = new TRopeDataPacket;
            rq->Write((char)PKT_GETDEBUGINFO);
            ui32 crc32 = CalcChecksum(rq->GetChain());
            host->Send(addr, rq.Release(), crc32, nullptr, PP_HIGH);
        }
        for (;;) {
            TAutoPtr<TRequest> ptr = host->GetRequest();
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

    void Kill(const TUdpAddress& addr) {
        TIntrusivePtr<IUdpHost> host = CreateUdpHost(0);
        host->Kill(addr);
    }

    void StopAllNetLibaThreads() {
        PanicAttack = true; // AAAA!!!!
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
        if (PanicAttack)
            return nullptr;

        TIntrusivePtr<NNetlibaSocket::ISocket> socket = NNetlibaSocket::CreateSocket();
        socket->Open(port);
        if (!socket->IsValid())
            return nullptr;

        return CreateHttpUdpRequester(socket);
    }

    IRequester* CreateHttpUdpRequester(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
        if (PanicAttack)
            return nullptr;

        TIntrusivePtr<TUdpHttp> res(new TUdpHttp);
        if (!res->Start(socket))
            return nullptr;
        return res.Release();
    }

}
