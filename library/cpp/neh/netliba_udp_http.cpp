#include "netliba_udp_http.h"
#include "utils.h"

#include <library/cpp/netliba/v6/cpu_affinity.h>
#include <library/cpp/netliba/v6/stdafx.h>
#include <library/cpp/netliba/v6/udp_client_server.h>
#include <library/cpp/netliba/v6/udp_socket.h>

#include <library/cpp/netliba/v6/block_chain.h> // depend on another headers

#include <util/system/hp_timer.h>
#include <util/system/shmat.h>
#include <util/system/spinlock.h>
#include <util/system/thread.h>
#include <util/system/types.h>
#include <util/system/yassert.h>
#include <util/thread/lfqueue.h>

#include <atomic>

#if !defined(_win_)
#include <signal.h>
#include <pthread.h>
#endif

using namespace NNetliba;

namespace {
    const float HTTP_TIMEOUT = 15.0f;
    const size_t MIN_SHARED_MEM_PACKET = 1000;
    const size_t MAX_PACKET_SIZE = 0x70000000;

    NNeh::TAtomicBool PanicAttack;
    std::atomic<NHPTimer::STime> LastHeartbeat;
    std::atomic<double> HeartbeatTimeout;

    bool IsLocal(const TUdpAddress& addr) {
        return addr.IsIPv4() ? IsLocalIPv4(addr.GetIPv4()) : IsLocalIPv6(addr.Network, addr.Interface);
    }

    void StopAllNetLibaThreads() {
        PanicAttack = true; // AAAA!!!!
    }

    void ReadShm(TSharedMemory* shm, TVector<char>* data) {
        Y_ASSERT(shm);
        int dataSize = shm->GetSize();
        data->yresize(dataSize);
        memcpy(&(*data)[0], shm->GetPtr(), dataSize);
    }

    void ReadShm(TSharedMemory* shm, TString* data) {
        Y_ASSERT(shm);
        size_t dataSize = shm->GetSize();
        data->ReserveAndResize(dataSize);
        memcpy(data->begin(), shm->GetPtr(), dataSize);
    }

    template <class T>
    void EraseList(TLockFreeQueue<T*>* data) {
        T* ptr = 0;
        while (data->Dequeue(&ptr)) {
            delete ptr;
        }
    }

    enum EHttpPacket {
        PKT_REQUEST,
        PKT_PING,
        PKT_PING_RESPONSE,
        PKT_RESPONSE,
        PKT_LOCAL_REQUEST,
        PKT_LOCAL_RESPONSE,
        PKT_CANCEL,
    };
}

namespace NNehNetliba {
    TUdpHttpMessage::TUdpHttpMessage(const TGUID& reqId, const TUdpAddress& peerAddr)
        : ReqId(reqId)
        , PeerAddress(peerAddr)
    {
    }

    TUdpHttpRequest::TUdpHttpRequest(TAutoPtr<TRequest>& dataHolder, const TGUID& reqId, const TUdpAddress& peerAddr)
        : TUdpHttpMessage(reqId, peerAddr)
    {
        TBlockChainIterator reqData(dataHolder->Data->GetChain());
        char pktType;
        reqData.Read(&pktType, 1);
        ReadArr(&reqData, &Url);
        if (pktType == PKT_REQUEST) {
            ReadYArr(&reqData, &Data);
        } else if (pktType == PKT_LOCAL_REQUEST) {
            ReadShm(dataHolder->Data->GetSharedData(), &Data);
        } else {
            Y_ASSERT(0);
        }

        if (reqData.HasFailed()) {
            Y_ASSERT(0 && "wrong format, memory corruption suspected");
            Url = "";
            Data.clear();
        }
    }

    TUdpHttpResponse::TUdpHttpResponse(TAutoPtr<TRequest>& dataHolder, const TGUID& reqId, const TUdpAddress& peerAddr, EResult result, const char* error)
        : TUdpHttpMessage(reqId, peerAddr)
        , Ok(result)
    {
        if (result == TUdpHttpResponse::FAILED) {
            Error = error ? error : "request failed";
        } else if (result == TUdpHttpResponse::CANCELED) {
            Error = error ? error : "request cancelled";
        } else {
            TBlockChainIterator reqData(dataHolder->Data->GetChain());
            if (Y_UNLIKELY(reqData.HasFailed())) {
                Y_ASSERT(0 && "wrong format, memory corruption suspected");
                Ok = TUdpHttpResponse::FAILED;
                Data.clear();
                Error = "wrong response format";
            } else {
                char pktType;
                reqData.Read(&pktType, 1);
                TGUID guid;
                reqData.Read(&guid, sizeof(guid));
                Y_ASSERT(ReqId == guid);
                if (pktType == PKT_RESPONSE) {
                    ReadArr<TString>(&reqData, &Data);
                } else if (pktType == PKT_LOCAL_RESPONSE) {
                    ReadShm(dataHolder->Data->GetSharedData(), &Data);
                } else {
                    Y_ASSERT(0);
                }
            }
        }
    }

    class TUdpHttp: public IRequester {
        enum EDir {
            DIR_OUT,
            DIR_IN
        };

        struct TInRequestState {
            enum EState {
                S_WAITING,
                S_RESPONSE_SENDING,
                S_CANCELED,
            };

            TInRequestState()
                : State(S_WAITING)
            {
            }

            TInRequestState(const TUdpAddress& address)
                : State(S_WAITING)
                , Address(address)
            {
            }

            EState State;
            TUdpAddress Address;
        };

        struct TOutRequestState {
            enum EState {
                S_SENDING,
                S_WAITING,
                S_WAITING_PING_SENDING,
                S_WAITING_PING_SENT,
                S_CANCEL_AFTER_SENDING
            };

            TOutRequestState()
                : State(S_SENDING)
                , TimePassed(0)
                , PingTransferId(-1)
            {
            }

            EState State;
            TUdpAddress Address;
            double TimePassed;
            int PingTransferId;
            IEventsCollectorRef EventsCollector;
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
            TSendRequest() = default;

            TSendRequest(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket>* data, const TGUID& reqGuid, const IEventsCollectorRef& eventsCollector)
                : Addr(addr)
                , Data(*data)
                , ReqGuid(reqGuid)
                , EventsCollector(eventsCollector)
                , Crc32(CalcChecksum(Data->GetChain()))
            {
            }

            TUdpAddress Addr;
            TAutoPtr<TRopeDataPacket> Data;
            TGUID ReqGuid;
            IEventsCollectorRef EventsCollector;
            ui32 Crc32;
        };

        struct TSendResponse {
            TSendResponse() = default;

            TSendResponse(const TGUID& reqGuid, EPacketPriority prior, TVector<char>* data)
                : ReqGuid(reqGuid)
                , DataCrc32(0)
                , Priority(prior)
            {
                if (data && !data->empty()) {
                    data->swap(Data);
                    DataCrc32 = TIncrementalChecksumCalcer::CalcBlockSum(&Data[0], Data.ysize());
                }
            }

            TVector<char> Data;
            TGUID ReqGuid;
            ui32 DataCrc32;
            EPacketPriority Priority;
        };

        typedef THashMap<TGUID, TOutRequestState, TGUIDHash> TOutRequestHash;
        typedef THashMap<TGUID, TInRequestState, TGUIDHash> TInRequestHash;

    public:
        TUdpHttp(const IEventsCollectorRef& eventsCollector)
            : MyThread_(ExecServerThread, (void*)this)
            , AbortTransactions_(false)
            , Port_(0)
            , EventCollector_(eventsCollector)
            , ReportRequestCancel_(false)
            , ReporRequestAck_(false)
            , PhysicalCpu_(-1)
        {
        }

        ~TUdpHttp() override {
            if (MyThread_.Running()) {
                AtomicSet(KeepRunning_, 0);
                MyThread_.Join();
            }
        }

        bool Start(int port, int physicalCpu) {
            Y_ASSERT(Host_.Get() == nullptr);
            Port_ = port;
            PhysicalCpu_ = physicalCpu;
            MyThread_.Start();
            HasStarted_.Wait();
            return Host_.Get() != nullptr;
        }

        void EnableReportRequestCancel() override {
            ReportRequestCancel_ = true;
        }

        void EnableReportRequestAck() override {
            ReporRequestAck_ = true;
        }

        void SendRequest(const TUdpAddress& addr, const TString& url, const TString& data, const TGUID& reqId) override {
            Y_ABORT_UNLESS(
                data.size() < MAX_PACKET_SIZE,
                "data size is too large; data.size()=%" PRISZT ", MAX_PACKET_SIZE=%" PRISZT,
                data.size(), MAX_PACKET_SIZE);

            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            if (data.size() > MIN_SHARED_MEM_PACKET && IsLocal(addr)) {
                TIntrusivePtr<TSharedMemory> shm = new TSharedMemory;
                if (shm->Create(data.size())) {
                    ms->Write((char)PKT_LOCAL_REQUEST);
                    ms->WriteStroka(url);
                    memcpy(shm->GetPtr(), data.begin(), data.size());
                    ms->AttachSharedData(shm);
                }
            }
            if (ms->GetSharedData() == nullptr) {
                ms->Write((char)PKT_REQUEST);
                ms->WriteStroka(url);
                struct TStrokaStorage: public TThrRefBase, public TString {
                    TStrokaStorage(const TString& s)
                        : TString(s)
                    {
                    }
                };
                TStrokaStorage* ss = new TStrokaStorage(data);
                ms->Write((int)ss->size());
                ms->AddBlock(ss, ss->begin(), ss->size());
            }

            SendReqList_.Enqueue(new TSendRequest(addr, &ms, reqId, EventCollector_));
            Host_->CancelWait();
        }

        void CancelRequest(const TGUID& reqId) override {
            CancelReqList_.Enqueue(reqId);
            Host_->CancelWait();
        }

        void SendResponse(const TGUID& reqId, TVector<char>* data) override {
            if (data && data->size() > MAX_PACKET_SIZE) {
               Y_ABORT(
                    "data size is too large; data->size()=%" PRISZT ", MAX_PACKET_SIZE=%" PRISZT,
                    data->size(), MAX_PACKET_SIZE);
            }
            SendRespList_.Enqueue(new TSendResponse(reqId, PP_NORMAL, data));
            Host_->CancelWait();
        }

        void StopNoWait() override {
            AbortTransactions_ = true;
            AtomicSet(KeepRunning_, 0);
            // calcel all outgoing requests
            TGuard<TSpinLock> lock(Spn_);
            while (!OutRequests_.empty()) {
                // cancel without informing peer that we are cancelling the request
                FinishRequest(OutRequests_.begin(), TUdpHttpResponse::CANCELED, nullptr, "request canceled: inside TUdpHttp::StopNoWait()");
            }
        }

    private:
        void FinishRequest(TOutRequestHash::iterator i, TUdpHttpResponse::EResult ok, TRequestPtr data, const char* error = nullptr) {
            TOutRequestState& s = i->second;
            s.EventsCollector->AddResponse(new TUdpHttpResponse(data, i->first, s.Address, ok, error));
            OutRequests_.erase(i);
        }

        int SendWithHighPriority(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket> data) {
            ui32 crc32 = CalcChecksum(data->GetChain());
            return Host_->Send(addr, data.Release(), crc32, nullptr, PP_HIGH);
        }

        void ProcessIncomingPackets() {
            TVector<TGUID> failedRequests;
            for (;;) {
                TAutoPtr<TRequest> req = Host_->GetRequest();
                if (req.Get() == nullptr)
                    break;

                TBlockChainIterator reqData(req->Data->GetChain());
                char pktType;
                reqData.Read(&pktType, 1);
                switch (pktType) {
                    case PKT_REQUEST:
                    case PKT_LOCAL_REQUEST: {
                        TGUID reqId = req->Guid;
                        TInRequestHash::iterator z = InRequests_.find(reqId);
                        if (z != InRequests_.end()) {
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
                            InRequests_[reqId] = TInRequestState(req->Address);
                            EventCollector_->AddRequest(new TUdpHttpRequest(req, reqId, req->Address));
                        }
                    } break;
                    case PKT_PING: {
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        bool ok = InRequests_.find(guid) != InRequests_.end();
                        TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                        ms->Write((char)PKT_PING_RESPONSE);
                        ms->Write(guid);
                        ms->Write(ok);
                        SendWithHighPriority(req->Address, ms.Release());
                    } break;
                    case PKT_PING_RESPONSE: {
                        TGUID guid;
                        bool ok;
                        reqData.Read(&guid, sizeof(guid));
                        reqData.Read(&ok, sizeof(ok));
                        TOutRequestHash::iterator i = OutRequests_.find(guid);
                        if (i == OutRequests_.end())
                            ; //Y_ASSERT(0); // actually possible with some packet orders
                        else {
                            if (!ok) {
                                // can not delete request at this point
                                // since we can receive failed ping and response at the same moment
                                // consider sequence: client sends ping, server sends response
                                // and replies false to ping as reply is sent
                                // we can not receive failed ping_response earlier then response itself
                                // but we can receive them simultaneously
                                failedRequests.push_back(guid);
                            } else {
                                TOutRequestState& s = i->second;
                                switch (s.State) {
                                    case TOutRequestState::S_WAITING_PING_SENDING: {
                                        Y_ASSERT(s.PingTransferId >= 0);
                                        TTransferHash::iterator k = TransferHash_.find(s.PingTransferId);
                                        if (k != TransferHash_.end())
                                            TransferHash_.erase(k);
                                        else
                                            Y_ASSERT(0);
                                        s.PingTransferId = -1;
                                        s.TimePassed = 0;
                                        s.State = TOutRequestState::S_WAITING;
                                    } break;
                                    case TOutRequestState::S_WAITING_PING_SENT:
                                        s.TimePassed = 0;
                                        s.State = TOutRequestState::S_WAITING;
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
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        TOutRequestHash::iterator i = OutRequests_.find(guid);
                        if (i == OutRequests_.end()) {
                            ; //Y_ASSERT(0); // does happen
                        } else {
                            FinishRequest(i, TUdpHttpResponse::OK, req);
                        }
                    } break;
                    case PKT_CANCEL: {
                        TGUID guid;
                        reqData.Read(&guid, sizeof(guid));
                        TInRequestHash::iterator i = InRequests_.find(guid);
                        if (i == InRequests_.end()) {
                            ; //Y_ASSERT(0); // may happen
                        } else {
                            TInRequestState& s = i->second;
                            if (s.State != TInRequestState::S_CANCELED && ReportRequestCancel_)
                                EventCollector_->AddCancel(guid);
                            s.State = TInRequestState::S_CANCELED;
                        }
                    } break;
                    default:
                        Y_ASSERT(0);
                }
            }
            // cleanup failed requests
            for (size_t k = 0; k < failedRequests.size(); ++k) {
                const TGUID& guid = failedRequests[k];
                TOutRequestHash::iterator i = OutRequests_.find(guid);
                if (i != OutRequests_.end())
                    FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "failed udp ping");
            }
        }

        void AnalyzeSendResults() {
            TSendResult res;
            while (Host_->GetSendResult(&res)) {
                TTransferHash::iterator k = TransferHash_.find(res.TransferId);
                if (k != TransferHash_.end()) {
                    const TTransferPurpose& tp = k->second;
                    switch (tp.Dir) {
                        case DIR_OUT: {
                            TOutRequestHash::iterator i = OutRequests_.find(tp.Guid);
                            if (i != OutRequests_.end()) {
                                const TGUID& reqId = i->first;
                                TOutRequestState& s = i->second;
                                switch (s.State) {
                                    case TOutRequestState::S_SENDING:
                                        if (!res.Success) {
                                            FinishRequest(i, TUdpHttpResponse::FAILED, nullptr, "request failed: state S_SENDING");
                                        } else {
                                            if (ReporRequestAck_ && !!s.EventsCollector) {
                                                s.EventsCollector->AddRequestAck(reqId);
                                            }
                                            s.State = TOutRequestState::S_WAITING;
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
                                        } else {
                                            s.PingTransferId = -1;
                                            s.State = TOutRequestState::S_WAITING_PING_SENT;
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
                            TInRequestHash::iterator i = InRequests_.find(tp.Guid);
                            if (i != InRequests_.end()) {
                                Y_ASSERT(i->second.State == TInRequestState::S_RESPONSE_SENDING || i->second.State == TInRequestState::S_CANCELED);
                                InRequests_.erase(i);
                            }
                        } break;
                        default:
                            Y_ASSERT(0);
                            break;
                    }
                    TransferHash_.erase(k);
                }
            }
        }

        void SendPingsIfNeeded() {
            NHPTimer::STime tChk = PingsSendT_;
            float deltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (deltaT < 0.05) {
                return;
            }
            PingsSendT_ = tChk;
            deltaT = ClampVal(deltaT, 0.0f, HTTP_TIMEOUT / 3);

            {
                for (TOutRequestHash::iterator i = OutRequests_.begin(); i != OutRequests_.end();) {
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
                                TransferHash_[transId] = TTransferPurpose(DIR_OUT, guid);
                                s.State = TOutRequestState::S_WAITING_PING_SENDING;
                                s.PingTransferId = transId;
                            }
                            break;
                        case TOutRequestState::S_WAITING_PING_SENT:
                            s.TimePassed += deltaT;
                            if (s.TimePassed > HTTP_TIMEOUT) {
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
                TGuard<TSpinLock> lock(Spn_);
                DoSends();
            }
            Host_->Step();
            {
                TGuard<TSpinLock> lock(Spn_);
                DoSends();
                ProcessIncomingPackets();
                AnalyzeSendResults();
                SendPingsIfNeeded();
            }
        }

        void Wait() {
            Host_->Wait(0.1f);
        }

        void DoSendCancel(const TUdpAddress& addr, const TGUID& req) {
            TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
            ms->Write((char)PKT_CANCEL);
            ms->Write(req);
            SendWithHighPriority(addr, ms);
        }

        void DoSends() {
            {
                // cancelling requests
                TGUID reqGuid;
                while (CancelReqList_.Dequeue(&reqGuid)) {
                    TOutRequestHash::iterator i = OutRequests_.find(reqGuid);
                    if (i == OutRequests_.end()) {
                        AnticipateCancels_.insert(reqGuid);
                        continue; // cancelling non existing request is ok
                    }
                    TOutRequestState& s = i->second;
                    if (s.State == TOutRequestState::S_SENDING) {
                        // we are in trouble - have not sent request and we already have to cancel it, wait send
                        s.State = TOutRequestState::S_CANCEL_AFTER_SENDING;
                        s.EventsCollector->AddCancel(i->first);
                    } else {
                        DoSendCancel(s.Address, reqGuid);
                        FinishRequest(i, TUdpHttpResponse::CANCELED, nullptr, "request canceled: notify requested side");
                    }
                }
            }
            {
                // sending replies
                for (TSendResponse* rd = nullptr; SendRespList_.Dequeue(&rd); delete rd) {
                    TInRequestHash::iterator i = InRequests_.find(rd->ReqGuid);
                    if (i == InRequests_.end()) {
                        Y_ASSERT(0);
                        continue;
                    }
                    TInRequestState& s = i->second;
                    if (s.State == TInRequestState::S_CANCELED) {
                        // need not send response for the canceled request
                        InRequests_.erase(i);
                        continue;
                    }

                    Y_ASSERT(s.State == TInRequestState::S_WAITING);
                    s.State = TInRequestState::S_RESPONSE_SENDING;

                    TAutoPtr<TRopeDataPacket> ms = new TRopeDataPacket;
                    ui32 crc32 = 0;
                    int dataSize = rd->Data.ysize();
                    if (rd->Data.size() > MIN_SHARED_MEM_PACKET && IsLocal(s.Address)) {
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
                        // it is hard to avoid since in SendResponse() we don't know if shared mem will be used
                        // (peer address is not available there)
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

                    int transId = Host_->Send(s.Address, ms.Release(), crc32, nullptr, rd->Priority);
                    TransferHash_[transId] = TTransferPurpose(DIR_IN, rd->ReqGuid);
                }
            }
            {
                // sending requests
                for (TSendRequest* rd = nullptr; SendReqList_.Dequeue(&rd); delete rd) {
                    Y_ASSERT(OutRequests_.find(rd->ReqGuid) == OutRequests_.end());

                    {
                        TOutRequestState& s = OutRequests_[rd->ReqGuid];
                        s.State = TOutRequestState::S_SENDING;
                        s.Address = rd->Addr;
                        s.EventsCollector = rd->EventsCollector;
                    }

                    if (AnticipateCancels_.find(rd->ReqGuid) != AnticipateCancels_.end()) {
                        FinishRequest(OutRequests_.find(rd->ReqGuid), TUdpHttpResponse::CANCELED, nullptr, "Canceled (before transmit)");
                    } else {
                        TGUID pktGuid = rd->ReqGuid; // request packet id should match request id
                        int transId = Host_->Send(rd->Addr, rd->Data.Release(), rd->Crc32, &pktGuid, PP_NORMAL);
                        TransferHash_[transId] = TTransferPurpose(DIR_OUT, rd->ReqGuid);
                    }
                }
            }
            if (!AnticipateCancels_.empty()) {
                AnticipateCancels_.clear();
            }
        }

        void FinishOutstandingTransactions() {
            // wait all pending requests, all new requests are canceled
            while ((!OutRequests_.empty() || !InRequests_.empty() || !SendRespList_.IsEmpty() || !SendReqList_.IsEmpty()) && !PanicAttack) {
                Step();
                sleep(0);
            }
        }

        static void* ExecServerThread(void* param) {
            TUdpHttp* pThis = (TUdpHttp*)param;
            if (pThis->GetPhysicalCpu() >= 0) {
                BindToSocket(pThis->GetPhysicalCpu());
            }
            SetHighestThreadPriority();

            TIntrusivePtr<NNetlibaSocket::ISocket> socket = NNetlibaSocket::CreateSocket();
            socket->Open(pThis->Port_);
            if (socket->IsValid()) {
                pThis->Port_ = socket->GetPort();
                pThis->Host_ = CreateUdpHost(socket);
            } else {
                pThis->Host_ = nullptr;
            }

            pThis->HasStarted_.Signal();
            if (!pThis->Host_)
                return nullptr;

            NHPTimer::GetTime(&pThis->PingsSendT_);
            while (AtomicGet(pThis->KeepRunning_) && !PanicAttack) {
                if (HeartbeatTimeout.load(std::memory_order_acquire) > 0) {
                    NHPTimer::STime chk = LastHeartbeat.load(std::memory_order_acquire);
                    if (NHPTimer::GetTimePassed(&chk) > HeartbeatTimeout.load(std::memory_order_acquire)) {
                        StopAllNetLibaThreads();
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
            if (!pThis->AbortTransactions_ && !PanicAttack) {
                pThis->FinishOutstandingTransactions();
            }
            pThis->Host_ = nullptr;
            return nullptr;
        }

        int GetPhysicalCpu() const noexcept {
            return PhysicalCpu_;
        }

    private:
        TThread MyThread_;
        TAtomic KeepRunning_ = 1;
        bool AbortTransactions_;
        TSpinLock Spn_;
        TSystemEvent HasStarted_;

        NHPTimer::STime PingsSendT_;

        TIntrusivePtr<IUdpHost> Host_;
        int Port_;
        TOutRequestHash OutRequests_;
        TInRequestHash InRequests_;

        typedef THashMap<int, TTransferPurpose> TTransferHash;
        TTransferHash TransferHash_;

        // hold it here to not construct on every DoSends()
        typedef THashSet<TGUID, TGUIDHash> TAnticipateCancels;
        TAnticipateCancels AnticipateCancels_;

        TLockFreeQueue<TSendRequest*> SendReqList_;
        TLockFreeQueue<TSendResponse*> SendRespList_;
        TLockFreeQueue<TGUID> CancelReqList_;

        TIntrusivePtr<IEventsCollector> EventCollector_;

        bool ReportRequestCancel_;
        bool ReporRequestAck_;
        int PhysicalCpu_;
    };

    IRequesterRef CreateHttpUdpRequester(int port, const IEventsCollectorRef& ec, int physicalCpu) {
        TUdpHttp* udpHttp = new TUdpHttp(ec);
        IRequesterRef res(udpHttp);
        if (!udpHttp->Start(port, physicalCpu)) {
            if (port) {
                ythrow yexception() << "netliba can't bind port=" << port;
            } else {
                ythrow yexception() << "netliba can't bind random port";
            }
        }
        return res;
    }
}
