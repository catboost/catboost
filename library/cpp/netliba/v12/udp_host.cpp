#include "library/cpp/netliba/v12/udp_address.h"
#include "stdafx.h"
#include <util/digest/numeric.h>
#include <util/generic/cast.h>
#include <util/generic/guid.h>
#include <util/generic/vector.h>
#include <util/system/hp_timer.h>
#include <util/system/shmat.h>
#include <util/system/yield.h>
#include <util/system/spinlock.h>
#include <library/cpp/threading/chunk_queue/queue.h>
#include <util/system/thread.h>
#include "cpu_affinity.h"
#include "block_chain.h"
#include "ib_cs.h"
#include "net_acks.h"
#include "settings.h"
#include "udp_debug.h"
#include "udp_host.h"
#include "udp_host_connection.h"
#include "udp_host_protocol.h"
#include "udp_host_recv_completed.h"
#include "udp_http.h"
#include "udp_recv_packet.h"
#include "udp_socket.h"

#include <atomic>

namespace NNetliba_v12 {
    const float UDP_TRANSFER_TIMEOUT = 90;
    const float UDP_KEEP_CONNETION = 600;
    const float DEFAULT_MAX_WAIT_TIME = 1;
    const float STAT_UPDATE_TIME = 1;

    const float DEFAULT_MAX_SEND_RECV_LATENCY = 0.00005f;
    // траффик может идти, а новых данных для конкретного пакета может не добавляться.
    // это возможно когда мы прерываем процесс в момент передачи и перезапускаем его на том же порту,
    // тогда на приемнике повиснет пакет. Этот пакет мы зашибем по этому таймауту
    const float UDP_MAX_INPUT_DATA_WAIT = UDP_TRANSFER_TIMEOUT * 2;

    const size_t UDP_MAX_PACKETS_IN_QUEUE = 5U;
    const size_t UDP_MAX_NEW_PACKETS_BEFORE_FORCED_FLUSH = 15U; // big packets are limited by UDP_MAX_PACKETS_IN_QUEUE, small packets - by this constant

    const ui64 KILL_PASSPHRASE1 = 0x98ff9cefb11d9a4cul;
    const ui64 KILL_PASSPHRASE2 = 0xf7754c29e0be95eaul;

    //////////////////////////////////////////////////////////////////////////

    size_t TTransferHash(const TTransfer& t) {
        return CombineHashes<size_t>(THash<const IConnection*>()(t.Connection.Get()), THash<ui64>()(t.Id));
    }

    bool operator==(const TTransfer& lhv, const TTransfer& rhv) {
        return lhv.Connection == rhv.Connection && lhv.Id == rhv.Id;
    }
    bool operator!=(const TTransfer& lhv, const TTransfer& rhv) {
        return !(lhv == rhv);
    }

    ///////////////////////////////////////////////////////////////////////////////
    class TConnections {
        typedef THashMap<TGUID, TIntrusivePtr<IConnection>> TConnectionHash;

    public:
        using iterator = TConnectionHash::iterator;
        using const_iterator = TConnectionHash::const_iterator;
        typedef TVector<iterator> TRemover;
        TConnections()
            : CachedPos(ConnectionHash.begin())
            , CacheResetNum(0)
            , LoopPreemptedNum(0)
        {
        }

    private:
        TConnectionHash ConnectionHash;
        TActiveConnectionList ActiveConnectionList;
        TSendingConnectionsList SendingConnectionList;
        iterator CachedPos;
        ui64 CacheResetNum;
        ui64 LoopPreemptedNum;

    public:
        enum class ESentConnectionResult {
            SCR_CONT,
            SCR_DELETE,
            SCR_BREAK
        };

        enum class EIdleResult {
            ER_CONT,
            ER_DELETE,
            ER_YIELD
        };

        ui64 GetCacheResetNum() const {
            return CacheResetNum;
        }

        ui64 GetLoopPreemptedNum() const {
            return LoopPreemptedNum;
        }

        bool IsFinished() const {
            return ConnectionHash.begin() == CachedPos;
        }

        const TIntrusivePtr<IConnection>* FindPtr(const TGUID& guid) const {
            return ConnectionHash.FindPtr(guid);
        }
        const std::pair<iterator, bool> Insert(const TGUID& guid, TConnection* conn) {
            const std::pair<iterator, bool>& rv = ConnectionHash.insert(std::make_pair(guid, conn));
            if (!rv.second && rv.first == CachedPos) {
                CacheResetNum++;
                CachedPos = ConnectionHash.begin();
            }
            return rv;
        }
        TIntrusivePtr<IConnection>& operator[](const TGUID& guid) {
            return (*((Insert(guid, TIntrusivePtr<TConnection>().Get())).first)).second;
        }
        const_iterator Begin() const {
            return ConnectionHash.begin();
        }
        const_iterator End() const {
            return ConnectionHash.end();
        }
        size_t GetSize() const {
            return ConnectionHash.size();
        }
        size_t GetActiveSize() const {
            return ActiveConnectionList.Size();
        }
        size_t GetSendingSize() const {
            return SendingConnectionList.size();
        }

        template <typename F>
        void ForEachConnection(const F& function) {
            TVector<iterator> toRemove;
            iterator savedPos = CachedPos;
            iterator it = CachedPos;
            bool dropCache = false;

            for (; it != ConnectionHash.end(); ++it) {
                EIdleResult res = function(static_cast<const_iterator>(it));
                if (res == EIdleResult::ER_DELETE) {
                    toRemove.push_back(it);
                } else if (res == EIdleResult::ER_YIELD) {
                    LoopPreemptedNum++;
                    Y_ABORT_UNLESS(savedPos == CachedPos); //just to check nobody change me
                    CachedPos = it;
                    break;
                }
            }
            if (it == ConnectionHash.end()) {
                dropCache = true;
            }

            for (iterator& c : toRemove) {
                Y_ASSERT(dropCache || c != CachedPos);
                if (!dropCache && c == CachedPos) {
                    fprintf(stderr, "yielding and removing with same iterator! Wrong using!");
                    Y_ASSERT(false);
                    dropCache = true;
                }
                ConnectionHash.erase(c);
            }
            if (dropCache) {
                CachedPos = ConnectionHash.begin();
            }
        }

        void InsertToActive(TConnection* connection) {
            Y_ABORT_UNLESS(connection, "null connection inserted\n");
            if (connection->Inactivated == true) {
                ActiveConnectionList.PushBack(connection);
            }
            connection->Inactivated = false;
        }

        void EraseFromActive(TConnection* connection) {
            if (connection->Inactivated == false) {
                connection->Unlink();
            }
            connection->Inactivated = true;
        }

        void InsertToSending(TConnection* connection) {
            Y_ABORT_UNLESS(connection, "null connection inserted\n");
            if (connection->Sending == false) {
                SendingConnectionList.insert(SendingConnectionList.end(), connection);
            }
            connection->Sending = true;
        }

        template <typename F>
        void ForEachActiveConnection(const F& function) {
            for (TActiveConnectionList::iterator it = ActiveConnectionList.begin(); it != ActiveConnectionList.end();) {
                if (!function(it)) {
                    it->Inactivated = true;
                    (it++)->Unlink();
                } else {
                    it++;
                }
            }
        }

        template <typename F>
        bool ForEachSendingConnection(const F& function) {
            for (TSendingConnectionsList::iterator it = SendingConnectionList.begin(); it != SendingConnectionList.end();) {
                ESentConnectionResult res = function(it);
                if (res == ESentConnectionResult::SCR_DELETE) {
                    (*it)->Sending = false;
                    it = SendingConnectionList.erase(it);
                } else if (res == ESentConnectionResult::SCR_BREAK) {
                    return false;
                } else {
                    it++;
                }
            }
            TSendingConnectionsList::iterator it = SendingConnectionList.begin();
            if (it == SendingConnectionList.end())
                return false;
            TIntrusivePtr<TConnection> tmp = *it;
            SendingConnectionList.erase(it);
            SendingConnectionList.push_back(tmp);
            return true;
        }
    };

    static bool IBDetection = true;
    static bool XsPingSending = false;

    typedef std::function<void(TConnection* connection)> TConnectCb;
    typedef std::function<void(const TTransfer& transfer,
                               TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor)>
        TSendCb;
    typedef std::function<void(const TTransfer& transfer)> TCancelCb;

    class TTXUserQueue {
        TConnectCb ConnectCb;
        TSendCb SendCb;
        TCancelCb CancelCb;
        enum class ETXUserCmd {
            TXUC_CONNECT,
            TXUC_SEND,
            TXUC_CANCEL
        };
        struct TSendCmd: public TWithCustomAllocator {
            TSendCmd(const TTransfer& transfer, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor)
                : Transfer(transfer)
                , Data(std::move(data))
                , Pp(pp)
                , Tos(tos)
                , NetlibaColor(netlibaColor)
            {
            }
            const TTransfer Transfer;
            const TAutoPtr<TRopeDataPacket> Data;
            const EPacketPriority Pp;
            const TTos Tos;
            const ui8 NetlibaColor;
        };
        struct TCancelCmd: public TWithCustomAllocator {
            explicit TCancelCmd(const TTransfer& transfer)
                : Transfer(transfer)
            {
            }
            const TTransfer Transfer;
        };

    public:
        TTXUserQueue(const TConnectCb& connCb, const TSendCb& sendCb, const TCancelCb& cancelCb)
            : ConnectCb(connCb)
            , SendCb(sendCb)
            , CancelCb(cancelCb)
        {
        }
        ~TTXUserQueue() {
            if (!Queue.IsEmpty()) {
                fprintf(stderr, "TTXUserQueue destructed with no empty queue, memory leak...");
            }
            Y_ASSERT(Queue.IsEmpty());
        }

        void EnqueueConnect(TConnection* connection) {
            Queue.Enqueue(std::make_pair(ETXUserCmd::TXUC_CONNECT, connection));
        }

        void EnqueueSend(const TTransfer& transfer, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor) {
            TSendCmd* sendCmd = new TSendCmd(transfer, data, pp, tos, netlibaColor);
            Queue.Enqueue(std::make_pair(ETXUserCmd::TXUC_SEND, sendCmd));
        }

        void EnqueueCancel(const TTransfer& transfer) {
            TCancelCmd* cancelCmd = new TCancelCmd(transfer);
            Queue.Enqueue(std::make_pair(ETXUserCmd::TXUC_CANCEL, cancelCmd));
        }

        void DequeueAndRun() {
            std::pair<ETXUserCmd, void*> cmdPair;
            while (Queue.Dequeue(cmdPair)) {
                const ETXUserCmd cmdId = cmdPair.first;
                switch (cmdId) {
                    case ETXUserCmd::TXUC_CONNECT: {
                        TConnection* connection = reinterpret_cast<TConnection*>(cmdPair.second);
                        ConnectCb(connection);
                    } break;
                    case ETXUserCmd::TXUC_SEND: {
                        const TSendCmd* sendCmd = reinterpret_cast<const TSendCmd*>(cmdPair.second);
                        SendCb(sendCmd->Transfer, sendCmd->Data, sendCmd->Pp, sendCmd->Tos, sendCmd->NetlibaColor);
                        delete sendCmd;
                    } break;
                    case ETXUserCmd::TXUC_CANCEL: {
                        const TCancelCmd* cancelCmd = reinterpret_cast<const TCancelCmd*>(cmdPair.second);
                        CancelCb(cancelCmd->Transfer);
                        delete cancelCmd;
                    } break;
                    default:
                        Y_ABORT_UNLESS(false);
                        break;
                }
            }
        }

    private:
        NThreading::TOneOneQueue<std::pair<ETXUserCmd, void*>> Queue;
    };

    class TUdpHost: public IUdpHost {
        enum EFlashPacketResult {
            FPR_OK = 0,
            FPR_OVERFLOW = 1,
            FPR_OUT_TRANSFERS_CHANGED = 2
        };

        enum class ESentPacketResult {
            SPR_OK,
            SPR_OVERFLOW,
            SPR_STOP_SENDING_TRANSFER
        };

        TUdpSocket S;
        size_t NewPacketsAfterLastFlush;

        typedef TList<TTransfer> TSendOrder;
        // congestion control per peer
        TConnections Connections;
        TTos DefaultTos;
        NThreading::TOneOneQueue<TSendResult> SendResults;
        NThreading::TOneOneQueue<TUdpRequest*> ReceivedClientQueue;

        TSendOrder SendOrderLow, SendOrder, SendOrderHigh, SendOrderSystem;
        NHPTimer::STime CurrentT;
        TAtomic IsWaiting;
        float MaxWaitTime;
        std::atomic<float> MaxWaitTime2;
        float IBIdleTime;
        TIntrusivePtr<TRequesterPendingDataStats> TotalPendingDataStats;
        TColoredRequesterPendingDataStats ColoredPendingDataStats;
        TStatAggregator FailureStats;
        TIntrusivePtr<IIBClientServer> IB;
        typedef THashMap<TIBMsgHandle, TTransfer> TIBtoTransferHash;
        TIBtoTransferHash IBKeyToTransfer;
        float TimeSinceSocketStatsUpdate;
        const float UdpTransferTimeout;
        TThread UdpHostThread;
        TAtomic Run;
        TTXUserQueue TXUserQueue;
        TSpinLock StatsEnqueueLock;
        NThreading::TOneOneQueue<TRequesterPendingDataAllStatsCb> StatsReqQueue;
        NThreading::TOneOneQueue<TDebugStringCb> DebugReqQueue;
        //this event is used to block upper layer
        TManualEvent Event;
        //this enent is used to block Start function for thread starting time
        TManualEvent HasStarted;

        static void* ExecServerThread(void* param);

        void FailTransfersForConnection(TConnection* connection);

        void SuccessfulSend(const TTransfer& transfer);
        void FailedSend(const TTransfer& transfer);
        void CanceledSend(const TTransfer& transfer);

        void CheckConnectionsAndSendAcks();
        void SendAckForConnection(TConnection* connection, const float& deltaT);
        void InjectTransfers(TSendOrder* order, ui8 prio);
        ESentPacketResult SendTransferPacket(TConnection*, TUdpOutTransfer* xfer, ui64 transferId);
        bool SendCycle(ui8 prio);
        void RecvCycle();
        void OneStep();
        void StepLow();
        void WaitLow(float seconds);
        void ConnectLow(TConnection* connection);
        void SendLow(const TTransfer& transfer, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor);
        void CancelLow(const TTransfer& transfer);
        void CancelWaitLow();

        bool ProcessSystemPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd);
        bool ProcessInConnectionPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const ui8 extraFlags,
                                       const sockaddr_in6& fromAddress, const sockaddr_in6& dstAddress, TAutoPtr<TUdpRecvPacket> recvBuf);
        bool ProcessPingPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const sockaddr_in6& fromAddress, TConnection* connection);
        bool ProcessTransferPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, TAutoPtr<TUdpRecvPacket> recvBuf, TConnection* connection, const bool thatSideHasChanged, const TOptionsVector& opt);
        bool ProcessDataPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, TAutoPtr<TUdpRecvPacket> recvBuf, const TTransfer& transfer, const TOptionsVector& opt);
        bool ProcessCancelTransferPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const TTransfer& transfer, TConnection* connection);
        bool ProcessAcksPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const TTransfer& transfer, const bool thatSideHasChanged);

        bool ParseDataPacketHeader(const char* header, const char* headerEnd, TTransfer* transfer, int* packetId);
        ui8 FlushPacketsAndCheck(const TConnection* connection, const ui64 id);
        ui8 FlushPackets();
        bool ProcessIBRequest();
        void ProcessIBSendResults();
        void ProcessStatsRequest();
        void ProcessDebugRequests();
        std::pair<char*, ui8> GetPacketBuffer(const size_t bufSize, const TConnection* connection, const ui64 id);

        void AddToSendOrder(const TTransfer& transfer, EPacketPriority pp);
        bool CheckMTU(TConnection* connection, TUdpOutTransfer& xfer);

        TString GetConnectionsDebug() const;
        TString GetHostDebug() const;
        TString GetPendingDataStatsDebug(const TRequesterPendingDataStats& pds) const;
        void GetPendingDataSizeLow(TRequesterPendingDataAllStatsCb* res);

    public:
        TUdpHost(float udpTransferTimeout);
        ~TUdpHost() override;

        TUdpRequest* GetRequest() override;
        TIntrusivePtr<IConnection> Connect(const TUdpAddress& address, const TConnectionSettings& connectionSettings) override;
        TIntrusivePtr<IConnection> Connect(const TUdpAddress& address, const TUdpAddress& myAddress, const TConnectionSettings& connectionSettings) override;
        TTransfer Send(const TIntrusivePtr<IConnection>& connectionPtr, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor) override;
        bool GetSendResult(TSendResult* res) override;
        void Cancel(const TTransfer& transfer) override;
        void Step() override;
        void Wait(float seconds) override;
        void CancelWait() override;
        void GetAllPendingDataSize(TRequesterPendingDataAllStatsCb cb) override;
        void GetDebugInfo(TDebugStringCb cb) override;
        void Kill(const TUdpAddress&);

        bool Start(const TIntrusivePtr<ISocket>& socket);
        float GetFailRate() const override;
        bool IsLocal(const TUdpAddress& address) const override;
    };

    TUdpHost::TUdpHost(float udpTransferTimeout)
        : S(UDP_MAX_PACKETS_IN_QUEUE, true)
        , NewPacketsAfterLastFlush(0)
        , CurrentT(0)
        , IsWaiting(0)
        , MaxWaitTime(DEFAULT_MAX_WAIT_TIME)
        , MaxWaitTime2(DEFAULT_MAX_WAIT_TIME)
        , IBIdleTime(0)
        , TotalPendingDataStats(new TRequesterPendingDataStats)
        , TimeSinceSocketStatsUpdate(0)
        , UdpTransferTimeout(Min(UDP_TRANSFER_TIMEOUT, udpTransferTimeout))
        , UdpHostThread(TThread::TParams(ExecServerThread, (void*)this).SetName("nl12_udp_host"))
        , Run(1)
        , TXUserQueue([this](TConnection* connection) { ConnectLow(connection); },
                      [this](const TTransfer& transfer, TAutoPtr<TRopeDataPacket> data,
                             EPacketPriority pp, const TTos& tos, ui8 netlibaColor) { SendLow(transfer, data, pp, tos, netlibaColor); },
                      [this](const TTransfer& transfer) { CancelLow(transfer); }) {
    }

    void* TUdpHost::ExecServerThread(void* param) {
        TUdpHost* pThis = reinterpret_cast<TUdpHost*>(param);
        BindToSocket(0);
        SetHighestThreadPriority();
        pThis->HasStarted.Signal();
        while (AtomicAdd(pThis->Run, 0)) {
            pThis->StepLow();
            pThis->WaitLow(0.1f);
        }
        pThis->StepLow(); //one step in case of exit to empty TTXQueue
        return nullptr;
    }

    TUdpHost::~TUdpHost() {
        TUdpRequest* req = nullptr;
        AtomicSet(Run, 0);
        UdpHostThread.Join();
        while (ReceivedClientQueue.Dequeue(req))
            delete req;
    }

    bool TUdpHost::Start(const TIntrusivePtr<ISocket>& socket) {
        if (S.IsValid()) {
            Y_ASSERT(0);
            return false;
        }
        S.Open(socket);
        if (!S.IsValid())
            return false;

        if (IBDetection)
            IB = CreateIBClientServer();

        NHPTimer::GetTime(&CurrentT);
        UdpHostThread.Start();
        HasStarted.Wait();
        return true;
    }

    bool TUdpHost::ProcessIBRequest() {
        if (IB.Get()) {
            TAutoPtr<TIBRequest> r = IB->GetRequest();
            if (!!r) {
                const TIntrusivePtr<IConnection>* connection = Connections.FindPtr(r->ConnectionGuid);
                Y_ASSERT(connection); // hmm, old connection?
                TUdpRequest* result = nullptr;
                if (connection) {
                    result = new TUdpRequest;
                    result->IsHighPriority = false;
                    result->Connection = *connection;
                    result->Data = r->Data;
                }
                ReceivedClientQueue.Enqueue(result);
                Event.Signal();
                return true;
            }
        }
        return false;
    }

    void TUdpHost::ProcessIBSendResults() {
        if (IB.Get()) {
            TIBSendResult sr;
            while (IB->GetSendResult(&sr)) {
                TIBtoTransferHash::iterator z = IBKeyToTransfer.find(sr.Handle);
                if (z == IBKeyToTransfer.end()) {
                    Y_ABORT_UNLESS(0, "unknown handle returned from IB");
                }
                TTransfer transfer = z->second;
                IBKeyToTransfer.erase(z);

                TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
                TUdpOutTransfer* xferPtr = connection->GetSendQueue().Get(transfer.Id);
                if (!xferPtr) {
                    Y_ABORT_UNLESS(0, "IBKeyToTransferKey refers nonexisting xfer");
                }
                TUdpOutTransfer& xfer = *xferPtr;

                if (sr.Success) {
                    xfer.AckTracker.MarkAlive(); // got message from IB - mark connection as alive
                    SuccessfulSend(transfer);
                    //Y_ASSERT(SendResults.size() == 1);
                } else {
                    //printf("IB send failed, fall back to regular network\n");
                    // Houston, we got a problem
                    // IB failed to send, try to use regular network
                    AddToSendOrder(transfer, xfer.PacketPriority);
                }
            }
        }
    }

    TUdpRequest* TUdpHost::GetRequest() {
        TUdpRequest* result;
        if (ReceivedClientQueue.Dequeue(result)) {
            return result;
        }
        return nullptr; //No data
    }

    TIntrusivePtr<IConnection> TUdpHost::Connect(const TUdpAddress& address, const TConnectionSettings& connectionSettings) {
        return Connect(address, TUdpAddress(), connectionSettings);
    }

    TIntrusivePtr<IConnection> TUdpHost::Connect(const TUdpAddress& address, const TUdpAddress &myAddress, const TConnectionSettings& connectionSettings) {
        TGUID guid;
        CreateGuid(&guid);

        TConnection* connection = new TConnection(address, myAddress, connectionSettings, guid, UdpTransferTimeout);
        TIntrusivePtr<IConnection> result = connection;
        TXUserQueue.EnqueueConnect(connection);
        CancelWaitLow();
        return result;
    }

    void TUdpHost::ConnectLow(TConnection* connection) {
        Connections.Insert(connection->GetGuid(), connection);
        if (XsPingSending) {
            SendXsPing(S, connection, S.GetNetworkOrderPort(), 0);
        }
    }

    static ui8 ChooseTos(const int explicit_, const int default_) {
        return (ui8)(explicit_ == -1 ? default_ : explicit_);
    }

    TTransfer TUdpHost::Send(const TIntrusivePtr<IConnection>& connectionPtr, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor) {
        TConnection* connection = CheckedCast<TConnection*>(connectionPtr.Get());
        Y_ASSERT(connection);

        TTransfer transfer(connection, connection->GetNextTransferId());
        Y_ABORT_UNLESS(transfer.Id > 0, "transferId overflowed, wow you have counted to almost infinity!");
        TXUserQueue.EnqueueSend(transfer, data, pp, tos, netlibaColor);
        CancelWaitLow();
        return transfer;
    }
    void TUdpHost::SendLow(const TTransfer& transfer, TAutoPtr<TRopeDataPacket> data, EPacketPriority pp, const TTos& tos, ui8 netlibaColor) {
        TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
        std::pair<TUdpOutTransfer*, bool> p = connection->InsertSendTransfer(transfer.Id); // FailedSend needs created transfer
        Connections.InsertToActive(connection);
        Y_ASSERT(p.second);

        // shortcut for broken addresses or dead connection
        if (connection->GetAddress().Port == 0) {
            FailedSend(transfer);
            return;
        }

        Y_ASSERT(!connection->GetGuid().IsEmpty());

        TPeerLink& peerInfo = connection->GetAlivePeerLink();

        TUdpOutTransfer& xfer = *p.first;
        xfer.Data.Reset(data.Release());
        xfer.AckTracker.AttachCongestionControl(peerInfo.GetUdpCongestion().Get());
        xfer.PacketPriority = pp;
        xfer.DataTos = ChooseTos(tos.GetDataTos(), DefaultTos.GetDataTos());
        xfer.AckTos = ChooseTos(tos.GetAckTos(), DefaultTos.GetAckTos());
        xfer.NetlibaColor = netlibaColor;
        xfer.AttachStats(TotalPendingDataStats);
        xfer.AttachStats(ColoredPendingDataStats[netlibaColor]);
        xfer.AttachStats(connection->GetStatsPtr());

        // we don't support priorities (=service levels in IB terms) currently
        // so send only PP_NORMAL traffic over IB
        bool isSentOverIB = false;
        TIntrusivePtr<IIBPeer> ibPeer = peerInfo.GetIBPeer();
        if (pp == PP_NORMAL && ibPeer.Get() && xfer.Data->GetSharedData() == nullptr) {
            TGUID fakeUniqueGuid;
            CreateGuid(&fakeUniqueGuid);

            TIBMsgHandle hndl = IB->Send(ibPeer, xfer.Data.Get(), fakeUniqueGuid, connection->GetGuid());
            //fprintf(stderr, "TUdpHost::Send\tIB->Send returned %" PRIi64 "\n", (i64)hndl);

            if (hndl >= 0) {
                IBKeyToTransfer[hndl] = transfer;
                isSentOverIB = true;
            } else {
                // so we failed to use IB, ibPeer is either not connected yet or failed
                if (ibPeer->GetState() == IIBPeer::FAILED) {
                    //printf("Disconnect failed IB peer\n");
                    peerInfo.SetIBPeer(nullptr);
                }
            }
        }
        if (!isSentOverIB) {
            AddToSendOrder(transfer, pp);
        }
    }

    void TUdpHost::AddToSendOrder(const TTransfer& transfer, EPacketPriority pp) {
        if (pp == PP_LOW)
            SendOrderLow.push_back(transfer);
        else if (pp == PP_NORMAL)
            SendOrder.push_back(transfer);
        else if (pp == PP_HIGH)
            SendOrderHigh.push_back(transfer);
        else if (pp == PP_SYSTEM)
            SendOrderSystem.push_back(transfer);
        else
            Y_ASSERT(0);

        CancelWait();
    }

    bool TUdpHost::CheckMTU(TConnection* connection, TUdpOutTransfer& xfer) {
        Y_ASSERT(!xfer.AckTracker.IsInitialized());

        TIntrusivePtr<TCongestionControl> congestion = xfer.AckTracker.GetCongestionControl();
        Y_ASSERT(congestion.Get() != nullptr);
        if (!congestion->IsKnownMTU()) {
            TLameMTUDiscovery* md = congestion->GetMTUDiscovery();
            if (md->IsTimedOut()) {
                congestion->SetMTU(connection->GetSmallMtuUseXs() ? UDP_XSMALL_PACKET_SIZE : UDP_SMALL_PACKET_SIZE);

            } else {
                if (md->CanSend()) {
                    FlushPackets();
                    SendJumboPing(S, connection, S.GetNetworkOrderPort(), xfer.DataTos);
                    md->PingSent();
                }
                return false;
            }
        }

        // try to use large mtu, we could have selected small mtu due to connectivity problems
        if (congestion->GetMTU() == UDP_SMALL_PACKET_SIZE || congestion->GetMTU() == UDP_XSMALL_PACKET_SIZE || IB.Get() != nullptr) {
            // recheck every ~50mb
            int chkDenom = (50000000 / xfer.Data->GetSize()) | 1;
            if ((NetAckRnd() % chkDenom) == 0) {
                FlushPackets();

                //printf("send rechecking ping\n");
                if (congestion->GetMTU() == UDP_SMALL_PACKET_SIZE || congestion->GetMTU() == UDP_XSMALL_PACKET_SIZE) {
                    SendJumboPing(S, connection, S.GetNetworkOrderPort(), xfer.DataTos);
                } else {
                    SendIBOnlyPing(S, connection, S.GetNetworkOrderPort());
                }
            }
        }
        return true;
    }

    void TUdpHost::InjectTransfers(TSendOrder* order, ui8 prio) {
        for (TSendOrder::iterator z = order->begin(); z != order->end();) {
            const TTransfer& transfer = *z;
            TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
            TDeque<ui64>& transferIdQueue = connection->GetSendingTransfers(prio);
            transferIdQueue.push_back(transfer.Id);
            order->erase(z++);
            Connections.InsertToSending(connection);
        }
    }

    TUdpHost::ESentPacketResult TUdpHost::SendTransferPacket(TConnection* connection, TUdpOutTransfer* xfer, ui64 transferId) {
        NHPTimer::STime tCopy = CurrentT;
        float deltaT = (float)NHPTimer::GetTimePassed(&tCopy);
        deltaT = ClampVal(deltaT, 0.0f, UdpTransferTimeout / 3);

        bool isCanceled = false;
        const int packetId = xfer->AckTracker.GetPacketToSend(deltaT, &isCanceled);
        if (packetId == -1) {
            if (isCanceled) {
                if (xfer->TriedToSendAtLeastOnePacket) {
                    const ui8 err = FlushPackets(); // there must be FlushPackets before any Send.* function -
                    //it guarantee we have some space in packet buffer

                    if (err & TUdpHost::EFlashPacketResult::FPR_OUT_TRANSFERS_CHANGED) {
                        //also we must check _current_ transfer because FlushPackets() could remove it
                        if (!connection->IsSendTransferAlive(transferId)) {
                            return TUdpHost::ESentPacketResult::SPR_STOP_SENDING_TRANSFER;
                        }
                    }
                    SendCancelTransfer(S, connection, transferId, xfer->AckTos);
                    xfer->AckTracker.Congestion->ForceTimeAccount();
                } else {
                    xfer->AckTracker.AckAll(); // isn't necessary actually, but just to be sure...
                    CanceledSend(TTransfer(connection, transferId));
                }
            }
            return TUdpHost::ESentPacketResult::SPR_STOP_SENDING_TRANSFER;
        }
        const int dataSize = (packetId == xfer->PacketCount - 1) ? xfer->LastPacketSize : xfer->PacketSize;

        // I intentionally use very specific comparison here to minimize possible backward compatibility issues
        if (dataSize == UDP_SMALL_PACKET_SIZE && xfer->AckTracker.Congestion->GetMTU() == UDP_XSMALL_PACKET_SIZE) {
            // Cerr << "SendTransferPacker: dataSize " << dataSize << " > mtu " << xfer->AckTracker.Congestion->GetMTU()
            //      << ", xfer " << ui64(xfer) << " marked as failed" << Endl;
            FailedSend(TTransfer(connection, transferId));
            return TUdpHost::ESentPacketResult::SPR_STOP_SENDING_TRANSFER;
        }

        const std::pair<char* const, ui8>& packetBuffer = GetPacketBuffer(dataSize + PACKET_HEADERS_SIZE, connection, transferId);

        if (!packetBuffer.first) { //buffer overflow, or current transfer removed during packet flushing
            if (packetBuffer.second & TUdpHost::EFlashPacketResult::FPR_OUT_TRANSFERS_CHANGED) {
                //transfer removed
                return TUdpHost::ESentPacketResult::SPR_STOP_SENDING_TRANSFER;
            }
            //xfer->AckTracker.AddToResend(packetId); no need here - FlushPackets adds packetId to resend
            Y_ASSERT(packetBuffer.second & TUdpHost::EFlashPacketResult::FPR_OVERFLOW);
            return TUdpHost::ESentPacketResult::SPR_OVERFLOW;
        }

        xfer->TriedToSendAtLeastOnePacket = true;
        AddDataToPacketQueue(S, packetBuffer.first, connection, transferId, *xfer, packetId, dataSize);
        return TUdpHost::ESentPacketResult::SPR_OK;
    }

    bool TUdpHost::SendCycle(ui8 prio) {
        bool res1 = Connections.ForEachSendingConnection([prio, this](TSendingConnectionsList::iterator& sc) {
            typedef TConnections::ESentConnectionResult TSentConnectionResult;
            TConnection* connection = sc->Get();
            TDeque<ui64>& transferIdQueue = connection->GetSendingTransfers(prio);
            for (TDeque<ui64>::iterator z = transferIdQueue.begin(); z != transferIdQueue.end();) {
                const ui64 transferId = *z;
                TUdpOutTransfer* xfer = connection->GetSendQueue().Get(transferId);
                if (!xfer) {
                    z = transferIdQueue.erase(z);
                    if (transferIdQueue.empty()) {
                        return connection->HasAnySendingTransfers() ? TSentConnectionResult::SCR_CONT : TSentConnectionResult::SCR_DELETE; //delete connection if all queues are empty
                    } else {
                        continue;
                    }
                } else {
                    Y_ASSERT(connection->IsAlive());
                }
                const float deltaT = (float)NHPTimer::GetSeconds(CurrentT - xfer->LastTime);
                xfer->LastTime = CurrentT;

                if (!xfer->AckTracker.IsInitialized()) {
                    // Cerr << GetAddressAsString(connection->GetAddress()) << " Checking MTU for conn=" << ui64(connection) << ", xfer=" << ui64(xfer) << Endl;
                    if (!CheckMTU(connection, *xfer)) {
                        return TSentConnectionResult::SCR_CONT; //if we can`t get MTU for transfer we can`t get for connection
                    }
                    xfer->InitXfer();
                }
                xfer->AckTracker.Step(deltaT);
                MaxWaitTime = Min(MaxWaitTime, xfer->AckTracker.GetTimeToNextPacketTimeout());

                for (;;) {
                    if (!xfer->AckTracker.CanSend()) {
                        if (xfer->TriedToSendAtLeastOnePacket) {
                            //we can`t stop iteration if we start sending
                            break;
                        }
                        return TSentConnectionResult::SCR_CONT; //we can`t send more for current connection
                    }
                    TUdpHost::ESentPacketResult res2 = SendTransferPacket(connection, xfer, transferId);
                    if (res2 == TUdpHost::ESentPacketResult::SPR_STOP_SENDING_TRANSFER) {
                        break;
                    } else if (res2 == TUdpHost::ESentPacketResult::SPR_OVERFLOW) {
                        return TSentConnectionResult::SCR_BREAK;
                    }
                }
                ++z;
            }
            return TSentConnectionResult::SCR_CONT;
        });
        FlushPackets();
        return res1;
    }

    //Calls FlushPackets and checks status of current xfer
    //if got FPR_OUT_TRANSFERS_CHANGED for current xfer - current xfer is invalid. Return errors
    //if got FPR_OUT_TRANSFERS_CHANGED but not for current xfer - it is not error. Unset flag
    //pass FPR_OVERFLOW to err - it is not an error of particular xfer, it is an error of socket

    ui8 TUdpHost::FlushPacketsAndCheck(const TConnection* connection, const ui64 id) {
        ui8 err = FlushPackets();
        if (err & TUdpHost::EFlashPacketResult::FPR_OUT_TRANSFERS_CHANGED) {
            if (!connection->IsSendTransferAlive(id)) {
                return err;
            } else {
                //unset flag if it is not current xfer
                err &= ~TUdpHost::EFlashPacketResult::FPR_OUT_TRANSFERS_CHANGED;
            }
        }
        return err;
    }

    // Tries to get packet buffer in non-blocking way for given transfer
    // returns std::pair<nullptr, err> if _current_ transfer was removed during FlushPackets() or buffer ovrflow for _any_ transfers
    std::pair<char*, ui8> TUdpHost::GetPacketBuffer(const size_t bufSize, const TConnection* connection, const ui64 id) {
        Y_ASSERT(connection);
        ui8 err = TUdpHost::EFlashPacketResult::FPR_OK;
        // if small packet optimization is enabled then there could be thousands of packets added before we reach max udp packets in queue
        if (++NewPacketsAfterLastFlush > UDP_MAX_NEW_PACKETS_BEFORE_FORCED_FLUSH) {
            //printf("TUdpHost::GetPacketBuffer: too many small packets added, flushing\n");
            err = FlushPacketsAndCheck(connection, id);
            if (err)
                return std::make_pair(nullptr, err);
        }

        char* msgBuf = S.NewPacketBuffer(bufSize);
        if (!msgBuf) {
            err = FlushPacketsAndCheck(connection, id);
            if (err)
                return std::make_pair(nullptr, err);

            msgBuf = S.NewPacketBuffer(bufSize);
        }
        return std::make_pair(msgBuf, err);
    }

    // ui8 - error bitmap
    // FPR_OVERFLOW - there were non-sent packets because of udp/kernel buffer overflow.
    // FPR_OUT_TRANSFERS_CHANGED - some packets were not send because NO_ROUTE_TO_HOST
    // Informs all transfers about problems ("failed transfer" or "add packet to resend").
    // Clears TUdpSocket packet queue.
    ui8 TUdpHost::FlushPackets() {
        NewPacketsAfterLastFlush = 0;
        //The logic is we must inform caller about errors, but it is not good idea
        //to break flushing in case of NO_ROUTE_TO_HOST error. So, we will accumulate errors
        //as bitmask and pass it to caller
        ui8 errRet = TUdpHost::EFlashPacketResult::FPR_OK;
        for (;;) {
            size_t numSentPackets;
            TVector<std::pair<char*, size_t>> failedPackets;
            TUdpSocket::ESendError err = S.FlushPackets(&numSentPackets, &failedPackets);

            if (err == TUdpSocket::SEND_OK) {
                Y_ASSERT(S.IsPacketsQueueEmpty());
                return errRet;

                // most probably out of send buffer space (or something terrible has happened)
            } else if (err == TUdpSocket::SEND_BUFFER_OVERFLOW) {
                TVector<std::pair<char*, size_t>> notSentPackets;
                S.GetPacketsInQueue(&notSentPackets);

                for (size_t i = 0; i != notSentPackets.size(); ++i) {
                    const std::pair<char*, size_t>& notSentPacket = notSentPackets[i];
                    TTransfer transfer;
                    int packetId;
                    if (ParseDataPacketHeader(notSentPacket.first, notSentPacket.first + notSentPacket.second, &transfer, &packetId)) {
                        TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
                        connection->GetSendQueue().Get(transfer.Id)->AckTracker.AddToResend(packetId);
                        //fprintf(stderr, "Failed send\n");
                    }
                }
                MaxWaitTime = 0;
                S.ClearPacketsQueue(); // MUST be called only after flush (to simulate 100% flush in local connections)
                Y_ASSERT(S.IsPacketsQueueEmpty());
                errRet |= TUdpHost::EFlashPacketResult::FPR_OVERFLOW;
                return errRet;

            } else if (err == TUdpSocket::SEND_NO_ROUTE_TO_HOST || err == TUdpSocket::SEND_EINVAL) {
                const char* errText = (err == TUdpSocket::SEND_NO_ROUTE_TO_HOST) ? "No route to host" : "Error in value";
                for (size_t i = 0; i != failedPackets.size(); ++i) {
                    const std::pair<char*, size_t>& failedPacket = failedPackets[i];
                    TTransfer transfer;
                    int packetId;
                    if (ParseDataPacketHeader(failedPacket.first, failedPacket.first + failedPacket.second, &transfer, &packetId)) {
                        FailedSend(transfer);
                        fprintf(stderr, "%s, transfer: %" PRIu64 " failed, packetId: %" PRIi32 "\n",
                                errText, transfer.Id, packetId);
                    }
                    // packet is already dropped by socket queue
                }
                errRet |= TUdpHost::EFlashPacketResult::FPR_OUT_TRANSFERS_CHANGED;
                continue;

            } else {
                Y_ASSERT(false);
                break;
            }
        }
        Y_ASSERT(false); // unreachable
        return errRet;
    }

    bool TUdpHost::ParseDataPacketHeader(const char* header, const char* headerEnd, TTransfer* transfer, int* packetId) {
        EUdpCmd cmd;
        ui8 originalCmd;
        if (ReadBasicPacketHeader(&header, headerEnd, &cmd, &originalCmd)) {
            TGUID connectionGuid, thatSideGuid;
            TConnectionSettings settings;
            TOptionsVector opt;
            if (IsDataCmd(cmd) && ReadInConnectionPacketHeaderTail(&header, headerEnd, originalCmd, &connectionGuid, &thatSideGuid, &settings, &opt)) {
                ui64 transferId;
                if (ReadTransferHeader(&header, headerEnd, &transferId, opt)) {
                    int ackTos;
                    ui8 netlibaColor;
                    if (ReadPacketHeaderTail(&header, headerEnd, packetId, &ackTos, &netlibaColor, opt)) {
                        // hash map find only if everything else is OK
                        if (const TIntrusivePtr<IConnection>* connection = Connections.FindPtr(connectionGuid)) {
                            transfer->Connection = *connection;
                            transfer->Id = transferId;
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    void TUdpHost::FailTransfersForConnection(TConnection* connection) {
        for (TSendTransfers::TIdIterator i = connection->GetSendQueue().Begin(); i != connection->GetSendQueue().End();) {
            FailedSend(TTransfer(connection, *(i++)));
        }
        Y_ASSERT(connection->GetSendQueue().Empty());

        for (TRecvTransfers::TIdIterator i = connection->GetRecvQueue().Begin(); i != connection->GetRecvQueue().End();) {
            connection->FailedRecvTransfer(*(i++));
        }
        Y_ASSERT(connection->GetRecvQueue().Empty());
    }

    void TUdpHost::SuccessfulSend(const TTransfer& transfer) {
        CheckedCast<TConnection*>(transfer.Connection.Get())->SuccessfulSendTransfer(transfer.Id);
        SendResults.Enqueue(TSendResult(transfer, TSendResult::OK));
        Event.Signal();
    }

    void TUdpHost::FailedSend(const TTransfer& transfer) {
        CheckedCast<TConnection*>(transfer.Connection.Get())->FailedSendTransfer(transfer.Id);
        SendResults.Enqueue(TSendResult(transfer, TSendResult::FAILED));
        Event.Signal();
    }

    void TUdpHost::CanceledSend(const TTransfer& transfer) {
        CheckedCast<TConnection*>(transfer.Connection.Get())->CanceledSendTransfer(transfer.Id);
        SendResults.Enqueue(TSendResult(transfer, TSendResult::CANCELED));
        Event.Signal();
    }

    bool TUdpHost::GetSendResult(TSendResult* res) {
        if (SendResults.Dequeue(*res))
            return true;
        return false;
    }
    void TUdpHost::Cancel(const TTransfer& transfer) {
        TXUserQueue.EnqueueCancel(transfer);
        CancelWaitLow();
    }

    void TUdpHost::CancelLow(const TTransfer& transfer) {
        TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
        TUdpOutTransfer* xferPtr = connection->GetSendQueue().Get(transfer.Id);
        if (xferPtr) {
            // cancelling already finished transfer is no-op
            xferPtr->AckTracker.Cancel();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    static bool HasAllPackets(const TUdpInTransfer& res) {
        if (!res.HasLastPacket)
            return false;
        for (int i = res.GetPacketCount() - 1; i >= 0; --i) {
            if (!res.GetPacket(i))
                return false;
        }
        return true;
    }

    void TUdpHost::RecvCycle() {
        NHPTimer::STime now;
        NHPTimer::GetTime(&now);
        S.SetRecvLagTime(now - CurrentT);
        for (;;) {
            TSockAddrPair addr;
            TAutoPtr<TUdpRecvPacket> recvBuf = S.Recv(&addr);
            if (!recvBuf.Get()) { // no more packets in socket
                break;
            }

            const char* payloadStart = recvBuf->Data.get() + recvBuf->DataStart;
            const char* payloadEnd = payloadStart + recvBuf->DataSize;

            const char* pktData = payloadStart;
            EUdpCmd cmd;
            ui8 originalCmd;
            if (!ReadBasicPacketHeader(&pktData, payloadEnd, &cmd, &originalCmd)) {
                Y_ASSERT(false); // do not remove this yassert - it'll help you to find problems in protocol
                continue;
            }
            Y_ASSERT(recvBuf->DataStart != 0 || originalCmd == (ui8) * (recvBuf->Data.get() + CMD_POS));
            if (IsInConnectionCmd(cmd)) {
                ProcessInConnectionPacket(cmd, pktData, payloadEnd, originalCmd, addr.RemoteAddr, addr.MyAddr, recvBuf);

            } else if (IsSystemCmd(cmd)) {
                ProcessSystemPacket(cmd, pktData, payloadEnd);

            } else {
                // bug (forgotten cmd)? bad packet?
                Y_ASSERT(false);
            }
        }
    }

    bool TUdpHost::ProcessInConnectionPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const ui8 originalCmd,
                                             const sockaddr_in6& fromAddress, const sockaddr_in6& dstAddress, TAutoPtr<TUdpRecvPacket> recvBuf) {
        Y_ASSERT(IsInConnectionCmd(cmd));
        Y_ASSERT(pktData <= pktEnd);

        TGUID connectionGuid;
        TConnectionSettings settings;
        TOptionsVector opt;

        TGUID thatSideGuid;
        if (!ReadInConnectionPacketHeaderTail(&pktData, pktEnd, originalCmd, &connectionGuid, &thatSideGuid, &settings, &opt) ||
            connectionGuid.IsEmpty() || thatSideGuid.IsEmpty()) // validation
        {
            Y_ASSERT(false);
            return false;
        }

        TConnection* connection = nullptr;
        if (const TIntrusivePtr<IConnection>* c = Connections.FindPtr(connectionGuid)) {
            connection = CheckedCast<TConnection*>(c->Get());
            //Y_ASSERT(memcmp(&connection->GetWinsockAddress(), &fromAddress, sizeof(fromAddress)) == 0);

            // ignore imitator
            if (GetUdpAddress(fromAddress) != connection->GetAddress()) {
                Y_ASSERT(false);
                return true;
            }

        } else if (IsAckCmd(cmd)) {
            // OK, it seems we already deleted connection and got late duplicate ACK
            return true;

        } else {
            const TUdpAddress udpFromAddress = GetUdpAddress(fromAddress);
            const TUdpAddress toAddress = GetUdpAddress(dstAddress);

            connection = new TConnection(udpFromAddress, toAddress, settings, connectionGuid, UdpTransferTimeout);
            Connections[connectionGuid] = connection;
        }

        const bool thatSideHasChanged = !connection->CheckThatSideGuid(thatSideGuid);

        // actually GetAlivePeerLink would be enough
        connection->GetAlivePeerLink().GetUdpCongestion()->MarkAlive();

        Y_ASSERT(IsTransferCmd(cmd) || IsPingCmd(cmd));
        return IsTransferCmd(cmd) ? ProcessTransferPacket(cmd, pktData, pktEnd, recvBuf, connection, thatSideHasChanged, opt) : ProcessPingPacket(cmd, pktData, pktEnd, fromAddress, connection);
    }

    bool TUdpHost::ProcessTransferPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd,
                                         TAutoPtr<TUdpRecvPacket> recvBuf, TConnection* connection, const bool thatSideHasChanged, const TOptionsVector& opt) {
        Y_ASSERT(IsTransferCmd(cmd));
        Y_ASSERT(pktData <= pktEnd);

        ui64 transferId;
        if (!ReadTransferHeader(&pktData, pktEnd, &transferId, opt) || !transferId) // validation
        {
            Y_ASSERT(false);
            return false;
        }

        const TTransfer transfer(connection, transferId);

        if (IsDataCmd(cmd)) {
            return ProcessDataPacket(cmd, pktData, pktEnd, recvBuf, transfer, opt);
        } else if (IsAckCmd(cmd)) {
            return ProcessAcksPacket(cmd, pktData, pktEnd, transfer, thatSideHasChanged);
        } else if (IsCancelTransferCmd(cmd)) {
            return ProcessCancelTransferPacket(cmd, pktData, pktEnd, transfer, connection);
        } else {
            Y_ASSERT(false);
            return true;
        }
    }

    bool TUdpHost::ProcessDataPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd,
                                     TAutoPtr<TUdpRecvPacket> recvBuf, const TTransfer& transfer, const TOptionsVector& opt) {
        Y_ASSERT(IsDataCmd(cmd));
        Y_ASSERT(pktData <= pktEnd);

        int packetId;
        int ackTos;
        ui8 netlibaColor;

        if (!ReadPacketHeaderTail(&pktData, pktEnd, &packetId, &ackTos, &netlibaColor, opt) || packetId < 0 || !IsValidTos(ackTos)) {
            Y_ASSERT(false);
            return false;
        }

        TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
        Connections.InsertToActive(connection);
        bool isFailed = false, isCanceled = false;
        if (connection->IsRecvCompleted(transfer.Id, &isFailed, &isCanceled)) {
            Y_ASSERT(!connection->GetRecvQueue().Has(transfer.Id));

            if (isCanceled) {
                SendAckCanceled(S, connection, transfer.Id, packetId, DefaultTos.GetAckTos());
            } else if (isFailed) {
                ; // TODO: we cannot continue, at least we must reset everything, maybe ignore?
            } else {
                SendAckComplete(S, connection, transfer.Id, packetId, DefaultTos.GetAckTos()); // TODO: where to get ack tos?
            }
            return true;
        }

        std::pair<TUdpInTransfer*, bool> p = connection->InsertRecvTransfer(transfer.Id);
        if (p.second) {
            //printf("new input transfer\n");
            TUdpInTransfer& res = *p.first;
            res.AckTos = ChooseTos(ackTos, DefaultTos.GetAckTos());
            res.NetlibaColor = netlibaColor;
            res.HighPriority = opt.TransferOpt.IsHighPriority();
            res.PacketSize = 0;
            res.HasLastPacket = false;
            res.AttachStats(TotalPendingDataStats);
            res.AttachStats(ColoredPendingDataStats[netlibaColor]);
            res.AttachStats(connection->GetStatsPtr());
        }
        TUdpInTransfer& xfer = *p.first;
        xfer.TimeSinceLastRecv = 0;

        // got duplicate packet
        if (packetId < xfer.GetPacketCount() && xfer.GetPacket(packetId)) {
            xfer.NewPacketsToAck.push_back(packetId);
            return true;
        }

        if (!ReadDataPacket(cmd, &pktData, pktEnd, packetId, &xfer.SharedData, &xfer.PacketSize, opt)) {
            if (opt.TransferOpt.IsSharedMemoryRequired() && xfer.SharedData.Get() && !xfer.SharedData->GetPtr()) {
                SendAckResendNoShmem(S, connection, transfer.Id, DefaultTos.GetAckTos());
            }
            Y_ASSERT(false);
            return false;
        }

        const size_t dataSize = pktEnd - pktData;
        if (dataSize > xfer.PacketSize) {
            // Cerr << "ProcessDataPacket: dataSize " << dataSize << " > xfer.PacketSize " << xfer.PacketSize << Endl;
            Y_ASSERT(false);
            return false; // mem overrun protection
        }

        const bool isLastPacket = dataSize < xfer.PacketSize;
        if (isLastPacket) {
            if (xfer.HasLastPacket || packetId < xfer.GetPacketCount()) {
                Y_ASSERT(false);
                return false;
            }
            xfer.LastPacketSize = dataSize;
            xfer.HasLastPacket = true;
        }

        if (packetId >= xfer.GetPacketCount()) {
            xfer.SetPacketCount(packetId + 1);
        }

        TUdpRecvPacket* pkt1 = nullptr;
        if (xfer.PacketSize == UDP_SMALL_PACKET_SIZE || xfer.PacketSize == UDP_XSMALL_PACKET_SIZE) {
            // save memory by using smaller buffer at the cost of additional memcpy
            pkt1 = TUdpHostRecvBufAlloc::CreateNewSmallPacket(dataSize);
            memcpy(pkt1->Data.get(), pktData, dataSize);
            pkt1->DataStart = 0;
        } else {
            Y_ASSERT(xfer.PacketSize == UDP_PACKET_SIZE);
            pkt1 = recvBuf.Release();
            pkt1->DataStart = (int)(pktData - pkt1->Data.get()); // data offset in the packet;
        }
        pkt1->DataSize = dataSize;
        xfer.AssignPacket(packetId, pkt1);

        if (HasAllPackets(xfer)) {
            //printf("received\n");
            /*
            Cerr << GetAddressAsString(connection->GetAddress()) << " all packets received: " << xfer.GetPacketCount()
                << ", packet.dataSize=" << dataSize
                << ", xfer.PacketSize=" << xfer.PacketSize
                << ", xfer.LastPacketSize=" << xfer.LastPacketSize
                << ", packetopts.SmallMtuUseXs=" << opt.PacketOpt.IsSmallMtuUseXs()
                << ", *conn=" << size_t(connection)
                << Endl;
            */
            TUdpRequest* out = new TUdpRequest;
            out->Connection = connection;

            const int packetCount = xfer.GetPacketCount();
            out->Data.Reset(new TRopeDataPacket);
            for (int i = 0; i < packetCount; ++i) {
                TUdpRecvPacket* pkt2 = xfer.ExtractPacket(i);
                Y_ASSERT((size_t)pkt2->DataSize == ((i == packetCount - 1) ? xfer.LastPacketSize : xfer.PacketSize));
                out->Data->AddBlock(pkt2, pkt2->Data.get() + pkt2->DataStart, pkt2->DataSize);
            }
            out->Data->AttachSharedData(xfer.SharedData);
            out->IsHighPriority = xfer.HighPriority;
            ReceivedClientQueue.Enqueue(out);
            Event.Signal();
            xfer.EraseAllPackets();

            SendAckComplete(S, connection, transfer.Id, packetId, xfer.AckTos);
            connection->SuccessfulRecvTransfer(transfer.Id);

        } else {
            /*
            Cerr << GetAddressAsString(connection->GetAddress()) << " got packet: " << (packetId+1) << "/" << xfer.GetPacketCount()
                << ", packet.dataSize=" << dataSize
                << ", xfer.PacketSize=" << xfer.PacketSize
                << ", xfer.LastPacketSize=" << xfer.LastPacketSize
                << ", packetopts.SmallMtuUseXs=" << opt.PacketOpt.IsSmallMtuUseXs()
                << ", *conn=" << size_t(connection)
                << Endl;
            */
            xfer.NewPacketsToAck.push_back(packetId);
        }

        return true;
    }

    bool TUdpHost::ProcessAcksPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd,
                                     const TTransfer& transfer, const bool thatSideHasChanged) {
        Y_ASSERT(IsAckCmd(cmd));

        TConnection* connection = CheckedCast<TConnection*>(transfer.Connection.Get());
        TUdpOutTransfer* xferPtr = connection->GetSendQueue().Get(transfer.Id);
        if (!xferPtr) {
            return true;
        }
        TUdpOutTransfer& xfer = *xferPtr;

        if (!ReadAckPacketHeader(cmd, &pktData, pktEnd)) {
            Y_ASSERT(false);
            return false;
        }

        NHPTimer::STime tCopy = CurrentT;
        float deltaT = (float)NHPTimer::GetTimePassed(&tCopy);
        deltaT = ClampVal(deltaT, 0.0f, UdpTransferTimeout / 3);

        //printf("ack (%d) transferId = %" PRIu64 "\n", (int)cmd, transfer.Id);

        switch (cmd) {
            case ACK: {
                if (thatSideHasChanged) {
                    return true;
                }
                return ReadAndSetAcks(pktData, pktEnd, deltaT, &xfer.AckTracker);
            }
            case ACK_COMPLETE:
            case ACK_CANCELED: {
                int lastPacketId;
                if (!ReadAckCompleteAndCanceledTails(pktData, pktEnd, &lastPacketId)) {
                    Y_ASSERT(false);
                    return false;
                }

                if (!thatSideHasChanged && lastPacketId != -1) {
                    xfer.AckTracker.Ack(lastPacketId, deltaT, true); // update RTT
                }
                xfer.AckTracker.AckAll(); // acking packets is required, otherwise they will be treated as lost (look AckTracker destructor)

                if (cmd == ACK_COMPLETE) {
                    SuccessfulSend(transfer);
                } else {
                    CanceledSend(transfer);
                }

                return true;
            }
            case ACK_RESEND_NOSHMEM: {
                // abort execution here
                // failed to open shmem on recv side, need to transmit data without using shmem
                Y_ABORT_UNLESS(0, "not implemented yet");
                return true;
            }
            default:
                break;
        }
        Y_ASSERT(false && "Not a ACK cmd!");
        return false;
    }

    bool TUdpHost::ProcessCancelTransferPacket(const EUdpCmd cmd, const char*, const char*, const TTransfer& transfer, TConnection* connection) {
        switch (cmd) {
            case CANCEL_TRANSFER: {
                bool isFailed, isCanceled;

                ui8 tos = DefaultTos.GetAckTos();
                TUdpInTransfer* xferPtr = connection->GetRecvQueue().Get(transfer.Id);

                // there is an active transfer
                if (xferPtr) {
                    tos = xferPtr->AckTos;
                    connection->CanceledRecvTransfer(transfer.Id);

                    // transfer is finished or has not been started yet
                } else {
                    // may be we have not received any packet of this transfer, maybe we successfully received it, maybe we cancelled it...
                    const bool isCompleted = connection->IsRecvCompleted(transfer.Id, &isFailed, &isCanceled);

                    // already successfully received it or it failed or it was previously canceled
                    if (isCompleted) {
                        if (!isFailed && !isCanceled) {
                            SendAckComplete(S, connection, transfer.Id, -1, tos);
                            return true;
                        } else if (isFailed) {
                            // ignore
                            return true;
                        }
                        Y_ASSERT(isCanceled);

                        // we never received any packet of this transfer!
                    } else {
                        std::pair<TUdpInTransfer*, bool> p = connection->InsertRecvTransfer(transfer.Id);
                        Y_ASSERT(p.second);
                        connection->CanceledRecvTransfer(transfer.Id);
                        Connections.InsertToActive(connection);
                    }
                }

                // we will ignore upcoming packets and response with ACK_CANCELED on them
                Y_ASSERT(connection->IsRecvCompleted(transfer.Id, &isFailed, &isCanceled) && isCanceled);

                SendAckCanceled(S, connection, transfer.Id, -1, tos); // we always answer even if it's duplicate packet
                return true;
            }
            default:
                break;
        }
        Y_ASSERT(0 && "Not a cancel transfer cmd!");
        return false;
    }

    bool TUdpHost::ProcessPingPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd, const sockaddr_in6& fromAddress,
                                     TConnection* connection) {
        Y_ASSERT(IsPingCmd(cmd));
        Y_ASSERT(pktData <= pktEnd);

        switch (cmd) {
            case XS_PING:
            case PING: {
                sockaddr_in6 trueFromAddress = fromAddress;
                Y_ASSERT(trueFromAddress.sin6_family == AF_INET6);

                /*
                Cerr <<  GetAddressAsString(connection->GetAddress()) << " got ping " << int(cmd)
                    << ", *conn=" << size_t(connection)
                    << Endl;
                */
                // can not set MTU for fromAddress here since asymmetrical mtu is possible
                if (!ReadPing(pktData, pktEnd, &trueFromAddress.sin6_port)) {
                    Y_ASSERT(false);
                    return false;
                }

                if (IB.Get()) {
                    // For now just ignore XS pings over IB
                    if (cmd == PING) {
                        SendIBPong(S, connection, IB->GetConnectInfo(), trueFromAddress);
                    }
                } else {
                    if (cmd == XS_PING) {
                        connection->SetSmallMtuUseXs(true);
                        TPeerLink& peerInfo = connection->GetAlivePeerLink();
                        auto congestion = peerInfo.GetUdpCongestion();
                        if (congestion->GetMTU() == UDP_SMALL_PACKET_SIZE) {
                            // Cerr <<  GetAddressAsString(connection->GetAddress()) << " dropping MTU to " << int(UDP_XSMALL_PACKET_SIZE) << Endl;
                            congestion->SetMTU(UDP_XSMALL_PACKET_SIZE);
                        }
                    }
                    SendPong(S, connection, trueFromAddress, cmd == XS_PING);
                }
                return true;
            }
            case XS_PONG: {
                /*
                Cerr <<  GetAddressAsString(connection->GetAddress()) << " got xs pong"
                    << ", *conn=" << size_t(connection)
                    << Endl;
                */

                Connections.InsertToActive(connection);
                connection->SetSmallMtuUseXs(true);
                TPeerLink& peerInfo = connection->GetAlivePeerLink();
                auto congestion = peerInfo.GetUdpCongestion();
                if (congestion->GetMTU() == UDP_SMALL_PACKET_SIZE) {
                    // Cerr << GetAddressAsString(connection->GetAddress()) << " dropping MTU to " << int(UDP_XSMALL_PACKET_SIZE) << Endl;
                    congestion->SetMTU(UDP_XSMALL_PACKET_SIZE);
                }

                return true;
            }
            case PONG: {
                TPeerLink& peerInfo = connection->GetAlivePeerLink();
                Connections.InsertToActive(connection);
                peerInfo.GetUdpCongestion()->SetMTU(UDP_PACKET_SIZE);
                /*
                Cerr <<  GetAddressAsString(connection->GetAddress()) << " got pong"
                    << ", *conn=" << size_t(connection)
                    << Endl;
                */

                return true;
            }
            case PONG_IB: {
                TPeerLink& peerInfo = connection->GetAlivePeerLink();
                Connections.InsertToActive(connection);
                peerInfo.GetUdpCongestion()->SetMTU(UDP_PACKET_SIZE); // TODO: PONG_IB won't reach dst if mtu is small

                if (IB.Get() && !peerInfo.HasIBPeer()) {
                    TIBConnectInfo info;
                    sockaddr_in6 myAddress;
                    if (!ReadIBPong(pktData, pktEnd, &info, &myAddress)) {
                        Y_ASSERT(false);
                        return false;
                    }
                    peerInfo.SetIBPeer(IB->ConnectPeer(info, connection->GetAddress(), GetUdpAddress(myAddress)));
                }
                return true;
            }
            default:
                break;
        }
        Y_ASSERT(0 && "Not a ping cmd!");
        return false;
    }

    static void DumpConnection(const TConnection* connection);

    bool TUdpHost::ProcessSystemPacket(const EUdpCmd cmd, const char* pktData, const char* pktEnd) {
        Y_ASSERT(IsSystemCmd(cmd));
        Y_ASSERT(pktData <= pktEnd);

        switch (cmd) {
            case KILL: {
                if (ReadKill(pktData, pktEnd)) {
                    fprintf(stderr, "CONNECTIONS DEBUG: %s\n\n\n", GetConnectionsDebug().c_str());

                    for (TConnections::const_iterator i = Connections.Begin(); i != Connections.End(); ++i) {
                        DumpConnection(static_cast<TConnection*>(i->second.Get()));
                    }
                    abort();
                }
                Y_ASSERT(false);
                return false;
            }
            default:
                break;
        }
        Y_ASSERT(0 && "Not a system cmd!");
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////////

    static void DumpConnection(const TConnection* connection) {
        if (!connection->GetSendQueue().Empty() || !connection->GetRecvQueue().Empty()) {
            fprintf(stderr, "congestion timeout! sends: %d, recv: %d\n",
                    (int)connection->GetSendQueue().Size(), (int)connection->GetRecvQueue().Size());
        }

        fprintf(stderr, "CONNECTION: %s\n", GetGuidAsString(connection->GetGuid()).c_str());

        fprintf(stderr, "stats - in count: %d, out count: %d, in bytes: %d, out bytes: %d\n",
                connection->GetPendingDataSize().InpCount, connection->GetPendingDataSize().OutCount,
                (int)connection->GetPendingDataSize().InpDataSize, (int)connection->GetPendingDataSize().OutDataSize);

        fprintf(stderr, "printng debug info...\n%s\n", connection->GetDebugInfo().c_str());

        fprintf(stderr, "printing sends...\n");

        for (TSendTransfers::TIdIterator i1 = connection->GetSendQueue().Begin(); i1 != connection->GetSendQueue().End(); ++i1) {
            const TUdpOutTransfer& xfer = *connection->GetSendQueue().Get(*i1);

            fprintf(stderr, "send: %" PRIu64 ", packet count: %d, packet size: %d, last packet size: %d\n",
                    (ui64)*i1, xfer.PacketCount, xfer.PacketSize, xfer.LastPacketSize);

            fprintf(stderr, "ack tracker - packet count: %d, current packet: %d\n",
                    xfer.AckTracker.PacketCount, xfer.AckTracker.CurrentPacket);

            fprintf(stderr, "\tacks: \n");
            for (int i2 = 0; i2 != (int)xfer.AckTracker.GetAckReceived().size(); ++i2) {
                fprintf(stderr, "\t\t%d: %s\n", i2, xfer.AckTracker.GetAckReceived()[i2] ? "acked" : "not acked");
            }

            fprintf(stderr, "\tresend queue: \n");
            for (int i3 = 0; i3 != (int)xfer.AckTracker.ResendQueue.size(); ++i3) {
                fprintf(stderr, "\t\t%d\n", xfer.AckTracker.ResendQueue[i3]);
            }

            fprintf(stderr, "\tpackets in fly: \n");
            for (const auto& i4 : xfer.AckTracker.PacketsInFly) {
                fprintf(stderr, "\t\t%d\n", i4.first);
            }

            fprintf(stderr, "\tdropped packets: \n");
            for (const auto& droppedPacket : xfer.AckTracker.DroppedPackets) {
                fprintf(stderr, "\t\t%d\n", droppedPacket.first);
            }
        }

        fprintf(stderr, "printing recvs...\n");

        for (TRecvTransfers::TIdIterator i1 = connection->GetRecvQueue().Begin(); i1 != connection->GetRecvQueue().End(); ++i1) {
            const TUdpInTransfer& xfer = *connection->GetRecvQueue().Get(*i1);

            fprintf(stderr, "recv: %" PRIu64 ", packet count: %d, has last: %d, packet size: %d, last packet size: %d"
                            ", time since last recv: %f\n",
                    (ui64)*i1, xfer.GetPacketCount(), (int)xfer.HasLastPacket, (int)xfer.PacketSize, (int)xfer.LastPacketSize,
                    xfer.TimeSinceLastRecv);

            fprintf(stderr, "\tpackets:");
            for (int i2 = 0; i2 != xfer.GetPacketCount(); ++i2) {
                fprintf(stderr, "\t\t%d: %s\n", i2, xfer.GetPacket(i2) ? "received" : "not received");
            }

            fprintf(stderr, "\tnew packets to ack:");
            for (int i3 = 0; i3 != xfer.NewPacketsToAck.ysize(); ++i3) {
                fprintf(stderr, " %d, ", xfer.NewPacketsToAck[i3]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "\n");
    }
    void TUdpHost::SendAckForConnection(TConnection* connection, const float& deltaT) {
        for (TRecvTransfers::TIdIterator i = connection->GetRecvQueue().Begin(); i != connection->GetRecvQueue().End(); ++i) {
            TTransfer transfer(connection, *i);
            TUdpInTransfer& xfer = *connection->GetRecvQueue().Get(*i);
            xfer.TimeSinceLastRecv += deltaT;
            if (xfer.TimeSinceLastRecv > UDP_MAX_INPUT_DATA_WAIT) {
                fprintf(stderr, "recv %" PRIu64 " failed by timeout\n", (ui64)*i);
                connection->FailedRecvTransfer(transfer.Id);
                continue;
            }
#ifndef NDEBUG
            bool dummy;
            Y_ASSERT(!connection->IsRecvCompleted(transfer.Id, &dummy, &dummy)); // state "Complete & incomplete" is incorrect
#endif
            if (!xfer.NewPacketsToAck.empty()) {
                std::pair<char*, ui8> packetBuffer = GetPacketBuffer(UDP_PACKET_BUF_SIZE, connection, transfer.Id);
                if (!packetBuffer.first) { // buffer overflow, stop trying to send ACK, continue just checking keep alives.
                    fprintf(stderr, "can`t get packetBuffer to send ACK, err: %i\n", packetBuffer.second);
                    continue;
                }
                AddAcksToPacketQueue(S, packetBuffer.first, UDP_PACKET_BUF_SIZE, connection, transfer.Id, &xfer);
            }
        }
    }

    void TUdpHost::CheckConnectionsAndSendAcks() {
        NHPTimer::STime start;
        NHPTimer::GetTime(&start);
        Connections.ForEachActiveConnection([start, this](TActiveConnectionList::iterator& ac) {
            float stepDeltaT;
            TConnection* conn = &(*ac);
            const bool congestionTimeout = !conn->Step(UDP_KEEP_CONNETION, &MaxWaitTime, &stepDeltaT, start, &FailureStats);
            if (congestionTimeout) {
                FailTransfersForConnection(conn);
            } else {
                SendAckForConnection(conn, stepDeltaT);
            }
            if (congestionTimeout || conn->IsSleeping()) {
                return false; //detele me please
            }
            return true;
        });

        NHPTimer::GetTime(&start);

        Connections.ForEachConnection([start, this](TConnections::const_iterator c) {
            TConnection* connection = CheckedCast<TConnection*>(c->second.Get());
            NHPTimer::STime now = start;
            const float passed = (float)NHPTimer::GetTimePassed(&now);
            if (passed > DEFAULT_MAX_SEND_RECV_LATENCY) {
                return TConnections::EIdleResult::ER_YIELD;
            }

            // conegistionTimeout - connection is not used for UDP_KEEP_CONNETION seconds or congestion is not alive.
            // (there may be some transfer references in SendOrder so we destroy connection in next few steps).
            float stepDeltaT;
            const bool congestionTimeout = !connection->Step(UDP_KEEP_CONNETION, &MaxWaitTime, &stepDeltaT, now, &FailureStats);
            if (congestionTimeout || connection->IsSleeping()) {
                Connections.EraseFromActive(connection);
            }
            if (congestionTimeout) {
                FailTransfersForConnection(connection);
            }

            // Only 1 reference from Connections left.
            // For send connections: There are no refs from user handles (TSendResult) and no refs from SendOrder.
            // For recv connections: There are no refs from user handles (TRequest), connectionTimeout -- it's too old (or broken)
            if (c->second.RefCount() == 1 && congestionTimeout) {
                return TConnections::EIdleResult::ER_DELETE;
            }
            return TConnections::EIdleResult::ER_CONT;

        });

        FlushPackets();
        Y_ASSERT(S.IsPacketsQueueEmpty());
    }

    void TUdpHost::OneStep() {
        if (IB.Get()) {
            NHPTimer::STime tChk = CurrentT;
            float chkDeltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (IB->Step(tChk)) {
                IBIdleTime = -chkDeltaT;
            }
            if (chkDeltaT < 0.0005) {
                return;
            }
        }

        if (UseTOSforAcks) {
            DefaultTos.SetDataTos(0x60);
            DefaultTos.SetAckTos(0x20);
        } else {
            DefaultTos.SetDataTos(0);
            DefaultTos.SetAckTos(0);
        }

        RecvCycle();
        while (ProcessIBRequest())
            ; //TODO implement some limitations

        float deltaT = (float)NHPTimer::GetTimePassed(&CurrentT);
        TimeSinceSocketStatsUpdate += deltaT;
        deltaT = ClampVal(deltaT, 0.0f, UdpTransferTimeout / 3);

        MaxWaitTime = DEFAULT_MAX_WAIT_TIME;
        IBIdleTime += deltaT;

        CheckConnectionsAndSendAcks();

        // send data for outbound connections
        InjectTransfers(&SendOrderSystem, 0);
        InjectTransfers(&SendOrderHigh, 1);
        InjectTransfers(&SendOrder, 2);
        InjectTransfers(&SendOrderLow, 3);

        ui8 prio = 0;
        while (SendCycle(prio++)) {
            if (prio > PP_SYSTEM) {
                break;
            }
        }

        // roll send order to avoid exotic problems with lots of peers and high traffic
        //SendOrderSystem.splice(SendOrderSystem.end(), SendOrderSystem, SendOrderSystem.begin());
        //SendOrder.splice(SendOrder.end(), SendOrder, SendOrder.begin()); // sending data in order has lower delay and shorter queue
        if (TimeSinceSocketStatsUpdate > STAT_UPDATE_TIME) {
            S.UpdateStats();
            FailureStats.Update();
            TimeSinceSocketStatsUpdate = 0;
        }
    }

    //do nothing, keep it for compatibility
    void TUdpHost::Step() {
    }

    //TODO: not perfect but emulate old behavoiur
    void TUdpHost::Wait(float seconds) {
        if (SendResults.IsEmpty() && ReceivedClientQueue.IsEmpty()) {
            Event.Reset();
            if (SendResults.IsEmpty() && ReceivedClientQueue.IsEmpty()) {
                Event.Wait(ui32(seconds * 1000.0));
            }
        }
    }

    void TUdpHost::StepLow() {
        TXUserQueue.DequeueAndRun();
        ProcessIBSendResults();
        ProcessStatsRequest();
        ProcessDebugRequests();
        size_t i = 0;
        for (;;) { // it should break sometimes
            i++;
            OneStep();

            // reducing "CancelWait" packets between local connections
            if (MaxWaitTime != 0) {
                break;
            }
            if (Connections.IsFinished()) {
                break;
            }
            if (i > 10000) {
                fprintf(stderr, "too many OneStep() call, breaking loop. Adjust timeout?\n");
                break;
            }
        }
    }

    void TUdpHost::WaitLow(float seconds) {
        if (seconds < 1e-3)
            seconds = 0;
        if (seconds > MaxWaitTime)
            seconds = MaxWaitTime;
        if (IBIdleTime < 0.010) {
            seconds = 0;
        }
        if (seconds == 0) {
            ThreadYield();
        } else {
            AtomicAdd(IsWaiting, 1);
            if (seconds > MaxWaitTime2.load(std::memory_order_acquire))
                seconds = MaxWaitTime2.load(std::memory_order_acquire);
            MaxWaitTime2.store(DEFAULT_MAX_WAIT_TIME, std::memory_order_release);

            if (seconds == 0) {
                ThreadYield();
            } else {
                if (IB.Get()) {
                    for (float done = 0; done < seconds;) {
                        float deltaSleep = Min(seconds - done, 0.002f);
                        S.Wait(deltaSleep);
                        NHPTimer::STime tChk;
                        NHPTimer::GetTime(&tChk);
                        if (IB->Step(tChk)) {
                            IBIdleTime = 0;
                            break;
                        }
                        done += deltaSleep;
                    }
                } else {
                    S.Wait(seconds);
                }
            }
            AtomicAdd(IsWaiting, -1);
        }
    }

    void TUdpHost::CancelWait() {
        Event.Signal();
    }

    void TUdpHost::CancelWaitLow() {
        MaxWaitTime2.store(0, std::memory_order_release);
        if (AtomicAdd(IsWaiting, 0) == 1) {
            S.CancelWait();
        }
    }

    void TUdpHost::ProcessStatsRequest() {
        TRequesterPendingDataAllStatsCb cb;
        while (StatsReqQueue.Dequeue(cb)) {
            if (cb)
                cb(*TotalPendingDataStats, ColoredPendingDataStats.DeepCopy());
        }
    }

    void TUdpHost::GetAllPendingDataSize(TRequesterPendingDataAllStatsCb cb) {
        {
            TGuard<TSpinLock> lock(StatsEnqueueLock);
            StatsReqQueue.Enqueue(cb);
        }
        CancelWaitLow();
    }

    float TUdpHost::GetFailRate() const {
        return FailureStats.GetResult();
    }

    bool TUdpHost::IsLocal(const TUdpAddress& address) const {
        return S.IsLocal(address);
    }

    ///////////////////////////////////////////////////////////////////////////////
    TString TUdpHost::GetHostDebug() const {
        TString result;
        result += "Loop preempted by timeout:\t";
        result += ToString(Connections.GetLoopPreemptedNum());
        result += "\nCache reseted during preempted state:\t";
        result += ToString(Connections.GetCacheResetNum());
        result += "\nFailRateAgregate:\t";
        result += ToString(FailureStats.GetResult());
        result += "\nUdpTransferTimeout:\t";
        result += ToString(UdpTransferTimeout);
        result += "\n";
        return result;
    }

    TString TUdpHost::GetConnectionsDebug() const {
        TString result;
        result += "Total:\t";
        result += ToString(Connections.GetSize());
        result += "\nActive:\t";
        result += ToString(Connections.GetActiveSize());
        result += "\nSending:\t";
        result += ToString(Connections.GetSendingSize());
        result += "\n";
        for (TConnections::const_iterator i = Connections.Begin(); i != Connections.End(); ++i) {
            const TConnection* connection = CheckedCast<const TConnection*>(i->second.Get());
            result += connection->GetDebugInfo();
            result += "\n";
            result += GetPendingDataStatsDebug(connection->GetPendingDataSize());
            result += "\n";
        }
        return result;
    }

    TString TUdpHost::GetPendingDataStatsDebug(const TRequesterPendingDataStats& pds) const {
        char buf[1000];
        snprintf(buf, sizeof(buf), "\tPending data size: %" PRIu64 "\n\t\tin packets: %d, size %" PRIu64 "\n\t\tout packets: %d, size %" PRIu64 "\n",
                pds.InpDataSize + pds.OutDataSize,
                pds.InpCount, pds.InpDataSize,
                pds.OutCount, pds.OutDataSize);
        return buf;
    }

    void TUdpHost::GetDebugInfo(TDebugStringCb cb) {
        DebugReqQueue.Enqueue(cb);
        CancelWaitLow();
    }
    void TUdpHost::ProcessDebugRequests() {
        TDebugStringCb cb;
        while (DebugReqQueue.Dequeue(cb)) {
            if (!cb)
                continue;
            const TRequesterPendingDataStats& pds = *TotalPendingDataStats;
            TString res;
            char buf[1000];
            snprintf(buf, sizeof(buf), "Receiving %d transfers, sending %d system prior, sending %d high prior, %d regular, %d low prior\n",
                    pds.InpCount, (int)SendOrderSystem.size(), (int)SendOrderHigh.size(), (int)SendOrder.size(), (int)SendOrderLow.size());
            res += buf;

            res += "Total pending data stats:\n";
            res += GetPendingDataStatsDebug(*TotalPendingDataStats);

            for (TColoredRequesterPendingDataStats::const_iterator i = ColoredPendingDataStats.Begin(); i != ColoredPendingDataStats.End(); ++i) {
                const TRequesterPendingDataStats& p = *(i->second);
                if (p.InpCount || p.OutCount) {
                    snprintf(buf, sizeof(buf), "Pending data stats for color \"%d\":\n", (int)i->first);
                    res += buf;
                    res += GetPendingDataStatsDebug(p);
                }
            }

            res += "\nSocket info:\n";
            res += S.GetSockDebug();

            res += "\nHost info:\n";
            res += GetHostDebug();

            res += "\nConnections info:\n";
            res += GetConnectionsDebug();

            cb(res);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    TIntrusivePtr<IUdpHost> CreateUdpHost(int port, float udpTransferTimeout) {
        TIntrusivePtr<ISocket> socket = NNetlibaSocket::CreateBestRecvSocket();
        socket->Open(port);
        if (!socket->IsValid())
            return nullptr;
        return CreateUdpHost(socket, udpTransferTimeout);
    }

    TIntrusivePtr<IUdpHost> CreateUdpHost(const TIntrusivePtr<ISocket>& socket, float udpTransferTimeout) {
        TIntrusivePtr<TUdpHost> res = new TUdpHost(udpTransferTimeout);
        if (!res->Start(socket))
            return nullptr;
        return res.Get();
    }

    void SetUdpMaxBandwidthPerIP(float f) {
        f = Max(0.0f, f);
        TCongestionControl::MaxPacketRate = f / UDP_PACKET_SIZE;
    }

    void SetUdpSlowStart(bool enable) {
        TCongestionControl::StartWindowSize = enable ? 0.5f : 3;
    }

    void DisableIBDetection() {
        IBDetection = false;
    }

    void EnableXsPing() {
        XsPingSending = true;
    }

}
