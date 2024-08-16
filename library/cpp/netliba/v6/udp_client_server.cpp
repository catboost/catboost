#include "stdafx.h"
#include "udp_client_server.h"
#include "net_acks.h"
#include <util/generic/guid.h>
#include <util/system/hp_timer.h>
#include <util/datetime/cputimer.h>
#include <util/system/yield.h>
#include <util/system/unaligned_mem.h>
#include "block_chain.h"
#include <util/system/shmat.h>
#include "udp_debug.h"
#include "udp_socket.h"
#include "ib_cs.h"

#include <library/cpp/netliba/socket/socket.h>

#include <util/random/random.h>
#include <util/system/sanitizers.h>

#include <atomic>

namespace NNetliba {
    // rely on UDP checksum in packets, check crc only for complete packets
    // UPDATE: looks like UDP checksum is not enough, network errors do happen, we saw 600+ retransmits of a ~1MB data packet

    const float UDP_TRANSFER_TIMEOUT = 90.0f;
    const float DEFAULT_MAX_WAIT_TIME = 1;
    const float UDP_KEEP_PEER_INFO = 600;
    // траффик может идти, а новых данных для конкретного пакета может не добавляться.
    // это возможно когда мы прерываем процесс в момент передачи и перезапускаем его на том же порту,
    // тогда на приемнике повиснет пакет. Этот пакет мы зашибем по этому таймауту
    const float UDP_MAX_INPUT_DATA_WAIT = UDP_TRANSFER_TIMEOUT * 2;

    constexpr int UDP_PACKET_SIZE_FULL = 8900;  // used for ping to detect jumbo-frame support
    constexpr int UDP_PACKET_SIZE = 8800;       // max data in packet
    constexpr int UDP_PACKET_SIZE_SMALL = 1350; // 1180 would be better taking into account that 1280 is guaranteed ipv6 minimum MTU
    constexpr int UDP_PACKET_BUF_SIZE = UDP_PACKET_SIZE + 100;

    //////////////////////////////////////////////////////////////////////////
    struct TUdpCompleteInTransfer {
        TGUID PacketGuid;
    };

    //////////////////////////////////////////////////////////////////////////
    struct TUdpRecvPacket {
        int DataStart, DataSize;
        ui32 BlockSum;
        // Data[] should be last member in struct, this fact is used to create truncated TUdpRecvPacket in CreateNewSmallPacket()
        char Data[UDP_PACKET_BUF_SIZE];
    };

    struct TUdpInTransfer {
    private:
        TVector<TUdpRecvPacket*> Packets;

    public:
        sockaddr_in6 ToAddress;
        int PacketSize, LastPacketSize;
        bool HasLastPacket;
        TVector<int> NewPacketsToAck;
        TCongestionControlPtr Congestion;
        float TimeSinceLastRecv;
        int Attempt;
        TGUID PacketGuid;
        int Crc32;
        TIntrusivePtr<TSharedMemory> SharedData;
        TRequesterPendingDataStats* Stats;

        TUdpInTransfer()
            : PacketSize(0)
            , LastPacketSize(0)
            , HasLastPacket(false)
            , TimeSinceLastRecv(0)
            , Attempt(0)
            , Crc32(0)
            , Stats(nullptr)
        {
            Zero(ToAddress);
        }
        ~TUdpInTransfer() {
            if (Stats) {
                Stats->InpCount -= 1;
            }
            EraseAllPackets();
        }
        void EraseAllPackets() {
            for (int i = 0; i < Packets.ysize(); ++i) {
                ErasePacket(i);
            }
            Packets.clear();
            HasLastPacket = false;
        }
        void AttachStats(TRequesterPendingDataStats* stats) {
            Stats = stats;
            Stats->InpCount += 1;
            Y_ASSERT(Packets.empty());
        }
        void ErasePacket(int id) {
            TUdpRecvPacket* pkt = Packets[id];
            if (pkt) {
                if (Stats) {
                    Stats->InpDataSize -= PacketSize;
                }
                TRopeDataPacket::FreeBuf((char*)pkt);
                Packets[id] = nullptr;
            }
        }
        void AssignPacket(int id, TUdpRecvPacket* pkt) {
            ErasePacket(id);
            if (pkt && Stats) {
                Stats->InpDataSize += PacketSize;
            }
            Packets[id] = pkt;
        }
        int GetPacketCount() const {
            return Packets.ysize();
        }
        void SetPacketCount(int n) {
            Packets.resize(n, nullptr);
        }
        const TUdpRecvPacket* GetPacket(int id) const {
            return Packets[id];
        }
        TUdpRecvPacket* ExtractPacket(int id) {
            TUdpRecvPacket* res = Packets[id];
            if (res) {
                if (Stats) {
                    Stats->InpDataSize -= PacketSize;
                }
                Packets[id] = nullptr;
            }
            return res;
        }
    };

    struct TUdpOutTransfer {
        sockaddr_in6 ToAddress;
        TAutoPtr<TRopeDataPacket> Data;
        int PacketCount;
        int PacketSize, LastPacketSize;
        TAckTracker AckTracker;
        int Attempt;
        TGUID PacketGuid;
        int Crc32;
        EPacketPriority PacketPriority;
        TRequesterPendingDataStats* Stats;

        TUdpOutTransfer()
            : PacketCount(0)
            , PacketSize(0)
            , LastPacketSize(0)
            , Attempt(0)
            , Crc32(0)
            , PacketPriority(PP_LOW)
            , Stats(nullptr)
        {
            Zero(ToAddress);
        }
        ~TUdpOutTransfer() {
            if (Stats) {
                Stats->OutCount -= 1;
                Stats->OutDataSize -= Data->GetSize();
            }
        }
        void AttachStats(TRequesterPendingDataStats* stats) {
            Stats = stats;
            Stats->OutCount += 1;
            Stats->OutDataSize += Data->GetSize();
        }
    };

    struct TTransferKey {
        TUdpAddress Address;
        int Id;
    };
    inline bool operator==(const TTransferKey& a, const TTransferKey& b) {
        return a.Address == b.Address && a.Id == b.Id;
    }
    struct TTransferKeyHash {
        int operator()(const TTransferKey& k) const {
            return (ui32)k.Address.Interface + (ui32)k.Address.Port * (ui32)389461 + (ui32)k.Id;
        }
    };

    struct TUdpAddressHash {
        int operator()(const TUdpAddress& addr) const {
            return (ui32)addr.Interface + (ui32)addr.Port * (ui32)389461;
        }
    };

    class TUdpHostRevBufAlloc: public TNonCopyable {
        TUdpRecvPacket* RecvPktBuf;

        void AllocNewBuf() {
            RecvPktBuf = (TUdpRecvPacket*)TRopeDataPacket::AllocBuf(sizeof(TUdpRecvPacket));
        }

    public:
        TUdpHostRevBufAlloc() {
            AllocNewBuf();
        }
        ~TUdpHostRevBufAlloc() {
            FreeBuf(RecvPktBuf);
        }
        void FreeBuf(TUdpRecvPacket* pkt) {
            TRopeDataPacket::FreeBuf((char*)pkt);
        }
        TUdpRecvPacket* ExtractPacket() {
            TUdpRecvPacket* res = RecvPktBuf;
            AllocNewBuf();
            return res;
        }
        TUdpRecvPacket* CreateNewSmallPacket(int sz) {
            int pktStructSz = sizeof(TUdpRecvPacket) - Y_ARRAY_SIZE(RecvPktBuf->Data) + sz;
            TUdpRecvPacket* pkt = (TUdpRecvPacket*)TRopeDataPacket::AllocBuf(pktStructSz);
            return pkt;
        }
        int GetBufSize() const {
            return Y_ARRAY_SIZE(RecvPktBuf->Data);
        }
        char* GetDataPtr() const {
            return RecvPktBuf->Data;
        }
    };

    static TAtomic transferIdCounter = (long)(GetCycleCount() & 0x1fffffff);
    inline int GetTransferId() {
        int res = AtomicAdd(transferIdCounter, 1);
        while (res < 0) {
            // negative transfer ids are treated as errors, so wrap transfer id
            AtomicCas(&transferIdCounter, 0, transferIdCounter);
            res = AtomicAdd(transferIdCounter, 1);
        }
        return res;
    }

    static bool IBDetection = true;
    class TUdpHost: public IUdpHost {
        struct TPeerLink {
            TIntrusivePtr<TCongestionControl> UdpCongestion;
            TIntrusivePtr<IIBPeer> IBPeer;
            double TimeNoActiveTransfers;

            TPeerLink()
                : TimeNoActiveTransfers(0)
            {
            }
            bool Update(float deltaT, const TUdpAddress& toAddress, float* maxWaitTime) {
                bool updateOk = UdpCongestion->UpdateAlive(toAddress, deltaT, UDP_TRANSFER_TIMEOUT, maxWaitTime);
                return updateOk;
            }
            void StartSleep(const TUdpAddress& toAddress, float* maxWaitTime) {
                //printf("peer_link start sleep, IBPeer = %p, refs = %d\n", IBPeer.Get(), (int)IBPeer.RefCount());
                UdpCongestion->UpdateAlive(toAddress, 0, UDP_TRANSFER_TIMEOUT, maxWaitTime);
                UdpCongestion->MarkAlive();
                TimeNoActiveTransfers = 0;
            }
            bool UpdateSleep(float deltaT) {
                TimeNoActiveTransfers += deltaT;
                if (IBPeer.Get()) {
                    //printf("peer_link update sleep, IBPeer = %p, refs = %d\n", IBPeer.Get(), (int)IBPeer.RefCount());
                    if (IBPeer->GetState() == IIBPeer::OK) {
                        return true;
                    }
                    //printf("Drop broken IB connection\n");
                    IBPeer = nullptr;
                }
                return (TimeNoActiveTransfers < UDP_KEEP_PEER_INFO);
            }
        };

        TNetSocket s;
        typedef THashMap<TTransferKey, TUdpInTransfer, TTransferKeyHash> TUdpInXferHash;
        typedef THashMap<TTransferKey, TUdpOutTransfer, TTransferKeyHash> TUdpOutXferHash;
        // congestion control per peer
        typedef THashMap<TUdpAddress, TPeerLink, TUdpAddressHash> TPeerLinkHash;
        typedef THashMap<TTransferKey, TUdpCompleteInTransfer, TTransferKeyHash> TUdpCompleteInXferHash;
        typedef THashMap<TUdpAddress, TIntrusivePtr<TPeerQueueStats>, TUdpAddressHash> TQueueStatsHash;
        TUdpInXferHash RecvQueue;
        TUdpCompleteInXferHash RecvCompleted;
        TUdpOutXferHash SendQueue;
        TPeerLinkHash CongestionTrack, CongestionTrackHistory;
        TList<TRequest*> ReceivedList;
        NHPTimer::STime CurrentT;
        TList<TSendResult> SendResults;
        TList<TTransferKey> SendOrderLow, SendOrder, SendOrderHighPrior;
        TAtomic IsWaiting;
        float MaxWaitTime;
        std::atomic<float> MaxWaitTime2;
        float IBIdleTime;
        TVector<TTransferKey> RecvCompletedQueue, KeepCompletedQueue;
        float TimeSinceCompletedQueueClean, TimeSinceCongestionHistoryUpdate;
        TRequesterPendingDataStats PendingDataStats;
        TQueueStatsHash PeerQueueStats;
        TIntrusivePtr<IIBClientServer> IB;
        typedef THashMap<TIBMsgHandle, TTransferKey> TIBtoTransferKeyHash;
        TIBtoTransferKeyHash IBKeyToTransferKey;

        char PktBuf[UDP_PACKET_BUF_SIZE];
        TUdpHostRevBufAlloc RecvBuf;

        TPeerLink& GetPeerLink(const TUdpAddress& ip) {
            TPeerLinkHash::iterator z = CongestionTrack.find(ip);
            if (z == CongestionTrack.end()) {
                z = CongestionTrackHistory.find(ip);
                if (z == CongestionTrackHistory.end()) {
                    TPeerLink& res = CongestionTrack[ip];
                    Y_ASSERT(res.UdpCongestion.Get() == nullptr);
                    res.UdpCongestion = new TCongestionControl;
                    TQueueStatsHash::iterator zq = PeerQueueStats.find(ip);
                    if (zq != PeerQueueStats.end()) {
                        res.UdpCongestion->AttachQueueStats(zq->second);
                    }
                    return res;
                } else {
                    TPeerLink& res = CongestionTrack[z->first];
                    res = z->second;
                    CongestionTrackHistory.erase(z);
                    return res;
                }
            } else {
                Y_ASSERT(CongestionTrackHistory.find(ip) == CongestionTrackHistory.end());
                return z->second;
            }
        }
        void SucceededSend(int id) {
            SendResults.push_back(TSendResult(id, true));
        }
        void FailedSend(int id) {
            SendResults.push_back(TSendResult(id, false));
        }
        void SendData(TList<TTransferKey>* order, float deltaT, bool needCheckAlive);
        void RecvCycle();

    public:
        TUdpHost()
            : CurrentT(0)
            , IsWaiting(0)
            , MaxWaitTime(DEFAULT_MAX_WAIT_TIME)
            , MaxWaitTime2(DEFAULT_MAX_WAIT_TIME)
            , IBIdleTime(0)
            , TimeSinceCompletedQueueClean(0)
            , TimeSinceCongestionHistoryUpdate(0)
        {
        }
        ~TUdpHost() override {
            for (TList<TRequest*>::const_iterator i = ReceivedList.begin(); i != ReceivedList.end(); ++i)
                delete *i;
        }

        bool Start(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket);

        TRequest* GetRequest() override {
            if (ReceivedList.empty()) {
                if (IB.Get()) {
                    return IB->GetRequest();
                }
                return nullptr;
            }
            TRequest* res = ReceivedList.front();
            ReceivedList.pop_front();
            return res;
        }

        void AddToSendOrder(const TTransferKey& transferKey, EPacketPriority pp) {
            if (pp == PP_LOW)
                SendOrderLow.push_back(transferKey);
            else if (pp == PP_NORMAL)
                SendOrder.push_back(transferKey);
            else if (pp == PP_HIGH)
                SendOrderHighPrior.push_back(transferKey);
            else
                Y_ASSERT(0);

            CancelWait();
        }

        int Send(const TUdpAddress& addr, TAutoPtr<TRopeDataPacket> data, int crc32, TGUID* packetGuid, EPacketPriority pp) override {
            if (addr.Port == 0) {
                // shortcut for broken addresses
                if (packetGuid && packetGuid->IsEmpty())
                    CreateGuid(packetGuid);
                int reqId = GetTransferId();
                FailedSend(reqId);
                return reqId;
            }
            TTransferKey transferKey;
            transferKey.Address = addr;
            transferKey.Id = GetTransferId();
            Y_ASSERT(SendQueue.find(transferKey) == SendQueue.end());

            TPeerLink& peerInfo = GetPeerLink(transferKey.Address);

            TUdpOutTransfer& xfer = SendQueue[transferKey];
            GetWinsockAddr(&xfer.ToAddress, transferKey.Address);
            xfer.Crc32 = crc32;
            xfer.PacketPriority = pp;
            if (!packetGuid || packetGuid->IsEmpty()) {
                CreateGuid(&xfer.PacketGuid);
                if (packetGuid)
                    *packetGuid = xfer.PacketGuid;
            } else {
                xfer.PacketGuid = *packetGuid;
            }
            xfer.Data.Reset(data.Release());
            xfer.AttachStats(&PendingDataStats);
            xfer.AckTracker.AttachCongestionControl(peerInfo.UdpCongestion.Get());

            bool isSentOverIB = false;
            // we don't support priorities (=service levels in IB terms) currently
            // so send only PP_NORMAL traffic over IB
            if (pp == PP_NORMAL && peerInfo.IBPeer.Get() && xfer.Data->GetSharedData() == nullptr) {
                TIBMsgHandle hndl = IB->Send(peerInfo.IBPeer, xfer.Data.Get(), xfer.PacketGuid);
                if (hndl >= 0) {
                    IBKeyToTransferKey[hndl] = transferKey;
                    isSentOverIB = true;
                } else {
                    // so we failed to use IB, ibPeer is either not connected yet or failed
                    if (peerInfo.IBPeer->GetState() == IIBPeer::FAILED) {
                        //printf("Disconnect failed IB peer\n");
                        peerInfo.IBPeer = nullptr;
                    }
                }
            }
            if (!isSentOverIB) {
                AddToSendOrder(transferKey, pp);
            }

            return transferKey.Id;
        }

        bool GetSendResult(TSendResult* res) override {
            if (SendResults.empty()) {
                if (IB.Get()) {
                    TIBSendResult sr;
                    if (IB->GetSendResult(&sr)) {
                        TIBtoTransferKeyHash::iterator z = IBKeyToTransferKey.find(sr.Handle);
                        if (z == IBKeyToTransferKey.end()) {
                            Y_ABORT_UNLESS(0, "unknown handle returned from IB");
                        }
                        TTransferKey transferKey = z->second;
                        IBKeyToTransferKey.erase(z);

                        TUdpOutXferHash::iterator i = SendQueue.find(transferKey);
                        if (i == SendQueue.end()) {
                            Y_ABORT_UNLESS(0, "IBKeyToTransferKey refers nonexisting xfer");
                        }
                        if (sr.Success) {
                            TUdpOutTransfer& xfer = i->second;
                            xfer.AckTracker.MarkAlive(); // do we really need this?
                            *res = TSendResult(transferKey.Id, sr.Success);
                            SendQueue.erase(i);
                            return true;
                        } else {
                            //printf("IB send failed, fall back to regular network\n");
                            // Houston, we got a problem
                            // IB failed to send, try to use regular network
                            TUdpOutTransfer& xfer = i->second;
                            AddToSendOrder(transferKey, xfer.PacketPriority);
                        }
                    }
                }
                return false;
            }
            *res = SendResults.front();
            SendResults.pop_front();
            return true;
        }

        void Step() override;
        void IBStep() override;

        void Wait(float seconds) override {
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
                if (seconds > MaxWaitTime2)
                    seconds = MaxWaitTime2;
                MaxWaitTime2 = DEFAULT_MAX_WAIT_TIME;

                if (seconds == 0) {
                    ThreadYield();
                } else {
                    if (IB.Get()) {
                        for (float done = 0; done < seconds;) {
                            float deltaSleep = Min(seconds - done, 0.002f);
                            s.Wait(deltaSleep);
                            NHPTimer::STime tChk;
                            NHPTimer::GetTime(&tChk);
                            if (IB->Step(tChk)) {
                                IBIdleTime = 0;
                                break;
                            }
                            done += deltaSleep;
                        }
                    } else {
                        s.Wait(seconds);
                    }
                }
                AtomicAdd(IsWaiting, -1);
            }
        }

        void CancelWait() override {
            MaxWaitTime2 = 0;
            if (AtomicAdd(IsWaiting, 0) == 1) {
                s.SendSelfFakePacket();
            }
        }

        void GetPendingDataSize(TRequesterPendingDataStats* res) override {
            *res = PendingDataStats;
#ifndef NDEBUG
            TRequesterPendingDataStats chk;
            for (TUdpOutXferHash::const_iterator i = SendQueue.begin(); i != SendQueue.end(); ++i) {
                TRopeDataPacket* pckt = i->second.Data.Get();
                if (pckt) {
                    chk.OutDataSize += pckt->GetSize();
                    ++chk.OutCount;
                }
            }
            for (TUdpInXferHash::const_iterator i = RecvQueue.begin(); i != RecvQueue.end(); ++i) {
                const TUdpInTransfer& tr = i->second;
                for (int p = 0; p < tr.GetPacketCount(); ++p) {
                    if (tr.GetPacket(p)) {
                        chk.InpDataSize += tr.PacketSize;
                    }
                }
                ++chk.InpCount;
            }
            Y_ASSERT(memcmp(&chk, res, sizeof(chk)) == 0);
#endif
        }
        TString GetDebugInfo() override;
        TString GetPeerLinkDebug(const TPeerLinkHash& ch);
        void Kill(const TUdpAddress& addr) override;
        TIntrusivePtr<IPeerQueueStats> GetQueueStats(const TUdpAddress& addr) override;
    };

    bool TUdpHost::Start(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
        if (s.IsValid()) {
            Y_ASSERT(0);
            return false;
        }
        s.Open(socket);
        if (!s.IsValid())
            return false;

        if (IBDetection)
            IB = CreateIBClientServer();

        NHPTimer::GetTime(&CurrentT);
        return true;
    }

    static bool HasAllPackets(const TUdpInTransfer& res) {
        if (!res.HasLastPacket)
            return false;
        for (int i = res.GetPacketCount() - 1; i >= 0; --i) {
            if (!res.GetPacket(i))
                return false;
        }
        return true;
    }

    // grouped acks, first int - packet_id, second int - bit mask for 32 packets preceding packet_id
    const int SIZEOF_ACK = 8;
    static int WriteAck(TUdpInTransfer* p, int* dst, int maxAcks) {
        int ackCount = 0;
        if (p->NewPacketsToAck.size() > 1)
            Sort(p->NewPacketsToAck.begin(), p->NewPacketsToAck.end());
        int lastAcked = 0;
        for (size_t idx = 0; idx < p->NewPacketsToAck.size(); ++idx) {
            int pkt = p->NewPacketsToAck[idx];
            if (idx == p->NewPacketsToAck.size() - 1 || pkt > lastAcked + 30) {
                *dst++ = pkt;
                int bitMask = 0;
                int backPackets = Min(pkt, 32);
                for (int k = 0; k < backPackets; ++k) {
                    if (p->GetPacket(pkt - k - 1))
                        bitMask |= 1 << k;
                }
                *dst++ = bitMask;
                if (++ackCount >= maxAcks)
                    break;
                lastAcked = pkt;
                //printf("sending ack %d (mask %x)\n", pkt, bitMask);
            }
        }
        p->NewPacketsToAck.clear();
        return ackCount;
    }

    static void AckPacket(TUdpOutTransfer* p, int pkt, float deltaT, bool updateRTT) {
        if (pkt < 0 || pkt >= p->PacketCount) {
            Y_ASSERT(0);
            return;
        }
        p->AckTracker.Ack(pkt, deltaT, updateRTT);
    }

    static void ReadAcks(TUdpOutTransfer* p, const int* acks, int ackCount, float deltaT) {
        for (int i = 0; i < ackCount; ++i) {
            int pkt = *acks++;
            int bitMask = *acks++;
            bool updateRTT = i == ackCount - 1; // update RTT using only last packet in the pack
            AckPacket(p, pkt, deltaT, updateRTT);
            for (int k = 0; k < 32; ++k) {
                if (bitMask & (1 << k))
                    AckPacket(p, pkt - k - 1, deltaT, false);
            }
        }
    }

    using namespace NNetlibaSocket::NNetliba;

    const ui64 KILL_PASSPHRASE1 = 0x98ff9cefb11d9a4cul;
    const ui64 KILL_PASSPHRASE2 = 0xf7754c29e0be95eaul;

    template <class T>
    inline T Read(char** data) {
        T res = ReadUnaligned<T>(*data);
        *data += sizeof(T);
        return res;
    }
    template <class T>
    inline void Write(char** data, T res) {
        WriteUnaligned<T>(*data, res);
        *data += sizeof(T);
    }

    static void RequireResend(const TNetSocket& s, const sockaddr_in6& toAddress, int transferId, int attempt) {
        char buf[100], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        Write(&pktData, transferId);
        Write(&pktData, (char)ACK_RESEND);
        Write(&pktData, attempt);
        s.SendTo(buf, (int)(pktData - buf), toAddress, FF_ALLOW_FRAG);
    }

    static void RequireResendNoShmem(const TNetSocket& s, const sockaddr_in6& toAddress, int transferId, int attempt) {
        char buf[100], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        Write(&pktData, transferId);
        Write(&pktData, (char)ACK_RESEND_NOSHMEM);
        Write(&pktData, attempt);
        s.SendTo(buf, (int)(pktData - buf), toAddress, FF_ALLOW_FRAG);
    }

    static void AckComplete(const TNetSocket& s, const sockaddr_in6& toAddress, int transferId, const TGUID& packetGuid, int packetId) {
        char buf[100], *pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        Write(&pktData, transferId);
        Write(&pktData, (char)ACK_COMPLETE);
        Write(&pktData, packetGuid);
        Write(&pktData, packetId); // we need packetId to update RTT
        s.SendTo(buf, (int)(pktData - buf), toAddress, FF_ALLOW_FRAG);
    }

    static void SendPing(TNetSocket& s, const sockaddr_in6& toAddress, int selfNetworkOrderPort) {
        char pktBuf[UDP_PACKET_SIZE_FULL];
        char* pktData = pktBuf + UDP_LOW_LEVEL_HEADER_SIZE;
        if (NSan::MSanIsOn()) {
            Zero(pktBuf);
        }
        Write(&pktData, (int)0);
        Write(&pktData, (char)PING);
        Write(&pktData, selfNetworkOrderPort);
        s.SendTo(pktBuf, UDP_PACKET_SIZE_FULL, toAddress, FF_DONT_FRAG);
    }

    // not MTU discovery, just figure out IB address of the peer
    static void SendFakePing(TNetSocket& s, const sockaddr_in6& toAddress, int selfNetworkOrderPort) {
        char buf[100];
        char* pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        Write(&pktData, (int)0);
        Write(&pktData, (char)PING);
        Write(&pktData, selfNetworkOrderPort);
        s.SendTo(buf, (int)(pktData - buf), toAddress, FF_ALLOW_FRAG);
    }

    void TUdpHost::SendData(TList<TTransferKey>* order, float deltaT1, bool needCheckAlive) {
        for (TList<TTransferKey>::iterator z = order->begin(); z != order->end();) {
            // pick connection to send
            const TTransferKey& transferKey = *z;
            TUdpOutXferHash::iterator i = SendQueue.find(transferKey);
            if (i == SendQueue.end()) {
                z = order->erase(z);
                continue;
            }
            ++z;

            // perform sending
            int transferId = transferKey.Id;
            TUdpOutTransfer& xfer = i->second;

            if (!xfer.AckTracker.IsInitialized()) {
                TIntrusivePtr<TCongestionControl> congestion = xfer.AckTracker.GetCongestionControl();
                Y_ASSERT(congestion.Get() != nullptr);
                if (!congestion->IsKnownMTU()) {
                    TLameMTUDiscovery* md = congestion->GetMTUDiscovery();
                    if (md->IsTimedOut()) {
                        congestion->SetMTU(UDP_PACKET_SIZE_SMALL);
                    } else {
                        if (md->CanSend()) {
                            SendPing(s, xfer.ToAddress, s.GetNetworkOrderPort());
                            md->PingSent();
                        }
                        continue;
                    }
                }
                // try to use large mtu, we could have selected small mtu due to connectivity problems
                if (congestion->GetMTU() == UDP_PACKET_SIZE_SMALL || IB.Get() != nullptr) {
                    // recheck every ~50mb
                    int chkDenom = (50000000 / xfer.Data->GetSize()) | 1;
                    if ((NetAckRnd() % chkDenom) == 0) {
                        //printf("send rechecking ping\n");
                        if (congestion->GetMTU() == UDP_PACKET_SIZE_SMALL) {
                            SendPing(s, xfer.ToAddress, s.GetNetworkOrderPort());
                        } else {
                            SendFakePing(s, xfer.ToAddress, s.GetNetworkOrderPort());
                        }
                    }
                }
                xfer.PacketSize = congestion->GetMTU();
                xfer.LastPacketSize = xfer.Data->GetSize() % xfer.PacketSize;
                xfer.PacketCount = xfer.Data->GetSize() / xfer.PacketSize + 1;
                xfer.AckTracker.SetPacketCount(xfer.PacketCount);
            }

            xfer.AckTracker.Step(deltaT1);
            MaxWaitTime = Min(MaxWaitTime, xfer.AckTracker.GetTimeToNextPacketTimeout());
            if (needCheckAlive && !xfer.AckTracker.IsAlive()) {
                FailedSend(transferId);
                SendQueue.erase(i);
                continue;
            }
            bool sendBufferOverflow = false;
            while (xfer.AckTracker.CanSend()) {
                NHPTimer::STime tCopy = CurrentT;
                float deltaT2 = (float)NHPTimer::GetTimePassed(&tCopy);
                deltaT2 = ClampVal(deltaT2, 0.0f, UDP_TRANSFER_TIMEOUT / 3);

                int pkt = xfer.AckTracker.GetPacketToSend(deltaT2);
                if (pkt == -1) {
                    break;
                }

                int dataSize = xfer.PacketSize;
                if (pkt == xfer.PacketCount - 1)
                    dataSize = xfer.LastPacketSize;

                char* pktData = PktBuf + UDP_LOW_LEVEL_HEADER_SIZE;
                Write(&pktData, transferId);
                char pktType = xfer.PacketSize == UDP_PACKET_SIZE ? DATA : DATA_SMALL;
                TSharedMemory* shm = xfer.Data->GetSharedData();
                if (shm) {
                    if (pktType == DATA)
                        pktType = DATA_SHMEM;
                    else
                        pktType = DATA_SMALL_SHMEM;
                }
                Write(&pktData, pktType);
                Write(&pktData, xfer.Attempt);
                Write(&pktData, pkt);
                if (pkt == 0) {
                    Write(&pktData, xfer.PacketGuid);
                    Write(&pktData, xfer.Crc32);
                    if (shm) {
                        Write(&pktData, shm->GetId());
                        Write(&pktData, shm->GetSize());
                    }
                }
                TBlockChainIterator dataReader(xfer.Data->GetChain());
                dataReader.Seek(pkt * xfer.PacketSize);
                dataReader.Read(pktData, dataSize);
                pktData += dataSize;
                int sendSize = (int)(pktData - PktBuf);
                TNetSocket::ESendError sendErr = s.SendTo(PktBuf, sendSize, xfer.ToAddress, FF_ALLOW_FRAG);
                if (sendErr != TNetSocket::SEND_OK) {
                    if (sendErr == TNetSocket::SEND_NO_ROUTE_TO_HOST) {
                        FailedSend(transferId);
                        SendQueue.erase(i);
                        break;
                    } else {
                        // most probably out of send buffer space (or something terrible has happened)
                        xfer.AckTracker.AddToResend(pkt);
                        sendBufferOverflow = true;
                        MaxWaitTime = 0;
                        //printf("failed send\n");
                        break;
                    }
                }
            }
            if (sendBufferOverflow)
                break;
        }
    }

    void TUdpHost::RecvCycle() {
        for (;;) {
            sockaddr_in6 fromAddress;
            int rv = RecvBuf.GetBufSize();
            bool recvOk = s.RecvFrom(RecvBuf.GetDataPtr(), &rv, &fromAddress);
            if (!recvOk)
                break;

            NHPTimer::STime tCopy = CurrentT;
            float deltaT = (float)NHPTimer::GetTimePassed(&tCopy);
            deltaT = ClampVal(deltaT, 0.0f, UDP_TRANSFER_TIMEOUT / 3);

            //int fromIP = fromAddress.sin_addr.s_addr;

            TTransferKey k;
            char* pktData = RecvBuf.GetDataPtr() + UDP_LOW_LEVEL_HEADER_SIZE;
            GetUdpAddress(&k.Address, fromAddress);
            k.Id = Read<int>(&pktData);
            int transferId = k.Id;
            int cmd = Read<char>(&pktData);
            Y_ASSERT(cmd == (int)*(RecvBuf.GetDataPtr() + CMD_POS));
            switch (cmd) {
                case DATA:
                case DATA_SMALL:
                case DATA_SHMEM:
                case DATA_SMALL_SHMEM: {
                    int attempt = Read<int>(&pktData);
                    int packetId = Read<int>(&pktData);
                    //printf("data packet %d (trans ID = %d)\n", packetId, transferId);
                    TUdpCompleteInXferHash::iterator itCompl = RecvCompleted.find(k);
                    if (itCompl != RecvCompleted.end()) {
                        Y_ASSERT(RecvQueue.find(k) == RecvQueue.end());
                        const TUdpCompleteInTransfer& complete = itCompl->second;
                        bool sendAckComplete = true;
                        if (packetId == 0) {
                            // check packet GUID
                            char* tmpPktData = pktData;
                            TGUID packetGuid;
                            packetGuid = Read<TGUID>(&tmpPktData);
                            if (packetGuid != complete.PacketGuid) {
                                // we are receiving new data with the same transferId
                                // in this case we have to flush all the information about previous transfer
                                // and start over
                                //printf("same transferId for a different packet\n");
                                RecvCompleted.erase(itCompl);
                                sendAckComplete = false;
                            }
                        }
                        if (sendAckComplete) {
                            AckComplete(s, fromAddress, transferId, complete.PacketGuid, packetId);
                            break;
                        }
                    }
                    TUdpInXferHash::iterator rq = RecvQueue.find(k);
                    if (rq == RecvQueue.end()) {
                        //printf("new input transfer\n");
                        TUdpInTransfer& res = RecvQueue[k];
                        res.ToAddress = fromAddress;
                        res.Attempt = attempt;
                        res.Congestion = GetPeerLink(k.Address).UdpCongestion.Get();
                        res.PacketSize = 0;
                        res.HasLastPacket = false;
                        res.AttachStats(&PendingDataStats);
                        rq = RecvQueue.find(k);
                        Y_ASSERT(rq != RecvQueue.end());
                    }
                    TUdpInTransfer& res = rq->second;
                    res.Congestion->MarkAlive();
                    res.TimeSinceLastRecv = 0;

                    if (packetId == 0) {
                        TGUID packetGuid;
                        packetGuid = Read<TGUID>(&pktData);
                        int crc32 = Read<int>(&pktData);
                        res.Crc32 = crc32;
                        res.PacketGuid = packetGuid;
                        if (cmd == DATA_SHMEM || cmd == DATA_SMALL_SHMEM) {
                            // link to attached shared memory
                            TGUID shmemId = Read<TGUID>(&pktData);
                            int shmemSize = Read<int>(&pktData);
                            if (res.SharedData.Get() == nullptr) {
                                res.SharedData = new TSharedMemory;
                                if (!res.SharedData->Open(shmemId, shmemSize)) {
                                    res.SharedData = nullptr;
                                    RequireResendNoShmem(s, res.ToAddress, transferId, res.Attempt);
                                    break;
                                }
                            }
                        }
                    }
                    if (attempt != res.Attempt) {
                        RequireResend(s, res.ToAddress, transferId, res.Attempt);
                        break;
                    } else {
                        if (res.PacketSize == 0) {
                            res.PacketSize = (cmd == DATA || cmd == DATA_SHMEM ? UDP_PACKET_SIZE : UDP_PACKET_SIZE_SMALL);
                        } else {
                            // check that all data is of same size
                            Y_ASSERT(cmd == DATA || cmd == DATA_SMALL);
                            Y_ASSERT(res.PacketSize == (cmd == DATA ? UDP_PACKET_SIZE : UDP_PACKET_SIZE_SMALL));
                        }

                        int dataSize = (int)(RecvBuf.GetDataPtr() + rv - pktData);

                        Y_ASSERT(dataSize <= res.PacketSize);
                        if (dataSize > res.PacketSize)
                            break; // mem overrun protection
                        if (packetId >= res.GetPacketCount())
                            res.SetPacketCount(packetId + 1);
                        {
                            TUdpRecvPacket* pkt = nullptr;
                            if (res.PacketSize == UDP_PACKET_SIZE_SMALL) {
                                // save memory by using smaller buffer at the cost of additional memcpy
                                pkt = RecvBuf.CreateNewSmallPacket(dataSize);
                                memcpy(pkt->Data, pktData, dataSize);
                                pkt->DataStart = 0;
                                pkt->DataSize = dataSize;
                            } else {
                                int dataStart = (int)(pktData - RecvBuf.GetDataPtr()); // data offset in the packet
                                pkt = RecvBuf.ExtractPacket();
                                pkt->DataStart = dataStart;
                                pkt->DataSize = dataSize;
                            }
                            // calc packet sum, will be used to calc whole message crc
                            pkt->BlockSum = TIncrementalChecksumCalcer::CalcBlockSum(pkt->Data + pkt->DataStart, pkt->DataSize);
                            res.AssignPacket(packetId, pkt);
                        }

                        if (dataSize != res.PacketSize) {
                            res.LastPacketSize = dataSize;
                            res.HasLastPacket = true;
                        }

                        if (HasAllPackets(res)) {
                            //printf("received\n");
                            TRequest* out = new TRequest;
                            out->Address = k.Address;
                            out->Guid = res.PacketGuid;
                            TIncrementalChecksumCalcer incCS;
                            int packetCount = res.GetPacketCount();
                            out->Data.Reset(new TRopeDataPacket);
                            for (int i = 0; i < packetCount; ++i) {
                                TUdpRecvPacket* pkt = res.ExtractPacket(i);
                                Y_ASSERT(pkt->DataSize == ((i == packetCount - 1) ? res.LastPacketSize : res.PacketSize));
                                out->Data->AddBlock((char*)pkt, pkt->Data + pkt->DataStart, pkt->DataSize);
                                incCS.AddBlockSum(pkt->BlockSum, pkt->DataSize);
                            }
                            out->Data->AttachSharedData(res.SharedData);
                            res.EraseAllPackets();

                            int crc32 = incCS.CalcChecksum(); // CalcChecksum(out->Data->GetChain());
#ifdef SIMULATE_NETWORK_FAILURES
                            bool crcOk = crc32 == res.Crc32 ? (RandomNumber<size_t>() % 10) != 0 : false;
#else
                            bool crcOk = crc32 == res.Crc32;
#endif
                            if (crcOk) {
                                ReceivedList.push_back(out);
                                Y_ASSERT(RecvCompleted.find(k) == RecvCompleted.end());
                                TUdpCompleteInTransfer& complete = RecvCompleted[k];
                                RecvCompletedQueue.push_back(k);
                                complete.PacketGuid = res.PacketGuid;
                                AckComplete(s, res.ToAddress, transferId, complete.PacketGuid, packetId);
                                RecvQueue.erase(rq);
                            } else {
                                //printf("crc failed, require resend\n");
                                delete out;
                                ++res.Attempt;
                                res.NewPacketsToAck.clear();
                                RequireResend(s, res.ToAddress, transferId, res.Attempt);
                            }
                        } else {
                            res.NewPacketsToAck.push_back(packetId);
                        }
                    }
                } break;
                case ACK: {
                    TUdpOutXferHash::iterator i = SendQueue.find(k);
                    if (i == SendQueue.end())
                        break;
                    TUdpOutTransfer& xfer = i->second;
                    if (!xfer.AckTracker.IsInitialized())
                        break;
                    xfer.AckTracker.MarkAlive();
                    int attempt = Read<int>(&pktData);
                    Y_ASSERT(attempt <= xfer.Attempt);
                    if (attempt != xfer.Attempt)
                        break;
                    ReadAcks(&xfer, (int*)pktData, (int)(RecvBuf.GetDataPtr() + rv - pktData) / SIZEOF_ACK, deltaT);
                    break;
                }
                case ACK_COMPLETE: {
                    TUdpOutXferHash::iterator i = SendQueue.find(k);
                    if (i == SendQueue.end())
                        break;
                    TUdpOutTransfer& xfer = i->second;
                    xfer.AckTracker.MarkAlive();
                    TGUID packetGuid;
                    packetGuid = Read<TGUID>(&pktData);
                    int packetId = Read<int>(&pktData);
                    if (packetGuid == xfer.PacketGuid) {
                        xfer.AckTracker.Ack(packetId, deltaT, true); // update RTT
                        xfer.AckTracker.AckAll();                    // acking packets is required, otherwise they will be treated as lost (look AckTracker destructor)
                        SucceededSend(transferId);
                        SendQueue.erase(i);
                    } else {
                        // peer asserts that he has received this packet but packetGuid is wrong
                        // try to resend everything
                        // ++xfer.Attempt; // should not do this, only sender can modify attempt number, otherwise cycle is possible with out of order packets
                        xfer.AckTracker.Resend();
                    }
                    break;
                } break;
                case ACK_RESEND: {
                    TUdpOutXferHash::iterator i = SendQueue.find(k);
                    if (i == SendQueue.end())
                        break;
                    TUdpOutTransfer& xfer = i->second;
                    xfer.AckTracker.MarkAlive();
                    int attempt = Read<int>(&pktData);
                    if (xfer.Attempt != attempt) {
                        // reset current tranfser & initialize new one
                        xfer.Attempt = attempt;
                        xfer.AckTracker.Resend();
                    }
                    break;
                }
                case ACK_RESEND_NOSHMEM: {
                    // abort execution here
                    // failed to open shmem on recv side, need to transmit data without using shmem
                    Y_ABORT_UNLESS(0, "not implemented yet");
                    break;
                }
                case PING: {
                    sockaddr_in6 trueFromAddress = fromAddress;
                    int port = Read<int>(&pktData);
                    Y_ASSERT(trueFromAddress.sin6_family == AF_INET6);
                    trueFromAddress.sin6_port = port;
                    // can not set MTU for fromAddress here since asymmetrical mtu is possible
                    char* pktData2 = PktBuf + UDP_LOW_LEVEL_HEADER_SIZE;
                    Write(&pktData2, (int)0);
                    Write(&pktData2, (char)PONG);
                    if (IB.Get()) {
                        const TIBConnectInfo& ibConnectInfo = IB->GetConnectInfo();
                        Write(&pktData2, ibConnectInfo);
                        Write(&pktData2, trueFromAddress);
                    }
                    s.SendTo(PktBuf, pktData2 - PktBuf, trueFromAddress, FF_ALLOW_FRAG);
                    break;
                }
                case PONG: {
                    TPeerLink& peerInfo = GetPeerLink(k.Address);
                    peerInfo.UdpCongestion->SetMTU(UDP_PACKET_SIZE);
                    int dataSize = (int)(RecvBuf.GetDataPtr() + rv - pktData);
                    if (dataSize == sizeof(TIBConnectInfo) + sizeof(sockaddr_in6)) {
                        if (IB.Get() != nullptr && peerInfo.IBPeer.Get() == nullptr) {
                            TIBConnectInfo info = Read<TIBConnectInfo>(&pktData);
                            sockaddr_in6 myAddress = Read<sockaddr_in6>(&pktData);
                            TUdpAddress myUdpAddress;
                            GetUdpAddress(&myUdpAddress, myAddress);
                            peerInfo.IBPeer = IB->ConnectPeer(info, k.Address, myUdpAddress);
                        }
                    }
                    break;
                }
                case KILL: {
                    ui64 p1 = Read<ui64>(&pktData);
                    ui64 p2 = Read<ui64>(&pktData);
                    int restSize = (int)(RecvBuf.GetDataPtr() + rv - pktData);
                    if (restSize == 0 && p1 == KILL_PASSPHRASE1 && p2 == KILL_PASSPHRASE2) {
                        abort();
                    }
                    break;
                }
                default:
                    Y_ASSERT(0);
                    break;
            }
        }
    }

    void TUdpHost::IBStep() {
        if (IB.Get()) {
            NHPTimer::STime tChk = CurrentT;
            float chkDeltaT = (float)NHPTimer::GetTimePassed(&tChk);
            if (IB->Step(tChk)) {
                IBIdleTime = -chkDeltaT;
            }
        }
    }

    void TUdpHost::Step() {
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
            s.SetTOS(0x20);
        } else {
            s.SetTOS(0);
        }

        RecvCycle();

        float deltaT = (float)NHPTimer::GetTimePassed(&CurrentT);
        deltaT = ClampVal(deltaT, 0.0f, UDP_TRANSFER_TIMEOUT / 3);

        MaxWaitTime = DEFAULT_MAX_WAIT_TIME;
        IBIdleTime += deltaT;

        bool needCheckAlive = false;

        // update alive ports
        const float INACTIVE_CONGESTION_UPDATE_INTERVAL = 1;
        TimeSinceCongestionHistoryUpdate += deltaT;
        if (TimeSinceCongestionHistoryUpdate > INACTIVE_CONGESTION_UPDATE_INTERVAL) {
            for (TPeerLinkHash::iterator i = CongestionTrackHistory.begin(); i != CongestionTrackHistory.end();) {
                TPeerLink& pl = i->second;
                if (!pl.UpdateSleep(TimeSinceCongestionHistoryUpdate)) {
                    TPeerLinkHash::iterator k = i++;
                    CongestionTrackHistory.erase(k);
                    needCheckAlive = true;
                } else {
                    ++i;
                }
            }
            TimeSinceCongestionHistoryUpdate = 0;
        }
        for (TPeerLinkHash::iterator i = CongestionTrack.begin(); i != CongestionTrack.end();) {
            const TUdpAddress& addr = i->first;
            TPeerLink& pl = i->second;
            if (pl.UdpCongestion->GetTransferCount() == 0) {
                pl.StartSleep(addr, &MaxWaitTime);
                CongestionTrackHistory[i->first] = i->second;
                TPeerLinkHash::iterator k = i++;
                CongestionTrack.erase(k);
            } else if (!pl.Update(deltaT, addr, &MaxWaitTime)) {
                TPeerLinkHash::iterator k = i++;
                CongestionTrack.erase(k);
                needCheckAlive = true;
            } else {
                ++i;
            }
        }

        // send acks on received data
        for (TUdpInXferHash::iterator i = RecvQueue.begin(); i != RecvQueue.end();) {
            const TTransferKey& transKey = i->first;
            int transferId = transKey.Id;
            TUdpInTransfer& xfer = i->second;
            xfer.TimeSinceLastRecv += deltaT;
            if (xfer.TimeSinceLastRecv > UDP_MAX_INPUT_DATA_WAIT || (needCheckAlive && !xfer.Congestion->IsAlive())) {
                TUdpInXferHash::iterator k = i++;
                RecvQueue.erase(k);
                continue;
            }
            Y_ASSERT(RecvCompleted.find(i->first) == RecvCompleted.end()); // state "Complete & incomplete" is incorrect
            if (!xfer.NewPacketsToAck.empty()) {
                char* pktData = PktBuf + UDP_LOW_LEVEL_HEADER_SIZE;
                Write(&pktData, transferId);
                Write(&pktData, (char)ACK);
                Write(&pktData, xfer.Attempt);
                int acks = WriteAck(&xfer, (int*)pktData, (int)(xfer.PacketSize - (pktData - PktBuf)) / SIZEOF_ACK);
                pktData += acks * SIZEOF_ACK;
                s.SendTo(PktBuf, (int)(pktData - PktBuf), xfer.ToAddress, FF_ALLOW_FRAG);
            }
            ++i;
        }

        if (UseTOSforAcks) {
            s.SetTOS(0x60);
        }

        // send data for outbound connections
        SendData(&SendOrderHighPrior, deltaT, needCheckAlive);
        SendData(&SendOrder, deltaT, needCheckAlive);
        SendData(&SendOrderLow, deltaT, needCheckAlive);

        // roll send order to avoid exotic problems with lots of peers and high traffic
        SendOrderHighPrior.splice(SendOrderHighPrior.end(), SendOrderHighPrior, SendOrderHighPrior.begin());
        //SendOrder.splice(SendOrder.end(), SendOrder, SendOrder.begin()); // sending data in order has lower delay and shorter queue

        // clean completed queue
        TimeSinceCompletedQueueClean += deltaT;
        if (TimeSinceCompletedQueueClean > UDP_TRANSFER_TIMEOUT * 1.5) {
            for (size_t i = 0; i < KeepCompletedQueue.size(); ++i) {
                TUdpCompleteInXferHash::iterator k = RecvCompleted.find(KeepCompletedQueue[i]);
                if (k != RecvCompleted.end())
                    RecvCompleted.erase(k);
            }
            KeepCompletedQueue.clear();
            KeepCompletedQueue.swap(RecvCompletedQueue);
            TimeSinceCompletedQueueClean = 0;
        }
    }

    TString TUdpHost::GetPeerLinkDebug(const TPeerLinkHash& ch) {
        TString res;
        char buf[1000];
        for (const auto& i : ch) {
            const TUdpAddress& ip = i.first;
            const TCongestionControl& cc = *i.second.UdpCongestion;
            IIBPeer* ibPeer = i.second.IBPeer.Get();
            snprintf(buf, sizeof(buf), "%s\tIB: %d, RTT: %g  Timeout: %g  Window: %g  MaxWin: %g  FailRate: %g  TimeSinceLastRecv: %g  Transfers: %d  MTU: %d\n",
                    GetAddressAsString(ip).c_str(),
                    ibPeer ? ibPeer->GetState() : -1,
                    cc.GetRTT() * 1000, cc.GetTimeout() * 1000, cc.GetWindow(), cc.GetMaxWindow(), cc.GetFailRate(),
                    cc.GetTimeSinceLastRecv() * 1000, cc.GetTransferCount(), cc.GetMTU());
            res += buf;
        }
        return res;
    }

    TString TUdpHost::GetDebugInfo() {
        TString res;
        char buf[1000];
        snprintf(buf, sizeof(buf), "Receiving %d msgs, sending %d high prior, %d regular msgs, %d low prior msgs\n",
                RecvQueue.ysize(), (int)SendOrderHighPrior.size(), (int)SendOrder.size(), (int)SendOrderLow.size());
        res += buf;

        TRequesterPendingDataStats pds;
        GetPendingDataSize(&pds);
        snprintf(buf, sizeof(buf), "Pending data size: %" PRIu64 "\n", pds.InpDataSize + pds.OutDataSize);
        res += buf;
        snprintf(buf, sizeof(buf), "  in packets: %d, size %" PRIu64 "\n", pds.InpCount, pds.InpDataSize);
        res += buf;
        snprintf(buf, sizeof(buf), "  out packets: %d, size %" PRIu64 "\n", pds.OutCount, pds.OutDataSize);
        res += buf;

        res += "\nCongestion info:\n";
        res += GetPeerLinkDebug(CongestionTrack);
        res += "\nCongestion info history:\n";
        res += GetPeerLinkDebug(CongestionTrackHistory);

        return res;
    }

    static void SendKill(const TNetSocket& s, const sockaddr_in6& toAddress) {
        char buf[100];
        char* pktData = buf + UDP_LOW_LEVEL_HEADER_SIZE;
        Write(&pktData, (int)0);
        Write(&pktData, (char)KILL);
        Write(&pktData, KILL_PASSPHRASE1);
        Write(&pktData, KILL_PASSPHRASE2);
        s.SendTo(buf, (int)(pktData - buf), toAddress, FF_ALLOW_FRAG);
    }

    void TUdpHost::Kill(const TUdpAddress& addr) {
        sockaddr_in6 target;
        GetWinsockAddr(&target, addr);
        SendKill(s, target);
    }

    TIntrusivePtr<IPeerQueueStats> TUdpHost::GetQueueStats(const TUdpAddress& addr) {
        TQueueStatsHash::iterator zq = PeerQueueStats.find(addr);
        if (zq != PeerQueueStats.end()) {
            return zq->second.Get();
        }
        TPeerQueueStats* res = new TPeerQueueStats;
        PeerQueueStats[addr] = res;
        // attach to existing congestion tracker
        TPeerLinkHash::iterator z;
        z = CongestionTrack.find(addr);
        if (z != CongestionTrack.end()) {
            z->second.UdpCongestion->AttachQueueStats(res);
        }
        z = CongestionTrackHistory.find(addr);
        if (z != CongestionTrackHistory.end()) {
            z->second.UdpCongestion->AttachQueueStats(res);
        }
        return res;
    }

    //////////////////////////////////////////////////////////////////////////

    TIntrusivePtr<IUdpHost> CreateUdpHost(int port) {
        TIntrusivePtr<NNetlibaSocket::ISocket> socket = NNetlibaSocket::CreateBestRecvSocket();
        socket->Open(port);
        if (!socket->IsValid())
            return nullptr;
        return CreateUdpHost(socket);
    }

    TIntrusivePtr<IUdpHost> CreateUdpHost(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
        if (!InitLocalIPList()) {
            Y_ASSERT(0 && "Can not determine self IP address");
            return nullptr;
        }
        TIntrusivePtr<TUdpHost> res = new TUdpHost;
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

}
