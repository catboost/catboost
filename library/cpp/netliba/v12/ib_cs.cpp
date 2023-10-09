#include "stdafx.h"
#include "ib_cs.h"
#include "ib_buffers.h"
#include "ib_mem.h"
#include <util/generic/deque.h>
#include <util/digest/murmur.h>

/*
Questions
 does rdma work?
 what is RC latency?
   3us if measured by completion event arrival
   2.3us if bind to socket 0 & use inline send
 memory region - can we use memory from some offset?
   yes
 is send_inplace supported and is it faster?
   yes, supported, 1024 bytes limit, inline is faster (2.4 vs 2.9)
 is srq a penalty compared to regular rq?
   rdma is faster anyway, so why bother

collective ops
  support asymmetric configurations by additional transfers (overlap 1 or 2 hosts is allowed)

remove commented stuff all around

next gen
  shared+registered large mem blocks for easy transfer
  no crc calcs
  direct channel exposure
    make ui64 packet id? otherwise we could get duplicate id (highly improbable but possible)
  lock free allocation in ib_mem
*/

namespace NNetliba_v12 {
    const int WELCOME_QKEY = 0x13081976;

    const int MAX_SEND_COUNT = (128 - 10) / 4;
    const int QP_SEND_QUEUE_SIZE = (MAX_SEND_COUNT * 2 + 10) + 10;
    const int WELCOME_QP_SEND_SIZE = 10000;

    const int MAX_SRQ_WORK_REQUESTS = 10000;
    const int MAX_CQ_EVENTS = MAX_SRQ_WORK_REQUESTS; //1000;

    const double CHANNEL_CHECK_INTERVAL = 1.;

    const int TRAFFIC_SL = 4; // 4 is mandatory for RoCE to work, it's the only lossless priority(?)
    const int CONNECT_SL = 1;

    class TIBClientServer: public IIBClientServer {
        enum ECmd {
            CMD_HANDSHAKE,
            CMD_HANDSHAKE_ACK,
            CMD_CONFIRM,
            CMD_DATA_TINY,
            CMD_DATA_INIT,
            CMD_BUFFER_READY,
            CMD_DATA_COMPLETE,
            CMD_KEEP_ALIVE,
        };
#pragma pack(1)
        struct TCmdHandshake {
            char Command;
            int QPN, PSN;
            TGUID SocketId;
            TUdpAddress MyAddress; // address of the handshake sender as viewed from receiver
        };
        struct TCmdHandshakeAck {
            char Command;
            int QPN, PSN;
            int YourQPN;
        };
        struct TCmdConfirm {
            char Command;
        };
        struct TCmdDataTiny {
            struct THeader {
                char Command;
                ui16 Size;
                TGUID PacketGuid;
                TGUID ConnectionGuid;
            } Header;
            typedef char TDataVec[SMALL_PKT_SIZE - sizeof(THeader)];
            TDataVec Data;

            static int GetMaxDataSize() {
                return sizeof(TDataVec);
            }
        };
        struct TCmdDataInit {
            char Command;
            size_t Size;
            TGUID PacketGuid;
            TGUID ConnectionGuid;
        };
        struct TCmdBufferReady {
            char Command;
            TGUID PacketGuid;
            ui64 RemoteAddr;
            ui32 RemoteKey;
        };
        struct TCmdDataComplete {
            char Command;
            TGUID PacketGuid;
            ui64 DataHash;
        };
        struct TCmdKeepAlive {
            char Command;
        };
#pragma pack()

        struct TCompleteInfo {
            enum {
                CI_DATA_TINY,
                CI_RDMA_COMPLETE,
                CI_DATA_SENT,
                CI_KEEP_ALIVE,
                CI_IGNORE,
            };
            int Type;
            int BufId;
            TIBMsgHandle MsgHandle;

            TCompleteInfo(int t, int bufId, TIBMsgHandle msg)
                : Type(t)
                , BufId(bufId)
                , MsgHandle(msg)
            {
            }
        };
        struct TPendingQueuedSend {
            TGUID PacketGuid;
            TGUID ConnectionGuid;
            TIBMsgHandle MsgHandle;
            TRopeDataPacket* Data;

            TPendingQueuedSend()
                : MsgHandle(0)
            {
            }
            TPendingQueuedSend(const TGUID& packetGuid, const TGUID& connectionGuid, TIBMsgHandle msgHandle, TRopeDataPacket* data)
                : PacketGuid(packetGuid)
                , ConnectionGuid(connectionGuid)
                , MsgHandle(msgHandle)
                , Data(data)
            {
            }
        };
        struct TQueuedSend {
            TGUID PacketGuid;
            TIBMsgHandle MsgHandle;
            TIntrusivePtr<TIBMemBlock> MemBlock;
            ui64 RemoteAddr;
            ui32 RemoteKey;

            TQueuedSend() {
            }
            TQueuedSend(const TGUID& packetGuid, TIBMsgHandle msgHandle)
                : PacketGuid(packetGuid)
                , MsgHandle(msgHandle)
                , RemoteAddr(0)
                , RemoteKey(0)
            {
            }
        };
        struct TQueuedRecv {
            TGUID PacketGuid, ConnectionGuid;
            TIntrusivePtr<TIBMemBlock> Data;

            TQueuedRecv() {
            }
            TQueuedRecv(const TGUID& packetGuid, const TGUID& connectionGuid, TPtrArg<TIBMemBlock> data)
                : PacketGuid(packetGuid)
                , ConnectionGuid(connectionGuid)
                , Data(data)
            {
            }
        };
        struct TIBPeer: public IIBPeer {
            TUdpAddress PeerAddress;
            TIntrusivePtr<TRCQueuePair> QP;
            EState State;
            int SendCount;
            NHPTimer::STime LastRecv;
            TDeque<TPendingQueuedSend> PendingSendQueue;
            // these lists have limited size and potentially just circle buffers
            TDeque<TQueuedSend> SendQueue;
            TDeque<TQueuedRecv> RecvQueue;
            TDeque<TCompleteInfo> OutMsgs;

            TIBPeer(const TUdpAddress& peerAddress, TPtrArg<TRCQueuePair> qp)
                : PeerAddress(peerAddress)
                , QP(qp)
                , State(CONNECTING)
                , SendCount(0)
            {
                NHPTimer::GetTime(&LastRecv);
            }
            ~TIBPeer() override {
                //printf("IBPeer destroyed\n");
            }
            EState GetState() override {
                return State;
            }
            TDeque<TQueuedSend>::iterator GetSend(const TGUID& packetGuid) {
                for (TDeque<TQueuedSend>::iterator z = SendQueue.begin(); z != SendQueue.end(); ++z) {
                    if (z->PacketGuid == packetGuid) {
                        return z;
                    }
                }
                Y_ABORT_UNLESS(0, "no send by guid");
                return SendQueue.begin();
            }
            TDeque<TQueuedSend>::iterator GetSend(TIBMsgHandle msgHandle) {
                for (TDeque<TQueuedSend>::iterator z = SendQueue.begin(); z != SendQueue.end(); ++z) {
                    if (z->MsgHandle == msgHandle) {
                        return z;
                    }
                }
                Y_ABORT_UNLESS(0, "no send by handle");
                return SendQueue.begin();
            }
            TDeque<TQueuedRecv>::iterator GetRecv(const TGUID& packetGuid) {
                for (TDeque<TQueuedRecv>::iterator z = RecvQueue.begin(); z != RecvQueue.end(); ++z) {
                    if (z->PacketGuid == packetGuid) {
                        return z;
                    }
                }
                Y_ABORT_UNLESS(0, "no recv by guid");
                return RecvQueue.begin();
            }
            void PostRDMA(TQueuedSend& qs) {
                Y_ASSERT(qs.RemoteAddr != 0 && qs.MemBlock.Get() != nullptr);
                QP->PostRDMAWrite(qs.RemoteAddr, qs.RemoteKey,
                                  qs.MemBlock->GetMemRegion(), 0, qs.MemBlock->GetData(), qs.MemBlock->GetSize());
                OutMsgs.push_back(TCompleteInfo(TCompleteInfo::CI_RDMA_COMPLETE, 0, qs.MsgHandle));
                //printf("Post rdma write, size %d\n", qs.Data->GetSize());
            }
            void PostSend(TIBBufferPool& bp, const void* data, size_t len, int t, TIBMsgHandle msgHandle) {
                int bufId = bp.PostSend(QP, data, len);
                OutMsgs.push_back(TCompleteInfo(t, bufId, msgHandle));
            }
        };

        TIntrusivePtr<TIBPort> Port;
        TIntrusivePtr<TIBMemPool> MemPool;
        TIntrusivePtr<TIBMemPool::TCopyResultStorage> CopyResults;
        TIntrusivePtr<TComplectionQueue> CQ;
        TIBBufferPool BP;
        TIntrusivePtr<TUDQueuePair> WelcomeQP;
        int WelcomeQPN;
        TIBConnectInfo ConnectInfo;
        TDeque<TIBSendResult> SendResults;
        TDeque<TIBRequest*> ReceivedList;
        typedef THashMap<int, TIntrusivePtr<TIBPeer>> TPeerChannelHash;
        TPeerChannelHash Channels;
        TIBMsgHandle MsgCounter;
        NHPTimer::STime LastCheckTime;

        ~TIBClientServer() override {
            for (auto& z : ReceivedList) {
                delete z;
            }
        }
        TIBPeer* GetChannelByQPN(int qpn) {
            TPeerChannelHash::iterator z = Channels.find(qpn);
            if (z == Channels.end()) {
                return nullptr;
            }
            return z->second.Get();
        }

        // IIBClientServer
        TIBRequest* GetRequest() override {
            if (ReceivedList.empty()) {
                return nullptr;
            }
            TIBRequest* res = ReceivedList.front();
            ReceivedList.pop_front();
            return res;
        }
        bool GetSendResult(TIBSendResult* res) override {
            if (SendResults.empty()) {
                return false;
            }
            *res = SendResults.front();
            SendResults.pop_front();
            return true;
        }
        void StartSend(TPtrArg<TIBPeer> peer, const TGUID& packetGuid, const TGUID& connectionGuid, TIBMsgHandle msgHandle, TRopeDataPacket* data) {
            int sz = data->GetSize();
            if (sz <= TCmdDataTiny::GetMaxDataSize()) {
                TCmdDataTiny dataTiny;
                dataTiny.Header.Command = CMD_DATA_TINY;
                dataTiny.Header.Size = (ui16)sz;
                dataTiny.Header.PacketGuid = packetGuid;
                dataTiny.Header.ConnectionGuid = connectionGuid;
                TBlockChainIterator bc(data->GetChain());
                bc.Read(dataTiny.Data, sz);

                peer->PostSend(BP, &dataTiny, sizeof(dataTiny.Header) + sz, TCompleteInfo::CI_DATA_TINY, msgHandle);
                //printf("Send CMD_DATA_TINY\n");
            } else {
                MemPool->CopyData(data, msgHandle, peer, CopyResults);
                peer->SendQueue.push_back(TQueuedSend(packetGuid, msgHandle));
                {
                    TQueuedSend& msg = peer->SendQueue.back();
                    TCmdDataInit dataInit;
                    dataInit.Command = CMD_DATA_INIT;
                    dataInit.PacketGuid = msg.PacketGuid;
                    dataInit.Size = data->GetSize();
                    dataInit.ConnectionGuid = connectionGuid;
                    peer->PostSend(BP, &dataInit, sizeof(dataInit), TCompleteInfo::CI_IGNORE, 0);
                    //printf("Send CMD_DATA_INIT\n");
                }
            }
            ++peer->SendCount;
        }
        void SendCompleted(TPtrArg<TIBPeer> peer, TIBMsgHandle msgHandle) {
            SendResults.push_back(TIBSendResult(msgHandle, true));
            if (--peer->SendCount < MAX_SEND_COUNT) {
                if (!peer->PendingSendQueue.empty()) {
                    TPendingQueuedSend& qs = peer->PendingSendQueue.front();
                    StartSend(peer, qs.PacketGuid, qs.ConnectionGuid, qs.MsgHandle, qs.Data);
                    //printf("Sending pending %d\n", qs.MsgHandle);
                    peer->PendingSendQueue.pop_front();
                }
            }
        }
        void SendFailed(TPtrArg<TIBPeer> peer, TIBMsgHandle msgHandle) {
            //printf("IB SendFailed()\n");
            SendResults.push_back(TIBSendResult(msgHandle, false));
            --peer->SendCount;
        }
        void PeerFailed(TPtrArg<TIBPeer> peer) {
            //printf("PeerFailed(), peer %p, state %d (%d pending, %d queued, %d out, %d sendcount)\n",
            //    peer.Get(), peer->State,
            //    (int)peer->PendingSendQueue.size(),
            //    (int)peer->SendQueue.size(),
            //    (int)peer->OutMsgs.size(),
            //    peer->SendCount);
            peer->State = IIBPeer::FAILED;
            while (!peer->PendingSendQueue.empty()) {
                TPendingQueuedSend& qs = peer->PendingSendQueue.front();
                SendResults.push_back(TIBSendResult(qs.MsgHandle, false));
                peer->PendingSendQueue.pop_front();
            }
            while (!peer->SendQueue.empty()) {
                TQueuedSend& qs = peer->SendQueue.front();
                SendFailed(peer, qs.MsgHandle);
                peer->SendQueue.pop_front();
            }
            while (!peer->OutMsgs.empty()) {
                TCompleteInfo& cc = peer->OutMsgs.front();
                //printf("Don't wait completion for sent packet (QPN %d), bufId %d\n", peer->QP->GetQPN(), cc.BufId);
                if (cc.Type == TCompleteInfo::CI_DATA_TINY) {
                    SendFailed(peer, cc.MsgHandle);
                }
                BP.FreeBuf(cc.BufId);
                peer->OutMsgs.pop_front();
            }
            {
                Y_ASSERT(peer->SendCount == 0);
                //printf("Remove peer %p from hash (QPN %d)\n", peer.Get(), peer->QP->GetQPN());
                TPeerChannelHash::iterator z = Channels.find(peer->QP->GetQPN());
                if (z == Channels.end()) {
                    Y_ABORT_UNLESS(0, "peer failed for unregistered peer");
                }
                Channels.erase(z);
            }
        }
        TIBMsgHandle Send(TPtrArg<IIBPeer> peerArg, TRopeDataPacket* data, const TGUID& packetGuid, const TGUID& connectionGuid) override {
            TIBPeer* peer = static_cast<TIBPeer*>(peerArg.Get()); // trust me, I'm professional
            if (peer == nullptr || peer->State != IIBPeer::OK) {
                return -1;
            }
            Y_ASSERT(Channels.find(peer->QP->GetQPN())->second == peer);
            TIBMsgHandle msgHandle = ++MsgCounter;
            if (peer->SendCount >= MAX_SEND_COUNT) {
                peer->PendingSendQueue.push_back(TPendingQueuedSend(packetGuid, connectionGuid, msgHandle, data));
            } else {
                //printf("Sending direct %d\n", msgHandle);
                StartSend(peer, packetGuid, connectionGuid, msgHandle, data);
            }
            return msgHandle;
        }
        void ParsePacket(ibv_wc* wc, NHPTimer::STime tCurrent) {
            if (wc->status != IBV_WC_SUCCESS) {
                TIBPeer* peer = GetChannelByQPN(wc->qp_num);
                if (peer) {
                    //printf("failed recv packet (status %d)\n", wc->status);
                    PeerFailed(peer);
                } else {
                    //printf("Ignoring recv error for closed/non existing QPN %d\n", wc->qp_num);
                }
                return;
            }

            TIBRecvPacketProcess pkt(BP, *wc);

            TIBPeer* peer = GetChannelByQPN(wc->qp_num);
            if (peer) {
                Y_ASSERT(peer->State != IIBPeer::FAILED);
                peer->LastRecv = tCurrent;
                char cmdId = *(const char*)pkt.GetData();
                switch (cmdId) {
                    case CMD_CONFIRM:
                        //printf("got confirm\n");
                        Y_ASSERT(peer->State == IIBPeer::CONNECTING);
                        peer->State = IIBPeer::OK;
                        break;
                    case CMD_DATA_TINY:
                        //printf("Recv CMD_DATA_TINY\n");
                        {
                            const TCmdDataTiny& dataTiny = *(TCmdDataTiny*)pkt.GetData();
                            TIBRequest* req = new TIBRequest;
                            req->ConnectionGuid = dataTiny.Header.ConnectionGuid;
                            req->Data = new TRopeDataPacket;
                            req->Data->Write(dataTiny.Data, dataTiny.Header.Size);
                            ReceivedList.push_back(req);
                        }
                        break;
                    case CMD_DATA_INIT:
                        //printf("Recv CMD_DATA_INIT\n");
                        {
                            const TCmdDataInit& data = *(TCmdDataInit*)pkt.GetData();
                            TIntrusivePtr<TIBMemBlock> blk = MemPool->Alloc(data.Size);
                            peer->RecvQueue.push_back(TQueuedRecv(data.PacketGuid, data.ConnectionGuid, blk));
                            TCmdBufferReady ready;
                            ready.Command = CMD_BUFFER_READY;
                            ready.PacketGuid = data.PacketGuid;

                            ready.RemoteAddr = reinterpret_cast<ui64>(blk->GetData()) / sizeof(char);
                            ready.RemoteKey = blk->GetMemRegion()->GetRKey();

                            peer->PostSend(BP, &ready, sizeof(ready), TCompleteInfo::CI_IGNORE, 0);
                            //printf("Send CMD_BUFFER_READY\n");
                        }
                        break;
                    case CMD_BUFFER_READY:
                        //printf("Recv CMD_BUFFER_READY\n");
                        {
                            const TCmdBufferReady& ready = *(TCmdBufferReady*)pkt.GetData();
                            TDeque<TQueuedSend>::iterator z = peer->GetSend(ready.PacketGuid);
                            TQueuedSend& qs = *z;
                            qs.RemoteAddr = ready.RemoteAddr;
                            qs.RemoteKey = ready.RemoteKey;
                            if (qs.MemBlock.Get()) {
                                peer->PostRDMA(qs);
                            }
                        }
                        break;
                    case CMD_DATA_COMPLETE:
                        //printf("Recv CMD_DATA_COMPLETE\n");
                        {
                            const TCmdDataComplete& cmd = *(TCmdDataComplete*)pkt.GetData();
                            TDeque<TQueuedRecv>::iterator z = peer->GetRecv(cmd.PacketGuid);
                            TQueuedRecv& qr = *z;
#ifdef _DEBUG
                            Y_ABORT_UNLESS(MurmurHash<ui64>(qr.Data->GetData(), qr.Data->GetSize()) == cmd.DataHash || cmd.DataHash == 0, "RDMA data hash mismatch");
#endif
                            TIBRequest* req = new TIBRequest;

                            req->ConnectionGuid = qr.ConnectionGuid;
                            req->Data = new TRopeDataPacket;
                            req->Data->AddBlock(qr.Data.Get(), qr.Data->GetData(), qr.Data->GetSize());
                            ReceivedList.push_back(req);
                            peer->RecvQueue.erase(z);
                        }
                        break;
                    case CMD_KEEP_ALIVE:
                        break;
                    default:
                        Y_ASSERT(0);
                        break;
                }
            } else {
                // can get here
                //printf("Ignoring packet for closed/non existing QPN %d\n", wc->qp_num);
            }
        }
        void OnComplete(ibv_wc* wc, NHPTimer::STime tCurrent) {
            TIBPeer* peer = GetChannelByQPN(wc->qp_num);
            if (peer) {
                if (!peer->OutMsgs.empty()) {
                    peer->LastRecv = tCurrent;
                    if (wc->status != IBV_WC_SUCCESS) {
                        //printf("completed with status %d\n", wc->status);
                        PeerFailed(peer);
                    } else {
                        const TCompleteInfo& cc = peer->OutMsgs.front();
                        switch (cc.Type) {
                            case TCompleteInfo::CI_DATA_TINY:
                                //printf("Completed data_tiny\n");
                                SendCompleted(peer, cc.MsgHandle);
                                break;
                            case TCompleteInfo::CI_RDMA_COMPLETE:
                                //printf("Completed rdma_complete\n");
                                {
                                    TDeque<TQueuedSend>::iterator z = peer->GetSend(cc.MsgHandle);
                                    TQueuedSend& qs = *z;

                                    TCmdDataComplete complete;
                                    complete.Command = CMD_DATA_COMPLETE;
                                    complete.PacketGuid = qs.PacketGuid;
#ifdef _DEBUG
                                    complete.DataHash = MurmurHash<ui64>(qs.MemBlock->GetData(), qs.MemBlock->GetSize());
#else
                                    complete.DataHash = 0;
#endif

                                    peer->PostSend(BP, &complete, sizeof(complete), TCompleteInfo::CI_DATA_SENT, qs.MsgHandle);
                                    //printf("Send CMD_DATA_COMPLETE\n");
                                }
                                break;
                            case TCompleteInfo::CI_DATA_SENT:
                                //printf("Completed data_sent\n");
                                {
                                    TDeque<TQueuedSend>::iterator z = peer->GetSend(cc.MsgHandle);
                                    TIBMsgHandle msgHandle = z->MsgHandle;
                                    peer->SendQueue.erase(z);
                                    SendCompleted(peer, msgHandle);
                                }
                                break;
                            case TCompleteInfo::CI_KEEP_ALIVE:
                                break;
                            case TCompleteInfo::CI_IGNORE:
                                //printf("Completed ignored\n");
                                break;
                            default:
                                Y_ASSERT(0);
                                break;
                        }
                        peer->OutMsgs.pop_front();
                        BP.FreeBuf(wc->wr_id);
                    }
                } else {
                    Y_ABORT_UNLESS(0, "got completion without outstanding messages");
                }
            } else {
                //printf("Got completion for non existing qpn %d, bufId %d (status %d)\n", wc->qp_num, (int)wc->wr_id, (int)wc->status);
                if (wc->status == IBV_WC_SUCCESS) {
                    Y_ABORT_UNLESS(0, "only errors should go unmatched");
                }
                // no need to free buf since it has to be freed in PeerFailed()
            }
        }
        void ParseWelcomePacket(ibv_wc* wc) {
            TIBRecvPacketProcess pkt(BP, *wc);

            char cmdId = *(const char*)pkt.GetUDData();
            switch (cmdId) {
                case CMD_HANDSHAKE: {
                    //printf("got handshake\n");
                    const TCmdHandshake& handshake = *(TCmdHandshake*)pkt.GetUDData();
                    if (handshake.SocketId != ConnectInfo.SocketId) {
                        // connection attempt from wrong IB subnet
                        break;
                    }
                    TIntrusivePtr<TRCQueuePair> rcQP;
                    rcQP = new TRCQueuePair(Port->GetCtx(), CQ, BP.GetSRQ(), QP_SEND_QUEUE_SIZE);

                    int qpn = rcQP->GetQPN();
                    Y_ASSERT(Channels.find(qpn) == Channels.end());
                    TIntrusivePtr<TIBPeer>& peer = Channels[qpn];
                    peer = new TIBPeer(handshake.MyAddress, rcQP);

                    ibv_ah_attr peerAddr;
                    TIntrusivePtr<TAddressHandle> ahPeer;
                    Port->GetAHAttr(wc, pkt.GetGRH(), &peerAddr);
                    ahPeer = new TAddressHandle(Port->GetCtx(), &peerAddr);

                    peerAddr.sl = TRAFFIC_SL;
                    rcQP->Init(peerAddr, handshake.QPN, handshake.PSN);

                    TCmdHandshakeAck handshakeAck;
                    handshakeAck.Command = CMD_HANDSHAKE_ACK;
                    handshakeAck.PSN = rcQP->GetPSN();
                    handshakeAck.QPN = rcQP->GetQPN();
                    handshakeAck.YourQPN = handshake.QPN;
                    // if ack gets lost we'll create new Peer Channel
                    // and this one will be erased in Step() by timeout counted from LastRecv
                    BP.PostSend(WelcomeQP, ahPeer, wc->src_qp, WELCOME_QKEY, &handshakeAck, sizeof(handshakeAck));
                    //printf("send handshake_ack\n");
                } break;
                case CMD_HANDSHAKE_ACK: {
                    //printf("got handshake_ack\n");
                    const TCmdHandshakeAck& handshakeAck = *(TCmdHandshakeAck*)pkt.GetUDData();
                    TIBPeer* peer = GetChannelByQPN(handshakeAck.YourQPN);
                    if (peer) {
                        ibv_ah_attr peerAddr;
                        Port->GetAHAttr(wc, pkt.GetGRH(), &peerAddr);

                        peerAddr.sl = TRAFFIC_SL;
                        peer->QP->Init(peerAddr, handshakeAck.QPN, handshakeAck.PSN);

                        peer->State = IIBPeer::OK;

                        TCmdConfirm confirm;
                        confirm.Command = CMD_CONFIRM;
                        peer->PostSend(BP, &confirm, sizeof(confirm), TCompleteInfo::CI_IGNORE, 0);
                        //printf("send confirm\n");
                    } else {
                        // respective QPN was deleted or never existed
                        // silently ignore and peer channel on remote side
                        // will not get into confirmed state and will be deleted
                    }
                } break;
                default:
                    Y_ASSERT(0);
                    break;
            }
        }
        bool Step(NHPTimer::STime tCurrent) override {
            bool rv = false;
            // only have to process completions, everything is done on completion of something
            ibv_wc wcArr[10];
            for (;;) {
                int wcCount = CQ->Poll(wcArr, Y_ARRAY_SIZE(wcArr));
                if (wcCount == 0) {
                    break;
                }
                rv = true;
                for (int z = 0; z < wcCount; ++z) {
                    ibv_wc& wc = wcArr[z];
                    if (wc.opcode & IBV_WC_RECV) {
                        // received msg
                        if ((int)wc.qp_num == WelcomeQPN) {
                            if (wc.status != IBV_WC_SUCCESS) {
                                Y_ABORT_UNLESS(0, "ud recv op completed with error %d\n", (int)wc.status);
                            }
                            Y_ASSERT(wc.opcode == IBV_WC_RECV | IBV_WC_SEND);
                            ParseWelcomePacket(&wc);
                        } else {
                            ParsePacket(&wc, tCurrent);
                        }
                    } else {
                        // send completion
                        if ((int)wc.qp_num == WelcomeQPN) {
                            // ok
                            BP.FreeBuf(wc.wr_id);
                        } else {
                            OnComplete(&wc, tCurrent);
                        }
                    }
                }
            }
            {
                TIntrusivePtr<TIBMemBlock> memBlock;
                i64 msgHandle;
                TIntrusivePtr<TIBPeer> peer;
                while (CopyResults->GetCopyResult(&memBlock, &msgHandle, &peer)) {
                    if (peer->GetState() != IIBPeer::OK) {
                        continue;
                    }
                    TDeque<TQueuedSend>::iterator z = peer->GetSend(msgHandle);
                    if (z == peer->SendQueue.end()) {
                        Y_ABORT_UNLESS(0, "peer %p, copy completed, msg %d not found?\n", peer.Get(), (int)msgHandle);
                        continue;
                    }
                    TQueuedSend& qs = *z;
                    qs.MemBlock = memBlock;
                    if (qs.RemoteAddr != 0) {
                        peer->PostRDMA(qs);
                    }
                    rv = true;
                }
            }
            {
                NHPTimer::STime t1 = LastCheckTime;
                if (NHPTimer::GetTimePassed(&t1) > CHANNEL_CHECK_INTERVAL) {
                    for (TPeerChannelHash::iterator z = Channels.begin(); z != Channels.end();) {
                        TIntrusivePtr<TIBPeer> peer = z->second;
                        ++z; // peer can be removed from Channels
                        Y_ASSERT(peer->State != IIBPeer::FAILED);
                        NHPTimer::STime t2 = peer->LastRecv;
                        double timeSinceLastRecv = NHPTimer::GetTimePassed(&t2);
                        if (timeSinceLastRecv > CHANNEL_CHECK_INTERVAL) {
                            if (peer->State == IIBPeer::CONNECTING) {
                                Y_ASSERT(peer->OutMsgs.empty() && peer->SendCount == 0);
                                // if handshake does not seem to work out - close connection
                                //printf("IB connecting timed out\n");
                                PeerFailed(peer);
                            } else {
                                // if we have outmsg we hope that IB will report us if there are any problems
                                // with connectivity
                                if (peer->OutMsgs.empty()) {
                                    //printf("Sending keep alive\n");
                                    TCmdKeepAlive keep;
                                    keep.Command = CMD_KEEP_ALIVE;
                                    peer->PostSend(BP, &keep, sizeof(keep), TCompleteInfo::CI_KEEP_ALIVE, 0);
                                }
                            }
                        }
                    }
                    LastCheckTime = t1;
                }
            }
            return rv;
        }
        IIBPeer* ConnectPeer(const TIBConnectInfo& info, const TUdpAddress& peerAddr, const TUdpAddress& myAddr) override {
            for (auto& channel : Channels) {
                TIntrusivePtr<TIBPeer> peer = channel.second;
                if (peer->PeerAddress == peerAddr) {
                    return peer.Get();
                }
            }
            TIntrusivePtr<TRCQueuePair> rcQP;
            rcQP = new TRCQueuePair(Port->GetCtx(), CQ, BP.GetSRQ(), QP_SEND_QUEUE_SIZE);

            int qpn = rcQP->GetQPN();
            Y_ASSERT(Channels.find(qpn) == Channels.end());
            TIntrusivePtr<TIBPeer>& peer = Channels[qpn];
            peer = new TIBPeer(peerAddr, rcQP);

            TCmdHandshake handshake;
            handshake.Command = CMD_HANDSHAKE;
            handshake.PSN = rcQP->GetPSN();
            handshake.QPN = rcQP->GetQPN();
            handshake.SocketId = info.SocketId;
            handshake.MyAddress = myAddr;

            TIntrusivePtr<TAddressHandle> serverAH;
            if (info.LID != 0) {
                serverAH = new TAddressHandle(Port, info.LID, CONNECT_SL);
            } else {
                TUdpAddress local = myAddr;
                local.Port = 0;
                TUdpAddress remote = peerAddr;
                remote.Port = 0;
                //printf("local Addr %s\n", GetAddressAsString(local).c_str());
                //printf("remote Addr %s\n", GetAddressAsString(remote).c_str());
                sockaddr_in6 remoteAddr;
                GetWinsockAddrForMLNX(&remoteAddr, remote);
                sockaddr_in6 localAddr;
                GetWinsockAddrForMLNX(&localAddr, local);
                // CRAP - somehow prevent connecting machines from different RoCE isles
                serverAH = new TAddressHandle(Port, (sockaddr&)remoteAddr, (sockaddr&)localAddr, CONNECT_SL);
                if (!serverAH->IsValid()) {
                    return nullptr;
                }
            }
            BP.PostSend(WelcomeQP, serverAH, info.QPN, WELCOME_QKEY, &handshake, sizeof(handshake));
            //printf("send handshake\n");

            return peer.Get();
        }
        const TIBConnectInfo& GetConnectInfo() override {
            return ConnectInfo;
        }

    public:
        TIBClientServer(TPtrArg<TIBPort> port)
            : Port(port)
            , MemPool(GetIBMemPool())
            , CQ(new TComplectionQueue(port->GetCtx(), MAX_CQ_EVENTS))
            , BP(port->GetCtx(), MAX_SRQ_WORK_REQUESTS)
            , WelcomeQP(new TUDQueuePair(port, CQ, BP.GetSRQ(), WELCOME_QP_SEND_SIZE))
            , WelcomeQPN(WelcomeQP->GetQPN())
            , MsgCounter(1)
        {
            CopyResults = new TIBMemPool::TCopyResultStorage;
            CreateGuid(&ConnectInfo.SocketId);
            ibv_gid addr;
            port->GetGID(&addr);
            ConnectInfo.Interface = addr.global.interface_id;
            ConnectInfo.Subnet = addr.global.subnet_prefix;
            //printf("connect addr subnet %lx, iface %lx\n", addr.global.subnet_prefix, addr.global.interface_id);
            ConnectInfo.LID = port->GetLID();
            ConnectInfo.QPN = WelcomeQPN;

            WelcomeQP->Init(WELCOME_QKEY);

            NHPTimer::GetTime(&LastCheckTime);
        }
    };

    IIBClientServer* CreateIBClientServer() {
        TIntrusivePtr<TIBPort> port = GetIBDevice();
        if (port.Get() == nullptr) {
            return nullptr;
        }
        return new TIBClientServer(port);
    }
}
