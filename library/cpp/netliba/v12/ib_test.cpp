#include "stdafx.h"
#include "ib_test.h"
#include "ib_buffers.h"
#include "udp_socket.h"
#include "udp_address.h"
#include <util/system/hp_timer.h>

namespace NNetliba_v12 {
    struct TWelcomeSocketAddr {
        int LID;
        int QPN;
    };

    struct TRCQueuePairHandshake {
        int QPN, PSN;
    };

    class TIPSocket {
        TUdpSocket s;

        static constexpr int HDR_SIZE = UDP_LOW_LEVEL_HEADER_SIZE;

    public:
        TIPSocket()
            : s(1, false)
        {
        }

        void Init(int port) {
            s.Open(port);
            if (!s.IsValid()) {
                Y_ASSERT(0 && "Can not open socket");
            }
        }
        bool IsValid() {
            return s.IsValid();
        }
        void Respond(const TWelcomeSocketAddr& info) {
            char buf[10000];
            size_t sz = sizeof(buf);
            TSockAddrPair addr;
            bool rv = s.RecvFrom(buf, &sz, &addr);
            if (rv && strcmp(buf + HDR_SIZE, "Hello_IB") == 0) {
                printf("send welcome info\n");
                memcpy(buf + HDR_SIZE, &info, sizeof(info));
                TUdpSocket::ESendError err = s.SendTo(buf, sizeof(info) + HDR_SIZE, addr, 0, FF_ALLOW_FRAG); // TODO: choose tos
                if (err != TUdpSocket::SEND_OK) {
                    printf("SendTo() fail %d\n", err);
                }
            }
        }
        void Request(const char* hostName, int port, TWelcomeSocketAddr* res) {
            TUdpAddress addr1 = CreateAddress(hostName, port);
            printf("addr = %s\n", GetAddressAsString(addr1).c_str());

            sockaddr_in6 sockAddr;
            GetWinsockAddr(&sockAddr, addr1);

            for (;;) {
                char buf[10000];
                size_t sz = sizeof(buf);
                TSockAddrPair addr2;
                bool rv = s.RecvFrom(buf, &sz, &addr2);
                if (rv) {
                    if (sz == sizeof(TWelcomeSocketAddr) + HDR_SIZE) {
                        *res = *(TWelcomeSocketAddr*)(buf + HDR_SIZE);
                        break;
                    }
                    printf("Get unexpected %d bytes from somewhere?\n", (int)sz);
                }

                strcpy(buf + HDR_SIZE, "Hello_IB");
                TUdpSocket::ESendError err = s.SendTo(buf, strlen(buf + HDR_SIZE) + 1 + HDR_SIZE, {sockAddr, addr2.MyAddr}, 0, FF_ALLOW_FRAG); // TODO: choose tos
                if (err != TUdpSocket::SEND_OK) {
                    printf("SendTo() fail %d\n", err);
                }

                Sleep(TDuration::MilliSeconds(100));
            }
        }
    };

    // can hang if opposite side exits, but it's only basic test
    static void WaitForRecv(TIBBufferPool* bp, TPtrArg<TComplectionQueue> cq, ibv_wc* wc) {
        for (;;) {
            if (cq->Poll(wc, 1) == 1) {
                if (wc->opcode & IBV_WC_RECV) {
                    break;
                }
                bp->FreeBuf(wc->wr_id);
            }
        }
    }

    void RunIBTest(bool isClient, const char* serverName) {
        TIntrusivePtr<TIBPort> port = GetIBDevice();
        if (port.Get() == nullptr) {
            printf("No IB device found\n");
            return;
        }

        const int IP_PORT = 13666;
        const int WELCOME_QKEY = 0x1113013;
        const int MAX_SRQ_WORK_REQUESTS = 100;
        const int MAX_CQ_EVENTS = 1000;
        const int QP_SEND_QUEUE_SIZE = 3;

        TIntrusivePtr<TComplectionQueue> cq = new TComplectionQueue(port->GetCtx(), MAX_CQ_EVENTS);

        TIBBufferPool bp(port->GetCtx(), MAX_SRQ_WORK_REQUESTS);

        if (!isClient) {
            // server
            TIPSocket ipSocket;
            ipSocket.Init(IP_PORT);
            if (!ipSocket.IsValid()) {
                printf("UDP port %d is not available\n", IP_PORT);
                return;
            }

            TIntrusivePtr<TComplectionQueue> cqRC = new TComplectionQueue(port->GetCtx(), MAX_CQ_EVENTS);

            TIntrusivePtr<TUDQueuePair> welcomeQP = new TUDQueuePair(port, cq, bp.GetSRQ(), QP_SEND_QUEUE_SIZE);
            welcomeQP->Init(WELCOME_QKEY);

            TWelcomeSocketAddr info;
            info.LID = port->GetLID();
            info.QPN = welcomeQP->GetQPN();

            for (;;) {
                ipSocket.Respond(info);
                // poll srq
                ibv_wc wc;
                if (cq->Poll(&wc, 1) == 1 && (wc.opcode & IBV_WC_RECV)) {
                    printf("Got IB handshake\n");

                    TRCQueuePairHandshake remoteHandshake;
                    ibv_ah_attr clientAddr;
                    {
                        TIBRecvPacketProcess pkt(bp, wc);
                        remoteHandshake = *(TRCQueuePairHandshake*)pkt.GetUDData();
                        port->GetAHAttr(&wc, pkt.GetGRH(), &clientAddr);
                    }

                    TIntrusivePtr<TAddressHandle> ahPeer;
                    ahPeer = new TAddressHandle(port->GetCtx(), &clientAddr);

                    TIntrusivePtr<TRCQueuePair> rcTest = new TRCQueuePair(port->GetCtx(), cqRC, bp.GetSRQ(), QP_SEND_QUEUE_SIZE);
                    rcTest->Init(clientAddr, remoteHandshake.QPN, remoteHandshake.PSN);

                    TRCQueuePairHandshake handshake;
                    handshake.PSN = rcTest->GetPSN();
                    handshake.QPN = rcTest->GetQPN();
                    bp.PostSend(welcomeQP, ahPeer, wc.src_qp, WELCOME_QKEY, &handshake, sizeof(handshake));

                    WaitForRecv(&bp, cqRC, &wc);

                    {
                        TIBRecvPacketProcess pkt(bp, wc);
                        printf("Got RC ping: %s\n", pkt.GetData());
                        const char* ret = "Puk";
                        bp.PostSend(rcTest, ret, strlen(ret) + 1);
                    }

                    for (int i = 0; i < 5; ++i) {
                        WaitForRecv(&bp, cqRC, &wc);
                        TIBRecvPacketProcess pkt(bp, wc);
                        printf("Got RC ping: %s\n", pkt.GetData());
                        const char* ret = "Fine";
                        bp.PostSend(rcTest, ret, strlen(ret) + 1);
                    }
                }
            }
        } else {
            // client
            ibv_wc wc;

            TIPSocket ipSocket;
            ipSocket.Init(0);
            if (!ipSocket.IsValid()) {
                printf("Failed to create UDP socket\n");
                return;
            }

            printf("Connecting to %s\n", serverName);
            TWelcomeSocketAddr info;
            ipSocket.Request(serverName, IP_PORT, &info);
            printf("Got welcome info, lid %d, qpn %d\n", info.LID, info.QPN);

            TIntrusivePtr<TUDQueuePair> welcomeQP = new TUDQueuePair(port, cq, bp.GetSRQ(), QP_SEND_QUEUE_SIZE);
            welcomeQP->Init(WELCOME_QKEY);

            TIntrusivePtr<TRCQueuePair> rcTest = new TRCQueuePair(port->GetCtx(), cq, bp.GetSRQ(), QP_SEND_QUEUE_SIZE);

            TRCQueuePairHandshake handshake;
            handshake.PSN = rcTest->GetPSN();
            handshake.QPN = rcTest->GetQPN();
            TIntrusivePtr<TAddressHandle> serverAH = new TAddressHandle(port, info.LID, 0);
            bp.PostSend(welcomeQP, serverAH, info.QPN, WELCOME_QKEY, &handshake, sizeof(handshake));

            WaitForRecv(&bp, cq, &wc);

            ibv_ah_attr serverAddr;
            TRCQueuePairHandshake remoteHandshake;
            {
                TIBRecvPacketProcess pkt(bp, wc);
                printf("Got handshake response\n");
                remoteHandshake = *(TRCQueuePairHandshake*)pkt.GetUDData();
                port->GetAHAttr(&wc, pkt.GetGRH(), &serverAddr);
            }

            rcTest->Init(serverAddr, remoteHandshake.QPN, remoteHandshake.PSN);

            char hiAndy[] = "Hi, Andy";
            bp.PostSend(rcTest, hiAndy, sizeof(hiAndy));
            WaitForRecv(&bp, cq, &wc);
            {
                TIBRecvPacketProcess pkt(bp, wc);
                printf("Got RC pong: %s\n", pkt.GetData());
            }

            for (int i = 0; i < 5; ++i) {
                char howAreYou[] = "How are you?";
                bp.PostSend(rcTest, howAreYou, sizeof(howAreYou));

                WaitForRecv(&bp, cq, &wc);
                {
                    TIBRecvPacketProcess pkt(bp, wc);
                    printf("Got RC pong: %s\n", pkt.GetData());
                }
            }
        }
    }
}
