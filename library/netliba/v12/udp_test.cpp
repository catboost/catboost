#include "stdafx.h"
#include "udp_test.h"
#include "udp_host.h"
#include "udp_http.h"
#include "cpu_affinity.h"

#include <util/system/hp_timer.h>
#include <util/datetime/cputimer.h>
#include <util/random/random.h>
#include <util/random/fast.h>

namespace NNetliba_v12 {
    //static void PacketLevelTest(bool client)
    //{
    //    int port = client ? 0 : 13013;
    //    TIntrusivePtr<IUdpHost> host = CreateUdpHost(&port);
    //
    //    if(host == 0) {
    //        exit(-1);
    //    }
    //    TUdpAddress serverAddr = CreateAddress("localhost", 13013);
    //    vector<char> dummyPacket;
    //    dummyPacket.resize(10000);
    //    srand(GetCycleCount());
    //
    //    for (int i = 0; i < dummyPacket.size(); ++i)
    //        dummyPacket[i] = rand();
    //    bool cont = true, hasReply = true;
    //    int reqCount = 1;
    //    for (int i = 0; cont; ++i) {
    //        host->Step();
    //        if (client) {
    //            //while (host->HasPendingData(serverAddr))
    //            //    Sleep(0);
    //            if (hasReply) {
    //                printf("request %d\n", reqCount);
    //                *(int*)&dummyPacket[0] = reqCount;
    //                host->Send(serverAddr, dummyPacket, 0, PP_NORMAL);
    //                hasReply = false;
    //                ++reqCount;
    //            }else
    //                sleep(0);
    //
    //            TRequest *req;
    //            while (req = host->GetRequest()) {
    //                int n = *(int*)&req->Data[0];
    //                printf("received response %d\n", n);
    //                Y_ASSERT(memcmp(&req->Data[4], &dummyPacket[4], dummyPacket.size() - 4) == 0);
    //                delete req;
    //                hasReply = true;
    //            }
    //            TSendResult sr;
    //            while (host->GetSendResult(&sr)) {
    //                if (!sr.Success) {
    //                    printf("Send failed!\n");
    //                    //Sleep(INFINITE);
    //                    hasReply = true;
    //                }
    //            }
    //        } else {
    //            while (TRequest *req = host->GetRequest()) {
    //                int n = *(int*)&req->Data[0];
    //                printf("responding %d\n", n);
    //                host->Send(req->Address, req->Data, 0, PP_NORMAL);
    //                delete req;
    //            }
    //            TSendResult sr;
    //            while (host->GetSendResult(&sr)) {
    //                if (!sr.Success) {
    //                    printf("Send failed!\n");
    //                    sleep(0);
    //                }
    //            }
    //            sleep(0);
    //        }
    //    }
    //}

    static void SessionLevelTest(bool client, const char* serverName, int packetSize, int packetsInFly, int srcPort) {
        BindToSocket(0);
        TIntrusivePtr<IRequester> reqHost;
        //    reqHost = CreateHttpUdpRequester(13013);
        const int port = client ? srcPort : 13013;
        reqHost = CreateHttpUdpRequester(port);
        if (!reqHost) {
            fprintf(stderr, "netliba failed to create http udp requester on port %d\n", port);
            exit(EXIT_FAILURE);
        }

        TUdpAddress serverAddr = CreateAddress(serverName, 13013);
        TVector<char> dummyPacket;
        dummyPacket.resize(packetSize);
        TReallyFastRng32 rr(RandomNumber<ui64>());
        for (size_t i = 0; i < dummyPacket.size(); ++i)
            dummyPacket[i] = (char)rr.Uniform(256);
        bool cont = true;
        NHPTimer::STime t;
        NHPTimer::GetTime(&t);
        THashMap<TGUID, bool, TGUIDHash> seenReqs;
        if (client) {
            THashMap<TGUID, bool, TGUIDHash> reqList;
            int packetsSentCount = 0;
            TUdpHttpRequest* req;
            for (int i = 1; cont; ++i) {
                for (;;) {
                    req = reqHost->GetRequest();
                    if (req == nullptr)
                        break;
                    req->Data.resize(10);
                    reqHost->SendResponse(req->ReqId, &req->Data, req->Colors);
                    delete req;
                }
                while (TUdpHttpResponse* res = reqHost->GetResponse()) {
                    THashMap<TGUID, bool, TGUIDHash>::iterator z = reqList.find(res->ReqId);
                    if (z == reqList.end()) {
                        printf("Unexpected response\n");
                        abort();
                    }
                    reqList.erase(z);
                    if (res->Ok) {
                        ++packetsSentCount;
                        //Y_ASSERT(res->Data == dummyPacket);
                        NHPTimer::STime tChk = t;
                        if (NHPTimer::GetTimePassed(&tChk) > 1) {
                            printf("packet size = %d\n", dummyPacket.ysize());
                            double passedTime = NHPTimer::GetTimePassed(&t);
                            double rate = packetsSentCount / passedTime;
                            printf("packet rate %g, transfer %gmb\n", rate, rate * dummyPacket.size() / 1000000);
                            packetsSentCount = 0;
                        }
                    } else {
                        printf("Failed request!\n");
                        //Sleep(INFINITE);
                    }
                    delete res;
                }
                while (reqList.ysize() < packetsInFly) {
                    *(int*)&dummyPacket[0] = i;
                    TVector<char> fakePacket = dummyPacket;
                    TGUID req2 = reqHost->SendRequest(TConnectionAddress(serverAddr), "blaxuz", &fakePacket);
                    reqList[req2];
                }
                reqHost->GetAsyncEvent().Wait();
            }
        } else {
            TUdpHttpRequest* req;
            for (;;) {
                req = reqHost->GetRequest();
                if (req) {
                    if (seenReqs.find(req->ReqId) != seenReqs.end()) {
                        printf("Request %s recieved twice!\n", GetGuidAsString(req->ReqId).c_str());
                    }
                    seenReqs[req->ReqId];
                    req->Data.resize(10);
                    reqHost->SendResponse(req->ReqId, &req->Data, req->Colors);
                    delete req;
                } else {
                    reqHost->GetAsyncEvent().Wait();
                }
            }
        }
    }

    void RunUdpTest(bool client, const char* serverName, int packetSize, int packetsInFly, int srcPort) {
        //PacketLevelTest(client);
        SessionLevelTest(client, serverName, packetSize, packetsInFly, srcPort);
    }
}
