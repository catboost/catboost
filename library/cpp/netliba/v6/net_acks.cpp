#include "stdafx.h"
#include "net_acks.h"
#include <util/datetime/cputimer.h>

#include <atomic>

namespace NNetliba {
    const float RTT_AVERAGE_OVER = 15;

    float TCongestionControl::StartWindowSize = 3;
    float TCongestionControl::MaxPacketRate = 0; // unlimited

    bool UseTOSforAcks = false; //true;//

    void EnableUseTOSforAcks(bool enable) {
        UseTOSforAcks = enable;
    }

    float CONG_CTRL_CHANNEL_INFLATE = 1;

    void SetCongCtrlChannelInflate(float inflate) {
        CONG_CTRL_CHANNEL_INFLATE = inflate;
    }

    //////////////////////////////////////////////////////////////////////////
    TPingTracker::TPingTracker()
        : AvrgRTT(CONG_CTRL_INITIAL_RTT)
        , AvrgRTT2(CONG_CTRL_INITIAL_RTT * CONG_CTRL_INITIAL_RTT)
        , RTTCount(0)
    {
    }

    void TPingTracker::RegisterRTT(float rtt) {
        Y_ASSERT(rtt > 0);
        float keep = RTTCount / (RTTCount + 1);
        AvrgRTT *= keep;
        AvrgRTT += (1 - keep) * rtt;
        AvrgRTT2 *= keep;
        AvrgRTT2 += (1 - keep) * Sqr(rtt);
        RTTCount = Min(RTTCount + 1, RTT_AVERAGE_OVER);
        //static int n;
        //if ((++n % 1024) == 0)
        //    printf("Average RTT = %g (sko = %g)\n", GetRTT() * 1000, GetRTTSKO() * 1000);
    }

    void TPingTracker::IncreaseRTT() {
        const float F_RTT_DECAY_RATE = 1.1f;
        AvrgRTT *= F_RTT_DECAY_RATE;
        AvrgRTT2 *= Sqr(F_RTT_DECAY_RATE);
    }

    //////////////////////////////////////////////////////////////////////////
    void TAckTracker::Resend() {
        CurrentPacket = 0;
        for (TPacketHash::const_iterator i = PacketsInFly.begin(); i != PacketsInFly.end(); ++i)
            Congestion->Failure(); // not actually correct but simplifies logic a lot
        PacketsInFly.clear();
        DroppedPackets.clear();
        ResendQueue.clear();
        for (size_t i = 0; i < AckReceived.size(); ++i)
            AckReceived[i] = false;
    }

    int TAckTracker::SelectPacket() {
        if (!ResendQueue.empty()) {
            int res = ResendQueue.back();
            ResendQueue.pop_back();
            //printf("resending packet %d\n", res);
            return res;
        }
        if (CurrentPacket == PacketCount) {
            return -1;
        }
        return CurrentPacket++;
    }

    TAckTracker::~TAckTracker() {
        for (TPacketHash::const_iterator i = PacketsInFly.begin(); i != PacketsInFly.end(); ++i)
            Congestion->Failure();
        // object will be incorrect state after this (failed packets are not added to resend queue), but who cares
    }

    int TAckTracker::GetPacketToSend(float deltaT) {
        int res = SelectPacket();
        if (res == -1) {
            // needed to count time even if we don't have anything to send
            Congestion->HasTriedToSend();
            return res;
        }
        Congestion->LaunchPacket();
        PacketsInFly[res] = -deltaT; // deltaT is time since last Step(), so for the timing to be correct we should subtract it
        return res;
    }

    // called on SendTo() failure
    void TAckTracker::AddToResend(int pkt) {
        //printf("AddToResend(%d)\n", pkt);
        TPacketHash::iterator i = PacketsInFly.find(pkt);
        if (i != PacketsInFly.end()) {
            PacketsInFly.erase(i);
            Congestion->FailureOnSend();
            ResendQueue.push_back(pkt);
        } else
            Y_ASSERT(0);
    }

    void TAckTracker::Ack(int pkt, float deltaT, bool updateRTT) {
        Y_ASSERT(pkt >= 0 && pkt < PacketCount);
        if (AckReceived[pkt])
            return;
        AckReceived[pkt] = true;
        //printf("Ack received for %d\n", pkt);
        TPacketHash::iterator i = PacketsInFly.find(pkt);
        if (i == PacketsInFly.end()) {
            for (size_t k = 0; k < ResendQueue.size(); ++k) {
                if (ResendQueue[k] == pkt) {
                    ResendQueue[k] = ResendQueue.back();
                    ResendQueue.pop_back();
                    break;
                }
            }
            TPacketHash::iterator z = DroppedPackets.find(pkt);
            if (z != DroppedPackets.end()) {
                // late packet arrived
                if (updateRTT) {
                    float ping = z->second + deltaT;
                    Congestion->RegisterRTT(ping);
                }
                DroppedPackets.erase(z);
            } else {
                // Y_ASSERT(0); // ack on nonsent packet, possible in resend scenario
            }
            return;
        }
        if (updateRTT) {
            float ping = i->second + deltaT;
            //printf("Register RTT %g\n", ping * 1000);
            Congestion->RegisterRTT(ping);
        }
        PacketsInFly.erase(i);
        Congestion->Success();
    }

    void TAckTracker::AckAll() {
        for (TPacketHash::const_iterator i = PacketsInFly.begin(); i != PacketsInFly.end(); ++i) {
            int pkt = i->first;
            AckReceived[pkt] = true;
            Congestion->Success();
        }
        PacketsInFly.clear();
    }

    void TAckTracker::Step(float deltaT) {
        float timeoutVal = Congestion->GetTimeout();

        //static int n;
        //if ((++n % 1024) == 0)
        //    printf("timeout = %g, window = %g, fail_rate %g, pkt_rate = %g\n", timeoutVal * 1000, Congestion->GetWindow(), Congestion->GetFailRate(), (1 - Congestion->GetFailRate()) * Congestion->GetWindow() / Congestion->GetRTT());

        TimeToNextPacketTimeout = 1000;
        // для окон меньше единицы мы кидаем рандом один раз за RTT на то, можно ли пускать пакет
        // поэтому можно ждать максимум RTT, после этого надо кинуть новый random
        if (Congestion->GetWindow() < 1)
            TimeToNextPacketTimeout = Congestion->GetRTT();

        for (auto& droppedPacket : DroppedPackets) {
            float& t = droppedPacket.second;
            t += deltaT;
        }

        for (TPacketHash::iterator i = PacketsInFly.begin(); i != PacketsInFly.end();) {
            float& t = i->second;
            t += deltaT;
            if (t > timeoutVal) {
                //printf("packet %d timed out (timeout = %g)\n", i->first, timeoutVal);
                ResendQueue.push_back(i->first);
                DroppedPackets[i->first] = i->second;
                TPacketHash::iterator k = i++;
                PacketsInFly.erase(k);
                Congestion->Failure();
            } else {
                TimeToNextPacketTimeout = Min(TimeToNextPacketTimeout, timeoutVal - t);
                ++i;
            }
        }
    }

    static std::atomic<ui32> netAckRndVal = (ui32)GetCycleCount();
    ui32 NetAckRnd() {
        const auto nextNetAckRndVal = static_cast<ui32>(((ui64)netAckRndVal.load(std::memory_order_acquire) * 279470273) % 4294967291);
        netAckRndVal.store(nextNetAckRndVal, std::memory_order_release);
        return nextNetAckRndVal;
    }
}
