#pragma once

#include "net_test.h"
#include "net_queue_stat.h"

#include <util/system/spinlock.h>

namespace NNetliba {
    const float MIN_PACKET_RTT_SKO = 0.001f; // avoid drops due to small hiccups

    const float CONG_CTRL_INITIAL_RTT = 0.24f; //0.01f; // taking into account Las Vegas 10ms estimate is too optimistic

    const float CONG_CTRL_WINDOW_GROW = 0.005f;
    const float CONG_CTRL_WINDOW_SHRINK = 0.9f;
    const float CONG_CTRL_WINDOW_SHRINK_RTT = 0.95f;
    const float CONG_CTRL_RTT_MIX_RATE = 0.9f;
    const int CONG_CTRL_RTT_SEQ_COUNT = 8;
    const float CONG_CTRL_MIN_WINDOW = 0.01f;
    const float CONG_CTRL_LARGE_TIME_WINDOW = 10000.0f;
    const float CONG_CTRL_TIME_WINDOW_LIMIT_PERIOD = 0.4f; // in seconds
    const float CONG_CTRL_MINIMAL_SEND_INTERVAL = 1;
    const float CONG_CTRL_MIN_FAIL_INTERVAL = 0.001f;
    const float CONG_CTRL_ALLOWED_BURST_SIZE = 3;
    const float CONG_CTRL_MIN_RTT_FOR_BURST_REDUCTION = 0.002f;

    const float LAME_MTU_TIMEOUT = 0.3f;
    const float LAME_MTU_INTERVAL = 0.05f;

    const float START_CHECK_PORT_DELAY = 0.5;
    const float FINISH_CHECK_PORT_DELAY = 10;
    const int N_PORT_TEST_COUNT_LIMIT = 256; // or 512

    // if enabled all acks are sent with different TOS, so they end up in different queue
    // this allows us to limit window based on minimal RTT observed and 1G link assumption
    extern bool UseTOSforAcks;

    class TPingTracker {
        float AvrgRTT, AvrgRTT2; // RTT statistics
        float RTTCount;

    public:
        TPingTracker();
        float GetRTT() const {
            return AvrgRTT;
        }
        float GetRTTSKO() const {
            float sko = sqrt(fabs(Sqr(AvrgRTT) - AvrgRTT2));
            float minSKO = Max(MIN_PACKET_RTT_SKO, AvrgRTT * 0.05f);
            return Max(minSKO, sko);
        }
        float GetTimeout() const {
            return GetRTT() + GetRTTSKO() * 3;
        }
        void RegisterRTT(float rtt);
        void IncreaseRTT();
    };

    ui32 NetAckRnd();

    class TLameMTUDiscovery: public TThrRefBase {
        enum EState {
            NEED_PING,
            WAIT,
        };

        float TimePassed, TimeSinceLastPing;
        EState State;

    public:
        TLameMTUDiscovery()
            : TimePassed(0)
            , TimeSinceLastPing(0)
            , State(NEED_PING)
        {
        }
        bool CanSend() {
            return State == NEED_PING;
        }
        void PingSent() {
            State = WAIT;
            TimeSinceLastPing = 0;
        }
        bool IsTimedOut() const {
            return TimePassed > LAME_MTU_TIMEOUT;
        }
        void Step(float deltaT) {
            TimePassed += deltaT;
            TimeSinceLastPing += deltaT;
            if (TimeSinceLastPing > LAME_MTU_INTERVAL)
                State = NEED_PING;
        }
    };

    struct TPeerQueueStats: public IPeerQueueStats {
        int Count;

        TPeerQueueStats()
            : Count(0)
        {
        }
        int GetPacketCount() override {
            return Count;
        }
    };

    // pretend we have multiple channels in parallel
    // not exact approximation since N channels should have N distinct windows
    extern float CONG_CTRL_CHANNEL_INFLATE;

    class TCongestionControl: public TThrRefBase {
        float Window, PacketsInFly, FailRate;
        float MinRTT, MaxWindow;
        bool FullSpeed, DoCountTime;
        TPingTracker PingTracker;
        double TimeSinceLastRecv;
        TAdaptiveLock PortTesterLock;
        TIntrusivePtr<TPortUnreachableTester> PortTester;
        int ActiveTransferCount;
        float AvrgRTT;
        int HighRTTCounter;
        float WindowFraction, FractionRecalc;
        float TimeWindow;
        double TimeSinceLastFail;
        float VirtualPackets;
        int MTU;
        TIntrusivePtr<TLameMTUDiscovery> MTUDiscovery;
        TIntrusivePtr<TPeerQueueStats> QueueStats;

        void CalcMaxWindow() {
            if (MTU == 0)
                return;
            MaxWindow = 125000000 / MTU * Max(0.001f, MinRTT);
        }

    public:
        static float StartWindowSize, MaxPacketRate;

    public:
        TCongestionControl()
            : Window(StartWindowSize * CONG_CTRL_CHANNEL_INFLATE)
            , PacketsInFly(0)
            , FailRate(0)
            , MinRTT(10)
            , MaxWindow(10000)
            , FullSpeed(false)
            , DoCountTime(false)
            , TimeSinceLastRecv(0)
            , ActiveTransferCount(0)
            , AvrgRTT(0)
            , HighRTTCounter(0)
            , WindowFraction(0)
            , FractionRecalc(0)
            , TimeWindow(CONG_CTRL_LARGE_TIME_WINDOW)
            , TimeSinceLastFail(0)
            , MTU(0)
        {
            VirtualPackets = Max(Window - CONG_CTRL_ALLOWED_BURST_SIZE, 0.f);
        }
        bool CanSend() {
            bool res = VirtualPackets + PacketsInFly + WindowFraction <= Window;
            FullSpeed |= !res;
            res &= TimeWindow > 0;
            return res;
        }
        void LaunchPacket() {
            PacketsInFly += 1.0f;
            TimeWindow -= 1.0f;
        }
        void RegisterRTT(float RTT) {
            if (RTT < 0)
                return;
            RTT = ClampVal(RTT, 0.0001f, 1.0f);
            if (RTT < MinRTT && MTU != 0) {
                MinRTT = RTT;
                CalcMaxWindow();
            }
            MinRTT = Min(MinRTT, RTT);

            PingTracker.RegisterRTT(RTT);
            if (AvrgRTT == 0)
                AvrgRTT = RTT;
            if (RTT > AvrgRTT) {
                ++HighRTTCounter;
                if (HighRTTCounter >= CONG_CTRL_RTT_SEQ_COUNT) {
                    //printf("Too many high RTT in a row\n");
                    if (FullSpeed) {
                        float windowSubtract = Window * ((1 - CONG_CTRL_WINDOW_SHRINK_RTT) / CONG_CTRL_CHANNEL_INFLATE);
                        Window = Max(CONG_CTRL_MIN_WINDOW, Window - windowSubtract);
                        VirtualPackets = Max(0.f, VirtualPackets - windowSubtract);
                        //printf("reducing window by RTT , new window %g\n", Window);
                    }
                    // reduce no more then twice per RTT
                    HighRTTCounter = Min(0, CONG_CTRL_RTT_SEQ_COUNT - (int)(Window * 0.5));
                }
            } else {
                HighRTTCounter = Min(0, HighRTTCounter);
            }

            float rttMixRate = CONG_CTRL_RTT_MIX_RATE;
            AvrgRTT = AvrgRTT * rttMixRate + RTT * (1 - rttMixRate);
        }
        void Success() {
            PacketsInFly -= 1;
            Y_ASSERT(PacketsInFly >= 0);
            // FullSpeed should be correct at this point
            // we assume that after UpdateAlive() we send all packets first then we listen for acks and call Success()
            // FullSpeed is set in CanSend() during send if we are using full window
            // do not increaese window while send rate is limited by virtual packets (ie start of transfer)
            if (FullSpeed && VirtualPackets == 0) {
                // there are 2 requirements for window growth
                // 1) growth should be proportional to window size to ensure constant FailRate
                // 2) growth should be constant to ensure fairness among different flows
                // so lets make it square root :)
                Window += sqrt(Window / CONG_CTRL_CHANNEL_INFLATE) * CONG_CTRL_WINDOW_GROW;
                if (UseTOSforAcks) {
                    Window = Min(Window, MaxWindow);
                }
            }
            FailRate *= 0.99f;
        }
        void FailureOnSend() {
            //printf("Failure on send\n");
            PacketsInFly -= 1;
            Y_ASSERT(PacketsInFly >= 0);
            // not a congestion event, do not modify Window
            // do not set FullSpeed since we are not using full Window
        }
        void Failure() {
            //printf("Congestion failure\n");
            PacketsInFly -= 1;
            Y_ASSERT(PacketsInFly >= 0);
            // account limited number of fails per segment
            if (TimeSinceLastFail > CONG_CTRL_MIN_FAIL_INTERVAL) {
                TimeSinceLastFail = 0;
                if (Window <= CONG_CTRL_MIN_WINDOW) {
                    // ping dead hosts less frequently
                    if (PingTracker.GetRTT() / CONG_CTRL_MIN_WINDOW < CONG_CTRL_MINIMAL_SEND_INTERVAL)
                        PingTracker.IncreaseRTT();
                    Window = CONG_CTRL_MIN_WINDOW;
                    VirtualPackets = 0;
                } else {
                    float windowSubtract = Window * ((1 - CONG_CTRL_WINDOW_SHRINK) / CONG_CTRL_CHANNEL_INFLATE);
                    Window = Max(CONG_CTRL_MIN_WINDOW, Window - windowSubtract);
                    VirtualPackets = Max(0.f, VirtualPackets - windowSubtract);
                }
            }
            FailRate = FailRate * 0.99f + 0.01f;
        }
        bool HasPacketsInFly() const {
            return PacketsInFly > 0;
        }
        float GetTimeout() const {
            return PingTracker.GetTimeout();
        }
        float GetWindow() const {
            return Window;
        }
        float GetRTT() const {
            return PingTracker.GetRTT();
        }
        float GetFailRate() const {
            return FailRate;
        }
        float GetTimeSinceLastRecv() const {
            return TimeSinceLastRecv;
        }
        int GetTransferCount() const {
            return ActiveTransferCount;
        }
        float GetMaxWindow() const {
            return UseTOSforAcks ? MaxWindow : -1;
        }
        void MarkAlive() {
            TimeSinceLastRecv = 0;

            with_lock (PortTesterLock) {
                PortTester = nullptr;
            }

        }
        void HasTriedToSend() {
            DoCountTime = true;
        }
        bool IsAlive() const {
            return TimeSinceLastRecv < 1e6f;
        }
        void Kill() {
            TimeSinceLastRecv = 1e6f;
        }
        bool UpdateAlive(const TUdpAddress& toAddress, float deltaT, float timeout, float* resMaxWaitTime) {
            if (!FullSpeed) {
                // create virtual packets during idle to avoid burst on transmit start
                if (AvrgRTT > CONG_CTRL_MIN_RTT_FOR_BURST_REDUCTION) {
                    VirtualPackets = Max(VirtualPackets, Window - PacketsInFly - CONG_CTRL_ALLOWED_BURST_SIZE);
                }
            } else {
                if (VirtualPackets > 0) {
                    if (Window <= CONG_CTRL_ALLOWED_BURST_SIZE) {
                        VirtualPackets = 0;
                    }
                    float xRTT = AvrgRTT == 0 ? CONG_CTRL_INITIAL_RTT : AvrgRTT;
                    float virtualPktsPerSecond = Window / xRTT;
                    VirtualPackets = Max(0.f, VirtualPackets - deltaT * virtualPktsPerSecond);
                    *resMaxWaitTime = Min(*resMaxWaitTime, 0.001f); // need to update virtual packets counter regularly
                }
            }
            float currentRTT = GetRTT();
            FractionRecalc += deltaT;
            if (FractionRecalc > currentRTT) {
                int cycleCount = (int)(FractionRecalc / currentRTT);
                FractionRecalc -= currentRTT * cycleCount;
                WindowFraction = (NetAckRnd() & 1023) * (1 / 1023.0f) / cycleCount;
            }

            if (MaxPacketRate > 0 && AvrgRTT > 0) {
                float maxTimeWindow = CONG_CTRL_TIME_WINDOW_LIMIT_PERIOD * MaxPacketRate;
                TimeWindow = Min(maxTimeWindow, TimeWindow + MaxPacketRate * deltaT);
            } else
                TimeWindow = CONG_CTRL_LARGE_TIME_WINDOW;

            // guarantee minimal send rate
            if (currentRTT > CONG_CTRL_MINIMAL_SEND_INTERVAL * Window) {
                Window = Max(CONG_CTRL_MIN_WINDOW, currentRTT / CONG_CTRL_MINIMAL_SEND_INTERVAL);
                VirtualPackets = 0;
            }

            TimeSinceLastFail += deltaT;

            //static int n;
            //if ((++n & 127) == 0)
            //    printf("window = %g, fly = %g, VirtualPkts = %g, deltaT = %g, FailRate = %g FullSpeed = %d AvrgRTT = %g\n",
            //        Window, PacketsInFly, VirtualPackets, deltaT * 1000, FailRate, (int)FullSpeed, AvrgRTT * 1000);

            if (PacketsInFly > 0 || FullSpeed || DoCountTime) {
                // считаем время только когда есть пакеты в полете
                TimeSinceLastRecv += deltaT;
                if (TimeSinceLastRecv > START_CHECK_PORT_DELAY) {
                    if (TimeSinceLastRecv < FINISH_CHECK_PORT_DELAY) {
                        TIntrusivePtr<TPortUnreachableTester> portTester;
                        with_lock (PortTesterLock) {
                            portTester = PortTester;
                        }

                        if (!portTester && AtomicGet(ActivePortTestersCount) < N_PORT_TEST_COUNT_LIMIT) {
                            portTester = new TPortUnreachableTester();
                            with_lock (PortTesterLock) {
                                PortTester = portTester;
                            }

                            if (portTester->IsValid()) {
                                portTester->Connect(toAddress);
                            } else {
                                with_lock (PortTesterLock) {
                                    PortTester = nullptr;
                                }
                            }
                        }
                        if (portTester && !portTester->Test(deltaT)) {
                            Kill();
                            return false;
                        }
                    } else {
                        with_lock (PortTesterLock) {
                            PortTester = nullptr;
                        }
                    }
                }
                if (TimeSinceLastRecv > timeout) {
                    Kill();
                    return false;
                }
            }

            FullSpeed = false;
            DoCountTime = false;

            if (MTUDiscovery.Get())
                MTUDiscovery->Step(deltaT);

            return true;
        }
        bool IsKnownMTU() const {
            return MTU != 0;
        }
        int GetMTU() const {
            return MTU;
        }
        TLameMTUDiscovery* GetMTUDiscovery() {
            if (MTUDiscovery.Get() == nullptr)
                MTUDiscovery = new TLameMTUDiscovery;
            return MTUDiscovery.Get();
        }
        void SetMTU(int sz) {
            MTU = sz;
            MTUDiscovery = nullptr;
            CalcMaxWindow();
        }
        void AttachQueueStats(TIntrusivePtr<TPeerQueueStats> s) {
            if (s.Get()) {
                s->Count = ActiveTransferCount;
            }
            Y_ASSERT(QueueStats.Get() == nullptr);
            QueueStats = s;
        }
        friend class TCongestionControlPtr;
    };

    class TCongestionControlPtr {
        TIntrusivePtr<TCongestionControl> Ptr;

        void Inc() {
            if (Ptr.Get()) {
                ++Ptr->ActiveTransferCount;
                if (Ptr->QueueStats.Get()) {
                    Ptr->QueueStats->Count = Ptr->ActiveTransferCount;
                }
            }
        }
        void Dec() {
            if (Ptr.Get()) {
                --Ptr->ActiveTransferCount;
                if (Ptr->QueueStats.Get()) {
                    Ptr->QueueStats->Count = Ptr->ActiveTransferCount;
                }
            }
        }

    public:
        TCongestionControlPtr() {
        }
        ~TCongestionControlPtr() {
            Dec();
        }
        TCongestionControlPtr(TCongestionControl* p)
            : Ptr(p)
        {
            Inc();
        }
        TCongestionControlPtr& operator=(const TCongestionControlPtr& a) {
            Dec();
            Ptr = a.Ptr;
            Inc();
            return *this;
        }
        TCongestionControlPtr& operator=(TCongestionControl* a) {
            Dec();
            Ptr = a;
            Inc();
            return *this;
        }
        operator TCongestionControl*() const {
            return Ptr.Get();
        }
        TCongestionControl* operator->() const {
            return Ptr.Get();
        }
        TIntrusivePtr<TCongestionControl> Get() const {
            return Ptr;
        }
    };

    class TAckTracker {
        struct TFlyingPacket {
            float T;
            int PktId;
            TFlyingPacket()
                : T(0)
                , PktId(-1)
            {
            }
            TFlyingPacket(float t, int pktId)
                : T(t)
                , PktId(pktId)
            {
            }
        };
        int PacketCount, CurrentPacket;
        typedef THashMap<int, float> TPacketHash;
        TPacketHash PacketsInFly, DroppedPackets;
        TVector<int> ResendQueue;
        TCongestionControlPtr Congestion;
        TVector<bool> AckReceived;
        float TimeToNextPacketTimeout;

        int SelectPacket();

    public:
        TAckTracker()
            : PacketCount(0)
            , CurrentPacket(0)
            , TimeToNextPacketTimeout(1000)
        {
        }
        ~TAckTracker();
        void AttachCongestionControl(TCongestionControl* p) {
            Congestion = p;
        }
        TIntrusivePtr<TCongestionControl> GetCongestionControl() const {
            return Congestion.Get();
        }
        void SetPacketCount(int n) {
            Y_ASSERT(PacketCount == 0);
            PacketCount = n;
            AckReceived.resize(n, false);
        }
        void Resend();
        bool IsInitialized() {
            return PacketCount != 0;
        }
        int GetPacketToSend(float deltaT);
        void AddToResend(int pkt); // called when failed to send packet
        void Ack(int pkt, float deltaT, bool updateRTT);
        void AckAll();
        void MarkAlive() {
            Congestion->MarkAlive();
        }
        bool IsAlive() const {
            return Congestion->IsAlive();
        }
        void Step(float deltaT);
        bool CanSend() const {
            return Congestion->CanSend();
        }
        float GetTimeToNextPacketTimeout() const {
            return TimeToNextPacketTimeout;
        }
    };
}
