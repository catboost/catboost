#pragma once

#include <util/datetime/base.h>
#include <util/generic/algorithm.h>
#include <util/generic/guid.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/intrlist.h>
#include <util/string/cast.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/shmat.h>
#include <util/system/yassert.h>
#include "block_chain.h"
#include "circular_pod_buffer.h"
#include "ib_cs.h"
#include "net_acks.h"
#include "settings.h"
#include "udp_address.h"
#include "udp_debug.h"
#include "udp_host.h"
#include "udp_host_recv_completed.h"
#include "udp_recv_packet.h"
#include "udp_socket.h"

#include <library/cpp/netliba/socket/allocator.h>

#include <array>

namespace NNetliba_v12 {
    constexpr int PACKET_HEADERS_SIZE = 128; // 128 bytes are enough for any header (100 is not enough for PONG_IB)
    constexpr int UDP_PACKET_SIZE = UDP_PACKET_BUF_SIZE - PACKET_HEADERS_SIZE;
    constexpr int UDP_SMALL_PACKET_SIZE = 1350; // 1180 would be better taking into account that 1280 is guaranteed ipv6 minimum MTU
    // NOTE (torkve) netliba had UDP_SMALL_PACKET_SIZE hardcoded for ages, and we cannot change this value, while preserving
    //               backward compatibility. Thus we have XS packet size which is turned on by an transfer option
    constexpr int UDP_XSMALL_PACKET_SIZE = 1180;

    ///////////////////////////////////////////////////////////////////////////////

    extern const float UDP_KEEP_CONNETION;

    ///////////////////////////////////////////////////////////////////////////////

    struct TUdpInTransfer: public TThrRefBase, public TWithCustomAllocator {
    private:
        TVector<TUdpRecvPacket*, TCustomAllocator<TUdpRecvPacket*>> Packets;

    public:
        size_t PacketSize, LastPacketSize;
        bool HasLastPacket;
        TVector<int, TCustomAllocator<int>> NewPacketsToAck;
        float TimeSinceLastRecv;
        ui8 AckTos;
        ui8 NetlibaColor;
        bool HighPriority;
        TIntrusivePtr<TPosixSharedMemory> SharedData;
        TIntrusivePtr<TRequesterPendingDataStats> Stats[3];
        size_t StatsEnd;

        TUdpInTransfer()
            : PacketSize(0)
            , LastPacketSize(0)
            , HasLastPacket(false)
            , TimeSinceLastRecv(0)
            , AckTos(0)
            , NetlibaColor(DEFAULT_NETLIBA_COLOR)
            , HighPriority(false)
            , StatsEnd(0)
        {
        }

        ~TUdpInTransfer() override {
            for (size_t i = 0; i != StatsEnd; ++i) {
                if (Stats[i].Get())
                    AddToInpCount(Stats[i].Get(), -1);
            }
            EraseAllPackets();
        }

        void AttachStats(TIntrusivePtr<TRequesterPendingDataStats> stats) {
            Y_ASSERT(stats.Get());
            Y_ASSERT(Find(Stats, Stats + StatsEnd, stats.Get()) == Stats + StatsEnd && "Duplicate stats attach!");
            Y_ABORT_UNLESS(StatsEnd < Y_ARRAY_SIZE(Stats), "Please increase Stats array size");

            Stats[StatsEnd++] = stats;
            AddToInpCount(stats.Get(), 1);

            Y_ASSERT(Packets.empty());
        }

        int GetPacketCount() const {
            return Packets.ysize();
        }
        void SetPacketCount(int n) {
            Packets.resize(n);
        }

        const TUdpRecvPacket* GetPacket(int id) const {
            Y_ASSERT(id < GetPacketCount());
            return Packets[id];
        }

        void AssignPacket(int id, TUdpRecvPacket* pkt) {
            ErasePacket(id);
            if (pkt) {
                Y_ASSERT(pkt->DataSize == (int)PacketSize || pkt->DataSize == (int)LastPacketSize);
                for (size_t i = 0; i != StatsEnd; ++i) {
                    AddToInpDataSize(Stats[i].Get(), pkt->DataSize);
                }
            }
            Packets[id] = pkt;
        }
        TUdpRecvPacket* ExtractPacket(int id) {
            TUdpRecvPacket* res = Packets[id];
            if (res) {
                for (size_t i = 0; i != StatsEnd; ++i) {
                    SubFromInpDataSize(Stats[i].Get(), res->DataSize);
                }

                Packets[id] = nullptr;

                if (HasLastPacket && id == GetPacketCount() - 1) {
                    HasLastPacket = false;
                }
            }
            return res;
        }

        void ErasePacket(int id) {
            delete ExtractPacket(id);
        }
        void EraseAllPackets() {
            for (int i = 0; i < Packets.ysize(); ++i) {
                ErasePacket(i);
            }
            Packets.clear();
            HasLastPacket = false;
        }

    private:
        void AddToInpCount(TRequesterPendingDataStats* stats, const int value) const {
            stats->InpCount += value;
        }
        void AddToInpDataSize(TRequesterPendingDataStats* stats, const ui64 value) const {
            stats->InpDataSize += value;
        }
        void SubFromInpDataSize(TRequesterPendingDataStats* stats, const ui64 value) const {
            stats->InpDataSize -= value;
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    struct TUdpOutTransfer: public TThrRefBase, public TWithCustomAllocator {
        TAutoPtr<TRopeDataPacket> Data;
        int PacketCount;
        int PacketSize, LastPacketSize;
        TAckTracker AckTracker;
        EPacketPriority PacketPriority;
        ui8 DataTos;
        ui8 AckTos;
        ui8 NetlibaColor;
        bool IBTriedToPing;
        bool TriedToSendAtLeastOnePacket;
        TIntrusivePtr<TRequesterPendingDataStats> Stats[3];
        size_t StatsEnd;
        NHPTimer::STime LastTime;

        TUdpOutTransfer()
            : PacketCount(0)
            , PacketSize(0)
            , LastPacketSize(0)
            , PacketPriority(PP_LOW)
            , DataTos(0)
            , AckTos(0)
            , NetlibaColor(DEFAULT_NETLIBA_COLOR)
            , IBTriedToPing(false)
            , TriedToSendAtLeastOnePacket(false)
            , StatsEnd(0)
        {
            NHPTimer::GetTime(&LastTime);
        }

        ~TUdpOutTransfer() override {
            for (size_t i = 0; i != StatsEnd; ++i) {
                if (Stats[i].Get())
                    DecStats(Stats[i].Get());
            }
        }
        void AttachStats(TIntrusivePtr<TRequesterPendingDataStats> stats) {
            Y_ASSERT(stats.Get());
            Y_ASSERT(Find(Stats, Stats + StatsEnd, stats.Get()) == Stats + StatsEnd && "Duplicate stats attach!");
            Y_ABORT_UNLESS(StatsEnd < Y_ARRAY_SIZE(Stats), "Please increase Stats array size");

            Stats[StatsEnd++] = stats;
            IncStats(stats.Get());
        }
        void InitXfer() {
            PacketSize = AckTracker.GetCongestionControl()->GetMTU();
            //fprintf(stderr, "MTU is: %i\n", (int)xfer.PacketSize);
            /*
            Cerr << "InitXfer AckTracker.GetCongestionControl()->GetMTU(): "
                << ui64(&AckTracker) << "." << ui64(AckTracker.GetCongestionControl().Get())
                << " = " << PacketSize << Endl;
            */
            LastPacketSize = Data->GetSize() % PacketSize;
            PacketCount = Data->GetSize() / PacketSize + 1;
            AckTracker.SetPacketCount(PacketCount);
            TriedToSendAtLeastOnePacket = false;
        }

    private:
        void IncStats(TRequesterPendingDataStats* stats) const {
            stats->OutCount++;
            stats->OutDataSize += Data->GetSize();
        }
        void DecStats(TRequesterPendingDataStats* stats) const {
            stats->OutCount--;
            stats->OutDataSize -= Data->GetSize();
        }
    };

    //////////////////////////////////////////////////////////////////////////

    template <class T>
    class TTransfers {
    public:
        class TIdIterator: public std::iterator<std::output_iterator_tag, const ui64> {
        public:
            TIdIterator(const typename THashMap<ui64, TIntrusivePtr<T>>::const_iterator& currentOld,
                        const typename THashMap<ui64, TIntrusivePtr<T>>::const_iterator& endOld,
                        const TCircularPodBuffer<T*>* window,
                        const size_t numActiveInWindow,
                        const ui64 latest)
                : CurrentOld(currentOld)
                , EndOld(endOld)
                , Window(window)
                , CurrentWindowIdx(0)
                , NumActiveInWindowLeft(numActiveInWindow)
                , Latest(latest)
            {
                SkipToNextActiveWindowPos(false);
                Y_ASSERT(CurrentWindowIdx == Window->Size() || (*Window)[CurrentWindowIdx]);
            }

            void operator++() {
                Y_ASSERT(!AtEnd());
                if (CurrentOld == EndOld) {
                    SkipToNextActiveWindowPos(true);
                } else {
                    ++CurrentOld;
                }
                Y_ASSERT(CurrentOld != EndOld || AtEnd() || (*Window)[CurrentWindowIdx]);
            }

            TIdIterator operator++(int) {
                TIdIterator copy(*this);
                ++(*this);
                return copy;
            }

            ui64 operator*() const {
                Y_ASSERT(!AtEnd());
                return CurrentOld == EndOld ? GetIdByWindowIndex(Latest, CurrentWindowIdx) : CurrentOld->first;
            }

            bool operator==(const TIdIterator& rhv) const {
                return (rhv.AtEnd() && AtEnd() == rhv.AtEnd()) ||
                       (CurrentOld == rhv.CurrentOld &&
                        EndOld == rhv.EndOld &&
                        Window == rhv.Window &&
                        CurrentWindowIdx == rhv.CurrentWindowIdx &&
                        NumActiveInWindowLeft == rhv.NumActiveInWindowLeft &&
                        Latest == rhv.Latest);
            }

            bool operator!=(const TIdIterator& rhv) const {
                return !(*this == rhv);
            }

        private:
            bool AtEnd() const {
                return CurrentOld == EndOld && CurrentWindowIdx == Window->Size();
            }

            bool SkipToNextActiveWindowPos(bool skipCurrent) {
                NumActiveInWindowLeft -= (size_t) !!skipCurrent;
                if (!NumActiveInWindowLeft) {
                    CurrentWindowIdx = Window->Size();
                    return false;
                }

                for (CurrentWindowIdx += (size_t) !!skipCurrent; CurrentWindowIdx < Window->Size(); ++CurrentWindowIdx) {
                    if ((*Window)[CurrentWindowIdx]) {
                        return true;
                    }
                }
                Y_ASSERT(false);
                return false;
            }

            typename THashMap<ui64, TIntrusivePtr<T>>::const_iterator CurrentOld;
            typename THashMap<ui64, TIntrusivePtr<T>>::const_iterator EndOld;
            const TCircularPodBuffer<T*>* Window;
            ui64 CurrentWindowIdx;
            size_t NumActiveInWindowLeft;
            /*const*/ ui64 Latest;
        };

        ///////////////////////////////////////////////////////////////////////////

        friend class TConnection;

        static constexpr int WINDOW_SIZE = 128; // window size must be equal to max expected simultanious active transfers

        static size_t GetWindowIndexById(const size_t latest, const ui64 id) {
            Y_ASSERT(id <= latest);
            return WINDOW_SIZE - 1 - (latest - id);
        }
        static size_t GetIdByWindowIndex(const size_t latest, const ui64 windowIdx) {
            Y_ASSERT(windowIdx < WINDOW_SIZE);
            return latest - (WINDOW_SIZE - 1 - windowIdx);
        }

        TTransfers()
            : Latest(0)
            , Window(WINDOW_SIZE)
            , NumActiveInWindow(0)
        {
            Clear();
        }

        ~TTransfers() {
            Clear();
        }

        bool Empty() const {
            return NumActiveInWindow == 0 && Old.empty();
        }
        size_t Size() const {
            return NumActiveInWindow + Old.size();
        }

        bool Has(const ui64 id) const {
            return Get(id) != nullptr;
        }
        T* Get(const ui64 id) {
            return const_cast<T*>(static_cast<const TTransfers<T>*>(this)->Get(id));
        }
        const T* Get(const ui64 id) const {
            if (Y_LIKELY(IsInWindow(id))) {
                return Window[GetWindowIndex(id)];
            } else if (id > Latest) {
                return nullptr;
            } else if (const TIntrusivePtr<T>* o = Old.FindPtr(id)) {
                return o->Get();
            } else {
                return nullptr;
            }
        }

        TIdIterator Begin() const {
            return TIdIterator(Old.begin(), Old.end(), &Window, NumActiveInWindow, Latest);
        }
        TIdIterator End() const {
            return TIdIterator(Old.end(), Old.end(), &Window, 0, Latest);
        }

    private:
        bool IsInWindow(const ui64 id) const {
            return GetIdByWindowIndex(Latest, 0) <= id && id <= Latest;
        }
        size_t GetWindowIndex(const ui64 id) const {
            Y_ASSERT(IsInWindow(id));
            return GetWindowIndexById(Latest, id);
        }

        // must implement std::map::insert semantics!
        std::pair<T*, bool> Insert(const ui64 id) {
            // most common case for recv first (but without Y_LIKELY because we are not so sure)
            if (IsInWindow(id)) {
                T*& val = Window[GetWindowIndex(id)];
                const bool active = !!val;
                if (!active) {
                    val = new T();
                    NumActiveInWindow++;
                }
                return std::make_pair(val, !active);

                // second most common for recv and the only one for send
            } else if (id > Latest) {
                for (; Latest < id && NumActiveInWindow > 0; ++Latest) {
                    if (!!Window.Front()) {
                        // do not delete front element - move in to Old with automatic memory management
                        Old[GetIdByWindowIndex(Latest, 0)] = Window.Front();
                        --NumActiveInWindow;
                    }

                    Window.PopFront();
                    Y_ABORT_UNLESS(Window.PushBack(nullptr), "");
                    Y_ASSERT(Window.Full());
                }

                // optimization for large id - Latest deltas
                if (NumActiveInWindow == 0) {
                    Latest = id;
                }

                Window.Back() = new T();
                NumActiveInWindow++;
                return std::make_pair(Window.Back(), true);

            } else {
                TIntrusivePtr<T>& t = Old[id];
                const bool active = !!t;
                if (!active) {
                    t = new T();
                }
                return std::make_pair(t.Get(), !active);
            }
        }

        // it is allowed to erase elements which were already passed by iterator or are its current position.
        bool Erase(const ui64 id) {
            const bool has = Has(id);
            if (has) {
                if (Y_LIKELY(IsInWindow(id))) {
                    T*& w = Window[GetWindowIndex(id)];
                    delete w;
                    w = nullptr;
                    NumActiveInWindow--;
                } else {
                    Old.erase(id); // TIntrusivePtr will destroy T
                }
            }
            return has;
        }

        void Clear() {
            for (size_t i = 0; i != Window.Size() && NumActiveInWindow > 0; ++i) {
                Erase(GetIdByWindowIndex(Latest, i));
            }

            Latest = WINDOW_SIZE;
            Window.Clear();

            Y_ASSERT(Window.Capacity() == WINDOW_SIZE);
            for (size_t i = 1; i <= Latest; ++i) {
                Y_ABORT_UNLESS(Window.PushBack(nullptr), "");
            }
            Y_ASSERT(Window.Full());
            Y_ASSERT(GetIdByWindowIndex(Latest, 0) == 1);
            Y_ASSERT(NumActiveInWindow == 0);

            Old = THashMap<ui64, TIntrusivePtr<T>>(); // shrinks buckets memory
        }

        // we use T* in circular buffer to reduce memory usage for large windows (instead we could use TMaybe<T>).
        // we also expect that mostly Old will be empty
        ui64 Latest;
        TCircularPodBuffer<T*> Window; // (Latest - Window.size(), Latest] transfers
        size_t NumActiveInWindow;
        THashMap<ui64, TIntrusivePtr<T>> Old; // [1, Latest - Window.size()] transfers
    };

    using TRecvTransfers = TTransfers<TUdpInTransfer>;
    using TSendTransfers = TTransfers<TUdpOutTransfer>;

    ///////////////////////////////////////////////////////////////////////////////

    struct TPeerLink {
        friend class TConnection;

        TPeerLink(const TUdpAddress& toAddress, const TConnectionSettings& connectionSettings, const float udpTransferTimeout)
            : UdpCongestion(new TCongestionControl(connectionSettings.GetInflateCongestion() ? 10 : 1))
            , ToAddress(toAddress)
            , TimeSleeping(0)
            , CongestionSettings(connectionSettings)
            , UdpTransferTimeout(udpTransferTimeout)
        {
        }

        TIntrusivePtr<TCongestionControl> GetUdpCongestion() {
            TimeSleeping = 0;
            return UdpCongestion;
        }
        TIntrusivePtr<IIBPeer> GetIBPeer() {
            return IBPeer;
        }

        bool HasIBPeer() const {
            return IBPeer.Get();
        }
        void SetIBPeer(const TIntrusivePtr<IIBPeer>& ibPeer) {
            IBPeer = ibPeer;
        }

        bool IsSleeping() const {
            return TimeSleeping > 0;
        }
        void MakeAlive() {
            TimeSleeping = 0;

            if (UdpCongestion->IsAlive()) {
                UdpCongestion->MarkAlive(); // resets timeout

            } else {
                // This is tricky: congestion thinks that connection is dead,
                // but we either received a new packet or user forces us to start a new transfer.
                // OK, we reset congestion and start from scratch.
                // fprintf(stderr, "Congestion reset (%s)\n", GetAddressAsString(ToAddress).c_str());
                *this = TPeerLink(ToAddress, CongestionSettings, UdpTransferTimeout);

                // for future changes: we need to reset MTU to small size (1300) packets,
                // we may be dead because of network changes which reduced MTU from 9000 to 1300.
            }
        }

        // if method returns false - this is really bad: it means that connection is dead.
        bool Update(float deltaT, float* maxWaitTime) {
            Y_ASSERT(!IsSleeping());
            return UdpCongestion->UpdateAlive(ToAddress, deltaT, UdpTransferTimeout, maxWaitTime);
        }

        // if method returns false this means that we've been sleeping too long - this link seems to be not needed anymore.
        bool Sleep(float deltaT, float maxSleepTime, float* maxWaitTime) {
            Y_ASSERT(UdpCongestion->IsAlive());
            return IsSleeping() ? UpdateSleeping(deltaT, maxSleepTime) : StartSleeping(maxWaitTime);
        }

        void ForceTimeAccount() {
            UdpCongestion->ForceTimeAccount();
        }

        TString GetDebugInfo() const {
            char buf[1000];
            const TCongestionControl& cc = *UdpCongestion;
            snprintf(buf, sizeof(buf), "IB: %d, RTT: %g, Timeout: %g, Window: %g, MaxWin: %g, FailRate: %g, TimeSinceLastRecv: %g, MTU: %d, Sleeping: %g, Alive: %d",
                    IBPeer.Get() ? IBPeer->GetState() : -1,
                    cc.GetRTT() * 1000, cc.GetTimeout() * 1000, cc.GetWindow(), cc.GetMaxWindow(), cc.GetFailRate(),
                    cc.GetTimeSinceLastRecv() * 1000, cc.GetMTU(), TimeSleeping, (int)cc.IsAlive());
            return buf;
        }

    private:
        bool StartSleeping(float* maxWaitTime) {
            //printf("peer_link start sleep, IBPeer = %p, refs = %d\n", IBPeer.Get(), (int)IBPeer.RefCount());
            UdpCongestion->UpdateAlive(ToAddress, 0, UdpTransferTimeout, maxWaitTime); // TODO: do we actually need this line?
            UdpCongestion->MarkAlive();
            TimeSleeping = 1e-9; // any small non-zero value
            return true;
        }

        bool UpdateSleeping(float deltaT, float maxSleepTime) {
            TimeSleeping += deltaT;

            if (TimeSleeping > maxSleepTime) {
                IBPeer = nullptr;
                return false;
            }

            if (IBPeer.Get()) {
                //printf("peer_link update sleep, IBPeer = %p, refs = %d\n", IBPeer.Get(), (int)IBPeer.RefCount());
                if (IBPeer->GetState() == IIBPeer::OK) {
                    return true;
                }
                //printf("Drop broken IB connection\n");
                IBPeer = nullptr;
            }

            return true;
        }

        TIntrusivePtr<TCongestionControl> UdpCongestion;
        TIntrusivePtr<IIBPeer> IBPeer;
        TUdpAddress ToAddress;
        double TimeSleeping;
        TConnectionSettings CongestionSettings;
        float UdpTransferTimeout;
    };

    //////////////////////////////////////////////////////////////////////////
    class TConnections;
    class TConnection;
    using TActiveConnectionList = TIntrusiveList<TConnection>;
    using TSendingConnectionsList = TDeque<TIntrusivePtr<TConnection>>;
    template <class TItem>
    class TActiveConnectionListItem: public TIntrusiveListItem<TItem> {
        friend class TConnections;
        bool Inactivated = true;
        bool Sending = false;

    protected:
        bool IsInactivated() const {
            return Inactivated;
        }
    };

    class TConnection: public IConnection, public TActiveConnectionListItem<TConnection> {
    public:
        TConnection(const TUdpAddress& address, const TUdpAddress& myAddress, const TConnectionSettings& connectionSettings, const TGUID& guid, float udpTransferTimeout)
            : Address(address)
            , MyAddress(myAddress)
            , Guid(guid)
            , Stats(new TRequesterPendingDataStats)
            , SmallMtuUseXs(false)
            , TransferId(1) // start with 1, do not use 0
            , PeerLink(address, connectionSettings, udpTransferTimeout)
        {
            Y_ASSERT(!Guid.IsEmpty());
            GetWinsockAddr(&WinsockAddress, address);
            GetWinsockAddr(&WinsockMyAddress, myAddress);
            CreateGuid(&ThisSideGuid);
            NHPTimer::GetTime(&CurrentTime);
        }

        const TUdpAddress& GetAddress() const override {
            return Address;
        }
        const TUdpAddress& GetMyAddress() const {
            return MyAddress;
        }
        const sockaddr_in6& GetWinsockAddress() const override {
            return WinsockAddress;
        }
        const sockaddr_in6& GetWinsockMyAddress() const {
            return WinsockMyAddress;
        }

        const TGUID& GetGuid() const override {
            return Guid;
        }

        TRequesterPendingDataStats GetPendingDataSize() const override {
            return *Stats;
        }
        TIntrusivePtr<TRequesterPendingDataStats> GetStatsPtr() {
            return Stats;
        }

        bool IsAlive() const override {
            return PeerLink.UdpCongestion->IsAlive();
        }

        bool IsSendTransferAlive(const ui64 id) const {
            return GetSendQueue().Get(id) != nullptr;
        }

        TConnectionSettings GetSettings() const override {
            return PeerLink.CongestionSettings;
        }

        const TGUID& GetThisSideGuid() const {
            return ThisSideGuid;
        }
        const TGUID& GetThatSideGuid() const {
            return ThatSideGuid;
        }
        bool CheckThatSideGuid(const TGUID& newThatSideGuid) {
            Y_ASSERT(!newThatSideGuid.IsEmpty());
            const TGUID originalThatSideGuid = ThatSideGuid;
            ThatSideGuid = newThatSideGuid;

            const bool isFirstConnection = originalThatSideGuid.IsEmpty();
            if (isFirstConnection) {
                Y_ASSERT(InQueue.Empty());
                return true;
            }

            if (originalThatSideGuid == newThatSideGuid) {
                return true;
            }

            //fprintf(stderr, "Connection reset %s -> %s\n", GetGuidAsString(originalThatSideGuid).c_str(), GetGuidAsString(newThatSideGuid).c_str());

            // If method returns false then:
            // This is not connection handshake - receiver has changed it's guid - netliba has been restarted
            // or it's new connection, because old one was dropped by timeout!
            // It's too difficult to properly set internal state of AckTracker for storing some ACKs
            // in the middle of transfer, so in worst case (receiver instance has been restarted)
            // we just ignore received ACKs and ask to resend everything.
            //
            // There could be timeout problems in upper levels since we may already got ACK_COMPLETED
            // just before receiver was restarted, so we would never get response.

            // we must clear it or "holes" mechanism will break up!
            RecvCompleted.Clear();
            InQueue.Clear();
            // Resend all packets.
            // Worst thing we could have - ACK received from previous instance for packet which is unknown to current.
            // TODO: we can actually send message twice - to old instance and to new. Is it OK?
            for (TSendTransfers::TIdIterator i = OutQueue.Begin(); i != OutQueue.End(); ++i) {
                OutQueue.Get(*i)->AckTracker.Resend();
            }
            return false;
        }

        ui64 GetNextTransferId() {
            return TransferId++;
        }

        TRecvTransfers& GetRecvQueue() {
            return InQueue;
        }
        TSendTransfers& GetSendQueue() {
            return OutQueue;
        }
        TDeque<ui64>& GetSendingTransfers(ui8 prio) {
            Y_ASSERT(prio <= PP_SYSTEM);
            return SendingTransferIDs[prio];
        }
        bool HasAnySendingTransfers() const {
            for (const auto& sendingTransferID : SendingTransferIDs) {
                if (!sendingTransferID.empty())
                    return true;
            }
            return false;
        }

        const TRecvTransfers& GetRecvQueue() const {
            return InQueue;
        }
        const TSendTransfers& GetSendQueue() const {
            return OutQueue;
        }

        std::pair<TUdpOutTransfer*, bool> InsertSendTransfer(const ui64 transferId) {
            return OutQueue.Insert(transferId);
        }
        std::pair<TUdpInTransfer*, bool> InsertRecvTransfer(const ui64 transferId) {
            std::pair<TUdpInTransfer*, bool> r = InQueue.Insert(transferId);
            if (r.second) {
                RecvCompleted.NewTransfer(transferId);
                Y_ASSERT(RecvCompleted.GetNumActive() == InQueue.Size());
            }
            return r;
        }

        void SuccessfulSendTransfer(const ui64 transferId) {
            const bool r = CompletedSendTransfer(transferId);
            Y_ASSERT(r);
            Y_UNUSED(r);
        }
        void SuccessfulRecvTransfer(const ui64 transferId) {
            const bool r = CompletedRecvTransfer(transferId, false, false);
            Y_ASSERT(r);
            Y_UNUSED(r);
        }
        void FailedSendTransfer(const ui64 transferId) {
            // do not check return value, we may call this method multiple times for each failed packet of same transfer
            CompletedSendTransfer(transferId);
        }
        void FailedRecvTransfer(const ui64 transferId) {
            CompletedRecvTransfer(transferId, true, false);
        }
        void CanceledSendTransfer(const ui64 transferId) {
            CompletedSendTransfer(transferId);
        }
        void CanceledRecvTransfer(const ui64 transferId) {
            CompletedRecvTransfer(transferId, false, true);
        }

        bool IsRecvCompleted(const ui64 transferId, bool* isFailed, bool* isCanceled) const {
            return RecvCompleted.IsCompleted(transferId, isFailed, isCanceled);
        }
        bool IsSleeping() const {
            return PeerLink.IsSleeping();
        }
        bool GetSmallMtuUseXs() const {
            return SmallMtuUseXs;
        }
        void SetSmallMtuUseXs(bool smallMtuUseXs) {
            SmallMtuUseXs = smallMtuUseXs;
        }

        bool Step(const float maxSleepTime, float* maxWaitTime, float* stepDeltaTime, const NHPTimer::STime now, TStatAggregator* failureStat) {
            const float deltaT = (float)NHPTimer::GetSeconds(now - CurrentTime);
            *stepDeltaTime = deltaT;
            CurrentTime = now;

            RecvCompleted.Cleanup(GetGuid());

            if (!IsAlive()) {
                Y_ASSERT(!PeerLink.IsSleeping());
                Y_ASSERT(InQueue.Empty() && OutQueue.Empty());
                return false; // kill me plz!

            } else if (InQueue.Empty() && OutQueue.Empty()) {
                return PeerLink.Sleep(deltaT, maxSleepTime, maxWaitTime);
            } else {
                //If we are here - we must be active. It is important to handle unexpected killed senders, or some
                //network error
                Y_ASSERT(IsInactivated() == false);
                // we had to stop sleeping after inserting new element to Udp(In|Out)Queue
                if (Y_UNLIKELY(PeerLink.IsSleeping())) {
                    Y_ASSERT(false);
                    PeerLink.MakeAlive(); // not necessary, just to be sure
                }
                failureStat->AddPoint(GetFailRate());
                return PeerLink.Update(deltaT, maxWaitTime);
            }
        }

        TPeerLink& GetAlivePeerLink() {
            PeerLink.MakeAlive();
            return PeerLink;
        }

        TString GetDebugInfo() const {
            TString result;
            result += "Connection: ";
            result += GetGuidAsString(GetGuid());
            result += "\n\tThis side guid: ";
            result += GetGuidAsString(GetThisSideGuid());
            result += "\n\tThat side guid: ";
            result += GetGuidAsString(GetThatSideGuid());
            result += "\n\tsource: ";
            result += GetAddressAsString(GetMyAddress());
            result += "\n\tdestination: ";
            result += GetAddressAsString(GetAddress());
            result += "\n\tactive send transfers: ";
            result += ToString(OutQueue.Size());
            result += "\n\tsending transfers: ";
            for (int i = 0; i < NSendQueues; i++) {
                result += "(" + ToString(i) + ") " + ToString(SendingTransferIDs[i].size()) + " ";
            }
            result += ", recv transfers: ";
            result += ToString(InQueue.Size());
            result += "\n\trecv completed: ";
            result += RecvCompleted.GetDebugInfo();
            result += "\n\tcongestion: ";
            result += PeerLink.GetDebugInfo();
            return result;
        }

        float GetFailRate() const {
            return PeerLink.UdpCongestion->GetFailRate();
        }

    private:
        bool CompletedSendTransfer(const ui64 transferId) {
            return OutQueue.Erase(transferId);
        }
        bool CompletedRecvTransfer(const ui64 transferId, const bool isFailed, const bool isCanceled) {
            const bool r = InQueue.Erase(transferId);

            Y_ASSERT((int)isFailed + (int)isCanceled < 2);

            RecvCompleted.MarkCompleted(transferId, isFailed, isCanceled);
            Y_ASSERT(RecvCompleted.GetNumActive() == InQueue.Size());

            return r;
        }

        TUdpAddress Address;
        TUdpAddress MyAddress;
        sockaddr_in6 WinsockAddress;
        sockaddr_in6 WinsockMyAddress;
        TGUID Guid;
        TIntrusivePtr<TRequesterPendingDataStats> Stats;
        bool SmallMtuUseXs;

        NHPTimer::STime CurrentTime;

        // used only while sending
        static const int NSendQueues = PP_SYSTEM + 1;
        std::array<TDeque<ui64>, NSendQueues> SendingTransferIDs;
        TSendTransfers OutQueue;
        ui64 TransferId;
        TGUID ThisSideGuid;
        //TPeerLink CongestionTrack;

        // used only while receiving
        TRecvTransfers InQueue;
        TRecvCompleted RecvCompleted;
        TGUID ThatSideGuid;

        TPeerLink PeerLink;
    };
}
