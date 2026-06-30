#pragma once

#include <util/draft/holder_vector.h>
#include <util/generic/noncopyable.h>
#include <util/generic/deque.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/network/init.h>
#include <util/system/defaults.h>
#include <util/system/hp_timer.h>
#include "circular_pod_buffer.h"
#include "paged_pod_buffer.h"
#include "socket.h"
#include "udp_address.h"
#include "udp_recv_packet.h"
#include "local_ip_params.h"

namespace NNetliba_v12 {
    constexpr int UDP_NETLIBA_VERSION_V12 = 112;                                         // 100 (offset) + 12 (netliba version)
    constexpr int UDP_LOW_LEVEL_HEADER_SIZE = 8 + 1 + 2;                                 // crc + version + packet size
    constexpr int UDP_PACKET_BUF_SIZE = UDP_MAX_PACKET_SIZE - UDP_LOW_LEVEL_HEADER_SIZE; // for user

    struct TSockAddrPair {
        sockaddr_in6 RemoteAddr;
        sockaddr_in6 MyAddr;
    };

    ///////////////////////////////////////////////////////////////////////////////

    class TUdpSocket: public TNonCopyable {
    public:
        enum ESendError {
            SEND_OK,
            SEND_BUFFER_OVERFLOW,
            SEND_NO_ROUTE_TO_HOST,
            SEND_EINVAL
        };

    private:
        TIntrusivePtr<NNetlibaSocket::ISocket> S;

        ui32 PortCrc;
        ui32 LastLocalIpCrc[2];
        TLocalIpParams IpParams;

        size_t MaxUdpPacketsInQueue; // udp packets, does not include small packet optimization!
        bool UseSmallPacketsOptimization;

        // used for sendmmsg data buffering
        TCircularPodBuffer<sockaddr_in6> UdpPacketsAddresses;
        TCircularPodBuffer<char> UdpPacketsAuxData;
        TCircularPodBuffer<char> PacketsDataBuffer;
        TPagedPodBuffer<TIoVec> PacketsIoVecs;
        TVector<TMMsgHdr> UdpPacketsHeaders; // can't use circular buffer here, or old packets may never get sent
        size_t UdpPacketsHeadersBegin;
        size_t LastUdpPacketSize;

        char* LastReturnedPacketBuffer;
        size_t LastReturnPacketBufferDataSize;

        // used for small packet optimization
        THolder<TUdpRecvPacket> RecvContUdpPacket;
        int RecvContUdpPacketSize;
        TSockAddrPair RecvContAddress;

        struct TSendPacketsStat {
            ui64 Total = 0;
            ui64 Merged = 0;
        } SendStat;
        float MergeRatio = 0.0;

        bool CrcMatches(ui64 expectedCrc, ui64 crc, const sockaddr_in6& addr);
        bool CheckPacketIntegrity(const char* buf, const size_t size, const TSockAddrPair& addr);

        size_t GetNumUdpPacketsInQueue() const;

        bool TryToApplySmallPacketsOptimization(const TIoVec& v, const TSockAddrPair& addr, ui8 tos, size_t mtu);
        void ForgetHeadUdpPackets(const size_t numPackets);

        TUdpRecvPacket* RecvContinuationPacket(TSockAddrPair* addr);
        void CacheContinuationUdpPacket(const TUdpRecvPacket& pkt, const size_t pktSize, const TSockAddrPair& addr);

    public:
        TUdpSocket(const size_t maxUdpPacketsInQueue, const bool useSmallPacketsOptimization);
        ~TUdpSocket();

        void Open(int port);
        void Open(const TIntrusivePtr<ISocket>& socket);
        void Close();
        bool IsValid() const;

        int GetNetworkOrderPort() const;

        void Wait(float timeoutSec) const;
        void CancelWait(const TUdpAddress& address = TUdpAddress());

        // Legacy: first UDP_LOW_LEVEL_HEADER_SIZE bytes of buf are used for low-level header.
        // RecvFrom ignores small packet optimization and returns only first packet!
        ESendError SendTo(const char* buf, size_t size, const TSockAddrPair& addr, ui8 tos, const EFragFlag frag);
        bool RecvFrom(char* buf, size_t* size, TSockAddrPair* addr);

        // Packet queue send interface: buf contains only user data
        // Do not mix with SendTo calls (OK if IsPacketsQueueEmpty())!
        // After each call NewPacketBuffer copy data to returned buffer and call AddPacketToQueue with proper data size.
        char* NewPacketBuffer(const size_t maxSize);
        void AddPacketToQueue(size_t size, const TSockAddrPair& addr, ui8 tos, size_t mtu);
        TUdpSocket::ESendError FlushPackets(size_t* numSentPackets, TVector<std::pair<char*, size_t>>* failedPackets);

        bool IsPacketsQueueEmpty() const;
        void GetPacketsInQueue(TVector<std::pair<char*, size_t>>* packets) const; // O(n)
        void ClearPacketsQueue();

        // TUdpRecvPacket == NULL means - no packet available.
        // Packet payload starts from TUdpRecvPacket::DataStart offset.
        TUdpRecvPacket* Recv(TSockAddrPair* addr);
        float GetAndResetMergeRatio();
        void SetRecvLagTime(NHPTimer::STime time);
        TString GetSockDebug() const;
        void UpdateStats();
        bool IsLocal(const TUdpAddress& address) const;
    };
}
