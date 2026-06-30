#include <util/system/platform.h>
#include <util/datetime/cputimer.h>
#include <util/string/cast.h>
#include <util/random/random.h>
#include "stdafx.h"
#include "udp_socket.h"
#include "block_chain.h"
#include "udp_address.h"
#include "udp_recv_packet.h"
#include "crc32c.h"
#include "settings.h"

//#define SIMULATE_NETWORK_FAILURES
//#define SIMULATE_NO_ROUTE_TO_HOST

namespace NNetliba_v12 {
    static const ui32 PROTO_REV = 1;
    static inline ui32 CalcPortChecksum(const sockaddr_in6& addr);
    static inline ui64 CalcCrc(const char* buf, int size, const sockaddr_in6& addr);

    ///////////////////////////////////////////////////////////////////////////////

    TUdpSocket::TUdpSocket(const size_t maxUdpPacketsInQueue, const bool useSmallPacketsOptimization)
        : PortCrc(0)
        , MaxUdpPacketsInQueue(maxUdpPacketsInQueue)
        , UseSmallPacketsOptimization(useSmallPacketsOptimization)
        , UdpPacketsAddresses(MaxUdpPacketsInQueue)
        , UdpPacketsAuxData(CTRL_BUFFER_SIZE * MaxUdpPacketsInQueue)
        , PacketsDataBuffer(UDP_MAX_PACKET_SIZE * MaxUdpPacketsInQueue)
        , PacketsIoVecs(UDP_MAX_PACKET_SIZE / UDP_LOW_LEVEL_HEADER_SIZE)
        , UdpPacketsHeadersBegin(0)
        , LastUdpPacketSize(0)
        , LastReturnedPacketBuffer(nullptr)
        , LastReturnPacketBufferDataSize(0)
        , RecvContUdpPacketSize(0)
    {
#ifdef _win_
        UseSmallPacketsOptimization = false;
#endif
        Y_ABORT_UNLESS(MaxUdpPacketsInQueue > 0, "WAT?");
        Y_ABORT_UNLESS(!UseSmallPacketsOptimization || MaxUdpPacketsInQueue > 1, "For small packets optimization use packets queue with at least 2 elements");

        UdpPacketsHeaders.reserve(MaxUdpPacketsInQueue * 2);

        Zero(RecvContAddress);
    }

    TUdpSocket::~TUdpSocket() {
    }

    void TUdpSocket::Open(int port) {
        TIntrusivePtr<ISocket> s = CreateSocket();
        s->Open(port);
        Open(s);
    }

    // caller code assumes that packet fragmentation is forbidden!
    void TUdpSocket::Open(const TIntrusivePtr<ISocket>& socket) {
        if (IpParams.Init()) {
            LastLocalIpCrc[IPv4] = IpParams.GetLocaIpCrcs(IPv4)[0];
            LastLocalIpCrc[IPv6] = IpParams.GetLocaIpCrcs(IPv6)[0];
        } else {
            fprintf(stderr, "Unable to init ip params\n");
            return;
        }
        S = socket;
        if (IsValid()) {
            PortCrc = CalcPortChecksum(S->GetSelfAddress());
        }
    }

    void TUdpSocket::Close() {
        if (IsValid()) {
            S->Close();
            S = nullptr;
        }
    }

    bool TUdpSocket::IsValid() const {
        return S.Get() && S->IsValid();
    }

    int TUdpSocket::GetNetworkOrderPort() const {
        return S->GetNetworkOrderPort();
    }

    void TUdpSocket::Wait(float timeoutSec) const {
        S->Wait(timeoutSec, UDP_NETLIBA_VERSION_V12);
    }

    void TUdpSocket::CancelWait(const TUdpAddress& address) {
        if (address == TUdpAddress()) {
            S->CancelWait(UDP_NETLIBA_VERSION_V12);
        } else {
            sockaddr_in6 a;
            GetWinsockAddr(&a, address);
            S->CancelWaitHost(a);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    static constexpr int UDP_CRC_SIZE = 8;
    static constexpr int UDP_VERSION_OFFSET = 0 + UDP_CRC_SIZE;
    static constexpr int UDP_PACKET_DATA_SIZE_OFFSET = UDP_VERSION_OFFSET + 1;
    static constexpr int UDP_DATA_OFFSET = UDP_PACKET_DATA_SIZE_OFFSET + 2;

    static_assert(UDP_LOW_LEVEL_HEADER_SIZE == UDP_DATA_OFFSET, "expect UDP_LOW_LEVEL_HEADER_SIZE == UDP_DATA_OFFSET");

    ///////////////////////////////////////////////////////////////////////////////

    static ui8 ReadVersion(const char* buf) {
        return *(ui8*)(buf + UDP_VERSION_OFFSET);
    }
    static void WriteVersion(char* buf) {
        // We use version field to distinguish between netliba v12 and old netliba v6 packets.
        // At this offset netliba v6 had CMD field (from TUdpHost) which was less than 100.
        *(ui8*)(buf + UDP_VERSION_OFFSET) = (char)UDP_NETLIBA_VERSION_V12;
    }

    static ui16 ReadPacketDataSize(const char* buf) {
        return *(ui16*)(buf + UDP_PACKET_DATA_SIZE_OFFSET);
    }
    static void WritePacketDataSize(char* buf, const size_t size) {
        Y_ASSERT(size <= 1 << 16);
        *(ui16*)(buf + UDP_PACKET_DATA_SIZE_OFFSET) = (ui16)size;
    }

    static void WritePacketHeader(char* packet, const size_t packetSize, const sockaddr_in6& toAddress) {
        WriteVersion(packet);
        WritePacketDataSize(packet, (ui16)(packetSize - UDP_LOW_LEVEL_HEADER_SIZE));
        *(ui64*)packet = CalcCrc(packet + UDP_CRC_SIZE, packetSize - UDP_CRC_SIZE, toAddress);
    }

    //TODO: refactoring CreateSendMsgHdr!
    //for a while do not use TSockAddrPair here - CreateSendMsgHdr will cast const sockaddr_in6& toAddr to void* !!!
    static inline TMsgHdr CreateNl12SendMsgHdr(const sockaddr_in6& toAddr, const sockaddr_in6& fromAddr, const TIoVec& iov, ui8 tos,
                                               void* ctrlBuffer, size_t bufferLen) {
#ifdef _win_
        Y_UNUSED(ctrlBuffer);
        Y_UNUSED(bufferLen);
        i64 winTos = tos;
        //NOTE: We can't handle ip alias in windows
        const TMsgHdr& hdr = CreateSendMsgHdr(toAddr, iov, &winTos); //there is no controll buffer in Windows, just write Tos
        return hdr;
#else
        //TODO: check, it may be expencive to zero buffer here
        memset(ctrlBuffer, 0, bufferLen);
        TMsgHdr hdr = CreateSendMsgHdr(toAddr, iov, ctrlBuffer);
        if (ctrlBuffer && (AddSockAuxData(&hdr, tos, fromAddr, ctrlBuffer, bufferLen) == nullptr)) {
            fprintf(stderr, "BUG! Can not attach controll buffer!\n");
        }
        return hdr;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////////

    static inline ui32 CalcPortChecksum(const sockaddr_in6& addr) {
        return addr.sin6_port;
    }

    static inline ui64 CalcCrc(const char* buf, int size, const sockaddr_in6& addr) {
        return CalcChecksum(buf, size) + CalcAddressChecksum(addr) + CalcPortChecksum(addr) + PROTO_REV;
    }

    bool TUdpSocket::CrcMatches(ui64 expectedCrc, ui64 crc, const sockaddr_in6& addr) {
        Y_ASSERT(addr.sin6_family == AF_INET6);

        // determine our ip address family based on the sender address
        // address family can not change in network, so sender address type determines type of our address used
        const EIpType ipType = GetIpType(addr);
        if (crc + LastLocalIpCrc[ipType] == expectedCrc) {
            return true;
        }

        // crc failed
        // check if packet was sent to different IP address
        const TVector<ui32>& localIpCrcs = IpParams.GetLocaIpCrcs(ipType);
        for (size_t i = 0; i < localIpCrcs.size(); ++i) {
            const ui32 otherIpCrc = localIpCrcs[i];
            if (crc + otherIpCrc == expectedCrc) {
                LastLocalIpCrc[ipType] = otherIpCrc;
                return true;
            }
        }

        // crc is really failed, discard packet
        return false;
    }

    bool TUdpSocket::CheckPacketIntegrity(const char* buf, const size_t size, const TSockAddrPair& addr) {
        // ignore empty or small packets
        if (size < UDP_LOW_LEVEL_HEADER_SIZE) {
            return false;
        }

        if (ReadVersion(buf) != UDP_NETLIBA_VERSION_V12) {
            fprintf(stderr, "NETLIBA::TUdpSocket: version mismatch\n");
            return false;
        }

        const size_t packetDataSize = ReadPacketDataSize(buf);
        if (packetDataSize + UDP_LOW_LEVEL_HEADER_SIZE > size) {
            fprintf(stderr, "NETLIBA::TUdpSocket: bad packet size in header\n");
            return false;
        }

        const ui64 expectedCrc = *(ui64*)buf;

        const ui64 crc = CalcChecksum(buf + UDP_CRC_SIZE, packetDataSize + (UDP_LOW_LEVEL_HEADER_SIZE - UDP_CRC_SIZE)) + PROTO_REV;
        if (!CrcMatches(expectedCrc, crc + PortCrc, addr.RemoteAddr)) {
            fprintf(stderr, "NETLIBA::TUdpSocket: udp packet crc failure %s, expected %" PRIu64 ", %" PRIu64 ", %u \n",
                    GetAddressAsString(GetUdpAddress(addr.RemoteAddr)).c_str(), expectedCrc, crc, PortCrc);
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////////

    static TUdpSocket::ESendError GetSendErrorFromErrno(const int e) {
        if (e == EHOSTUNREACH || e == ENETUNREACH) {
            return TUdpSocket::SEND_NO_ROUTE_TO_HOST;
        } else if (e == EINVAL) {
            return TUdpSocket::SEND_EINVAL;
        } else if (e == EAGAIN || e == EWOULDBLOCK || e == ENOMEM || e == ENOBUFS || e == EMSGSIZE) {
            return TUdpSocket::SEND_BUFFER_OVERFLOW;
        } else {
            fprintf(stderr, "got unhandled errno: %d\n", e);
        }
        return TUdpSocket::SEND_EINVAL; //transfer will be canceled
    }

    TUdpSocket::ESendError TUdpSocket::SendTo(const char* buf, size_t size, const TSockAddrPair& addr, ui8 tos, const EFragFlag frag) {
        Y_ASSERT(IsPacketsQueueEmpty());
        Y_ASSERT(UDP_LOW_LEVEL_HEADER_SIZE <= size && size <= UDP_MAX_PACKET_SIZE);
        //fprintf(stderr, "SendTo: %s -> %s\n", GetAddressAsString(GetUdpAddress(addr.RemoteAddr)).c_str(), GetAddressAsString(GetUdpAddress(addr.MyAddr)).c_str());

        WritePacketHeader((char*)buf, size, addr.RemoteAddr);

#ifdef SIMULATE_NETWORK_FAILURES
        if ((RandomNumber<size_t>() % 3) == 0)
            return SEND_OK; // packet lost
        if ((RandomNumber<size_t>() % 3) == 0)
            (char&)(buf[RandomNumber<size_t>() % size]) += RandomNumber<size_t>(); // packet broken
#endif

        char controllBuffer[CTRL_BUFFER_SIZE];
        const TIoVec iov = CreateIoVec((char*)buf, size); // store as local variable on the stack

        const TMsgHdr hdr = CreateNl12SendMsgHdr(addr.RemoteAddr, addr.MyAddr, iov, tos, controllBuffer, CTRL_BUFFER_SIZE);
        const int rv = S->SendMsg(&hdr, 0, frag);
        if (rv < 0)
            return GetSendErrorFromErrno(LastSystemError());

        Y_ASSERT((size_t)rv == size);
        return SEND_OK;
    }

    ///////////////////////////////////////////////////////////////////////////////

    static TAutoPtr<TUdpRecvPacket> ConvertToTUdpRecvPacket(const char* packetBuffer, const size_t packetSize, const size_t dataSize) {
        // memcpy, but what can we do? Do not use old interface (RecvFrom), prefer new (Recv)!
        TAutoPtr<TUdpRecvPacket> result = TUdpHostRecvBufAlloc::CreateNewPacket();
        memcpy(result->Data.get(), packetBuffer, packetSize);
        result->DataStart = 0;
        result->DataSize = dataSize;
        return result;
    }

    bool TUdpSocket::RecvFrom(char* buf, size_t* size, TSockAddrPair* addr) {
        Y_ASSERT(*size >= UDP_MAX_PACKET_SIZE);
        for (;;) {
            TAutoPtr<TUdpRecvPacket> result = RecvContinuationPacket(addr);
            if (!!result) {
                *size = result->DataSize;
                memcpy(buf, result->Data.get() + result->DataStart, *size);
                return true;
            }

            const TIoVec v = CreateIoVec(buf, *size);

            char controllBuffer[CTRL_BUFFER_SIZE];
            TMsgHdr hdr = CreateRecvMsgHdr(&addr->RemoteAddr, v, controllBuffer);

            const int rv = S->RecvMsg(&hdr, 0);
            if (rv < 0)
                return false;

            NNetlibaSocket::ExtractDestinationAddress(hdr, &addr->MyAddr);

            if (CheckPacketIntegrity(buf, rv, *addr)) {
                *size = ReadPacketDataSize(buf) + UDP_LOW_LEVEL_HEADER_SIZE;

                // it's packet with small packet optimization, we have to cache it for next calls.
                if ((size_t)rv != *size) {
                    CacheContinuationUdpPacket(*ConvertToTUdpRecvPacket(buf, rv, *size), rv, *addr);
                }
                return true;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    bool TUdpSocket::IsPacketsQueueEmpty() const {
        return UdpPacketsHeadersBegin == UdpPacketsHeaders.size();
    }

    size_t TUdpSocket::GetNumUdpPacketsInQueue() const {
        return UdpPacketsHeaders.size() - UdpPacketsHeadersBegin;
    }

    //////////////////////////////////////////////////////////////////////////////

    // 1 user packet or many packets if small packet optimization is enabled
    static void AddPacketsFromUdpPacket(const TMsgHdr& hdr, TVector<std::pair<char*, size_t>>* packets) {
        // iterating through small packets
        for (size_t s = 0; s != (size_t)hdr.msg_iovlen; ++s) {
            packets->push_back(std::make_pair((char*)hdr.msg_iov[s].iov_base + UDP_LOW_LEVEL_HEADER_SIZE,
                                              hdr.msg_iov[s].iov_len - UDP_LOW_LEVEL_HEADER_SIZE));
        }
    }

    void TUdpSocket::GetPacketsInQueue(TVector<std::pair<char*, size_t>>* packets) const {
        for (size_t p = 0; p != GetNumUdpPacketsInQueue(); ++p) {
            AddPacketsFromUdpPacket(UdpPacketsHeaders[UdpPacketsHeadersBegin + p].msg_hdr, packets);
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    char* TUdpSocket::NewPacketBuffer(const size_t maxSize) {
        Y_ASSERT(!LastReturnedPacketBuffer && "Call AddPacketToQueue after each NewPacketBuffer!");
        Y_ASSERT(maxSize <= UDP_PACKET_BUF_SIZE);
        if (GetNumUdpPacketsInQueue() == MaxUdpPacketsInQueue) { // doesn't count small packets!
            return nullptr;
        }

        // User didn't call ClearPacketsQueue after FlushPacketsQueue with partial sendmmsg.
        if (UdpPacketsHeaders.size() == UdpPacketsHeaders.capacity()) {
            UdpPacketsHeaders.erase(UdpPacketsHeaders.begin(), UdpPacketsHeaders.begin() + UdpPacketsHeadersBegin);
            UdpPacketsHeadersBegin = 0;
        }

        LastReturnedPacketBuffer = PacketsDataBuffer.GetContinuousRegion(UDP_LOW_LEVEL_HEADER_SIZE + maxSize);
        if (!LastReturnedPacketBuffer) { // may occur is small packet optimization is enabled
            return nullptr;
        }
        LastReturnPacketBufferDataSize = maxSize;
        return LastReturnedPacketBuffer + UDP_LOW_LEVEL_HEADER_SIZE;
    }

    bool TUdpSocket::TryToApplySmallPacketsOptimization(const TIoVec& v, const TSockAddrPair& addr, ui8 tos, size_t mtu) {
        SendStat.Total++;

        if (!UseSmallPacketsOptimization) {
            return false;
        }

        if (!GetNumUdpPacketsInQueue()) {
            return false;
        }

        TMsgHdr& last = UdpPacketsHeaders.back().msg_hdr;
        if (memcmp(last.msg_name, &addr.RemoteAddr, last.msg_namelen)) {
            return false;
        }

        sockaddr_in6 tmp;
        if (NNetlibaSocket::ExtractDestinationAddress(last, &tmp)) {
            if (memcmp(&tmp.sin6_addr, &addr.MyAddr.sin6_addr, sizeof(struct in6_addr))) {
                return false;
            }
        } else {
            //for outgoing connections self address is 0 (OS routing table), so no CMSG_DATA header attached
            //we can apply SPO only if outgoing IP address of current packets == 0
            if (*BreakAliasing<ui64>(addr.MyAddr.sin6_addr.s6_addr + 0) != 0u || *BreakAliasing<ui64>(addr.MyAddr.sin6_addr.s6_addr + 8) != 0u) {
                return false;
            }
        }

        ui8 tosInQueue;
        if (!NNetlibaSocket::ReadTos(last, &tosInQueue)) {
            return false;
        }

        if (tos != tosInQueue) {
            return false;
        }

        if (LastUdpPacketSize + v.iov_len > mtu) {
            return false;
        }

        // with large enough pages this check will always be successful.
        if (TIoVec* a = PacketsIoVecs.PushBackToContRegion(v, &last.msg_iov)) {
            Y_ASSERT(last.msg_iov + last.msg_iovlen == a);
            last.msg_iovlen++;
            LastUdpPacketSize += v.iov_len;
            SendStat.Merged++;
            //fprintf(stderr, "performing small packet optimization, total size = %d, total packets = %d\n", (int)LastUdpPacketSize, (int)last.msg_iovlen);
            return true;
        }
        return false;
    }

    void TUdpSocket::AddPacketToQueue(size_t size, const TSockAddrPair& addr, ui8 tos, size_t mtu) {
        Y_ASSERT(LastReturnedPacketBuffer);
        Y_ASSERT(size < LastReturnPacketBufferDataSize);

        char* packet = LastReturnedPacketBuffer;
        LastReturnedPacketBuffer = nullptr;

        const size_t packetSize = size + UDP_LOW_LEVEL_HEADER_SIZE;
        Y_ASSERT(packetSize <= UDP_MAX_PACKET_SIZE);

        WritePacketHeader(packet, packetSize, addr.RemoteAddr);

#ifdef SIMULATE_NETWORK_FAILURES
        // can't simulate packet loss - we must add packet to PacketsHeaders because we already allocated space in PacketsDataBuffer.
        if ((RandomNumber<size_t>() % 3) == 0)
            packet[RandomNumber<size_t>() % packetSize] += RandomNumber<size_t>(); // packet broken
#endif

        // Trying to append this packet to already existing in packet queue.
        const TIoVec iovValue = CreateIoVec(packet, packetSize);
        if (TryToApplySmallPacketsOptimization(iovValue, addr, tos, mtu)) {
            return;
        }

        // OK, creating new packet.
        LastUdpPacketSize = packetSize;

        TIoVec& iov = *PacketsIoVecs.PushBack(iovValue);

        Y_ASSERT(!UdpPacketsAddresses.Full());
        sockaddr_in6& name = *UdpPacketsAddresses.PushBack(addr.RemoteAddr);

        char* ctrlBuffer = UdpPacketsAuxData.GetContinuousRegion(CTRL_BUFFER_SIZE);

        UdpPacketsHeaders.resize(UdpPacketsHeaders.size() + 1);
        TMsgHdr& hdr = UdpPacketsHeaders.back().msg_hdr;

        hdr = CreateNl12SendMsgHdr(name, addr.MyAddr, iov, tos, ctrlBuffer, CTRL_BUFFER_SIZE);
    }

    void TUdpSocket::ForgetHeadUdpPackets(const size_t numPackets) {
        // necessary for TCircularPodBuffer::EraseBefore
        if (numPackets == 0) {
            return;
        }
        UdpPacketsAddresses.EraseHead(numPackets);
        UdpPacketsAuxData.EraseHead(numPackets * CTRL_BUFFER_SIZE);
        UdpPacketsHeadersBegin += size_t(numPackets);

        // necessary for TCircularPodBuffer::EraseBefore
        if (UdpPacketsHeadersBegin == UdpPacketsHeaders.size()) {
            PacketsDataBuffer.Clear();
            PacketsIoVecs.Clear();
            LastUdpPacketSize = 0;
        } else {
            Y_ASSERT(UdpPacketsHeaders[UdpPacketsHeadersBegin].msg_hdr.msg_iovlen);
            const TIoVec* iov = UdpPacketsHeaders[UdpPacketsHeadersBegin].msg_hdr.msg_iov;
            PacketsDataBuffer.EraseBefore((const char*)iov->iov_base);
            PacketsIoVecs.CleanupBefore(iov);
        }
    }

    struct TPacketsCountAdder {
        size_t operator()(const size_t base, const TMMsgHdr& next) const {
            return base + (size_t)next.msg_hdr.msg_iovlen;
        }
    };

    // Flushes packet until first error.
    // sentPackets is always updated, failedPacket is only set on non-SEND_BUFFER_OVERFLOW error.
    // Failed packet will be skipped on next call.
    TUdpSocket::ESendError TUdpSocket::FlushPackets(size_t* numSentPackets, TVector<std::pair<char*, size_t>>* failedPackets) {
        Y_ASSERT(!LastReturnedPacketBuffer);

        *numSentPackets = 0;
        failedPackets->clear();
        ESendError err = SEND_OK;

        const size_t totalUdpPackets = GetNumUdpPacketsInQueue();
        size_t sentUdpPackets = 0;

        while (sentUdpPackets < totalUdpPackets) {
            TMMsgHdr* udpPackets = &UdpPacketsHeaders[UdpPacketsHeadersBegin + sentUdpPackets];
            int sent = 0;

            if (S->IsSendMMsgSupported()) {
                const unsigned int udpPacketsLeft(totalUdpPackets - sentUdpPackets);
                sent = S->SendMMsg(udpPackets, udpPacketsLeft, 0);
                if (sent > 0 && (ui32)sent != udpPacketsLeft) {
                    //              printf("sendmmsg ate %d of %u packets\n", sent, udpPacketsLeft);
                }
            } else {
                const ssize_t rv = S->SendMsg(&udpPackets->msg_hdr, 0, FF_ALLOW_FRAG);
                sent = rv < 0 ? -1 : 1;
            }
#ifdef SIMULATE_NO_ROUTE_TO_HOST
            if (RandomNumber<size_t>() % 997 == 0) {
                err = TUdpSocket::SEND_NO_ROUTE_TO_HOST;
                break;
            }
#endif

            if (sent < 0) {
                err = GetSendErrorFromErrno(LastSystemError());
                break;
            }

            sentUdpPackets += (size_t)sent;
        }

        if (sentUdpPackets > 0 && UseSmallPacketsOptimization) {
            const TMMsgHdr* udpPackets = &UdpPacketsHeaders[UdpPacketsHeadersBegin];
            *numSentPackets = Accumulate(udpPackets, udpPackets + sentUdpPackets, 0, TPacketsCountAdder());
        } else {
            *numSentPackets = sentUdpPackets;
        }

        if (sentUdpPackets == totalUdpPackets) {
            ClearPacketsQueue();

        } else {
            ForgetHeadUdpPackets(sentUdpPackets);

            if (err == SEND_NO_ROUTE_TO_HOST || err == SEND_EINVAL) {
                AddPacketsFromUdpPacket(UdpPacketsHeaders[UdpPacketsHeadersBegin].msg_hdr, failedPackets);
                ForgetHeadUdpPackets(1);

            } else if (err == SEND_BUFFER_OVERFLOW) {
                if (!S->IncreaseSendBuff()) {
                    fprintf(stderr, "Socket, port: %d can`t adjust send buffer size (cur value: %d), "
                                    "please check net.core.wmem_max\n",
                            S->GetPort(), S->GetSendSysSocketSize());
                }
            } else {
                Y_ASSERT(false);
            }
        }
        return err;
    }

    void TUdpSocket::ClearPacketsQueue() {
        UdpPacketsAddresses.Clear();
        UdpPacketsAuxData.Clear();
        PacketsDataBuffer.Clear();
        PacketsIoVecs.Clear();
        UdpPacketsHeaders.resize(0);
        UdpPacketsHeadersBegin = 0;
        LastReturnedPacketBuffer = nullptr;
        LastReturnPacketBufferDataSize = 0;
        LastUdpPacketSize = 0;
    }

    ///////////////////////////////////////////////////////////////////////////////

    void TUdpSocket::CacheContinuationUdpPacket(const TUdpRecvPacket& pkt, const size_t pktSize, const TSockAddrPair& addr) {
        Y_ASSERT(!RecvContUdpPacket);
        Y_ASSERT((size_t)(pkt.DataStart + pkt.DataSize) < pktSize);

        RecvContUdpPacket.Reset(TUdpHostRecvBufAlloc::Clone(&pkt));
        RecvContUdpPacketSize = pktSize;
        RecvContAddress = addr;
    }

    TUdpRecvPacket* TUdpSocket::RecvContinuationPacket(TSockAddrPair* addr) {
        if (!!RecvContUdpPacket) {
            RecvContUdpPacket->DataStart += RecvContUdpPacket->DataSize;

            const int bytesLeft = RecvContUdpPacketSize - RecvContUdpPacket->DataStart;
            if (bytesLeft > 0) {
                const char* payload = RecvContUdpPacket->Data.get() + RecvContUdpPacket->DataStart;

                if (CheckPacketIntegrity(payload, (size_t)bytesLeft, RecvContAddress)) {
                    RecvContUdpPacket->DataSize = UDP_LOW_LEVEL_HEADER_SIZE + ReadPacketDataSize(payload);
                    *addr = RecvContAddress;
                    return TUdpHostRecvBufAlloc::Clone(RecvContUdpPacket.Get());
                }

                fprintf(stderr, "NETLIBA::TUdpSocket: continuation packet integrity check failed, ignoring tail!\n");
                // packet integrity check failed
                // ignore all continuation packets because we can't probably say where are their starts.
            }

            RecvContUdpPacket.Destroy();
            Zero(RecvContAddress);
            RecvContUdpPacketSize = 0;
        }
        return nullptr;
    }

    TUdpRecvPacket* TUdpSocket::Recv(TSockAddrPair* addr) {
        TAutoPtr<TUdpRecvPacket> result = nullptr;

        for (;;) {
            result = RecvContinuationPacket(addr);
            if (!!result) {
                break;
            }

            result = S->Recv(&addr->RemoteAddr, &addr->MyAddr, UDP_NETLIBA_VERSION_V12);
            if (!result) {
                return nullptr;
            }

            Y_ASSERT(result->DataStart == 0);
            const size_t recvSize = result->DataSize;

            // skip whole corrupted packet even with small packet optimization
            if (!CheckPacketIntegrity(result->Data.get(), recvSize, *addr)) {
                continue;
            }

            result->DataSize = UDP_LOW_LEVEL_HEADER_SIZE + ReadPacketDataSize(result->Data.get());

            if ((size_t)result->DataSize != recvSize) {
                CacheContinuationUdpPacket(*result, recvSize, *addr);
            }
            break;
        }

        // fixes offsets also for continuation packets
        result->DataStart += UDP_LOW_LEVEL_HEADER_SIZE;
        result->DataSize -= UDP_LOW_LEVEL_HEADER_SIZE;

        return result.Release();
    }

    float TUdpSocket::GetAndResetMergeRatio() {
        if (SendStat.Total == 0)
            return 0;
        //increase merged counter since we have uncounted first packet
        const float tmp = ((SendStat.Merged) ? ++SendStat.Merged : 0) / (float)SendStat.Total;
        SendStat = TUdpSocket::TSendPacketsStat();
        return tmp;
    }

    void TUdpSocket::SetRecvLagTime(NHPTimer::STime time) {
        S->SetRecvLagTime(time);
    }

    TString TUdpSocket::GetSockDebug() const {
        TString result;
        result += "SendSysSocketSize (SO_SNDBUF):\t";
        result += ToString(S->GetSendSysSocketSize());
        result += "\n";
        result += "SmallPacketsMergeRatio:\t";
        result += ToString(MergeRatio);
        result += "\n";
        return result;
    }

    void TUdpSocket::UpdateStats() {
        MergeRatio = GetAndResetMergeRatio();
    }

    bool TUdpSocket::IsLocal(const TUdpAddress& address) const {
        return IpParams.IsLocal(address);
    }

}
