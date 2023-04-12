#include "stdafx.h"
#include "udp_socket.h"
#include "block_chain.h"
#include "udp_address.h"

#include <util/datetime/cputimer.h>
#include <util/system/spinlock.h>
#include <util/random/random.h>

#include <library/cpp/netliba/socket/socket.h>

#include <errno.h>

//#define SIMULATE_NETWORK_FAILURES
// there is no explicit bit in the packet header for last packet of transfer
// last packet is just smaller then maximum size

namespace NNetliba {
    static bool LocalHostFound;
    enum {
        IPv4 = 0,
        IPv6 = 1
    };

    struct TIPv6Addr {
        ui64 Network, Interface;

        TIPv6Addr() {
            Zero(*this);
        }
        TIPv6Addr(ui64 n, ui64 i)
            : Network(n)
            , Interface(i)
        {
        }
    };
    inline bool operator==(const TIPv6Addr& a, const TIPv6Addr& b) {
        return a.Interface == b.Interface && a.Network == b.Network;
    }

    static ui32 LocalHostIP[2];
    static TVector<ui32> LocalHostIPList[2];
    static TVector<TIPv6Addr> LocalHostIPv6List;

    // Struct sockaddr_in6 does not have ui64-array representation
    // so we add it here. This avoids "strict aliasing" warnings
    typedef union {
        in6_addr Addr;
        ui64 Addr64[2];
    } TIPv6AddrUnion;

    static ui32 GetIPv6SuffixCrc(const sockaddr_in6& addr) {
        TIPv6AddrUnion a;
        a.Addr = addr.sin6_addr;
        ui64 suffix = a.Addr64[1];
        return (suffix & 0xffffffffll) + (suffix >> 32);
    }

    bool InitLocalIPList() {
        // Do not use TMutex here: it has a non-trivial destructor which will be called before
        // destruction of current thread, if its TThread declared as global/static variable.
        static TAdaptiveLock cs;
        TGuard lock(cs);

        if (LocalHostFound)
            return true;

        TVector<TUdpAddress> addrs;
        if (!GetLocalAddresses(&addrs))
            return false;
        for (int i = 0; i < addrs.ysize(); ++i) {
            const TUdpAddress& addr = addrs[i];
            if (addr.IsIPv4()) {
                LocalHostIPList[IPv4].push_back(addr.GetIPv4());
                LocalHostIP[IPv4] = addr.GetIPv4();
            } else {
                sockaddr_in6 addr6;
                GetWinsockAddr(&addr6, addr);

                LocalHostIPList[IPv6].push_back(GetIPv6SuffixCrc(addr6));
                LocalHostIP[IPv6] = GetIPv6SuffixCrc(addr6);
                LocalHostIPv6List.push_back(TIPv6Addr(addr.Network, addr.Interface));
            }
        }
        LocalHostFound = true;
        return true;
    }

    template <class T, class TElem>
    inline bool IsInSet(const T& c, const TElem& e) {
        return Find(c.begin(), c.end(), e) != c.end();
    }

    bool IsLocalIPv4(ui32 ip) {
        return IsInSet(LocalHostIPList[IPv4], ip);
    }
    bool IsLocalIPv6(ui64 network, ui64 iface) {
        return IsInSet(LocalHostIPv6List, TIPv6Addr(network, iface));
    }

    //////////////////////////////////////////////////////////////////////////
    void TNetSocket::Open(int port) {
        TIntrusivePtr<NNetlibaSocket::ISocket> theSocket = NNetlibaSocket::CreateSocket();
        theSocket->Open(port);
        Open(theSocket);
    }

    void TNetSocket::Open(const TIntrusivePtr<NNetlibaSocket::ISocket>& socket) {
        s = socket;
        if (IsValid()) {
            PortCrc = s->GetSelfAddress().sin6_port;
        }
    }

    void TNetSocket::Close() {
        if (IsValid()) {
            s->Close();
        }
    }

    void TNetSocket::SendSelfFakePacket() const {
        s->CancelWait();
    }

    inline ui32 CalcAddressCrc(const sockaddr_in6& addr) {
        Y_ASSERT(addr.sin6_family == AF_INET6);
        const ui64* addr64 = (const ui64*)addr.sin6_addr.s6_addr;
        const ui32* addr32 = (const ui32*)addr.sin6_addr.s6_addr;
        if (addr64[0] == 0 && addr32[2] == 0xffff0000ll) {
            // ipv4
            return addr32[3];
        } else {
            // ipv6
            return GetIPv6SuffixCrc(addr);
        }
    }

    TNetSocket::ESendError TNetSocket::SendTo(const char* buf, int size, const sockaddr_in6& toAddress, const EFragFlag frag) const {
        Y_ASSERT(size >= UDP_LOW_LEVEL_HEADER_SIZE);
        ui32 crc = CalcChecksum(buf + UDP_LOW_LEVEL_HEADER_SIZE, size - UDP_LOW_LEVEL_HEADER_SIZE);
        ui32 ipCrc = CalcAddressCrc(toAddress);
        ui32 portCrc = toAddress.sin6_port;
        *(ui32*)buf = crc + ipCrc + portCrc;
#ifdef SIMULATE_NETWORK_FAILURES
        if ((RandomNumber<size_t>() % 3) == 0)
            return true; // packet lost
        if ((RandomNumber<size_t>() % 3) == 0)
            (char&)(buf[RandomNumber<size_t>() % size]) += RandomNumber<size_t>(); // packet broken
#endif

        char tosBuffer[NNetlibaSocket::TOS_BUFFER_SIZE];
        void* t = NNetlibaSocket::CreateTos(Tos, tosBuffer);
        const NNetlibaSocket::TIoVec iov = NNetlibaSocket::CreateIoVec((char*)buf, size);
        NNetlibaSocket::TMsgHdr hdr = NNetlibaSocket::CreateSendMsgHdr(toAddress, iov, t);

        const int rv = s->SendMsg(&hdr, 0, frag);
        if (rv < 0) {
            if (errno == EHOSTUNREACH || errno == ENETUNREACH) {
                return SEND_NO_ROUTE_TO_HOST;
            } else {
                return SEND_BUFFER_OVERFLOW;
            }
        }
        Y_ASSERT(rv == size);
        return SEND_OK;
    }

    inline bool CrcMatches(ui32 pktCrc, ui32 crc, const sockaddr_in6& addr) {
        Y_ASSERT(LocalHostFound);
        Y_ASSERT(addr.sin6_family == AF_INET6);
        // determine our ip address family based on the sender address
        // address family can not change in network, so sender address type determines type of our address used
        const ui64* addr64 = (const ui64*)addr.sin6_addr.s6_addr;
        const ui32* addr32 = (const ui32*)addr.sin6_addr.s6_addr;
        yint ipType;
        if (addr64[0] == 0 && addr32[2] == 0xffff0000ll) {
            // ipv4
            ipType = IPv4;
        } else {
            // ipv6
            ipType = IPv6;
        }
        if (crc + LocalHostIP[ipType] == pktCrc) {
            return true;
        }
        // crc failed
        // check if packet was sent to different IP address
        for (int idx = 0; idx < LocalHostIPList[ipType].ysize(); ++idx) {
            ui32 otherIP = LocalHostIPList[ipType][idx];
            if (crc + otherIP == pktCrc) {
                LocalHostIP[ipType] = otherIP;
                return true;
            }
        }
        // crc is really failed, discard packet
        return false;
    }

    bool TNetSocket::RecvFrom(char* buf, int* size, sockaddr_in6* fromAddress) const {
        for (;;) {
            int rv;
            if (s->IsRecvMsgSupported()) {
                const NNetlibaSocket::TIoVec v = NNetlibaSocket::CreateIoVec(buf, *size);
                NNetlibaSocket::TMsgHdr hdr = NNetlibaSocket::CreateRecvMsgHdr(fromAddress, v);
                rv = s->RecvMsg(&hdr, 0);

            } else {
                sockaddr_in6 dummy;
                TAutoPtr<NNetlibaSocket::TUdpRecvPacket> pkt = s->Recv(fromAddress, &dummy, -1);
                rv = !!pkt ? pkt->DataSize - pkt->DataStart : -1;
                if (rv > 0) {
                    memcpy(buf, pkt->Data.get() + pkt->DataStart, rv);
                }
            }

            if (rv < 0)
                return false;
            // ignore empty packets
            if (rv == 0)
                continue;
            // skip small packets
            if (rv < UDP_LOW_LEVEL_HEADER_SIZE)
                continue;
            *size = rv;
            ui32 pktCrc = *(ui32*)buf;
            ui32 crc = CalcChecksum(buf + UDP_LOW_LEVEL_HEADER_SIZE, rv - UDP_LOW_LEVEL_HEADER_SIZE);
            if (!CrcMatches(pktCrc, crc + PortCrc, *fromAddress)) {
                // crc is really failed, discard packet
                continue;
            }
            return true;
        }
    }

    void TNetSocket::Wait(float timeoutSec) const {
        s->Wait(timeoutSec);
    }

    void TNetSocket::SetTOS(int n) const {
        Tos = n;
    }

    bool TNetSocket::Connect(const sockaddr_in6& addr) {
        // "connect" - meaningless operation
        // needed since port unreachable is routed only to "connected" udp sockets in ingenious FreeBSD
        if (s->Connect((sockaddr*)&addr, sizeof(addr)) < 0) {
            if (errno == EHOSTUNREACH || errno == ENETUNREACH) {
                return false;
            } else {
                Y_ASSERT(0);
            }
        }
        return true;
    }

    void TNetSocket::SendEmptyPacket() {
        NNetlibaSocket::TIoVec v;
        Zero(v);

        // darwin ignores packets with msg_iovlen == 0, also windows implementation uses sendto of first iovec.
        NNetlibaSocket::TMsgHdr hdr;
        Zero(hdr);
        hdr.msg_iov = &v;
        hdr.msg_iovlen = 1;

        s->SendMsg(&hdr, 0, FF_ALLOW_FRAG); // sends empty packet to connected address
    }

    bool TNetSocket::IsHostUnreachable() {
#ifdef _win_
        char buf[10000];
        sockaddr_in6 fromAddress;

        const NNetlibaSocket::TIoVec v = NNetlibaSocket::CreateIoVec(buf, Y_ARRAY_SIZE(buf));
        NNetlibaSocket::TMsgHdr hdr = NNetlibaSocket::CreateRecvMsgHdr(&fromAddress, v);

        const ssize_t rv = s->RecvMsg(&hdr, 0);
        if (rv < 0) {
            int err = WSAGetLastError();
            if (err == WSAECONNRESET)
                return true;
        }
#else
        int err = 0;
        socklen_t bufSize = sizeof(err);
        s->GetSockOpt(SOL_SOCKET, SO_ERROR, (char*)&err, &bufSize);
        if (err == ECONNREFUSED)
            return true;
#endif
        return false;
    }
}
