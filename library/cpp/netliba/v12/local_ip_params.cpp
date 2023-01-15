#include "stdafx.h"
#include "local_ip_params.h"

namespace NNetliba_v12 {
    static ui32 GetIPv6SuffixCrc(const sockaddr_in6& addr) {
        TIPv6AddrUnion a;
        a.Addr = addr.sin6_addr;
        const ui64 suffix = a.Addr64[1];
        return (suffix & 0xffffffffll) + (suffix >> 32);
    }

    ui32 CalcAddressChecksum(const sockaddr_in6& addr) {
        Y_ASSERT(addr.sin6_family == AF_INET6);
        if (GetIpType(addr) == IPv4) {
            TIPv6AddrUnion a;
            a.Addr = addr.sin6_addr;
            return a.Addr32[3];
        } else {
            return GetIPv6SuffixCrc(addr);
        }
    }

    static ui32 CalcAddressCrcImpl(const TUdpAddress& addr) {
        sockaddr_in6 addr6;
        GetWinsockAddr(&addr6, addr);
        return CalcAddressChecksum(addr6);
    }

    ui32 CalcAddressChecksum(const TUdpAddress& addr) {
        if (addr.IsIPv4()) {
            Y_ASSERT(addr.GetIPv4() == CalcAddressCrcImpl(addr));
            return addr.GetIPv4();
        }
        return CalcAddressCrcImpl(addr);
    }

    bool TLocalIpParams::Init() {
        TVector<TUdpAddress> addrs;

        if (!GetLocalAddresses(&addrs))
            return false;

        for (int i = 0; i < addrs.ysize(); ++i) {
            const TUdpAddress& addr = addrs[i];
            const ui32 crc = CalcAddressChecksum(addr);

            if (addr.IsIPv4()) {
                LocalIPListCrcs[IPv4].push_back(crc);
            } else {
                LocalIPListCrcs[IPv6].push_back(crc);
                LocaIPv6List.push_back(TIPv6Addr(addr.Network, addr.Interface));
            }
        }
        return true;
    }

    bool TLocalIpParams::IsLocal(const TUdpAddress& addr) const {
        if (addr.IsIPv4()) {
            return IsIn(LocalIPListCrcs[IPv4], addr.GetIPv4());
        } else {
            return IsIn(LocaIPv6List, TIPv6Addr(addr.Network, addr.Interface));
        }
    }

}
