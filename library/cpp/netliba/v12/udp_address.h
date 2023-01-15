#pragma once

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/defaults.h>
#include <util/str_stl.h>

struct sockaddr_in6;

namespace NNetliba_v12 {
    bool IsValidIPv6(const char* str);

    struct TUdpAddress {
        ui64 Network, Interface;
        int Scope, Port;

        TUdpAddress()
            : Network(0)
            , Interface(0)
            , Scope(0)
            , Port(0)
        {
        }
        bool IsIPv4() const {
            return (Network == 0 && (Interface & 0xffffffffll) == 0xffff0000ll);
        }
        ui32 GetIPv4() const {
            return Interface >> 32;
        }
    };
    inline bool operator==(const TUdpAddress& a, const TUdpAddress& b) {
        return a.Network == b.Network && a.Interface == b.Interface && a.Scope == b.Scope && a.Port == b.Port;
    }
    inline bool operator!=(const TUdpAddress& a, const TUdpAddress& b) {
        return !(a == b);
    }
    struct TUdpAddressHash {
        inline size_t operator()(const TUdpAddress& addr) const {
            ui64 result = addr.Interface + (ui64)addr.Port * 389461ULL;
#ifdef _i386_
            result = result % (1ULL << 32);
#endif
            return (size_t)result;
        }
    };
}

template <>
struct THash<NNetliba_v12::TUdpAddress> {
    inline size_t operator()(const NNetliba_v12::TUdpAddress& addr) const {
        return NNetliba_v12::TUdpAddressHash()(addr);
    }
};

namespace NNetliba_v12 {
    enum EUdpAddressType {
        UAT_ANY,
        UAT_IPV4,
        UAT_IPV6,
    };

    // accepts sockaddr_in & sockaddr_in6
    void GetUdpAddress(TUdpAddress* res, const sockaddr_in6& addr);
    TUdpAddress GetUdpAddress(const sockaddr_in6& addr);
    // generates sockaddr_in6 for both ipv4 & ipv6
    void GetWinsockAddr(sockaddr_in6* res, const TUdpAddress& addr);
    // generates sockaddr_in for ipv4, sockaddr_in6 for ipv6
    void GetWinsockAddrForMLNX(sockaddr_in6* res, const TUdpAddress& addr);
    // supports formats like hostname, hostname:124, 127.0.0.1, 127.0.0.1:80, fe34::12, [fe34::12]:80
    TUdpAddress CreateAddress(const TString& server, int defaultPort, EUdpAddressType type = UAT_ANY);
    TString GetAddressAsString(const TUdpAddress& addr);

    bool GetLocalAddresses(TVector<TUdpAddress>* addrs);
}
