#pragma once

#include <util/generic/vector.h>
#include "udp_address.h"

namespace NNetliba_v12 {
    ///////////////////////////////////////////////////////////////////////////////

    enum EIpType {
        IPv4 = 0,
        IPv6 = 1
    };

    ///////////////////////////////////////////////////////////////////////////////

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

    ///////////////////////////////////////////////////////////////////////////////

    // Struct sockaddr_in6 does not have ui64-array representation
    // so we add it here. This avoids "strict aliasing" warnings
    typedef union {
        in6_addr Addr;
        ui64 Addr64[2];
        ui32 Addr32[4];
    } TIPv6AddrUnion;

    inline EIpType GetIpType(const sockaddr_in6& addr) {
        TIPv6AddrUnion a;
        a.Addr = addr.sin6_addr;
        if (a.Addr64[0] == 0 && a.Addr32[2] == 0xffff0000ll) {
            return IPv4;
        } else {
            return IPv6;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    class TLocalIpParams: public TNonCopyable {
    public:
        bool Init();
        const TVector<ui32>& GetLocaIpCrcs(const EIpType ipType) const {
            Y_ASSERT((size_t)ipType < Y_ARRAY_SIZE(LocalIPListCrcs));
            return LocalIPListCrcs[ipType];
        }
        bool IsLocal(const TUdpAddress& address) const;

    private:
        TVector<ui32> LocalIPListCrcs[2];
        TVector<TIPv6Addr> LocaIPv6List;
    };

    ///////////////////////////////////////////////////////////////////////////////

    ui32 CalcAddressChecksum(const sockaddr_in6& addr);

    ///////////////////////////////////////////////////////////////////////////////

}
