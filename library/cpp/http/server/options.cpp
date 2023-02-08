#include "options.h"

#include <util/string/cast.h>
#include <util/digest/numeric.h>
#include <util/network/ip.h>
#include <util/network/socket.h>
#include <util/generic/hash_set.h>
#include <util/generic/yexception.h>

using TAddr = THttpServerOptions::TAddr;

static inline TString AddrToString(const TAddr& addr) {
    return addr.Addr + ":" + ToString(addr.Port);
}

static inline TNetworkAddress ToNetworkAddr(const TString& address, ui16 port) {
    if (address.empty() || address == TStringBuf("*")) {
        return TNetworkAddress(port);
    }

    return TNetworkAddress(address, port);
}

void THttpServerOptions::BindAddresses(TBindAddresses& ret) const {
    THashSet<TString> check;

    for (auto addr : BindSockaddr) {
        if (!addr.Port) {
            addr.Port = Port;
        }

        const TString straddr = AddrToString(addr);

        if (check.find(straddr) == check.end()) {
            check.insert(straddr);
            ret.push_back(ToNetworkAddr(addr.Addr, addr.Port));
        }
    }

    if (ret.empty()) {
        ret.push_back(Host ? TNetworkAddress(Host, Port) : TNetworkAddress(Port));
    }
}
