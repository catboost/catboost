#pragma once

#include <util/network/socket.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

namespace NDns {
    struct TResolveInfo {
        inline TResolveInfo(const TStringBuf& host, ui16 port)
            : Host(host)
            , Port(port)
        {
        }

        TStringBuf Host;
        ui16 Port;
    };

    struct TResolvedHost {
        inline TResolvedHost(const TString& host, const TNetworkAddress& addr) noexcept
            : Host(host)
            , Addr(addr)
            , Id(0)
        {
        }

        TString Host; //resolved hostname (from TResolveInfo, - before aliasing)
        TNetworkAddress Addr;
        size_t Id; //cache record id
    };

    // Resolving order:
    //   1. check local thread cache, return if found
    //   2. check global cache, return if found
    //   3. search alias for hostname, if found, continue resolving alias
    //   4. normal resolver
    const TResolvedHost* CachedResolve(const TResolveInfo& ri);

    //like previous, but at stage 4 use separate thread for resolving (created on first usage)
    //useful in green-threads with tiny stack
    const TResolvedHost* CachedThrResolve(const TResolveInfo& ri);

    //create alias for host, which can be used for static resolving (when alias is ip address)
    void AddHostAlias(const TString& host, const TString& alias);
}
