#pragma once

#include <util/network/socket.h>

#include <util/generic/string.h>
#include <util/generic/ptr.h>

namespace NDns {
    typedef TAutoPtr<TNetworkAddress> TNetworkAddressPtr;

    TNetworkAddressPtr ThreadedResolve(const TString& host, ui16 port);
}
