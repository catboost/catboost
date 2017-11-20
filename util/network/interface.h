#pragma once

#include "address.h"

#include <util/generic/vector.h>

namespace NAddr {
    struct TNetworkInterface {
        TString Name;
        IRemoteAddrRef Address;
        IRemoteAddrRef Mask;
    };

    using TNetworkInterfaceList = TVector<TNetworkInterface>;

    TNetworkInterfaceList GetNetworkInterfaces();
}
