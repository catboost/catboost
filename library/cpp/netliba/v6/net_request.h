#pragma once

#include <util/generic/guid.h>
#include "udp_address.h"

namespace NNetliba {
    class TRopeDataPacket;

    struct TRequest {
        TUdpAddress Address;
        TGUID Guid;
        TAutoPtr<TRopeDataPacket> Data;
    };

}
