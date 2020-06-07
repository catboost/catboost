#pragma once

#include <util/generic/ptr.h>

namespace NNetliba {
    struct IPeerQueueStats: public TThrRefBase {
        virtual int GetPacketCount() = 0;
    };
}
