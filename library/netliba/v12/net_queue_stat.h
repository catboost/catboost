#pragma once

#include <util/generic/ptr.h>

namespace NNetliba_v12 {
    struct IPeerQueueStats: public TThrRefBase {
        virtual int GetPacketCount() = 0;
    };
}
