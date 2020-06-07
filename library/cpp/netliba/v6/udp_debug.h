#pragma once

namespace NNetliba {
    struct TRequesterPendingDataStats {
        int InpCount, OutCount;
        ui64 InpDataSize, OutDataSize;

        TRequesterPendingDataStats() {
            memset(this, 0, sizeof(*this));
        }
    };

    struct TRequesterQueueStats {
        int ReqCount, RespCount;
        ui64 ReqQueueSize, RespQueueSize;

        TRequesterQueueStats() {
            memset(this, 0, sizeof(*this));
        }
    };
}
