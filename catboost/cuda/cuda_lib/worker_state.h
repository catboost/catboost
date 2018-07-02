#pragma once

#include <util/ysaveload.h>

namespace NCudaLib {
    struct TMemoryState {
        ui64 FreeGpuRam = 0;
        ui64 RequestedGpuRam = 0;
        ui64 FreePinnedRam = 0;
        ui64 RequestedPinnedRam = 0;

        Y_SAVELOAD_DEFINE(FreeGpuRam, RequestedGpuRam, FreePinnedRam, RequestedPinnedRam);
    };

    class IWorkerStateProvider {
    public:
        virtual ~IWorkerStateProvider() noexcept(false) {
        }

        virtual TMemoryState GetMemoryState() const = 0;
    };

}
