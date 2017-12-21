#pragma once

#include "remote_device_future.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/task.h>

namespace NCudaLib {
    struct TMemoryState {
        ui64 FreeGpuRam;
        ui64 RequestedGpuRam;
        ui64 FreePinnedRam;
        ui64 RequestedPinnedRam;
        Y_SAVELOAD_DEFINE(FreeGpuRam, RequestedGpuRam, FreePinnedRam, RequestedPinnedRam);
    };

    class TMemoryStateTask: public IGpuCommand {
    public:
        TMemoryStateTask()
            : IGpuCommand(EGpuHostCommandType::MemoryState)
        {
        }

        void Set(const TMemoryState& state) {
            Promise.SetValue(state);
        }

        TDeviceFuture<TMemoryState> GetResult() {
            Promise = NThreading::NewPromise<TMemoryState>();
            return TDeviceFuture<TMemoryState>(Promise.GetFuture());
        }

    private:
        NThreading::TPromise<TMemoryState> Promise;
    };
}
