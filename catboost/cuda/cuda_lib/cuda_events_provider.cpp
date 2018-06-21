#include "cuda_events_provider.h"

namespace NCatboostCuda {
}

void NCudaLib::TCudaEventsProvider::RequestHandle(bool disableTimming) {
    cudaEvent_t event;
    if (disableTimming) {
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync));
        FreeHandlesWithoutTiming.push_back(event);
    } else {
        CUDA_SAFE_CALL(cudaEventCreate(&event));
        FreeHandles.push_back(event);
    }
}

NCudaLib::TCudaEventPtr NCudaLib::TCudaEventsProvider::Create(bool disableTimming) {
    TGuard<TSpinLock> lock(Lock);

    bool needMore = disableTimming ? FreeHandlesWithoutTiming.size() == 0 : FreeHandles.size() == 0;
    if (needMore) {
        for (ui64 i = 0; i < RequestHandlesBatchSize; ++i) {
            RequestHandle(disableTimming);
        }
    }

    cudaEvent_t handle;

    if (disableTimming) {
        handle = FreeHandlesWithoutTiming.back();
        FreeHandlesWithoutTiming.pop_back();
    } else {
        handle = FreeHandles.back();
        FreeHandles.pop_back();
    }

    return MakeHolder<TCudaEvent>(handle, disableTimming, this);
}
