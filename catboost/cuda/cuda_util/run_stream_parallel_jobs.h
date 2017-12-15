#pragma once

#include <catboost/cuda/cuda_lib/cuda_manager.h>

template <class TFunc>
inline void RunInStreams(ui32 taskCount, ui32 streamCount, TFunc&& func) {
    TVector<TComputationStream> streams;
    auto& manager = NCudaLib::GetCudaManager();

    if (streamCount == 1) {
        streams.push_back(manager.DefaultStream());
    } else {
        for (ui32 i = 0; i < streamCount; ++i) {
            streams.push_back(manager.RequestStream());
        }
        manager.WaitComplete();
    }
    for (ui32 i = 0; i < taskCount; ++i) {
        func(i, streams[i % streamCount].GetId());
    }
    if (streams.size() > 1) {
        manager.WaitComplete();
    }
}
