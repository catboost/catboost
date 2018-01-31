#pragma once

#include "cuda_manager.h"
#include <catboost/cuda/utils/countdown_latch.h>
#include <library/threading/local_executor/local_executor.h>

namespace NCudaLib {
    //helper method for run per device subtask with child cudaManagers
    template <class TTask>
    inline void RunPerDeviceSubtasks(TTask&& task) {
        auto& manager = NCudaLib::GetCudaManager();
        {
            const auto threadCount = static_cast<const ui32>(NPar::LocalExecutor().GetThreadCount());
            const ui32 deviceCount = manager.GetDeviceCount();
            if (threadCount < deviceCount) {
                NPar::LocalExecutor().RunAdditionalThreads(deviceCount - threadCount);
            }
        }
        manager.WaitComplete();

        NCudaLib::TChildCudaManagerInitializer consistentChildrenGuard;
        TCountDownLatch latch(manager.GetDeviceCount());
        NPar::AsyncParallelFor(0, manager.GetDeviceCount(), [&](ui32 device) {
            {
                auto freeGuard = consistentChildrenGuard.Initialize(device);
                task(device);
            }
            latch.Countdown();
        });
        latch.Wait();
    }
}
