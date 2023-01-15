#pragma once

#include "cuda_manager.h"
#include <catboost/cuda/utils/countdown_latch.h>
#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCudaLib {
    //helper method for run per device subtask with child cudaManagers
    template <class TTask>
    inline void RunPerDeviceSubtasks(TTask&& task, bool useRemoteDevices = true) {
        auto& manager = NCudaLib::GetCudaManager();
        auto devices = manager.GetDevices(!useRemoteDevices);
        {
            const auto threadCount = static_cast<const ui32>(NPar::LocalExecutor().GetThreadCount());
            const ui32 deviceCount = devices.size();
            if (threadCount < deviceCount) {
                NPar::LocalExecutor().RunAdditionalThreads(deviceCount - threadCount);
            }
        }
        manager.WaitComplete();

        NCudaLib::TChildCudaManagerInitializer consistentChildrenGuard;

        TCountDownLatch latch(devices.size());
        NPar::AsyncParallelFor(0, devices.size(), [&](ui32 i) {
            {
                const ui32 dev = devices[i];
                auto freeGuard = consistentChildrenGuard.Initialize(dev);
                task(dev);
            }
            latch.Countdown();
        });
        latch.Wait();
    }
}
