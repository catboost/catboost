#pragma once

#include "base.h"

#include <library/cpp/cuda/exception/exception.h>

#include <util/system/guard.h>
#include <util/system/mutex.h>

#include <array>
#include <atomic>
#include <cstddef>

#include <cuda_runtime.h>


namespace NKernel {

    class TArchProps {
        // TODO: should we support more than 16 devices per host?
        constexpr static size_t MAX_DEVICE_ID = 16;

        static TArchProps Instance;

        std::array<cudaDeviceProp, MAX_DEVICE_ID> Props;
        std::array<std::atomic<bool>, MAX_DEVICE_ID> PropsCached = {};
        std::array<TMutex, MAX_DEVICE_ID> PropsMutexes = {};

        TArchProps() = default;

        const cudaDeviceProp& GetCachedPropsForDevice(int devId) {
            if (!PropsCached[devId]) {
                with_lock(PropsMutexes[devId]) {
                    if (!PropsCached[devId]) {
                        CUDA_SAFE_CALL(cudaGetDeviceProperties(&Props[devId], devId));
                        PropsCached[devId] = true;
                    }
                }
            }
            return Props[devId];
        }

        const cudaDeviceProp& GetCachedPropsForCurrentDevice() {
            return GetCachedPropsForDevice(NCuda::GetDevice());
        }

    public:

        static int MaxBlockCount(int dim = 0) {
            return Instance.GetCachedPropsForCurrentDevice().maxGridSize[dim];
        }

        static int SMCount() {
            return Instance.GetCachedPropsForCurrentDevice().multiProcessorCount;
        }

        static size_t GlobalMemorySize() {
            return Instance.GetCachedPropsForCurrentDevice().totalGlobalMem;
        }

        static int GetMajorVersion() {
            return Instance.GetCachedPropsForCurrentDevice().major;
        }

        static int GetMajorVersion(int devId) {
            return Instance.GetCachedPropsForDevice(devId).major;
        }

        static int GetMinorVersion(int devId) {
            return Instance.GetCachedPropsForDevice(devId).minor;
        }

        // FP16 is accessible if compability version is at least 5.3
        // https://h.yandex-team.ru/?https%3A%2F%2Fdocs.nvidia.com%2Fcuda%2Fcuda-c-programming-guide%2Findex.html%23arithmetic-instructions
        static bool HasFp16(int devId) {
            if (GetMajorVersion(devId) < 5) {
                return false;
            }
            return !(GetMajorVersion(devId) == 5 && GetMinorVersion(devId) < 3);
        }

        // Fast FP16 is accessible if compability version is not 6.1
        static bool HasFastFp16(int devId) {
            if (!HasFp16(devId)) {
                return false;
            }
            return GetMajorVersion(devId) != 6 || GetMinorVersion(devId) != 1;
        }
    };


}
