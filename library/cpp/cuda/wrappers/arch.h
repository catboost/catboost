#pragma once

#include "base.h"

#include <library/cpp/cuda/exception/exception.h>

#include <array>
#include <cstddef>

// temporary workaround for strict dependencies
#include <utility>

#include <cuda_runtime.h>


namespace NKernel {

    class TArchProps {
        // TODO: should we support more than 16 devices per host?
        constexpr static size_t MAX_DEVICE_ID = 16;

        static TArchProps Instance;

        std::array<cudaDeviceProp, MAX_DEVICE_ID> Props;
        std::array<bool, MAX_DEVICE_ID> PropsCached = {};

        TArchProps() = default;

        inline void CacheProps(int devId) {
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&Props[devId], devId));
            Instance.PropsCached[devId] = true;
        }

        inline void CachePropsIfNotCached(int devId) {
            if (!PropsCached[devId]) {
                CacheProps(devId);
            }
        }

        inline int GetCurrentDevice() {
            int devId = NCuda::GetDevice();
            if (!PropsCached[devId]) {
                CacheProps(devId);
            }
            return devId;
        }

    public:

        static int MaxBlockCount(int dim = 0) {
            int devId = Instance.GetCurrentDevice();
            return Instance.Props[devId].maxGridSize[dim];
        }

        static int SMCount() {
            int devId = Instance.GetCurrentDevice();
            return Instance.Props[devId].multiProcessorCount;
        }

        static size_t GlobalMemorySize() {
            int devId = Instance.GetCurrentDevice();
            return Instance.Props[devId].totalGlobalMem;
        }

        static int GetMajorVersion() {
            int devId = Instance.GetCurrentDevice();
            return Instance.Props[devId].major;
        }

        static int GetMajorVersion(int devId) {
            Instance.CachePropsIfNotCached(devId);
            return Instance.Props[devId].major;
        }

        static int GetMinorVersion(int devId) {
            Instance.CachePropsIfNotCached(devId);
            return Instance.Props[devId].minor;
        }

        // FP16 is accessible if compability version is at least 5.3
        // https://h.yandex-team.ru/?https%3A%2F%2Fdocs.nvidia.com%2Fcuda%2Fcuda-c-programming-guide%2Findex.html%23arithmetic-instructions
        static bool HasFp16(int devId) {
            Instance.CachePropsIfNotCached(devId);
            if (GetMajorVersion(devId) < 5) {
                return false;
            }
            return !(GetMajorVersion(devId) == 5 && GetMinorVersion(devId) < 3);
        }

        // Fast FP16 is accessible if compability version is not 6.1
        static bool HasFastFp16(int devId) {
            Instance.CachePropsIfNotCached(devId);
            if (!HasFp16(devId)) {
                return false;
            }
            return GetMajorVersion(devId) != 6 || GetMinorVersion(devId) != 1;
        }
    };


}
