#pragma once
#include "kernel.cuh"

#include <tuple>

namespace NKernel {

    #define NUM_CACHED_PROPS 16

    class TArchProps {
        static TArchProps Instance;

        cudaDeviceProp Props[NUM_CACHED_PROPS];
        bool PropsCached[NUM_CACHED_PROPS];

        TArchProps() {
            for (int i = 0; i < NUM_CACHED_PROPS; i++) {
                PropsCached[i] = false;
            }
        }

        inline void CacheProps(int devId) {
            cudaGetDeviceProperties(&Props[devId], devId);
            Instance.PropsCached[devId] = true;
        }

        inline void CachePropsIfNotCached(int devId) {
            if (!PropsCached[devId]) {
                CacheProps(devId);
            }
        }

        inline int GetCurrentDevice() {
            int devId = -1;
            cudaGetDevice(&devId);
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

        static ui64 GlobalMemorySize() {
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
