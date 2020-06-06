#pragma once
#include "kernel.cuh"

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
    };


}
