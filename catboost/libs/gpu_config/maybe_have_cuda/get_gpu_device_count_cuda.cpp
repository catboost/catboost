#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>

#include <catboost/libs/logging/logging.h>

#include <cuda_runtime.h>

namespace NCB {

    int GetGpuDeviceCount() {
        int deviceCount = 0;

        cudaError_t status;
        if (cudaSuccess != (status = cudaGetDeviceCount(&deviceCount))) {
            CATBOOST_WARNING_LOG << "Error " << int(status) << " (" << cudaGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
        return deviceCount;
    }

}
