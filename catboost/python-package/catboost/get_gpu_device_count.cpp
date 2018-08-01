#include "helpers.h"

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/libs/train_lib/train_model.h>

int GetGpuDeviceCount() {
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetLastError());
    if (TTrainerFactory::Has(ETaskType::GPU)) {
        cudaError_t status = cudaGetDeviceCount(&deviceCount);
        if (status != cudaSuccess) {
            MATRIXNET_WARNING_LOG << "Error " << int(status) << " (" << cudaGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
    }
    return deviceCount;
}
