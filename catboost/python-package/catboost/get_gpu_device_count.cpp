#include "helpers.h"

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/libs/train_lib/train_model.h>

int GetGpuDeviceCount() {
    int deviceCount = 0;
    if (TTrainerFactory::Has(ETaskType::GPU)) {
        cudaError_t status;
        if (cudaSuccess != (status = cudaGetLastError())) {
            MATRIXNET_WARNING_LOG << "Error " << int(status) << " (" << cudaGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
        if (cudaSuccess != (status = cudaGetDeviceCount(&deviceCount))) {
            MATRIXNET_WARNING_LOG << "Error " << int(status) << " (" << cudaGetErrorString(status) << ") ignored while obtaining device count" << Endl;
        }
    }
    return deviceCount;
}
