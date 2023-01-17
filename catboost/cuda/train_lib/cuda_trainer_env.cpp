
#include "cuda_trainer_env.h"


namespace NCB {

    TGpuTrainerEnv::TGpuTrainerEnv(const NCatboostOptions::TCatBoostOptions& options) {
        StopCudaManagerCallback = StartCudaManager(NCudaLib::CreateDeviceRequestConfig(options), options.LoggingLevel);
    }

    namespace {

    TTrainerEnvFactory::TRegistrator<TGpuTrainerEnv> GpuTrainerInitReg(ETaskType::GPU);

    }
}
