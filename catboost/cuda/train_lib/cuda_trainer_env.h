#pragma once

#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/libs/train_lib/trainer_env.h>


namespace NCB {

    struct TGpuTrainerEnv : public ITrainerEnv {
        TGpuTrainerEnv(const NCatboostOptions::TCatBoostOptions& options);
        THolder<TStopCudaManagerCallback> StopCudaManagerCallback;
    };

}
