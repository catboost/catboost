#include "train_template.h"
#include <catboost/cuda/targets/mse.h>

namespace NCatboostCuda {
    using TRmseTrainer = TGpuTrainer<TL2>;
    TGpuTrainerFactory::TRegistrator<TRmseTrainer> RmseRegistrator(ELossFunction::RMSE);
}
