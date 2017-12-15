#include "train_template.h"
#include <catboost/cuda/targets/qrmse.h>

namespace NCatboostCuda {
    using TQrmseTrainer = TGpuTrainer<TQueryRMSE>;
    TGpuTrainerFactory::TRegistrator<TQrmseTrainer> QrmseRegistrator(ELossFunction::QueryRMSE);
}
