#include "train_template.h"
#include <catboost/cuda/targets/cross_entropy.h>

namespace NCatboostCuda {
    using TLoglossTrainer = TGpuTrainer<TLogloss>;
    TGpuTrainerFactory::TRegistrator<TLoglossTrainer> LogLossRegistrator(ELossFunction::Logloss);

    using TCrossEntropyTrainer = TGpuTrainer<TCrossEntropy>;
    TGpuTrainerFactory::TRegistrator<TCrossEntropyTrainer> CrossEntropyRegistrator(ELossFunction::CrossEntropy);
}
