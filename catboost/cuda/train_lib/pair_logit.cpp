#include "train_template.h"
#include <catboost/cuda/targets/pair_logit.h>

namespace NCatboostCuda {
    using TPairLogitTrainer = TGpuTrainer<TPairLogit>;
    TGpuTrainerFactory::TRegistrator<TPairLogitTrainer> PairLogitRegistrator(ELossFunction::PairLogit);
}
