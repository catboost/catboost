#include "train_template.h"
#include <catboost/cuda/targets/pfound_f.h>

namespace NCatboostCuda {
    using TPFoundFTrainer = TPairwiseGpuTrainer<TPFoundF>;
    TGpuTrainerFactory::TRegistrator<TPFoundFTrainer> PFoundRegistrator(ELossFunction::YetiRankPairwise);
}
