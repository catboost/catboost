#include "train_template_pairwise.h"
#include <catboost/cuda/targets/pfound_f.h>

namespace NCatboostCuda {
    using TPFoundFTrainer = TPairwiseGpuTrainer<TPFoundF>;
    TGpuTrainerFactory::TRegistrator<TPFoundFTrainer> PFoundRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRankPairwise));
}
