#include "train_template.h"
#include <catboost/cuda/targets/yeti_rank.h>

namespace NCatboostCuda {
    using TYetiRankTrainer = TGpuTrainer<TYetiRank>;
    TGpuTrainerFactory::TRegistrator<TYetiRankTrainer> YetiRankRegistrator(ELossFunction::YetiRank);
}
