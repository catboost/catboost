#include "train_template_pointwise.h"
#include <catboost/cuda/targets/combination_targets_impl.h>

namespace NCatboostCuda {
    using TQCombinationTrainer = TGpuTrainer<TCombinationTargetsImpl>;
    TGpuTrainerFactory::TRegistrator<TQCombinationTrainer> CombinationRegistrator(GetTrainerFactoryKey(ELossFunction::Combination));
}
