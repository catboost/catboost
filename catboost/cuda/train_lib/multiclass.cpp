#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/multiclass_targets.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TMultiClassificationTargets>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassRegistrator(ELossFunction::MultiClass);
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllRegistrator(ELossFunction::MultiClassOneVsAll);
}
