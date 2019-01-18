#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/multiclass_targets.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TMultiClassificationTargets, TNonSymmetricTree>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassLossguideRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClass, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllLossguideRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClassOneVsAll, EGrowingPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClass, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClassOneVsAll, EGrowingPolicy::Levelwise));
}
