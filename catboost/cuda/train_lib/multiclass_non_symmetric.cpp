#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/multiclass_targets.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TMultiClassificationTargets, TNonSymmetricTree>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassLossguideRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClass, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllLossguideRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClassOneVsAll, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> RMSEWithUncertaintyLossguideRegistrator(GetTrainerFactoryKey(ELossFunction::RMSEWithUncertainty, EGrowPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClass, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClassOneVsAll, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> RMSEWithUncertaintyDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::RMSEWithUncertainty, EGrowPolicy::Depthwise));

}
