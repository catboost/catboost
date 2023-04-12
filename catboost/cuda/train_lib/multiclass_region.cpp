#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/multiclass_targets.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TMultiClassificationTargets, TRegionModel>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassRegionRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::MultiClass));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllRegionRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::MultiClassOneVsAll));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> RMSEWithUncertaintyOneVsAllRegionRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::RMSEWithUncertainty));
}
