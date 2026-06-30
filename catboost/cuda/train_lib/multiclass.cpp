#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/multiclass_targets.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TMultiClassificationTargets>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClass));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MulticlassOneVsAllRegistrator(GetTrainerFactoryKey(ELossFunction::MultiClassOneVsAll));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> RMSEWithUncertaintyRegistrator(GetTrainerFactoryKey(ELossFunction::RMSEWithUncertainty));

    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MultiLoglossRegistrator(GetTrainerFactoryKey(ELossFunction::MultiLogloss));
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MultiCrossEntropyRegistrator(GetTrainerFactoryKey(ELossFunction::MultiCrossEntropy));

    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> MultiRMSERegistrator(GetTrainerFactoryKey(ELossFunction::MultiRMSE));
}
