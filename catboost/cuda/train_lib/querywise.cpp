#include "train_template_pointwise.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TQuerywiseTrainer = TGpuTrainer<TQuerywiseTargetsImpl>;
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseRegistrator(GetTrainerFactoryKey(ELossFunction::QueryRMSE));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxRegistrator(GetTrainerFactoryKey(ELossFunction::QuerySoftMax));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitRegistrator(GetTrainerFactoryKey(ELossFunction::PairLogit));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRank));
}
