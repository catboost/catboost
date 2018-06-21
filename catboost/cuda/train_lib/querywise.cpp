#include "train_template.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TQuerywiseTrainer = TGpuTrainer<TQuerywiseTargetsImpl>;
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseRegistrator(ELossFunction::QueryRMSE);
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxRegistrator(ELossFunction::QuerySoftMax);
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitRegistrator(ELossFunction::PairLogit);
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankRegistrator(ELossFunction::YetiRank);
}
