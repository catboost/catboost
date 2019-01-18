#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TQuerywiseTrainer = TGpuTrainer<TQuerywiseTargetsImpl, TRegionModel>;
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::QueryRMSE));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::QuerySoftMax));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::PairLogit));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankRegistrator(GetTrainerFactoryKeyForRegion(ELossFunction::YetiRank));
}
