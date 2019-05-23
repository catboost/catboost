#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TQuerywiseTrainer = TGpuTrainer<TQuerywiseTargetsImpl, TNonSymmetricTree>;

    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseRegistrator(GetTrainerFactoryKey(ELossFunction::QueryRMSE, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxRegistrator(GetTrainerFactoryKey(ELossFunction::QuerySoftMax, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitRegistrator(GetTrainerFactoryKey(ELossFunction::PairLogit, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRank, EGrowPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::QueryRMSE, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::QuerySoftMax, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::PairLogit, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankDepthwiseRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRank, EGrowPolicy::Depthwise));
}
