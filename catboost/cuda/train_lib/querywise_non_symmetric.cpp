#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TQuerywiseTrainer = TGpuTrainer<TQuerywiseTargetsImpl, TNonSymmetricTree>;

    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseRegistrator(GetTrainerFactoryKey(ELossFunction::QueryRMSE, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxRegistrator(GetTrainerFactoryKey(ELossFunction::QuerySoftMax, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitRegistrator(GetTrainerFactoryKey(ELossFunction::PairLogit, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRank, EGrowingPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QrmseLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::QueryRMSE, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> QsoftmaxLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::QuerySoftMax, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> PairLogitLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::PairLogit, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TQuerywiseTrainer> YetiRankLevelwiseRegistrator(GetTrainerFactoryKey(ELossFunction::YetiRank, EGrowingPolicy::Levelwise));
}
