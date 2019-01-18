#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/pointwise_target_impl.h>

namespace NCatboostCuda {
    using TPointwiseTrainer = TGpuTrainer<TPointwiseTargetsImpl, TNonSymmetricTree>;

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoissonLossguide(GetTrainerFactoryKey(ELossFunction::Poisson, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMapeLossguide(GetTrainerFactoryKey(ELossFunction::MAPE, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMaeLossguide(GetTrainerFactoryKey(ELossFunction::MAE, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantileLossguide(GetTrainerFactoryKey(ELossFunction::Quantile, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantileLossguide(GetTrainerFactoryKey(ELossFunction::LogLinQuantile, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSELossguide(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLoglossLossguide(GetTrainerFactoryKey(ELossFunction::Logloss, EGrowingPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropyLossguide(GetTrainerFactoryKey(ELossFunction::CrossEntropy, EGrowingPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoissonLevelwise(GetTrainerFactoryKey(ELossFunction::Poisson, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMapeLevelwise(GetTrainerFactoryKey(ELossFunction::MAPE, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMaeLevelwise(GetTrainerFactoryKey(ELossFunction::MAE, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantileLevelwise(GetTrainerFactoryKey(ELossFunction::Quantile, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantileLevelwise(GetTrainerFactoryKey(ELossFunction::LogLinQuantile, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSELevelwise(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLoglossLevelwise(GetTrainerFactoryKey(ELossFunction::Logloss, EGrowingPolicy::Levelwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropyLevelwise(GetTrainerFactoryKey(ELossFunction::CrossEntropy, EGrowingPolicy::Levelwise));

    //    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRmseOT(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowingPolicy::ObliviousTree));

}
