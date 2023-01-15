#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/pointwise_target_impl.h>
#include <catboost/cuda/models/non_symmetric_tree.h>

namespace NCatboostCuda {
    using TPointwiseTrainer = TGpuTrainer<TPointwiseTargetsImpl, TRegionModel>;

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoisson(GetTrainerFactoryKeyForRegion(ELossFunction::Poisson));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMape(GetTrainerFactoryKeyForRegion(ELossFunction::MAPE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMae(GetTrainerFactoryKeyForRegion(ELossFunction::MAE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantile(GetTrainerFactoryKeyForRegion(ELossFunction::Quantile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantile(GetTrainerFactoryKeyForRegion(ELossFunction::LogLinQuantile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSE(GetTrainerFactoryKeyForRegion(ELossFunction::RMSE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogloss(GetTrainerFactoryKeyForRegion(ELossFunction::Logloss));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropy(GetTrainerFactoryKeyForRegion(ELossFunction::CrossEntropy));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorExpectile(GetTrainerFactoryKeyForRegion(ELossFunction::Expectile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorTweedie(GetTrainerFactoryKeyForRegion(ELossFunction::Tweedie));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorHuber(GetTrainerFactoryKeyForRegion(ELossFunction::Huber));
}
