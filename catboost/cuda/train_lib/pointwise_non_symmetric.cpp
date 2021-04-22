#include "train_template_pointwise_greedy_subsets_searcher.h"
#include <catboost/cuda/targets/pointwise_target_impl.h>

namespace NCatboostCuda {
    using TPointwiseTrainer = TGpuTrainer<TPointwiseTargetsImpl, TNonSymmetricTree>;

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoissonLossguide(GetTrainerFactoryKey(ELossFunction::Poisson, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMapeLossguide(GetTrainerFactoryKey(ELossFunction::MAPE, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMaeLossguide(GetTrainerFactoryKey(ELossFunction::MAE, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantileLossguide(GetTrainerFactoryKey(ELossFunction::Quantile, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantileLossguide(GetTrainerFactoryKey(ELossFunction::LogLinQuantile, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSELossguide(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLoglossLossguide(GetTrainerFactoryKey(ELossFunction::Logloss, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropyLossguide(GetTrainerFactoryKey(ELossFunction::CrossEntropy, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorExpectileLossguide(GetTrainerFactoryKey(ELossFunction::Expectile, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorTweedieLossguide(GetTrainerFactoryKey(ELossFunction::Tweedie, EGrowPolicy::Lossguide));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorHuberLossguide(GetTrainerFactoryKey(ELossFunction::Huber, EGrowPolicy::Lossguide));

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoissonDepthwise(GetTrainerFactoryKey(ELossFunction::Poisson, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMapeDepthwise(GetTrainerFactoryKey(ELossFunction::MAPE, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMaeDepthwise(GetTrainerFactoryKey(ELossFunction::MAE, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantileDepthwise(GetTrainerFactoryKey(ELossFunction::Quantile, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantileDepthwise(GetTrainerFactoryKey(ELossFunction::LogLinQuantile, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSEDepthwise(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLoglossDepthwise(GetTrainerFactoryKey(ELossFunction::Logloss, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropyDepthwise(GetTrainerFactoryKey(ELossFunction::CrossEntropy, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorExpectileDepthwise(GetTrainerFactoryKey(ELossFunction::Expectile, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorTweedieDepthwise(GetTrainerFactoryKey(ELossFunction::Tweedie, EGrowPolicy::Depthwise));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorHuberDepthwise(GetTrainerFactoryKey(ELossFunction::Huber, EGrowPolicy::Depthwise));

    //    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRmseOT(GetTrainerFactoryKey(ELossFunction::RMSE, EGrowPolicy::SymmetricTree));

}
