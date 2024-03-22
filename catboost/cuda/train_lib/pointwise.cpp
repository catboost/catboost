#include "train_template_pointwise.h"
#include <catboost/cuda/targets/pointwise_target_impl.h>

namespace NCatboostCuda {
    using TPointwiseTrainer = TGpuTrainer<TPointwiseTargetsImpl>;

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorRMSE(GetTrainerFactoryKey(ELossFunction::RMSE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoisson(GetTrainerFactoryKey(ELossFunction::Poisson));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMape(GetTrainerFactoryKey(ELossFunction::MAPE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMae(GetTrainerFactoryKey(ELossFunction::MAE));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantile(GetTrainerFactoryKey(ELossFunction::Quantile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantile(GetTrainerFactoryKey(ELossFunction::LogLinQuantile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogloss(GetTrainerFactoryKey(ELossFunction::Logloss));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorCrossEntropy(GetTrainerFactoryKey(ELossFunction::CrossEntropy));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLq(GetTrainerFactoryKey(ELossFunction::Lq));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorExpectile(GetTrainerFactoryKey(ELossFunction::Expectile));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorTweedie(GetTrainerFactoryKey(ELossFunction::Tweedie));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorHuber(GetTrainerFactoryKey(ELossFunction::Huber));
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorUserDefined(GetTrainerFactoryKey(ELossFunction::PythonUserDefinedPerObject));
}
