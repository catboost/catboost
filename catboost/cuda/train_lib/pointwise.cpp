#include "train_template.h"
#include <catboost/cuda/targets/pointwise_target_impl.h>

namespace NCatboostCuda {
    using TPointwiseTrainer = TGpuTrainer<TPointwiseTargetsImpl>;

    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorPoisson(ELossFunction::Poisson);
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMape(ELossFunction::MAPE);
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorMae(ELossFunction::MAE);
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorQuantile(ELossFunction::Quantile);
    TGpuTrainerFactory::TRegistrator<TPointwiseTrainer> RegistratorLogLinQuantile(ELossFunction::LogLinQuantile);

}
