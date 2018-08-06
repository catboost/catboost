#include "train_template_pointwise.h"
#include <catboost/cuda/targets/querywise_targets_impl.h>

namespace NCatboostCuda {
    using TMultiClassTrainer = TGpuTrainer<TQuerywiseTargetsImpl>;
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> QrmseRegistrator(ELossFunction::QueryRMSE);
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> QsoftmaxRegistrator(ELossFunction::QuerySoftMax);
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> PairLogitRegistrator(ELossFunction::PairLogit);
    TGpuTrainerFactory::TRegistrator<TMultiClassTrainer> YetiRankRegistrator(ELossFunction::YetiRank);
}
