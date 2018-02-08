#include "train_template.h"
#include <catboost/cuda/targets/qsoftmax.h>

namespace NCatboostCuda {
    using TQsoftmaxTrainer = TGpuTrainer<TQuerySoftMax>;
    TGpuTrainerFactory::TRegistrator<TQsoftmaxTrainer> QsoftmaxRegistrator(ELossFunction::QuerySoftMax);
}
