#include "train_template_pairwise.h"
#include <catboost/cuda/targets/query_cross_entropy.h>

namespace NCatboostCuda {
    using TQueryCrossEntropyTrainer = TPairwiseGpuTrainer<TQueryCrossEntropy>;
    TGpuTrainerFactory::TRegistrator<TQueryCrossEntropyTrainer> QueryCrossEntropyRegistrator(GetTrainerFactoryKey(ELossFunction::QueryCrossEntropy));

}
