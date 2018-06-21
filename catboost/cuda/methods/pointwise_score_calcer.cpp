#include "pointwise_scores_calcer.h"

namespace NCatboostCuda {
    template class TScoresCalcerOnCompressedDataSet<TFeatureParallelLayout>;
    template class TScoresCalcerOnCompressedDataSet<TDocParallelLayout>;
    template class TScoresCalcerOnCompressedDataSet<TSingleDevLayout>;
}
