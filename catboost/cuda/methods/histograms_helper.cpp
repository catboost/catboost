#include "histograms_helper.h"
#include <util/system/env.h>

bool IsReduceCompressed() {
    static const bool reduceCompressed = GetEnv("CB_COMPRESSED_REDUCE", "false") == "true";
    return reduceCompressed;
}

namespace NCatboostCuda {
    template class TComputeHistogramsHelper<TFeatureParallelLayout>;
    template class TComputeHistogramsHelper<TDocParallelLayout>;
    template class TComputeHistogramsHelper<TSingleDevLayout>;

    template class TFindBestSplitsHelper<TFeatureParallelLayout>;
    template class TFindBestSplitsHelper<TDocParallelLayout>;
    template class TFindBestSplitsHelper<TSingleDevLayout>;

    template class TScoreHelper<TFeatureParallelLayout>;
    template class TScoreHelper<TDocParallelLayout>;
    template class TScoreHelper<TSingleDevLayout>;
}
