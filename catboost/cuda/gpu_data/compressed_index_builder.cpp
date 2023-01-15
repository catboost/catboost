#include "compressed_index_builder.h"
#include "feature_layout_doc_parallel.h"
#include "feature_layout_feature_parallel.h"

namespace NCatboostCuda {
    template class TSharedCompressedIndexBuilder<TFeatureParallelLayout>;

    template class TSharedCompressedIndexBuilder<TDocParallelLayout>;
}
