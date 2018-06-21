#include "compressed_index_builder.h"

namespace NCatboostCuda {
    template class TSharedCompressedIndexBuilder<TFeatureParallelLayout>;

    template class TSharedCompressedIndexBuilder<TDocParallelLayout>;

    template class TSharedCompressedIndexBuilder<TSingleDevLayout>;
}
