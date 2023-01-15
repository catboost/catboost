#include "compressed_index.h"

namespace NCatboostCuda {
    template class TSharedCompressedIndex<TFeatureParallelLayout>;
    template class TSharedCompressedIndex<TDocParallelLayout>;
}
