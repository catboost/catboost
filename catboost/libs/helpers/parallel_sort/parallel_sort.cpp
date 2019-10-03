#include "parallel_sort.h"

#include <catboost/private/libs/index_range/index_range.h>

namespace NCB {

    void EquallyDivide(ui32 size, ui32 blockCount, TVector<ui32>* blockSizes) {
        TEqualRangesGenerator<ui32> generator({0, size}, blockCount);
        blockSizes->clear();
        blockSizes->reserve(blockCount);
        for (ui32 i = 0; i < blockCount; ++i) {
            blockSizes->emplace_back(generator.GetRange(i).GetSize());
        }
    }
}
