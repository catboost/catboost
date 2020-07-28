#include "ctr_helpers.h"

namespace NCB {
    TVector<TCompressedModelCtr> CompressModelCtrs(const TConstArrayRef<TModelCtr> neededCtrs) {
        TVector<TCompressedModelCtr> compressedModelCtrs;
        compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[0].Base.Projection, {&neededCtrs[0]}});
        for (size_t i = 1; i < neededCtrs.size(); ++i) {
            Y_ASSERT(neededCtrs[i - 1] < neededCtrs[i]); // needed ctrs should be sorted
            if (*(compressedModelCtrs.back().Projection) != neededCtrs[i].Base.Projection) {
                compressedModelCtrs.emplace_back(TCompressedModelCtr{&neededCtrs[i].Base.Projection, {}});
            }
            compressedModelCtrs.back().ModelCtrs.push_back(&neededCtrs[i]);
        }
        return compressedModelCtrs;
    }
}
