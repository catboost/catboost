#pragma once

#include <catboost/libs/model/model.h>

namespace NCB {
    struct TCompressedModelCtr {
        const TFeatureCombination* Projection;
        TVector<const TModelCtr*> ModelCtrs;
    };

    TVector<TCompressedModelCtr> CompressModelCtrs(const TVector<TModelCtr>& neededCtrs);
} // namespace NCB
