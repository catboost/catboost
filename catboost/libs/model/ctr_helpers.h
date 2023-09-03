#pragma once

#include "online_ctr.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>


struct TFeatureCombination;


namespace NCB {
    struct TCompressedModelCtr {
        const TFeatureCombination* Projection;
        TVector<const TModelCtr*> ModelCtrs;
    };

    TVector<TCompressedModelCtr> CompressModelCtrs(const TConstArrayRef<TModelCtr> neededCtrs);
} // namespace NCB
