#pragma once

#include <catboost/libs/options/enums.h>

#include <util/system/types.h>


namespace NCB {

    template <EFeatureType FeatureType>
    struct TFeatureIdx {
        ui32 Idx;

    public:
        explicit TFeatureIdx(ui32 idx)
            : Idx(idx)
        {}

        // save some typing
        ui32 operator*() const {
            return Idx;
        }
    };

    using TFloatFeatureIdx = TFeatureIdx<EFeatureType::Float>;
    using TCatFeatureIdx = TFeatureIdx<EFeatureType::Categorical>;

}

