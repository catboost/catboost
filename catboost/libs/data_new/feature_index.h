#pragma once

#include <catboost/libs/options/enums.h>

#include <util/system/types.h>
#include <util/str_stl.h>


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

        bool operator==(TFeatureIdx rhs) const {
            return Idx == rhs.Idx;
        }
    };

    using TFloatFeatureIdx = TFeatureIdx<EFeatureType::Float>;
    using TCatFeatureIdx = TFeatureIdx<EFeatureType::Categorical>;
    using TTextFeatureIdx = TFeatureIdx<EFeatureType::Text>;

}


template <EFeatureType FeatureType>
struct THash<NCB::TFeatureIdx<FeatureType>> {
    inline size_t operator()(NCB::TFeatureIdx<FeatureType> featureIdx) const {
        return THash<ui32>()(featureIdx.Idx);
    }
};

