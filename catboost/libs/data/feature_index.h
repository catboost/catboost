#pragma once

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>

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
    using TEmbeddingFeatureIdx = TFeatureIdx<EFeatureType::Embedding>;

    struct TFeatureIdxWithType {
        EFeatureType FeatureType;
        ui32 FeatureIdx; // per type

    public:
        explicit TFeatureIdxWithType(EFeatureType featureType = EFeatureType::Float, ui32 featureIndex = 0)
            : FeatureType(featureType)
            , FeatureIdx(featureIndex)
        {}

        bool operator==(const TFeatureIdxWithType& rhs) const {
            return (FeatureType == rhs.FeatureType) && (FeatureIdx == rhs.FeatureIdx);
        }

        SAVELOAD(FeatureType, FeatureIdx);
    };

}


template <EFeatureType FeatureType>
struct THash<NCB::TFeatureIdx<FeatureType>> {
    inline size_t operator()(NCB::TFeatureIdx<FeatureType> featureIdx) const {
        return THash<ui32>()(featureIdx.Idx);
    }
};

