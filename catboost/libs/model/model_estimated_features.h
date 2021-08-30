#pragma once

#include "hash.h"
#include "fwd.h"

#include <catboost/libs/helpers/guid.h>
#include <catboost/private/libs/options/enums.h>

#include <util/digest/multi.h>
#include <util/ysaveload.h>


struct TModelEstimatedFeature {
    int SourceFeatureId = 0;
    NCB::TGuid CalcerId;
    int LocalId = 0;
    EEstimatedSourceFeatureType SourceFeatureType;

public:
    TModelEstimatedFeature() = default;
    TModelEstimatedFeature(
        int sourceFeatureId,
        NCB::TGuid calcerId,
        int localId,
        EEstimatedSourceFeatureType sourceFeatureType
    )
        : SourceFeatureId(sourceFeatureId)
        , CalcerId(calcerId)
        , LocalId(localId)
        , SourceFeatureType(sourceFeatureType)
    {}

    TModelEstimatedFeature(
        int sourceFeatureId,
        int localId,
        EEstimatedSourceFeatureType sourceFeatureType
    )
        : SourceFeatureId(sourceFeatureId)
        , LocalId(localId)
        , SourceFeatureType(sourceFeatureType)
    {}

    bool operator==(const TModelEstimatedFeature& other) const {
        return std::tie(SourceFeatureId, CalcerId, LocalId, SourceFeatureType)
               == std::tie(other.SourceFeatureId, other.CalcerId, other.LocalId, other.SourceFeatureType);
    }

    bool operator!=(const TModelEstimatedFeature& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelEstimatedFeature& other) const {
        return std::tie(SourceFeatureId, CalcerId, LocalId, SourceFeatureType)
               < std::tie(other.SourceFeatureId, other.CalcerId, other.LocalId, other.SourceFeatureType);
    }

    ui64 GetHash() const {
        return MultiHash(SourceFeatureId, CalcerId, LocalId, SourceFeatureType);
    }

    Y_SAVELOAD_DEFINE(SourceFeatureId, CalcerId, LocalId, SourceFeatureType);
};


template <>
struct THash<TModelEstimatedFeature> {
    size_t operator()(const TModelEstimatedFeature& estimatedFeature) const noexcept {
        return estimatedFeature.GetHash();
    }
};


struct TEstimatedFeatureSplit {
    TModelEstimatedFeature ModelEstimatedFeature;
    float Split = 0.f;

public:
    TEstimatedFeatureSplit() = default;
    TEstimatedFeatureSplit(
        const TModelEstimatedFeature& modelEstimatedFeature,
        float split
    )
        : ModelEstimatedFeature(modelEstimatedFeature)
        , Split(split)
    {}

    bool operator==(const TEstimatedFeatureSplit& other) const {
        return std::tie(ModelEstimatedFeature, Split)
               == std::tie(other.ModelEstimatedFeature, other.Split);
    }

    bool operator<(const TEstimatedFeatureSplit& other) const {
        return std::tie(ModelEstimatedFeature, Split)
               < std::tie(other.ModelEstimatedFeature, other.Split);
    }

    ui64 GetHash() const {
        return MultiHash(ModelEstimatedFeature, Split);
    }

    Y_SAVELOAD_DEFINE(ModelEstimatedFeature, Split);

    inline void CanonizeFloatForFbs(float *value) {
        if (*value == -0.0f) {
            *value = 0.0f;
        }
    }

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        CanonizeFloatForFbs(&Split);
    }
};

template <>
struct THash<TEstimatedFeatureSplit> {
    inline size_t operator()(const TEstimatedFeatureSplit& split) const {
        return split.GetHash();
    }
};
