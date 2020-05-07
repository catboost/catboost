#pragma once

#include "exclusive_feature_bundling.h"
#include "feature_index.h"
#include "packed_binary_features.h"

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/types.h>


namespace NCB {

    struct TFeaturesGroupPart : public TFeatureIdxWithType{
        ui32 BucketCount;

    public:
        explicit TFeaturesGroupPart(
            EFeatureType featureType = EFeatureType::Float, ui32 featureIndex = 0, ui32 bucketCount = 0)
            : TFeatureIdxWithType(featureType, featureIndex)
            , BucketCount(bucketCount)
        {}

        bool operator==(const TFeaturesGroupPart& rhs) const {
            return (FeatureType == rhs.FeatureType)
                   && (FeatureIdx == rhs.FeatureIdx)
                   && (BucketCount == rhs.BucketCount);
        }

        SAVELOAD(FeatureType, FeatureIdx, BucketCount);
    };

    struct TFeaturesGroup {
        TVector<TFeaturesGroupPart> Parts;
        TVector<ui32> BucketOffsets;
        ui32 TotalBucketCount = 0;

    public:
        inline bool operator==(const TFeaturesGroup& rhs) const {
            return (Parts == rhs.Parts)
                && (BucketOffsets == rhs.BucketOffsets)
                && (TotalBucketCount == rhs.TotalBucketCount);
        }

        inline ui32 GetSizeInBytes() const {
            return Parts.size();
        }

        inline void Add(const TFeaturesGroupPart& part) {
            Parts.emplace_back(part);
            BucketOffsets.push_back(TotalBucketCount);
            TotalBucketCount += part.BucketCount;
        }

        inline void PopLastFeature() {
            TotalBucketCount -= Parts.back().BucketCount;
            BucketOffsets.pop_back();
            Parts.pop_back();
        }

        SAVELOAD(Parts, BucketOffsets, TotalBucketCount);
    };

    struct TFeaturesGroupIndex {
        ui32 GroupIdx;
        ui32 InGroupIdx;
    };

    inline ui8 GetPartValueFromGroup(ui32 groupValue, size_t partIdx) {
        return static_cast<ui8>(groupValue >> (partIdx * CHAR_BIT));
    }

    struct TFeaturesGroupingOptions {
        ui32 MaxFeaturesPerBundle = 4;
    };

    TVector<TFeaturesGroup> CreateFeatureGroups(
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TVector<TMaybe<TExclusiveBundleIndex>>& flatFeatureIndexToBundlePart,
        const TVector<TMaybe<TPackedBinaryIndex>>& flatFeatureIndexToPackedBinaryIndex,
        const TFeaturesGroupingOptions& options = {});
}
