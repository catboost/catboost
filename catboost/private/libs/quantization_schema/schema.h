#pragma once

#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/data/cat_feature_perfect_hash.h>

#include <library/cpp/json/json_value.h>

#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {
    struct TPoolQuantizationSchema {
        // Flat indices of float non-ignored features
        // Sorted from min to max
        TVector<size_t> FloatFeatureIndices;

        // Borders[i] are borders for feature FloatFeatureIndices[i]
        // Borders[i] is sorted from min to max
        TVector<TVector<float>> Borders;

        // NanModes[i] is NaN mode for feature FloatFeatureIndices[i]
        // NanModes[i] == EColumn::Forbidden iff there are no NaN's
        TVector<ENanMode> NanModes;

        // If ClassLabels is non-empty target data is serialized format contains indices in this array
        // Values can be Integers, Doubles or Strings
        TVector<NJson::TJsonValue> ClassLabels;

        // Flat indices of categorical non-ignored features
        // Sorted from min to max
        TVector<size_t> CatFeatureIndices;

        // For building perfect hash
        // FeaturesPerfectHash[i] are hashes for cat feature CatFeatureIndices[i]
        TVector<TMap<ui32, TValueWithCount>> FeaturesPerfectHash; // [catFeatureIdx]

    public:
        bool HasAvailableFeatures() const {
            return !FloatFeatureIndices.empty() || !CatFeatureIndices.empty();
        }
    };
}
