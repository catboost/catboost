#pragma once

#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>
#include <util/generic/string.h>

namespace NCB {
    struct TPoolQuantizationSchema {
        // Flat indices of non-ignored features
        // Sorted from min to max
        TVector<size_t> FeatureIndices;

        // Borders[i] are borders for feature FeatureIndices[i]
        // Borders[i] is sorted from min to max
        TVector<TVector<float>> Borders;

        // NanModes[i] is NaN mode for feature FeatureIndices[i]
        // NanModes[i] == EColumn::Forbidden iff there are no NaN's
        TVector<ENanMode> NanModes;

        // Class names for multi-classification
        // ClassNames[i] is name of class i
        TVector<TString> ClassNames;
    };
}
