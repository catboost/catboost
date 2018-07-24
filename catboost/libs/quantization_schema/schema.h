#pragma once

#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>
#include <util/generic/string.h>

namespace NCB {
    // TODO(yazevnul): replace `NCatboostCuda::TBinarizedFloatFeaturesMetaInfo` with this struct
    struct TPoolQuantizationSchema {
        TVector<size_t> FeatureIndices;

        // Each element is sorted (asc.) and each value is unique.
        TVector<TVector<float>> Borders;

        // TODO(yazevnul): maybe rename `ENanMode` to `ENanPolicy`?
        TVector<ENanMode> NanModes;

        // List of class names; Makes sence only for multiclassification.
        //
        // NOTE: order is important
        TVector<TString> ClassNames;
    };
}
