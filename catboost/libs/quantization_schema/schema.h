#pragma once

#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>

namespace NCB {
    // TODO(yazevnul): replace `NCatboostCuda::TBinarizedFloatFeaturesMetaInfo` with this struct
    struct TPoolQuantizationSchema {
        TVector<size_t> TrueFeatureIndices;

        // Each element is sorted (asc.) and each value is unique.
        TVector<TVector<float>> Borders;

        // TODO(yazevnul): maybe rename `ENanMode` to `ENanPolicy`?
        TVector<ENanMode> NanModes;
    };
}
