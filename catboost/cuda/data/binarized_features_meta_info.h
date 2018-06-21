#pragma once

#include <util/generic/vector.h>
#include <catboost/libs/options/enums.h>

namespace NCatboostCuda {
    struct TBinarizedFloatFeaturesMetaInfo {
        TVector<int> BinarizedFeatureIds;
        TVector<TVector<float>> Borders;
        TVector<ENanMode> NanModes;
    };

    TBinarizedFloatFeaturesMetaInfo LoadBordersFromFromFileInMatrixnetFormat(const TString& path);

}
