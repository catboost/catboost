#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/monoforest/enums.h>

#include <util/generic/fwd.h>

namespace NMonoForest {
    struct THumanReadableSplit {
        int FeatureIdx;
        EBinSplitType SplitType;
        float Border;
    };

    struct THumanReadableMonom {
        TVector<THumanReadableSplit> Splits;
        TVector<double> Value;
        double Weight;
    };

    TVector<THumanReadableMonom> ConvertFullModelToPolynom(const TFullModel& fullModel);
    TString ConvertFullModelToPolynomString(const TFullModel& fullModel);
}
