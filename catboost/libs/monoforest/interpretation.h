#pragma once

#include "enums.h"
#include "grid.h"
#include "polynom.h"

#include <util/generic/vector.h>

namespace NMonoForest {
    struct TBorderExplanation {
        float Border;
        double ProbabilityToSatisfy; // P(FeatureValue > Border) for Float features
                                     // P(FeatureValue = Border) for OneHot features
        TVector<double> ExpectedValueChange; // [dim]
    };

    struct TFeatureExplanation {
        int FeatureIdx;
        EFeatureType FeatureType;
        TVector<double> ExpectedBias; // [dim]
        TVector<TBorderExplanation> BordersExplanations;
    };

    TVector<TFeatureExplanation> ExplainFeatures(const TPolynom& polynom, const IGrid& grid);
}
