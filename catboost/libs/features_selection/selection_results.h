#pragma once

#include <library/cpp/json/writer/json_value.h>
#include <util/generic/vector.h>
#include <util/ysaveload.h>


namespace NCB {

    struct TFeaturesSelectionLossGraph {
        TVector<double> LossValues;  // 0 - with all features, 1 - without first eliminated feature, 2 - without two eliminited features, ...
        TVector<ui32> MainIndices;   // indices with precise loss value after model fitting (without using fstr)

        Y_SAVELOAD_DEFINE(
            LossValues,
            MainIndices
        );
    };

    struct TFeaturesSelectionSummary {
        TVector<ui32> SelectedFeatures;
        TVector<ui32> EliminatedFeatures;
        TFeaturesSelectionLossGraph LossGraph;

        Y_SAVELOAD_DEFINE(
            SelectedFeatures,
            EliminatedFeatures,
            LossGraph
        );
    };


    NJson::TJsonValue ToJson(const TFeaturesSelectionSummary& summary);
}
