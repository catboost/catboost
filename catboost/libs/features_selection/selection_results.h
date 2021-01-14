#pragma once

#include <library/cpp/json/writer/json_value.h>
#include <util/generic/vector.h>
#include <util/ysaveload.h>


namespace NCB {

    struct TFeaturesSelectionLossGraph {
        TVector<ui32> RemovedFeaturesCount;     // Number of removed features at each point of graph
        TVector<double> LossValues;             // Loss value at each point of graph
        TVector<ui32> MainIndices;              // indices with precise loss value after model fitting (without using fstr)

        Y_SAVELOAD_DEFINE(
            RemovedFeaturesCount,
            LossValues,
            MainIndices
        );
    };

    struct TFeaturesSelectionSummary {
        TVector<ui32> SelectedFeatures;
        TVector<TString> SelectedFeaturesNames;
        TVector<ui32> EliminatedFeatures;
        TVector<TString> EliminatedFeaturesNames;
        TFeaturesSelectionLossGraph LossGraph;

        Y_SAVELOAD_DEFINE(
            SelectedFeatures,
            SelectedFeaturesNames,
            EliminatedFeatures,
            EliminatedFeaturesNames,
            LossGraph
        );
    };


    NJson::TJsonValue ToJson(const TFeaturesSelectionSummary& summary);
}
