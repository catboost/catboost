#pragma once

#include <library/cpp/json/writer/json_value.h>
#include <util/generic/vector.h>
#include <util/ysaveload.h>


namespace NCB {

    struct TFeaturesSelectionLossGraph {
        TVector<ui32> RemovedEntitiesCount;     // Number of removed entities at each point of graph
        TVector<double> LossValues;             // Loss value at each point of graph
        TVector<ui32> MainIndices;              // indices with precise loss value after model fitting (without using fstr)

    public:
        Y_SAVELOAD_DEFINE(
            RemovedEntitiesCount,
            LossValues,
            MainIndices
        );
    };

    struct TFeaturesSelectionSummary {
        TVector<ui32> SelectedFeatures;
        TVector<TString> SelectedFeaturesNames;
        TVector<ui32> EliminatedFeatures;
        TVector<TString> EliminatedFeaturesNames;
        TVector<TString> SelectedFeaturesTags;                      // can be empty if tags are not used
        TVector<TString> EliminatedFeaturesTags;                    // can be empty if tags are not used
        TFeaturesSelectionLossGraph FeaturesLossGraph;
        TFeaturesSelectionLossGraph FeaturesTagsLossGraph;
        TFeaturesSelectionLossGraph FeaturesTagsCostGraph;

    public:
        Y_SAVELOAD_DEFINE(
            SelectedFeatures,
            SelectedFeaturesNames,
            EliminatedFeatures,
            EliminatedFeaturesNames,
            SelectedFeaturesTags,
            EliminatedFeaturesTags,
            FeaturesLossGraph,
            FeaturesTagsLossGraph,
            FeaturesTagsCostGraph
        );
    };


    NJson::TJsonValue ToJson(const TFeaturesSelectionSummary& summary);
}
