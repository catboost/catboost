#include "selection_results.h"

using namespace NJson;

namespace NCB {
    template <typename T>
    static TJsonValue ToJsonArray(const TVector<T>& arr) {
        TJsonValue jsonArray(JSON_ARRAY);
        for (auto x : arr) {
            jsonArray.AppendValue(x);
        }
        return jsonArray;
    }


    static TJsonValue ToJson(const TFeaturesSelectionLossGraph& lossGraph, const TString& entitiesName) {
        TJsonValue lossGraphJson(JSON_MAP);
        lossGraphJson["removed_" + entitiesName + "_count"] = ToJsonArray(lossGraph.RemovedEntitiesCount);
        lossGraphJson["loss_values"] = ToJsonArray(lossGraph.LossValues);
        lossGraphJson["main_indices"] = ToJsonArray(lossGraph.MainIndices);
        return lossGraphJson;
    }


    TJsonValue ToJson(const TFeaturesSelectionSummary& summary) {
        TJsonValue summaryJson(JSON_MAP);
        summaryJson["selected_features"] = ToJsonArray(summary.SelectedFeatures);
        summaryJson["selected_features_names"] = ToJsonArray(summary.SelectedFeaturesNames);
        summaryJson["eliminated_features"] = ToJsonArray(summary.EliminatedFeatures);
        summaryJson["eliminated_features_names"] = ToJsonArray(summary.EliminatedFeaturesNames);
        summaryJson["loss_graph"] = ToJson(summary.FeaturesLossGraph, "features");
        if (!summary.SelectedFeaturesTags.empty()) {
            summaryJson["selected_features_tags"] = ToJsonArray(summary.SelectedFeaturesTags);
            summaryJson["eliminated_features_tags"] = ToJsonArray(summary.EliminatedFeaturesTags);
            summaryJson["features_tags_loss_graph"] = ToJson(summary.FeaturesTagsLossGraph, "features_tags");
            summaryJson["features_tags_cost_graph"] = ToJson(summary.FeaturesTagsCostGraph, "features_tags");
        }

        return summaryJson;
    }
}
