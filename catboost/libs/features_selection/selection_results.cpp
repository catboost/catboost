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


    static TJsonValue ToJson(const TFeaturesSelectionLossGraph& lossGraph) {
        TJsonValue lossGraphJson(JSON_MAP);
        lossGraphJson["removed_features_count"] = ToJsonArray(lossGraph.RemovedFeaturesCount);
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
        summaryJson["loss_graph"] = ToJson(summary.LossGraph);
        return summaryJson;
    }
}
