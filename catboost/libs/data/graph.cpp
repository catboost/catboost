#include "graph.h"

#include <catboost/private/libs/data_types/pair.h>

TVector<TVector<ui32>> NCB::ConvertGraphToAdjMatrix(const TRawPairsData& graph, ui32 objectCount) {
    const TFlatPairsInfo* graphP = std::get_if<TFlatPairsInfo>(&graph);
    TVector<TVector<ui32>> matrix(objectCount);
    for (auto g: *graphP) {
        matrix[g.WinnerId].push_back(g.LoserId);
    }
    return matrix;
}

TVector<NCB::EFloatGraphFeatureType> NCB::GetAggregationTypeNames(EFeatureType featureType) {
    switch (featureType) {
        case EFeatureType::Float:
            return {NCB::EFloatGraphFeatureType::Mean, NCB::EFloatGraphFeatureType::Min, NCB::EFloatGraphFeatureType::Max};
        default:
            return {};
    }
}
