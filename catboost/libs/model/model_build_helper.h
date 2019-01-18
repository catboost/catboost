#pragma once

#include "model.h"

#include <util/generic/array_ref.h>


class TObliviousTreeBuilder {
public:
    TObliviousTreeBuilder(
        const TVector<TFloatFeature>& allFloatFeatures,
        const TVector<TCatFeature>& allCategoricalFeatures,
        int approxDimension);
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        const TVector<TVector<double>>& treeLeafValues,
        TConstArrayRef<double> treeLeafWeights);
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        TConstArrayRef<double> treeLeafValues,
        TConstArrayRef<double> treeLeafWeights);
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        const TVector<TVector<double>>& treeLeafValues) {

        AddTree(modelSplits, treeLeafValues, TVector<double>());
    }
    TObliviousTrees Build();
private:
    int ApproxDimension = 1;
    TVector<TVector<TModelSplit>> Trees;
    TVector<double> LeafValues;
    TVector<TVector<double>> LeafWeights;
    TVector<TFloatFeature> FloatFeatures;
    TVector<size_t> FloatFeaturesInternalIndexesMap;
    TVector<TCatFeature> CatFeatures;
    TVector<size_t> CatFeaturesInternalIndexesMap;
};
