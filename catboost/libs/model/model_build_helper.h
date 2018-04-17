#pragma once

#include <catboost/libs/model/model.h>

#include <catboost/libs/helpers/exception.h>

#include <util/generic/set.h>

class TObliviousTreeBuilder {
public:
    TObliviousTreeBuilder(const TVector<TFloatFeature>& allFloatFeatures, const TVector<TCatFeature>& allCategoricalFeatures, int approxDimension);
    void AddTree(
            const TVector<TModelSplit>& modelSplits,
            const TVector<TVector<double>>& treeLeafValues,
            const TVector<double>& treeLeafWeights);
    void AddTree(
            const TVector<TModelSplit>& modelSplits,
            const TVector<TVector<double>>& treeLeafValues) {
        AddTree(modelSplits, treeLeafValues, TVector<double>());
    }
    TObliviousTrees Build();
private:
    int ApproxDimension = 1;

    TVector<TVector<TModelSplit>> Trees;
    TVector<TVector<double>> LeafValues;
    TVector<TVector<double>> LeafWeights;
    TVector<TFloatFeature> FloatFeatures;
    TVector<TCatFeature> CatFeatures;
};
