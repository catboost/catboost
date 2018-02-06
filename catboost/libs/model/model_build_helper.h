#pragma once

#include <catboost/libs/model/model.h>

#include <catboost/libs/helpers/exception.h>

#include <util/generic/set.h>

class TObliviousTreeBuilder {
public:
    TObliviousTreeBuilder(const TVector<TFloatFeature>& allFloatFeatures, const TVector<TCatFeature>& allCategoricalFeatures);
    void AddTree(const TVector<TModelSplit>& modelSplits, const TVector<TVector<double>>& treeLeafValues);

    TObliviousTrees Build();
private:
    int ApproxDimension = 0;

    TVector<TVector<TModelSplit>> Trees;
    TVector<TVector<double>> LeafValues;
    TVector<TFloatFeature> FloatFeatures;
    TVector<TCatFeature> CatFeatures;
};
