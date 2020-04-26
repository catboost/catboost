#pragma once

#include "shap_values.h"
#include "util.h"

#include <catboost/libs/model/model.h>

void CalcObliviousExactShapValuesForLeafImplementation(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    TVector<TShapValue>* shapValues
);