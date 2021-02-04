#pragma once

#include "shap_prepared_trees.h"

#include <catboost/libs/model/model.h>

#include <util/generic/vector.h>

void PostProcessingIndependent(
    const TIndependentTreeShapParams& independentTreeShapParams,
    const TVector<TVector<TVector<double>>>& shapValuesForAllReferences,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t approxDimension,
    size_t flatFeatureCount,
    size_t documentIdx,
    bool calcInternalValues,
    const TVector<double>& bias,
    TVector<TVector<double>>* shapValues
);

void AddValuesToShapValuesByAllReferences(
    const TVector<TVector<TVector<double>>>& shapValueByDepthForLeaf,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClassByDepth,
    TVector<TVector<TVector<double>>>* shapValuesForAllReferences
);

void CalcObliviousShapValuesByDepthForLeaf(
    const TModelTrees& forest,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<TVector<double>>& weights,
    size_t documentLeafIdx,
    size_t treeIdx,
    bool isCalcForAllLeafes,
    TVector<TVector<TVector<double>>>* shapValueByDepthBetweenLeaves
);
