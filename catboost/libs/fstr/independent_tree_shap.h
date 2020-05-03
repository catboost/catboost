#pragma once

#include "shap_prepared_trees.h"


void CalcIndependentTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    const TIndependentTreeShapParams& independentTreeShapParams,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    bool calcInternalValues,
    size_t treeIdx,
    TVector<TVector<TVector<TVector<double>>>>* shapValueByDepthForLeaf
);
/*
void IndependentTreeShap(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndexes,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
);
*/

void PostProcessingIndependent(
    const TIndependentTreeShapParams& independentTreeShapParams,
    const TVector<TVector<TVector<double>>>& shapValuesForAllReferences,
    size_t approxDimension,
    size_t featureCount,
    size_t documentIdx,
    double bias,
    TVector<TVector<double>>* shapValues  
);

void SetValuesToShapValuesByAllReferences(
    const TVector<TVector<TVector<double>>>& shapValueByDepthForLeaf,
    const TVector<TVector<ui32>>& referenceIndices,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    size_t leafCount,
    bool isCalcForAllLeaves,
    TVector<TVector<TVector<double>>>* shapValuesForAllReferences
);

void CalcObliviousShapValuesByDepthForLeaf(
    const TModelTrees& forest,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<TVector<double>>& weights,
    size_t flatFeatureCount,
    size_t documentLeafIdx,
    size_t treeIdx,
    bool isCalcForAllLeafes,
    bool calcInternalValues,
    TVector<TVector<TVector<double>>>* shapValueByDepthBetweenLeaves
);