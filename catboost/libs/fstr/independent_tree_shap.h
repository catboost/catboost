#pragma once

#include "shap_prepared_trees.h"


void CalcIndependentTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    size_t treeIdx,
    TShapPreparedTrees* preparedTrees
);

void IndependentTreeShap(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndexes,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
);
