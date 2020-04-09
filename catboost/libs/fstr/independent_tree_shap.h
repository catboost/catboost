#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


namespace {
    using TTransformFunc = double(*)(double target, double approx);
}

struct TIndependentTreeShapParams {
    TVector<TVector<double>> TransformedTargetOfDataset; // [documentIdx][dim]
    TVector<TVector<double>> TargetOfDataset; // [documentIdx][dim]
    TVector<TVector<double>> ApproxOfDataset; // [dim][documentIdx]
    TVector<TVector<double>> ApproxOfReferenceDataset; // [dim][documentIdx]
    EModelOutputType ModelOutputType;
    TTransformFunc TransformFunction; 

    TVector<TVector<double>> Weights;
    TVector<TVector<TVector<TVector<TVector<double>>>>> ShapValueByDepthBetweenLeavesForAllTrees; // [treeIdx][leafIdx(foregroundLeafIdx)][leafIdx(referenceLeafIdx)][depth][dimension]
    TVector<TVector<NCB::NModelEvaluation::TCalcerIndexType>> ReferenceLeafIndicesForAllTrees; // [treeIdx][refIdx] -> leafIdx on refIdx
    TVector<TVector<TVector<ui32>>> ReferenceIndicesForAllTrees; // [treeIdx][leafIdx] -> TVector<ui32> ref Indices
    TVector<bool> IsCalcForAllLeafesForAllTrees;
    TVector<TVector<int>> BinFeatureCombinationClassByDepthForAllTrees;

public:
    TIndependentTreeShapParams(
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        const NCB::TDataProvider& referenceDataset,
        const TVector<int>& binFeatureCombinationClass,
        EModelOutputType modelOutputType,
        NPar::TLocalExecutor* localExecutor
    );

private:
    void InitTransformedData(
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        const NCatboostOptions::TLossDescription& metricDescription,
        NPar::TLocalExecutor* localExecutor
    );
};

struct TShapPreparedTrees;

void CalcIndependentTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    size_t treeIdx,
    TShapPreparedTrees* preparedTrees
);

void IndependentTreeShap(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    int flatFeatureCount,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndexes,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
);
