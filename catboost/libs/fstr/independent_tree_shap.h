#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


struct TContribution {
    TVector<double> PositiveContribution;
    TVector<double> NegativeContribution;

public:

    explicit TContribution(size_t approxDimension)
        : PositiveContribution(approxDimension)
        , NegativeContribution(approxDimension)
        {
        }
};

class TInternalIndependentTreeShapCalcer {
private:
    const TModelTrees& Forest;
    const TVector<int>& BinFeatureCombinationClassByDepth;
    const TVector<TVector<double>>& Weights;
    TVector<int> ListOfFeaturesDocumentLeaf;
    TVector<int> ListOfFeaturesDocumentLeafReference;
    size_t DocumentLeafIdx;
    size_t DocumentLeafIdxReference;
    size_t TreeIdx;
    int DepthOfTree;
    size_t ApproxDimension;
    const double* LeafValuesPtr;
    TVector<TVector<double>>& ShapValuesInternalByDepth;

public:
    TInternalIndependentTreeShapCalcer(
        const TModelTrees& forest,
        const TVector<int>& binFeatureCombinationClassByDepth,
        const TVector<TVector<double>>& weights,
        size_t classCount,
        size_t documentLeafIdx,
        size_t documentLeafIdxReference,
        size_t treeIdx,
        TVector<TVector<double>>* shapValuesInternalByDepth
    )
        : Forest(forest)
        , BinFeatureCombinationClassByDepth(binFeatureCombinationClassByDepth) 
        , Weights(weights) 
        , ListOfFeaturesDocumentLeaf(classCount) 
        , ListOfFeaturesDocumentLeafReference(classCount) 
        , DocumentLeafIdx(documentLeafIdx) 
        , DocumentLeafIdxReference(documentLeafIdxReference) 
        , TreeIdx(treeIdx) 
        , DepthOfTree(Forest.GetTreeSizes()[TreeIdx]) 
        , ApproxDimension(Forest.GetDimensionsCount()) 
        , LeafValuesPtr(Forest.GetFirstLeafPtrForTree(TreeIdx)) 
        , ShapValuesInternalByDepth(*shapValuesInternalByDepth) 
    { 
    }

    TContribution Calc(
        int depth = 0,
        size_t nodeIdx = 0,
        ui32 uniqueFeaturesCount = 0,
        ui32 featureMatchedForegroundCount = 0
    );
};

using TTransformFunc = double(*)(double target, double approx);

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
