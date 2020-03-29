#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


struct TShapValue {
    int Feature = -1;
    TVector<double> Value;

public:
    TShapValue() = default;

    TShapValue(int feature, int approxDimension)
        : Feature(feature)
        , Value(approxDimension)
    {
    }

    Y_SAVELOAD_DEFINE(Feature, Value);
};

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

struct TIndependentTreeShapParams {
    TConstArrayRef<TConstArrayRef<float>> TransformedTargetOfInputDataset;
    TConstArrayRef<TConstArrayRef<float>> TransformedTargetOfReferenceDataset;
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
        NPar::TLocalExecutor* localExecutor
    );
};

struct TShapPreparedTrees {
    TVector<TVector<TVector<TShapValue>>> ShapValuesByLeafForAllTrees; // [treeIdx][leafIdx][shapFeature] trees * 2^d * d
    TVector<TVector<double>> MeanValuesForAllTrees;
    TVector<double> AverageApproxByTree;
    TVector<int> BinFeatureCombinationClass;
    TVector<TVector<int>> CombinationClassFeatures;
    bool CalcShapValuesByLeafForAllTrees;
    bool CalcInternalValues;
    TVector<double> LeafWeightsForAllTrees;
    TVector<TVector<TVector<double>>> SubtreeWeightsForAllTrees;
    TMaybe<TIndependentTreeShapParams> IndependentTreeShapParams;

public:
    TShapPreparedTrees() = default;

    TShapPreparedTrees(
        const TVector<TVector<TVector<TShapValue>>>& shapValuesByLeafForAllTrees,
        const TVector<TVector<double>>& meanValuesForAllTrees
    )
        : ShapValuesByLeafForAllTrees(shapValuesByLeafForAllTrees)
        , MeanValuesForAllTrees(meanValuesForAllTrees)
    {
    }

    Y_SAVELOAD_DEFINE(
        ShapValuesByLeafForAllTrees,
        MeanValuesForAllTrees,
        AverageApproxByTree,
        BinFeatureCombinationClass,
        CombinationClassFeatures,
        CalcShapValuesByLeafForAllTrees,
        CalcInternalValues,
        LeafWeightsForAllTrees,
        SubtreeWeightsForAllTrees
    );
};

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    int flatFeatureCount,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndexes,
    ECalcShapValues modeCalcShapValues,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
);

TShapPreparedTrees PrepareTrees(const TFullModel& model, NPar::TLocalExecutor* localExecutor);
TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr if calc in mode no independent
    int logPeriod,
    EPreCalcShapValues mode,
    ECalcShapValues modeCalcShapValues,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues = false
);

// returned: ShapValues[documentIdx][dimenesion][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const NCB::TDataProviderPtr referenceDataset,
    int logPeriod,
    EPreCalcShapValues mode,
    ECalcShapValues modeCalcShapValues,
    NPar::TLocalExecutor* localExecutor
);

// returned: ShapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const NCB::TDataProviderPtr referenceDataset,
    int logPeriod,
    EPreCalcShapValues mode,
    ECalcShapValues modeCalcShapValues,
    NPar::TLocalExecutor* localExecutor
);

// TODO (to add parameter modeCalcShapValues)
// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor
);

void CalcShapValuesInternalForFeature(
    const TShapPreparedTrees& preparedTrees,
    const TFullModel& model,
    int logPeriod,
    ui32 start,
    ui32 end,
    ui32 featuresCount,
    const NCB::TObjectsDataProvider& objectsData,
    TVector<TVector<TVector<double>>>* shapValues, // [docIdx][featureIdx][dim]
    NPar::TLocalExecutor* localExecutor
);
