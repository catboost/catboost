#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
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

struct TIndependentTreeShapParams {
    TVector<TVector<double>> ProbabilitiesOfReferenceDataset; // [dim][documentIdx]
    TVector<TVector<double>> TransformedTargetOfDataset; // [dim][documentIdx]
    TVector<TVector<double>> TargetOfDataset; // [dim][documentIdx]
    TVector<TVector<double>> ApproxOfDataset; // [dim][documentIdx]
    TVector<TVector<double>> ApproxOfReferenceDataset; // [dim][documentIdx]
    EExplainableModelOutput ModelOutputType;
    TAtomicSharedPtr<IMetric> Metric;

    TVector<TVector<double>> Weights;
    TVector<TVector<TVector<TVector<TVector<double>>>>> ShapValueByDepthBetweenLeavesForAllTrees; // [treeIdx][leafIdx(foregroundLeafIdx)][leafIdx(referenceLeafIdx)][depth][dimension]
    TVector<TVector<NCB::NModelEvaluation::TCalcerIndexType>> ReferenceLeafIndicesForAllTrees; // [treeIdx][refIdx] -> leafIdx on refIdx
    TVector<TVector<TVector<ui32>>> ReferenceIndicesForAllTrees; // [treeIdx][leafIdx] -> TVector<ui32> ref Indices
    TVector<bool> IsCalcForAllLeafesForAllTrees;
    int FlatFeatureCount;

public:
    TIndependentTreeShapParams(
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        const NCB::TDataProvider& referenceDataset,
        EExplainableModelOutput modelOutputType,
        NPar::ILocalExecutor* localExecutor
    );

private:
    void InitTransformedData(
        const TFullModel& model,
        const NCB::TDataProvider& dataset,
        const NCatboostOptions::TLossDescription& metricDescription,
        NPar::ILocalExecutor* localExecutor
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
    TVector<TVector<TVector<TVector<double>>>> SubtreeValuesForAllTrees;
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
        SubtreeWeightsForAllTrees,
        SubtreeValuesForAllTrees
    );
};

TShapPreparedTrees PrepareTrees(const TFullModel& model, NPar::ILocalExecutor* localExecutor);

TShapPreparedTrees PrepareTreesWithoutIndependent(
    const TFullModel& model,
    i64 datasetObjectCount, // can be -1 if no dataset is provided
    bool needSumModelAndDatasetWeights,
    TConstArrayRef<double> leafWeightsFromDataset,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType
);

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr if using Independent Tree SHAP algorithm
    EPreCalcShapValues mode,
    NPar::ILocalExecutor* localExecutor,
    bool calcInternalValues = false,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    EExplainableModelOutput modelOutputType = EExplainableModelOutput::Raw,
    bool fstrOnTrainPool=false
);

