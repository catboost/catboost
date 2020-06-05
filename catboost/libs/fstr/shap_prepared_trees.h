#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/ysaveload.h>


struct TShapValue;

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

TShapPreparedTrees PrepareTrees(const TFullModel& model, NPar::TLocalExecutor* localExecutor);

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues = false,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

