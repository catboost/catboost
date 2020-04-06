#pragma once

#include "independent_tree_shap.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
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

TShapPreparedTrees PrepareTrees(const TFullModel& model, NPar::TLocalExecutor* localExecutor);
TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const NCB::TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr if using Independent Tree SHAP algorithm
    EPreCalcShapValues mode,
    ECalcTypeShapValues calcType,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues = false
);
