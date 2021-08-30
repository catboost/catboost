#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>


struct TShapPreparedTrees;


void ValidateFeaturePair(int flatFeatureCount, std::pair<int, int> featurePair);
void ValidateFeatureInteractionParams(
    const EFstrType fstrType,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    ECalcTypeShapValues calcType
);


// returned: ShapInteractionValues[featureIdx1][featureIdx2][dim][documentIdx]
TVector<TVector<TVector<TVector<double>>>> CalcShapInteractionValuesWithPreparedTrees(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    ECalcTypeShapValues calcType,
    NPar::ILocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
);

// returned: ShapInteractionValues[featureIdx1][featureIdx2][dim][documentIdx]
TVector<TVector<TVector<TVector<double>>>> CalcShapInteractionValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TMaybe<std::pair<int, int>>& pairOfFeatures,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);
