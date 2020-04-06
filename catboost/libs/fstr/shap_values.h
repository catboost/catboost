#pragma once

#include "shap_prepared_trees.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


struct TFixedFeatureParams {
    enum class EMode {
        FixedOn, FixedOff, NotFixed
    };

    int Feature = -1;
    EMode FixedFeatureMode = EMode::NotFixed;

public:
    TFixedFeatureParams() = default;

    TFixedFeatureParams(int feature, EMode fixedFeatureMode)
        : Feature(feature)
        , FixedFeatureMode(fixedFeatureMode)
        {
        }
};

struct TConditionsFeatureFraction {
    double HotConditionFeatureFraction;
    double ColdConditionFeatureFraction;

public:
    TConditionsFeatureFraction(
        const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
        int combinationClass,
        double conditionFeatureFraction,
        double hotCoefficient,
        double coldCoefficient
    );
};

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int flatFeatureCount,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndices,
    ECalcTypeShapValues calcType,
    size_t documentIdxInBlock,
    TVector<TVector<double>>* shapValues,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    size_t documentIdx = (size_t)(-1)
);

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    int flatFeatureCount,
    TConstArrayRef<NCB::NModelEvaluation::TCalcerIndexType> docIndices,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

void CalcShapValuesByLeaf(
    const TFullModel& model,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    ECalcTypeShapValues calcType,
    int logPeriod,
    bool calcInternalValues,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

// returned: ShapValues[documentIdx][dimension][feature]
TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr if using Independent Tree SHAP algorithm
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    EModelOutputType modelOutputType = EModelOutputType::Raw
);

// returned: ShapValues[documentIdx][feature]
TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const NCB::TDataProviderPtr referenceDataset, // can be nullptr if using Independent Tree SHAP algorithm
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular,
    EModelOutputType modelOutputType = EModelOutputType::Raw
);

// returned: ShapValues[featureIdx][dim][documentIdx]
TVector<TVector<TVector<double>>> CalcShapValueWithQuantizedData(
    const TFullModel& model,
    const TVector<TIntrusivePtr<NCB::NModelEvaluation::IQuantizedData>>& quantizedFeatures,
    const TVector<TVector<NCB::NModelEvaluation::TCalcerIndexType>>& indexes,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const size_t documentCount,
    int logPeriod,
    TShapPreparedTrees* preparedTrees,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);

// outputs for each document in order for each dimension in order an array of feature contributions
void CalcAndOutputShapValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
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
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType = ECalcTypeShapValues::Regular
);
