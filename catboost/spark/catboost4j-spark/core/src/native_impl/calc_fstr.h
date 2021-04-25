#pragma once

#include <catboost/private/libs/options/enums.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/fstr/shap_prepared_trees.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/fwd.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>

namespace NPar {
    class TLocalExecutor;
}

class TFullModel;
struct TCombinationClassFeatures;


EFstrType GetDefaultFstrType(const TFullModel& model) throw(yexception);

// needed for forwarding exceptions from C++ to JVM
TVector<TString> GetMaybeGeneratedModelFeatureIdsWrapper(
    const TFullModel& model,
    const NCB::TFeaturesLayoutPtr datasetFeaturesLayout // can be null
) throw(yexception);

bool PreparedTreesNeedLeavesWeightsFromDataset(const TFullModel& model) throw(yexception);

// needed for forwarding exceptions from C++ to JVM
TVector<double> CollectLeavesStatisticsWrapper(
    const NCB::TDataProviderPtr dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
) throw(yexception);


TShapPreparedTrees PrepareTreesWithoutIndependent(
    const TFullModel& model,
    i64 datasetObjectCount,
    bool needSumModelAndDatasetWeights,
    TConstArrayRef<double> leafWeightsFromDataset,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType,
    bool calcShapValuesByLeaf,
    NPar::TLocalExecutor* localExecutor
) throw(yexception);


// returned TVector is row-major matrix representation of Stats[featureIdx][metricIdx]
TVector<double> CalcFeatureEffectLossChangeMetricStatsWrapper(
    const TFullModel& model,
    const int featuresCount,
    const TShapPreparedTrees& preparedTrees,
    const NCB::TDataProviderPtr dataset,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
) throw(yexception);


TVector<double> CalcFeatureEffectLossChangeFromScores(
    const TFullModel& model,
    const TCombinationClassFeatures& combinationClassFeatures,
    TConstArrayRef<double> scoresMatrix // row-major matrix representation of Stats[featureIdx][metricIdx]
) throw(yexception);

TVector<double> CalcFeatureEffectAverageChangeWrapper(
    const TFullModel& model,
    TConstArrayRef<double> leafWeightsFromDataset // can be empty
) throw(yexception);

TVector<double> GetPredictionDiffWrapper(
    const TFullModel& model,
    const NCB::TRawObjectsDataProviderPtr objectsDataProvider,
    NPar::TLocalExecutor* localExecutor
) throw(yexception);


class TShapValuesResult {
public:
    TShapValuesResult(TVector<TVector<TVector<double>>>&& data)
        : Data(std::move(data))
    {}

    i32 GetObjectCount() const {
        return SafeIntegerCast<i32>(Data.size());
    }

    i32 GetShapValuesCount() const {
        return SafeIntegerCast<i32>(Data[0][0].size());
    }

    // returns matrix data of (dimension x (featureCount + 1)) size in row-major order
    TVector<double> Get(i32 objectIdx) const throw(yexception);

private:
    TVector<TVector<TVector<double>>> Data; // [objectIdx][dimension][featureIdx]
};

TShapValuesResult CalcShapValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    const TShapPreparedTrees& preparedTrees,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
) throw (yexception);


void GetSelectedFeaturesIndices(
    const TFullModel& model,
    const TString& feature1Name,
    const TString& feature2Name,
    TArrayRef<i32> featureIndices // out param
) throw (yexception);


class TShapInteractionValuesResult {
public:
    TShapInteractionValuesResult(TVector<TVector<TVector<TVector<double>>>>&& data)
        : Data(std::move(data))
    {}

    i32 GetObjectCount() const {
        return SafeIntegerCast<i32>(Data[0][0][0].size());
    }

    i32 GetShapInteractionValuesCount() const {
        return SafeIntegerCast<i32>(Data.size());
    }

    // returns matrix data of ((featureCount + 1) x (featureCount + 1)) size in row-major order
    TVector<double> Get(i32 objectIdx, i32 dimensionIdx = 0) const throw(yexception);

private:
    TVector<TVector<TVector<TVector<double>>>> Data; // [featureIdx1][featureIdx2][dim][objectIdx]
};

TShapInteractionValuesResult CalcShapInteractionValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    TConstArrayRef<i32> selectedFeatureIndices, // -1 if not selected
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
) throw (yexception);


void CalcInteraction(
    const TFullModel& model,
    TVector<i32>* firstIndices,
    TVector<i32>* secondIndices,
    TVector<double>* scores
) throw(yexception);
