#include "calc_fstr.h"

#include <catboost/private/libs/options/enum_helpers.h>

#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/fstr/compare_documents.h>
#include <catboost/libs/fstr/shap_interaction_values.h>
#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/fstr/util.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/xrange.h>
#include <util/system/yassert.h>

using namespace NCB;


EFstrType GetDefaultFstrType(const TFullModel& model) {
    return IsGroupwiseMetric(model.GetLossFunctionName())
       ? EFstrType::LossFunctionChange
       : EFstrType::PredictionValuesChange;
}

bool PreparedTreesNeedLeavesWeightsFromDataset(const TFullModel& model) {
    const auto leafWeightsOfModels = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
    if (!leafWeightsOfModels) {
        return true;
    }
    // needSumModelAndDatasetWeights
    return HasNonZeroApproxForZeroWeightLeaf(model);
}

TVector<double> CollectLeavesStatisticsWrapper(
    const TDataProviderPtr dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
) {
    return CollectLeavesStatistics(*dataset, model, localExecutor);
}

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
)  {
    TShapPreparedTrees preparedTrees = PrepareTreesWithoutIndependent(
        model,
        datasetObjectCount,
        needSumModelAndDatasetWeights,
        leafWeightsFromDataset,
        mode,
        calcInternalValues,
        calcType
    );
    if (calcShapValuesByLeaf) {
        CalcShapValuesByLeaf(
            model,
            /*fixedFeatureParams*/ Nothing(),
            /*logPeriod*/ 0,
            calcInternalValues,
            localExecutor,
            &preparedTrees,
            calcType
        );
    }
    return preparedTrees;
}

TVector<double> CalcFeatureEffectLossChangeMetricStatsWrapper(
    const TFullModel& model,
    const int featuresCount,
    const TShapPreparedTrees& preparedTrees,
    const TDataProviderPtr dataset,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
) {
    auto resultWithMetricHolders = CalcFeatureEffectLossChangeMetricStats(
        model,
        featuresCount,
        preparedTrees,
        dataset,
        calcType,
        /* randomSeed */0,
        localExecutor
    );
    TVector<double> result;
    auto statsPerMetricSize = resultWithMetricHolders[0].Stats.size();
    result.yresize(resultWithMetricHolders.size() * statsPerMetricSize);
    for (auto featureIdx : xrange(resultWithMetricHolders.size())) {
        for (auto statsIdx : xrange(statsPerMetricSize)) {
            result[featureIdx * statsPerMetricSize + statsIdx] = resultWithMetricHolders[featureIdx].Stats[statsIdx];
        }
    }
    return result;
}

TVector<double> CalcFeatureEffectLossChangeFromScores(
    const TFullModel& model,
    const TCombinationClassFeatures& combinationClassFeatures,
    TConstArrayRef<double> scoresMatrix // row-major matrix representation of Stats[featureIdx][metricIdx]
) {
    NCatboostOptions::TLossDescription metricDescription;
    NCatboostOptions::TLossDescription lossDescription;
    bool needYetiRankPairs = false;
    THolder<IMetric> metric;

    CreateMetricAndLossDescriptionForLossChange(
        model,
        &metricDescription,
        &lossDescription,
        &needYetiRankPairs,
        &metric
    );

    const auto featuresCount = combinationClassFeatures.size();
    TVector<TMetricHolder> scores(featuresCount + 1);
    Y_ASSERT(!(scoresMatrix.size() % (featuresCount + 1)));
    const auto statsPerMetricSize = scoresMatrix.size() / (featuresCount + 1);
    for (auto featureIdx : xrange(scores.size())) {
        scores[featureIdx].Stats.reserve(statsPerMetricSize);
        for (auto statsIdx : xrange(statsPerMetricSize)) {
            scores[featureIdx].Stats.push_back(
                scoresMatrix[featureIdx * statsPerMetricSize + statsIdx]
            );
        }
    }

    const TVector<std::pair<double, TFeature>> internalFeatureEffect = CalcFeatureEffectLossChangeFromScores(
        combinationClassFeatures,
        *metric,
        scores
    );
    return GetFeatureEffectForLinearIndices(internalFeatureEffect, model);
}

TVector<double> CalcFeatureEffectAverageChangeWrapper(
    const TFullModel& model,
    TConstArrayRef<double> leafWeightsFromDataset // can be empty
) {
    TConstArrayRef<double> leafWeights;
    if (leafWeightsFromDataset.empty()) {
        leafWeights = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
        CB_ENSURE(
            !leafWeights.empty(),
            "CalcFeatureEffectAverageChange requires either non-empty LeafWeights in model or provided dataset"
        );
    } else {
        leafWeights = leafWeightsFromDataset;
    }

    const TVector<std::pair<double, TFeature>> internalFeatureEffect = CalcFeatureEffectAverageChange(
        model,
        leafWeights
    );
    return GetFeatureEffectForLinearIndices(internalFeatureEffect, model);
}

TVector<double> GetPredictionDiffWrapper(
    const TFullModel& model,
    const TRawObjectsDataProviderPtr objectsDataProvider,
    NPar::TLocalExecutor* localExecutor
) {
    return GetPredictionDiff(model, TObjectsDataProviderPtr(objectsDataProvider), localExecutor);
}


TVector<double> TShapValuesResult::Get(i32 objectIdx) const {
    const TVector<TVector<double>>& perObjectData = Data[objectIdx];
    Y_ASSERT(perObjectData.size() > 0);
    const size_t perDimensionSize = perObjectData[0].size();

    TVector<double> result;
    result.yresize(perObjectData.size() * perDimensionSize);
    for (auto dimensionIdx : xrange(perObjectData.size())) {
        for (auto featureIdx : xrange(perDimensionSize)) {
            result[dimensionIdx * perDimensionSize + featureIdx] = perObjectData[dimensionIdx][featureIdx];
        }
    }
    return result;
}

TShapValuesResult CalcShapValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const TDataProviderPtr dataset,
    const TShapPreparedTrees& preparedTrees,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
) {
    return TShapValuesResult(
        CalcShapValuesWithPreparedTrees(
            model,
            *dataset,
            /*fixedFeatureParams*/ Nothing(),
            /*logPeriod*/ 0,
            preparedTrees,
            localExecutor,
            calcType
        )
    );
}


void GetSelectedFeaturesIndices(
    const TFullModel& model,
    const TString& feature1Name,
    const TString& feature2Name,
    TArrayRef<i32> featureIndices
) {
    CB_ENSURE_INTERNAL(featureIndices.size() == 2, "featureIndices must have 2 elements");

    featureIndices[0] = -1;
    featureIndices[1] = -1;
    const TFeaturesLayout layout = MakeFeaturesLayout(model);
    TConstArrayRef<TFeatureMetaInfo> metaInfoArray = layout.GetExternalFeaturesMetaInfo();
    for (auto featureIdx : xrange(SafeIntegerCast<i32>(metaInfoArray.size()))) {
        const auto& featureName = metaInfoArray[featureIdx].Name;
        if (feature1Name == featureName) {
            featureIndices[0] = featureIdx;
        }
        if (feature2Name == featureName) {
            featureIndices[1] = featureIdx;
        }
    }
    CB_ENSURE(featureIndices[0] != -1, "Feature with name '" << feature1Name << "' not found");
    CB_ENSURE(featureIndices[1] != -1, "Feature with name '" << feature2Name << "' not found");
}


TVector<double> TShapInteractionValuesResult::Get(i32 objectIdx, i32 dimensionIdx) const {
    const size_t shapInteractionValuesCount = Data.size();
    Y_ASSERT(shapInteractionValuesCount > 0);

    TVector<double> result;
    result.yresize(shapInteractionValuesCount * shapInteractionValuesCount);
    for (auto featureIdx1 : xrange(shapInteractionValuesCount)) {
        for (auto featureIdx2 : xrange(shapInteractionValuesCount)) {
            result[featureIdx1 * shapInteractionValuesCount + featureIdx2]
                = Data[featureIdx1][featureIdx2][dimensionIdx][objectIdx];
        }
    }
    return result;
}

TShapInteractionValuesResult CalcShapInteractionValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    TConstArrayRef<i32> selectedFeatureIndices,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
) {
    CB_ENSURE_INTERNAL(selectedFeatureIndices.size() == 2, "featureIndices must have 2 elements");

    TMaybe<std::pair<int, int>> pairOfFeatures;
    if (selectedFeatureIndices[0] != -1) {
        CB_ENSURE_INTERNAL(selectedFeatureIndices[1] != -1, "only one of selected feature indices != -1");
        pairOfFeatures = std::pair<int, int>{selectedFeatureIndices[0], selectedFeatureIndices[1]};
    } else {
        CB_ENSURE_INTERNAL(selectedFeatureIndices[1] == -1, "only one of selected feature indices != -1");
    }

    return TShapInteractionValuesResult(
        CalcShapInteractionValuesWithPreparedTrees(
            model,
            *dataset,
            pairOfFeatures,
            /*logPeriod*/ 0,
            calcType,
            localExecutor,
            preparedTrees
        )
    );
}


void CalcInteraction(
    const TFullModel& model,
    TVector<i32>* firstIndices,
    TVector<i32>* secondIndices,
    TVector<double>* scores
) {
    const TFeaturesLayout layout = MakeFeaturesLayout(model);

    TVector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    TVector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);

    firstIndices->clear();
    secondIndices->clear();
    scores->clear();
    for (const auto& value : interaction) {
        firstIndices->push_back(
            SafeIntegerCast<i32>(
                layout.GetExternalFeatureIdx(value.FirstFeature.Index, value.FirstFeature.Type)
            )
        );
        secondIndices->push_back(
            SafeIntegerCast<i32>(
                layout.GetExternalFeatureIdx(value.SecondFeature.Index, value.SecondFeature.Type)
            )
        );
        scores->push_back(value.Score);
    }
}
