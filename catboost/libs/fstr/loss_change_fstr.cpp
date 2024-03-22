#include "loss_change_fstr.h"
#include "util.h"

#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/yetirank_helpers.h>
#include <catboost/private/libs/target/data_providers.h>
#include <catboost/private/libs/pairs/util.h>

#include <util/string/split.h>

using namespace NCB;

TCombinationClassFeatures GetCombinationClassFeatures(const TFullModel& model) {
    NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);
    TVector<std::pair<TVector<int>, TFeature>> featuresCombinations;
    const TModelTrees& forest = *model.ModelTrees;

    for (const TFloatFeature& floatFeature : forest.GetFloatFeatures()) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        featuresCombinations.emplace_back();
        featuresCombinations.back().first = { floatFeature.Position.FlatIndex };
        featuresCombinations.back().second = TFeature(floatFeature);
    }
    for (const TOneHotFeature& oneHotFeature: forest.GetOneHotFeatures()) {
        featuresCombinations.emplace_back();
        featuresCombinations.back().first = {
            (int)layout.GetExternalFeatureIdx(oneHotFeature.CatFeatureIndex, EFeatureType::Categorical)
        };
        featuresCombinations.back().second = TFeature(oneHotFeature);
    }
    for (const TCtrFeature& ctrFeature : forest.GetCtrFeatures()) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        featuresCombinations.emplace_back();
        for (int catFeatureIdx : combination.CatFeatures) {
            featuresCombinations.back().first.push_back(
                layout.GetExternalFeatureIdx(catFeatureIdx, EFeatureType::Categorical)
            );
        }
        featuresCombinations.back().second = TFeature(ctrFeature);
    }
    for (const TEstimatedFeature& estimatedFeature: forest.GetEstimatedFeatures()) {
        featuresCombinations.emplace_back();
        featuresCombinations.back().first = {
            (int)layout.GetExternalFeatureIdx(
                estimatedFeature.ModelEstimatedFeature.SourceFeatureId,
                EstimatedSourceFeatureTypeToFeatureType(
                    estimatedFeature.ModelEstimatedFeature.SourceFeatureType
                )
            )
        };
        featuresCombinations.back().second = TFeature(
            estimatedFeature.ModelEstimatedFeature,
            GetEstimatedFeatureCalcerType(model, estimatedFeature.ModelEstimatedFeature)
        );
    }
    TVector<int> sortedBinFeatures(featuresCombinations.size());
    Iota(sortedBinFeatures.begin(), sortedBinFeatures.end(), 0);
    StableSort(
        sortedBinFeatures.begin(),
        sortedBinFeatures.end(),
        [featuresCombinations](int feature1, int feature2) {
            return featuresCombinations[feature1].first < featuresCombinations[feature2].first;
        }
    );
    TCombinationClassFeatures combinationClassFeatures;

    for (ui32 featureIdx = 0; featureIdx < featuresCombinations.size(); ++featureIdx) {
        int currentFeature = sortedBinFeatures[featureIdx];
        int previousFeature = featureIdx == 0 ? -1 : sortedBinFeatures[featureIdx - 1];
        if (featureIdx == 0 ||
            featuresCombinations[currentFeature].first != featuresCombinations[previousFeature].first)
        {
            combinationClassFeatures.push_back(featuresCombinations[currentFeature].second);
        }
    }
    return combinationClassFeatures;
}

i64 GetMaxObjectCountForFstrCalc(i64 objectCount, i32 featureCount) {
    return Min(objectCount, Max(i64(2e5), i64(2e9 / featureCount)));
}

const TDataProviderPtr GetSubsetForFstrCalc(
    const TDataProviderPtr dataset,
    NPar::ILocalExecutor* localExecutor)
{
    ui32 totalDocumentCount = dataset->ObjectsData->GetObjectCount();
    ui32 maxDocumentCount = SafeIntegerCast<ui32>(
        GetMaxObjectCountForFstrCalc(
            totalDocumentCount,
            SafeIntegerCast<i64>(dataset->ObjectsData->GetFeaturesLayout()->GetExternalFeatureCount())
        )
    );

    if (totalDocumentCount > maxDocumentCount) {
        ui32 foldCount = totalDocumentCount / maxDocumentCount;

        TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;

        testSubsets = NCB::Split(*dataset->ObjectsGrouping, foldCount, /*oldCvStyleSplit*/ true);

        auto subset = dataset->GetSubset(
            GetSubset(
                dataset->ObjectsGrouping,
                std::move(testSubsets[0]),
                NCB::EObjectsOrder::Ordered
            ),
            NSystemInfo::TotalMemorySize(),
            localExecutor
        );
        return subset;
    } else {
        return dataset;
    }
}

void CreateMetricAndLossDescriptionForLossChange(
    const TFullModel& model,
    NCatboostOptions::TLossDescription* metricDescription,
    NCatboostOptions::TLossDescription* lossDescription,
    bool* needYetiRankPairs,
    THolder<IMetric>* metric)
{
    CB_ENSURE(
        TryGetObjectiveMetric(model, metricDescription),
        "Cannot calculate LossFunctionChange feature importances without metric, need model with params"
    );
    CATBOOST_INFO_LOG << "Used " << *metricDescription << " metric for fstr calculation" << Endl;

    CB_ENSURE(TryGetLossDescription(model, lossDescription), "No loss_function in model params");

    // NDCG and PFound metrics are possible for YetiRank
    // PFound replace with PairLogit (with YetiRank generated pairs) due to quality
    // NDCG used for labels not in [0., 1.] and don't use YetiRank pairs
    *needYetiRankPairs =
        (IsYetiRankLossFunction(lossDescription->GetLossFunction())
         && metricDescription->LossFunction != ELossFunction::NDCG);
    if (*needYetiRankPairs) {
        TString lossDescription = "PairLogit:max_pairs=";
        lossDescription.append(ToString(MAX_PAIR_COUNT_ON_GPU));
        *metricDescription = NCatboostOptions::ParseLossDescription(lossDescription);
    }
    *metric = std::move(
        CreateMetricFromDescription(*metricDescription, model.ModelTrees->GetDimensionsCount())[0]
    );
    CB_ENSURE((*metric)->IsAdditiveMetric(), "LossFunctionChange support only additive metric");
}

TVector<TMetricHolder> CalcFeatureEffectLossChangeMetricStats(
    const TFullModel& model,
    const int featuresCount,
    const TShapPreparedTrees& preparedTrees,
    const TDataProviderPtr dataset,
    ECalcTypeShapValues calcType,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor)
{
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

    TRestorableFastRng64 rand(0);
    auto targetData = CreateModelCompatibleProcessedDataProvider(
        *dataset.Get(),
        { metricDescription },
        model,
        GetMonopolisticFreeCpuRam(),
        &rand,
        localExecutor,
        /* metricsThatRequireTargetCanBeSkipped */ false,
        /* skipMinMaxPairsCheck */ true
    ).TargetData;
    CB_ENSURE(targetData->GetTargetDimension() <= 1, "Multi-dimensional target fstr is unimplemented yet");

    ui32 documentCount = dataset->ObjectsData->GetObjectCount();
    const TObjectsDataProvider& objectsData = *dataset->ObjectsData;

    TVector<TMetricHolder> scores(featuresCount + 1);

    TConstArrayRef<TQueryInfo> targetQueriesInfo
        = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());
    TVector<TVector<double>> approx = ApplyModelMulti(
        model,
        objectsData,
        EPredictionType::RawFormulaVal,
        /*begin*/ 0,
        /*end*/ 0,
        localExecutor,
        dataset->RawTargetData.GetBaseline()
    );
    TVector<TQueryInfo> queriesInfo(targetQueriesInfo.begin(), targetQueriesInfo.end());

    ui32 blockCount = queriesInfo.empty() ? documentCount : queriesInfo.size();
    ui32 blockSize = Min(ui32(10000), ui32(1e6) / (featuresCount * approx.ysize())); // shapValues[blockSize][featuresCount][dim] double

    if (needYetiRankPairs) {
        ui32 maxQuerySize = 0;
        for (const auto& query : queriesInfo) {
            maxQuerySize = Max(maxQuerySize, query.GetSize());
        }
        blockSize = Min(blockSize, ui32(ceil(20000. / maxQuerySize)));
    }

    int approxDimension = model.ModelTrees->GetDimensionsCount();

    TProfileInfo profile(documentCount);
    TImportanceLogger importanceLogger(
        documentCount,
        "Process documents",
        "Started LossFunctionChange calculation",
        1
    );
    for (ui32 queryBegin = 0; queryBegin < blockCount; queryBegin += blockSize) {
        profile.StartIterationBlock();
        ui32 queryEnd = Min(blockCount, queryBegin + blockSize);
        ui32 begin, end;
        if (queriesInfo.empty()) {
            begin = queryBegin;
            end = queryEnd;
        } else {
            begin = queriesInfo[queryBegin].Begin;
            end = queriesInfo[queryEnd - 1].End;
        }
        if (needYetiRankPairs) {
            UpdatePairsForYetiRank(
                approx[0],
                *targetData->GetOneDimensionalTarget(),
                lossDescription,
                /*randomSeed*/ randomSeed,
                queryBegin,
                queryEnd,
                &queriesInfo,
                localExecutor
            );
        }
        scores.back().Add(
            dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
                approx,
                targetData->GetOneDimensionalTarget().GetOrElse(TConstArrayRef<float>()),
                GetWeights(*targetData),
                queriesInfo,
                queryBegin,
                queryEnd,
                *localExecutor
            )
        );
        TVector<TVector<TVector<double>>> shapValues;
        CalcShapValuesInternalForFeature(
            preparedTrees,
            model,
            0,
            begin,
            end,
            featuresCount,
            objectsData,
            &shapValues,
            localExecutor,
            calcType
        );

        for (int featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
            NPar::ILocalExecutor::TExecRangeParams blockParams(begin, end);
            blockParams.SetBlockCountToThreadCount();
            localExecutor->ExecRange([&](ui32 docIdx) {
                for (int dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] -= shapValues[docIdx - begin][featureIdx][dimensionIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
            scores[featureIdx].Add(
                dynamic_cast<const ISingleTargetEval*>(metric.Get())->Eval(
                    approx,
                    targetData->GetOneDimensionalTarget().GetOrElse(TConstArrayRef<float>()),
                    GetWeights(*targetData),
                    queriesInfo,
                    queryBegin,
                    queryEnd,
                    *localExecutor
                )
            );
            localExecutor->ExecRange([&](ui32 docIdx) {
                for (int dimensionIdx = 0; dimensionIdx < approxDimension; ++dimensionIdx) {
                    approx[dimensionIdx][docIdx] += shapValues[docIdx - begin][featureIdx][dimensionIdx];
                }
            }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
        }
        if (needYetiRankPairs) {
            for (ui32 queryIndex = queryBegin; queryIndex < queryEnd; ++queryIndex) {
                queriesInfo[queryIndex].Competitors.clear();
                queriesInfo[queryIndex].Competitors.shrink_to_fit();
            }
        }
        profile.FinishIterationBlock(end - begin);
        importanceLogger.Log(profile.GetProfileResults());
    }

    return scores;
}

TVector<std::pair<double, TFeature>> CalcFeatureEffectLossChangeFromScores(
    const TCombinationClassFeatures& combinationClassFeatures,
    const IMetric& metric,
    const TVector<TMetricHolder>& scores)
{
    int featuresCount = combinationClassFeatures.size();
    if (featuresCount == 0) {
        TVector<std::pair<double, TFeature>> result;
        return result;
    }

    TVector<std::pair<double, int>> featureScore(featuresCount);

    EMetricBestValue valueType;
    float bestValue;
    metric.GetBestValue(&valueType, &bestValue);
    for (int idx = 0; idx < featuresCount; ++idx) {
        double score = metric.GetFinalError(scores[idx]) - metric.GetFinalError(scores.back());
        switch(valueType) {
            case EMetricBestValue::Min:
                break;
            case EMetricBestValue::Max:
                score = -score;
                break;
            case EMetricBestValue::FixedValue:
                score = abs(metric.GetFinalError(scores[idx]) - bestValue)
                        - abs(metric.GetFinalError(scores.back()) - bestValue);
                break;
            default:
                ythrow TCatBoostException() << "unsupported bestValue metric type";
        }
        featureScore[idx].first = score;
        featureScore[idx].second = idx;
    }
    StableSort(featureScore.begin(), featureScore.end(), std::greater<std::pair<double, int>>());
    TVector<std::pair<double, TFeature>> result;

    for (const auto& score: featureScore) {
        result.emplace_back();
        result.back().first = score.first;
        result.back().second = combinationClassFeatures[score.second];
    }
    return result;
}

TVector<std::pair<double, TFeature>> CalcFeatureEffectLossChange(
    const TFullModel& model,
    const TDataProviderPtr dataProvider,
    NPar::ILocalExecutor* localExecutor,
    ECalcTypeShapValues calcType)
{
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

    const auto dataset = GetSubsetForFstrCalc(dataProvider, localExecutor);

    ui32 documentCount = dataset->ObjectsData->GetObjectCount();

    CATBOOST_INFO_LOG << "Selected " << documentCount << " documents from " << dataProvider->GetObjectCount()
                      << " for LossFunctionChange calculation." << Endl;

    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        dataset.Get(),
        /*referenceDataset*/ nullptr,
        EPreCalcShapValues::Auto,
        localExecutor,
        /*calcInternalValues*/ true,
        calcType
    );
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*logPeriod*/ 0,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees,
        calcType
    );

    auto combinationClassFeatures = GetCombinationClassFeatures(model);
    int featuresCount = combinationClassFeatures.size();

    auto scores = CalcFeatureEffectLossChangeMetricStats(
        model,
        featuresCount,
        preparedTrees,
        dataset,
        calcType,
        /* randomSeed */ 0,
        localExecutor
    );

    return CalcFeatureEffectLossChangeFromScores(combinationClassFeatures, *metric, scores);
}
