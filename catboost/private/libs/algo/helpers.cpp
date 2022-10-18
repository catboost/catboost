#include "helpers.h"

#include "learn_context.h"

#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/malloc/api/malloc.h>

#include <functional>


using namespace NCB;


template <class TFeature, EFeatureType FeatureType>
static TVector<TFeature> CreateFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    std::function<void(TFeature&)> setSpecificDataFunc
) {
    TVector<TFeature> features;

    const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

    for (auto flatFeatureIdx : xrange(featuresMetaInfo.size())) {
        const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
        if (featureMetaInfo.Type != FeatureType) {
            continue;
        }

        TFeature feature;
        feature.Position.Index = (int)*featuresLayout.GetInternalFeatureIdx<FeatureType>(flatFeatureIdx);
        feature.Position.FlatIndex = flatFeatureIdx;
        feature.FeatureId = featureMetaInfo.Name;

        if (featureMetaInfo.IsAvailable) {
            setSpecificDataFunc(feature);
        }

        features.push_back(std::move(feature));
    }

    return features;
}


TVector<TFloatFeature> CreateFloatFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
) {
    return CreateFeatures<TFloatFeature, EFeatureType::Float>(
        featuresLayout,
        [&] (TFloatFeature& floatFeature) {
            const auto floatFeatureIdx = TFloatFeatureIdx((ui32)floatFeature.Position.Index);
            auto nanMode = quantizedFeaturesInfo.GetNanMode(floatFeatureIdx);
            if (nanMode == ENanMode::Min) {
                floatFeature.NanValueTreatment = TFloatFeature::ENanValueTreatment::AsFalse;
                floatFeature.HasNans = true;
            } else if (nanMode == ENanMode::Max) {
                floatFeature.NanValueTreatment = TFloatFeature::ENanValueTreatment::AsTrue;
                floatFeature.HasNans = true;
            }

            floatFeature.Borders = quantizedFeaturesInfo.GetBorders(floatFeatureIdx);
        }
    );
}

TVector<TCatFeature> CreateCatFeatures(const NCB::TFeaturesLayout& featuresLayout) {
    return CreateFeatures<TCatFeature, EFeatureType::Categorical>(
        featuresLayout,
        [] (TCatFeature&) { }
    );
}


TVector<TTextFeature> CreateTextFeatures(const NCB::TFeaturesLayout& featuresLayout) {
    return CreateFeatures<TTextFeature, EFeatureType::Text>(
        featuresLayout,
        [] (TTextFeature&) { }
    );
}

TVector<TEmbeddingFeature> CreateEmbeddingFeatures(const NCB::TFeaturesLayout& featuresLayout) {
    return CreateFeatures<TEmbeddingFeature, EFeatureType::Embedding>(
        featuresLayout,
        [] (TEmbeddingFeature&) {
            //Dimension!!!
        }
    );
}

void ConfigureMalloc() {
#if !(defined(__APPLE__) && defined(__MACH__)) && !defined(__aarch64__) // there is no LF for MacOS and aarch64
    NMalloc::MallocInfo().SetParam("LB_LIMIT_TOTAL_SIZE", "1000000");
#endif
}


double CalcMetric(
    const IMetric& metric,
    const TTargetDataProviderPtr& targetData,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE(
        approx[0].size() == targetData->GetObjectCount(),
        "Approx size and object count must be equal"
    );
    const auto target = targetData->GetTarget().GetOrElse(TConstArrayRef<TConstArrayRef<float>>());
    const auto weights = GetWeights(*targetData);
    const auto queryInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());
    const auto& additiveStats = EvalErrors(
        To2DConstArrayRef<double>(approx),
        /*approxDelta*/{},
        /*isExpApprox*/false,
        target,
        weights,
        queryInfo,
        metric,
        localExecutor
    );
    return metric.GetFinalError(additiveStats);
}

TVector<const IMetric*> FilterTrainMetrics(
    const TVector<THolder<IMetric>>& metrics,
    bool calcAdditiveMetrics,
    bool calcNonAdditiveMetrics
) {
    TVector<bool> skipMetricOnTrain = GetSkipMetricOnTrain(metrics);
    TVector<const IMetric*> filtered;
    for (auto i : xrange(metrics.size())) {
        auto metric = metrics[i].Get();
        auto isAdditive = metric->IsAdditiveMetric();
        if (((isAdditive && calcAdditiveMetrics) || (!isAdditive && calcNonAdditiveMetrics)) &&
            !skipMetricOnTrain[i])
        {
            filtered.push_back(metric);
        }
    }
    return filtered;
}

TVector<const IMetric*> FilterTestMetrics(
    const TVector<THolder<IMetric>>& metrics,
    bool calcAllMetrics,
    bool calcAdditiveMetrics,
    bool calcNonAdditiveMetrics,
    bool hasTarget,
    TMaybe<int> trackerIdx,
    TMaybe<int>* filteredTrackerIdx
) {
    *filteredTrackerIdx = Nothing();
    TVector<const IMetric*> filtered;
    for (int i : xrange(metrics.size())) {
        auto metric = metrics[i].Get();
        auto isAdditive = metric->IsAdditiveMetric();

        const bool skipMetric = (!calcAllMetrics && (!trackerIdx || i != *trackerIdx))
            || (!hasTarget && metric->NeedTarget())
            || (isAdditive && !calcAdditiveMetrics) || (!isAdditive && !calcNonAdditiveMetrics);

        if (!skipMetric) {
            if (trackerIdx && i == trackerIdx) {
                *filteredTrackerIdx = filtered.size();
            }
            filtered.push_back(metric);
        }
    }
    return filtered;
}

TVector<int> FilterTestPools(const TTrainingDataProviders& trainingDataProviders, bool calcAllMetrics) {
    TVector<int> filtered;
    for (int i : xrange(trainingDataProviders.Test.size())) {
        const auto &testPool = trainingDataProviders.Test[i];
        bool skipPool = testPool == nullptr || testPool->GetObjectCount() == 0
            || !calcAllMetrics && i != SafeIntegerCast<int>(trainingDataProviders.Test.size() - 1);

        if (!skipPool) {
            filtered.push_back(i);
        }
    }
    return filtered;
}

void CalcErrorsLocally(
    const TTrainingDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics,
    bool calcErrorTrackerMetric,
    bool calcNonAdditiveMetricsOnly,
    TLearnContext* ctx
) {
    auto onLearn = [&] (TConstArrayRef<const IMetric*> trainMetrics) {
        const auto& targetData = trainingDataProviders.Learn->TargetData;

        auto weights = GetWeights(*targetData);
        auto queryInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());

        TVector<TVector<double>>* approx = &ctx->LearnProgress->AvrgApprox;
        bool isExpApprox = false;
        if (UseAveragingFoldAsFoldZero(*ctx) && trainMetrics.size() == 1) {
            const auto lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();
            isExpApprox = IsStoreExpApprox(lossFunction);
            approx = &ctx->LearnProgress->AveragingFold.BodyTailArr[0].Approx;
        }

        auto errors = EvalErrorsWithCaching(
            *approx,
            /*approxDelta*/{},
            isExpApprox,
            targetData->GetTarget().GetOrElse(TConstArrayRef<TConstArrayRef<float>>()),
            weights,
            queryInfo,
            trainMetrics,
            ctx->LocalExecutor
        );

        for (auto i : xrange(trainMetrics.size())) {
            auto metric = trainMetrics[i];
            ctx->LearnProgress->MetricsAndTimeHistory.AddLearnError(
                *metric,
                metric->GetFinalError(errors[i])
            );
        }
    };
    auto onTest = [&] (size_t testIdx,
        TConstArrayRef<const IMetric*> testMetrics,
        TMaybe<int> filteredTrackerIdx
    ) {
        const auto &targetData = trainingDataProviders.Test[testIdx]->TargetData;

        auto maybeTarget = targetData->GetTarget();
        auto weights = GetWeights(*targetData);
        auto queryInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());

        auto errors = EvalErrorsWithCaching(
            ctx->LearnProgress->TestApprox[testIdx],
            /*approxDelta*/{},
            /*isExpApprox*/false,
            maybeTarget.GetOrElse(TConstArrayRef<TConstArrayRef<float>>()),
            weights,
            queryInfo,
            testMetrics,
            ctx->LocalExecutor
        );

        for (int i : xrange(testMetrics.size())) {
            auto metric = testMetrics[i];
            const bool updateBestIteration = filteredTrackerIdx && (i == *filteredTrackerIdx)
                && (testIdx == (trainingDataProviders.Test.size() - 1));

            ctx->LearnProgress->MetricsAndTimeHistory.AddTestError(
                testIdx,
                *metric,
                metric->GetFinalError(errors[i]),
                updateBestIteration
            );
        }
    };

    IterateOverMetrics(
        trainingDataProviders,
        errors,
        calcAllMetrics,
        calcErrorTrackerMetric,
        /*calcAdditiveMetrics*/ !calcNonAdditiveMetricsOnly,
        /*calcNonAdditiveMetrics*/ true,
        onLearn,
        onTest
    );
}

void IterateOverMetrics(
    const NCB::TTrainingDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics, // bool value for each error
    bool calcErrorTrackerMetric,
    bool calcAdditiveMetrics,
    bool calcNonAdditiveMetrics,
    std::function<void(TConstArrayRef<const IMetric*> /*metrics*/)> onLearnCallback,
    std::function<
        void(size_t /*testIdx*/, TConstArrayRef<const IMetric*> /*metrics*/, TMaybe<int> /*filteredTrackerIdx*/)
    > onTestCallback
) {
    if (trainingDataProviders.Learn->GetObjectCount() > 0) {
        if (calcAllMetrics) {
            onLearnCallback(FilterTrainMetrics(errors, calcAdditiveMetrics, calcNonAdditiveMetrics));
        }
    }
    if (trainingDataProviders.GetTestSampleCount() > 0) {
        for (auto testIdx : FilterTestPools(trainingDataProviders, calcAllMetrics)) {
            const auto &targetData = trainingDataProviders.Test[testIdx]->TargetData;

            TMaybe<int> trackerIdx = calcErrorTrackerMetric ? TMaybe<int>(0) : Nothing();
            TMaybe<int> filteredTrackerIdx;
            auto testMetrics = FilterTestMetrics(
                errors,
                calcAllMetrics,
                calcAdditiveMetrics,
                calcNonAdditiveMetrics,
                targetData->GetTarget().Defined(),
                trackerIdx,
                &filteredTrackerIdx);

            onTestCallback(testIdx, testMetrics, filteredTrackerIdx);
        }
    }
}
