#include "helpers.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/logging/logging.h>

#include <library/malloc/api/malloc.h>

#include <functional>


using namespace NCB;


template <class TFeature, EFeatureType FeatureType>
static TVector<TFeature> CreateFeatures(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    std::function<void(TFeature&)> setSpecificDataFunc
) {
    const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

    TVector<TFeature> features;

    const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

    for (auto flatFeatureIdx : xrange(featuresMetaInfo.size())) {
        const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
        if (featureMetaInfo.Type != FeatureType) {
            continue;
        }

        TFeature feature;
        feature.FeatureIndex = (int)*featuresLayout.GetInternalFeatureIdx<FeatureType>(flatFeatureIdx);
        feature.FlatFeatureIndex = flatFeatureIdx;
        feature.FeatureId = featureMetaInfo.Name;

        if (featureMetaInfo.IsAvailable) {
            setSpecificDataFunc(feature);
        }

        features.push_back(std::move(feature));
    }

    return features;
}


TVector<TFloatFeature> CreateFloatFeatures(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
    return CreateFeatures<TFloatFeature, EFeatureType::Float>(
        quantizedFeaturesInfo,
        [&] (TFloatFeature& floatFeature) {
            const auto floatFeatureIdx = TFloatFeatureIdx((ui32)floatFeature.FeatureIndex);
            auto nanMode = quantizedFeaturesInfo.GetNanMode(floatFeatureIdx);
            if (nanMode == ENanMode::Min) {
                floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsFalse;
                floatFeature.HasNans = true;
            } else if (nanMode == ENanMode::Max) {
                floatFeature.NanValueTreatment = NCatBoostFbs::ENanValueTreatment_AsTrue;
                floatFeature.HasNans = true;
            }

            floatFeature.Borders = quantizedFeaturesInfo.GetBorders(floatFeatureIdx);
        }
    );
}

TVector<TCatFeature> CreateCatFeatures(const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
    return CreateFeatures<TCatFeature, EFeatureType::Categorical>(
        quantizedFeaturesInfo,
        [&] (TCatFeature&) { }
    );
}


void ConfigureMalloc() {
#if !(defined(__APPLE__) && defined(__MACH__)) // there is no LF for MacOS
    if (!NMalloc::MallocInfo().SetParam("LB_LIMIT_TOTAL_SIZE", "1000000")) {
        CATBOOST_DEBUG_LOG << "link with lfalloc for better performance" << Endl;
    }
#endif
}

void CalcErrors(
    const TTrainingForCPUDataProviders& trainingDataProviders,
    const TVector<THolder<IMetric>>& errors,
    bool calcAllMetrics,
    bool calcErrorTrackerMetric,
    TLearnContext* ctx
) {
    if (trainingDataProviders.Learn->GetObjectCount() > 0) {
        ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory.emplace_back();
        if (calcAllMetrics) {
            if (ctx->Params.SystemOptions->IsSingleHost()) {
                const auto& targetData = trainingDataProviders.Learn->TargetData;

                auto target = targetData->GetTarget().GetOrElse(TConstArrayRef<float>());
                auto weights = GetWeights(*targetData);
                auto queryInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());

                TVector<bool> skipMetricOnTrain = GetSkipMetricOnTrain(errors);
                for (int i = 0; i < errors.ysize(); ++i) {
                    if (!skipMetricOnTrain[i]) {
                        const auto& additiveStats = EvalErrors(
                            ctx->LearnProgress.AvrgApprox,
                            target,
                            weights,
                            queryInfo,
                            errors[i],
                            ctx->LocalExecutor
                        );
                        ctx->LearnProgress.MetricsAndTimeHistory.AddLearnError(*errors[i].Get(), errors[i]->GetFinalError(additiveStats));
                    }
                }
            } else {
                MapCalcErrors(ctx);
            }
        }
    }

    const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;

    if (trainingDataProviders.GetTestSampleCount() > 0) {
        ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.emplace_back(); // new [iter]
        for (size_t testIdx = 0; testIdx < trainingDataProviders.Test.size(); ++testIdx) {
            const auto& testDataPtr = trainingDataProviders.Test[testIdx];

            if (testDataPtr == nullptr || testDataPtr->GetObjectCount() == 0) {
                continue;
            }
            // Use only last testset for eval metric
            if (!calcAllMetrics && testIdx != trainingDataProviders.Test.size() - 1) {
                continue;
            }
            const auto& targetData = testDataPtr->TargetData;

            auto maybeTarget = targetData->GetTarget();
            auto target = maybeTarget.GetOrElse(TConstArrayRef<float>());
            auto weights = GetWeights(*targetData);
            auto queryInfo = targetData->GetGroupInfo().GetOrElse(TConstArrayRef<TQueryInfo>());;

            const auto& testApprox = ctx->LearnProgress.TestApprox[testIdx];
            for (int i = 0; i < errors.ysize(); ++i) {
                if (!calcAllMetrics && (i != errorTrackerMetricIdx)) {
                    continue;
                }
                if (!maybeTarget && errors[i]->NeedTarget()) {
                    continue;
                }

                const auto& additiveStats = EvalErrors(
                    testApprox,
                    target,
                    weights,
                    queryInfo,
                    errors[i],
                    ctx->LocalExecutor
                );
                bool updateBestIteration = (i == 0) && (testIdx == trainingDataProviders.Test.size() - 1);
                ctx->LearnProgress.MetricsAndTimeHistory.AddTestError(testIdx,
                                                                      *errors[i].Get(),
                                                                      errors[i]->GetFinalError(additiveStats),
                                                                      updateBestIteration);
            }
        }
    }
}
