#include "eval_feature.h"

#include "dir_helper.h"
#include "train_model.h"
#include "options_helper.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo/train.h>
#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/helpers/wx_test.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/features.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/feature_eval_options.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/private/libs/options/path_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/generic/maybe.h>
#include <util/generic/utility.h>
#include <util/stream/labeled.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/string/builder.h>
#include <util/system/hp_timer.h>
#include <util/ysaveload.h>

#include <cmath>
#include <numeric>


using namespace NCB;


TString ToString(const TFeatureEvaluationSummary& summary) {
    TString featureEvalTsv;

    TStringOutput featureEvalStream(featureEvalTsv);
    featureEvalStream << "p-value\tbest iteration in each fold\t";
    for (const auto& metricName : summary.MetricNames) {
        featureEvalStream << metricName << '\t';
    }
    featureEvalStream << "feature set" << Endl;
    for (ui32 featureSetIdx : xrange(summary.GetFeatureSetCount())) {
        featureEvalStream << summary.WxTest[featureSetIdx] << '\t';
        const auto& bestIterations = summary.BestBaselineIterations[featureSetIdx];
        featureEvalStream << JoinRange(",", bestIterations.begin(), bestIterations.end());
        featureEvalStream << '\t';
        for (double delta : summary.AverageMetricDelta[featureSetIdx]) {
            featureEvalStream << delta << '\t';
        }
        if (!summary.FeatureSets.empty()) {
            const auto& featureSet = summary.FeatureSets[featureSetIdx];
            featureEvalStream << JoinRange(",", featureSet.begin(), featureSet.end());
        }
        featureEvalStream << Endl;
    }
    return featureEvalTsv;
}


static TVector<EMetricBestValue> GetBestValueType(
    const TVector<THolder<IMetric>>& metrics
) {
    TVector<EMetricBestValue> bestValueType;
    for (const auto& metric : metrics) {
        EMetricBestValue valueType;
        float bestValue;
        metric->GetBestValue(&valueType, &bestValue);
        CB_ENSURE(
            EqualToOneOf(valueType, EMetricBestValue::Min, EMetricBestValue::Max),
            "Metric " + metric->GetDescription() + " has neither lower, nor upper bound"
        );
        bestValueType.push_back(valueType);
    }
    return bestValueType;
}

static ui32 GetBestIterationInFold(
    const TVector<EMetricBestValue>& bestValueType,
    const TVector<TVector<double>>& metricValues // [iterIdx][metricIdx]
) {
    ui32 bestIteration = 0;
    constexpr ui32 lossIdx = 0;
    for (ui32 iteration = 1; iteration < metricValues.size(); ++iteration) {
        if (bestValueType[lossIdx] == EMetricBestValue::Min) {
            if (metricValues[iteration][lossIdx] < metricValues[bestIteration][lossIdx]) {
                bestIteration = iteration;
            }
        } else {
            if (metricValues[iteration][lossIdx] > metricValues[bestIteration][lossIdx]) {
                bestIteration = iteration;
            }
        }
    }
    return bestIteration;
}


size_t TFeatureEvaluationSummary::GetFeatureSetCount() const {
    return Max<size_t>(1, FeatureSets.size());
}


void TFeatureEvaluationSummary::AppendFeatureSetMetrics(
    bool isTest,
    ui32 featureSetIdx,
    const TVector<TVector<double>>& metricValuesOnFold
) {
    const auto featureSetCount = GetFeatureSetCount();
    CB_ENSURE_INTERNAL(featureSetIdx < featureSetCount, "Feature set index is too large");
    const ui32 bestIteration = GetBestIterationInFold(MetricTypes, metricValuesOnFold);
    if (!isTest) {
        BestBaselineIterations[featureSetIdx].push_back(bestIteration);
    }
    auto& bestMetrics = BestMetrics[isTest];
    auto& featureSetBestMetrics = bestMetrics[featureSetIdx];
    const auto metricCount = MetricTypes.size();
    featureSetBestMetrics.resize(metricCount);
    for (auto metricIdx : xrange(metricCount)) {
        const double bestMetric = metricValuesOnFold[bestIteration][metricIdx];
        featureSetBestMetrics[metricIdx].push_back(bestMetric);
    }
}


NJson::TJsonValue TFeatureEvaluationSummary::CalcProcessorsSummary() const {
    const auto& history = ProcessorsUsage;
    CB_ENSURE_INTERNAL(!history.empty(), "Need some processors usage info");
    THashMap<TString, float> totalTime; // [processor name]
    THashMap<TString, ui32> totalIterations; // [processor name]
    ui32 previousIteration = history[0].Iteration;
    for (const auto& event : history) {
        const auto time = event.Time;
        const auto iteration = event.Iteration;
        const auto& processors = event.Processors;
        CB_ENSURE_INTERNAL(processors.IsArray(), "Processors should be json array");
        const auto& processorsArray = processors.GetArray();
        for (auto& processor : processorsArray) {
            const auto& processorString = processor.GetString();
            totalTime[processorString] += time;
            if (previousIteration >= iteration) {
                totalIterations[processorString] += iteration; // first snapshot in this training
            } else {
                totalIterations[processorString] += (iteration - previousIteration);
            }
        }
        previousIteration = iteration;
    }
    NJson::TJsonValue summary;
    auto& timeJson = summary.InsertValue("time", NJson::TJsonMap());
    auto& iterationsJson = summary.InsertValue("iterations", NJson::TJsonMap());
    for (const auto& [processor, time] : totalTime) {
        timeJson.InsertValue(processor, NJson::TJsonValue{time});
        iterationsJson.InsertValue(processor, NJson::TJsonValue{totalIterations.at(processor)});
    }
    return summary;
}


void TFeatureEvaluationSummary::CalcWxTestAndAverageDelta() {
    const auto featureSetCount = GetFeatureSetCount();
    const auto metricCount = MetricTypes.size();
    TVector<double> averageDelta(metricCount);
    WxTest.resize(featureSetCount);
    AverageMetricDelta.resize(featureSetCount);
    constexpr ui32 LossIdx = 0;
    for (auto featureSetIdx : xrange(featureSetCount)) {
        const auto& baselineMetrics = BestMetrics[/*isTest*/0][featureSetIdx];
        const auto& testedMetrics = FeatureSets.empty() ? baselineMetrics : BestMetrics[/*isTest*/1][featureSetIdx];
        WxTest[featureSetIdx] = ::WxTest(baselineMetrics[LossIdx], testedMetrics[LossIdx]).PValue;

        const auto foldCount = baselineMetrics.size();
        for (auto metricIdx : xrange(metricCount)) {
            const auto baselineAverage = Accumulate(baselineMetrics[metricIdx], 0.0) / foldCount;
            const auto testedAverage = Accumulate(testedMetrics[metricIdx], 0.0) / foldCount;
            if (MetricTypes[metricIdx] == EMetricBestValue::Min) {
                averageDelta[metricIdx] = - testedAverage + baselineAverage;
            } else {
                averageDelta[metricIdx] = + testedAverage - baselineAverage;
            }
        }
        AverageMetricDelta[featureSetIdx] = averageDelta;
    }
}


static void CreateLogFromHistory(
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const TVector<THolder<IMetric>>& metrics,
    const TFeatureEvaluationSummary::TMetricsHistory& metricsHistory,
    ui32 iterationCount,
    TLogger* logger
) {
    const TString testToken = "test";
    CB_ENSURE_INTERNAL(
        outputFileOptions.GetMetricPeriod() == 1,
        "Feature evaluation requires metric_period=1");
    constexpr int errorTrackerMetricIdx = 0;
    for (ui32 iteration = 0; iteration < iterationCount; ++iteration) {
        TOneInterationLogger oneIterLogger(*logger);
        for (int metricIdx = 0; metricIdx < metrics.ysize(); ++metricIdx) {
            const auto& metric = metrics[metricIdx];
            const auto& metricDescription = metric->GetDescription();
            const double metricOnTest = metricsHistory[iteration][metricIdx];
            oneIterLogger.OutputMetric(
                testToken,
                TMetricEvalResult(metricDescription, metricOnTest, metricIdx == errorTrackerMetricIdx)
            );
        }
    }
}

static TString MakeFoldDirName(
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    bool isTest,
    ui32 featureSetIdx,
    ui32 foldIdx
) {
    auto foldDir = TStringBuilder();
    if (!isTest) {
        foldDir << "Baseline_";
        const auto evalMode = featureEvalOptions.FeatureEvalMode;
        const auto featureSetCount = featureEvalOptions.FeaturesToEvaluate->size();
        if (featureSetCount > 0 && evalMode == NCB::EFeatureEvalMode::OneVsOthers) {
            foldDir << "set_" << featureSetIdx << "_";
        }
    } else {
        foldDir << "Testing_set_" << featureSetIdx << "_";
    }
    foldDir << "fold_" << foldIdx;
    return foldDir;
}

void TFeatureEvaluationSummary::CreateLogs(
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TVector<THolder<IMetric>>& metrics,
    ui32 iterationCount,
    bool isTest,
    ui32 foldRangeBegin,
    ui32 absoluteOffset
) {
    if (!outputFileOptions.AllowWriteFiles()) {
        return;
    }

    const ui32 featureSetCount = GetFeatureSetCount();
    const auto topLevelTrainDir = outputFileOptions.GetTrainDir();
    const auto& metricsHistory = MetricsHistory[isTest];
    const auto& featureStrengths = FeatureStrengths[isTest];
    const auto& regularFeatureStrengths = RegularFeatureStrengths[isTest];
    const auto& models = Models[isTest];
    const auto& metricsMetaJson = GetJsonMeta(
        iterationCount,
        outputFileOptions.GetName(),
        GetConstPointers(metrics),
        /*learnSetNames*/{"learn"},
        /*testSetNames*/{"test"},
        /*parametersName=*/ "",
        ELaunchMode::CV);
    const ui32 absoluteBegin = foldRangeBegin + featureEvalOptions.Offset;
    const ui32 absoluteEnd = absoluteBegin + featureEvalOptions.FoldCount;
    const bool useSetZeroAlways = !isTest && featureEvalOptions.FeatureEvalMode != NCB::EFeatureEvalMode::OneVsOthers;
    for (ui32 setIdx : xrange(featureSetCount)) {
        for (ui32 absoluteFoldIdx : xrange(absoluteBegin, absoluteEnd)) {
            const auto foldDir = MakeFoldDirName(
                featureEvalOptions,
                isTest,
                setIdx,
                absoluteFoldIdx);
            auto options = outputFileOptions;
            options.SetTrainDir(JoinFsPaths(topLevelTrainDir, foldDir));
            TLogger logger;
            InitializeFileLoggers(
                options,
                metricsMetaJson,
                /*namesPrefix*/"",
                /*isDetailedProfile*/false,
                &logger);
            CreateLogFromHistory(
                options,
                metrics,
                metricsHistory[useSetZeroAlways ? 0 : setIdx][absoluteFoldIdx - absoluteOffset],
                iterationCount,
                &logger
            );
            const auto fstrPath = options.CreateFstrIternalFullPath();
            if (!fstrPath.empty()) {
                OutputStrengthDescriptions(
                    featureStrengths[useSetZeroAlways ? 0 : setIdx][absoluteFoldIdx - absoluteOffset],
                    fstrPath);
            }
            const auto regularFstrPath = options.CreateFstrRegularFullPath();
            if (!regularFstrPath.empty()) {
                OutputStrengthDescriptions(
                    regularFeatureStrengths[useSetZeroAlways ? 0 : setIdx][absoluteFoldIdx - absoluteOffset],
                    regularFstrPath);
            }
            const auto outputModelPath = options.CreateResultModelFullPath();
            if (!outputModelPath.empty()) {
                TFileOutput modelFile(outputModelPath);
                models[useSetZeroAlways ? 0 : setIdx][absoluteFoldIdx - absoluteOffset].Save(&modelFile);
            }
        }
    }
}


bool TFeatureEvaluationSummary::HasHeaderInfo() const {
    return !MetricNames.empty();
}


void TFeatureEvaluationSummary::SetHeaderInfo(
    const TVector<THolder<IMetric>>& metrics,
    const TVector<TVector<ui32>>& featureSets
) {
    MetricTypes = GetBestValueType(metrics);
    MetricNames.clear();
    for (const auto& metric : metrics) {
        MetricNames.push_back(metric->GetDescription());
    }
    FeatureSets = featureSets;
    const ui32 featureSetCount = GetFeatureSetCount();
    ResizeRank2(2, featureSetCount, MetricsHistory);
    ResizeRank2(2, featureSetCount, FeatureStrengths);
    ResizeRank2(2, featureSetCount, RegularFeatureStrengths);
    ResizeRank2(2, featureSetCount, Models);
    ResizeRank2(2, featureSetCount, BestMetrics);
    BestBaselineIterations.resize(featureSetCount);
}


static bool IsObjectwiseEval(const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions) {
    return featureEvalOptions.FoldSizeUnit.Get() == ESamplingUnit::Object;
}

static ui64 FindQuantileTimestamp(TConstArrayRef<TGroupId> groupIds, TConstArrayRef<ui64> timestamps, double quantile) {
    TVector<ui64> groupTimestamps;
    groupTimestamps.reserve(groupIds.size());

    TGroupId lastGroupId = groupIds[0];
    groupTimestamps.push_back(timestamps[0]);
    for (auto idx : xrange((size_t)1, groupIds.size())) {
        if (groupIds[idx] != lastGroupId) {
            lastGroupId = groupIds[idx];
            groupTimestamps.push_back(timestamps[idx]);
        }
    }
    std::sort(groupTimestamps.begin(), groupTimestamps.end());
    const auto quantileTimestamp = groupTimestamps[groupTimestamps.size() * quantile];
    CATBOOST_INFO_LOG << "Quantile timestamp " << quantileTimestamp << Endl;
    return quantileTimestamp;
}

static void CreateFoldData(
    typename TTrainingDataProviders::TDataPtr srcData,
    ui64 cpuUsedRamLimit,
    const TVector<NCB::TArraySubsetIndexing<ui32>>& trainSubsets,
    const TVector<NCB::TArraySubsetIndexing<ui32>>& testSubsets,
    TVector<TTrainingDataProviders>* foldsData,
    TVector<TTrainingDataProviders>* testFoldsData,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE_INTERNAL(trainSubsets.size() == testSubsets.size(), "Number of train and test subsets do not match");
    const NCB::EObjectsOrder objectsOrder = NCB::EObjectsOrder::Ordered;
    const ui64 perTaskCpuUsedRamLimit = cpuUsedRamLimit / (2 * trainSubsets.size());

    TVector<std::function<void()>> tasks;
    for (ui32 foldIdx : xrange(trainSubsets.size())) {
        tasks.emplace_back(
            [&, foldIdx]() {
                (*foldsData)[foldIdx].Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        NCB::TArraySubsetIndexing<ui32>(trainSubsets[foldIdx]),
                        objectsOrder
                    ),
                    perTaskCpuUsedRamLimit,
                    localExecutor
                );
            }
        );
        tasks.emplace_back(
            [&, foldIdx]() {
                (*testFoldsData)[foldIdx].Test.emplace_back(
                    srcData->GetSubset(
                        GetSubset(
                            srcData->ObjectsGrouping,
                            NCB::TArraySubsetIndexing<ui32>(testSubsets[foldIdx]),
                            objectsOrder
                        ),
                        perTaskCpuUsedRamLimit,
                        localExecutor
                    )
                );
            }
        );
    }

    NCB::ExecuteTasksInParallel(&tasks, localExecutor);
}

static void TakeMiddleElements(
    ui32 offset,
    ui32 count,
    TVector<NCB::TArraySubsetIndexing<ui32>>* subsets
) {
    CB_ENSURE_INTERNAL(offset + count <= subsets->size(), "Dataset permutation logic failed");
    TVector<NCB::TArraySubsetIndexing<ui32>>(subsets->begin() + offset, subsets->end()).swap(*subsets);
    subsets->resize(count);
}

static void PrepareTimeSplitFolds(
    typename TTrainingDataProviders::TDataPtr srcData,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    ui64 cpuUsedRamLimit,
    TVector<TTrainingDataProviders>* foldsData,
    TVector<TTrainingDataProviders>* testFoldsData,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE(srcData->ObjectsData->GetGroupIds(), "Timesplit feature evaluation requires dataset with groups");
    CB_ENSURE(srcData->ObjectsData->GetTimestamp(), "Timesplit feature evaluation requires dataset with timestamps");

    const ui32 foldSize = featureEvalOptions.FoldSize;
    CB_ENSURE(foldSize > 0, "Fold size must be positive integer");
    // group subsets, groups maybe trivial
    const auto& objectsGrouping = *srcData->ObjectsGrouping;

    const auto timesplitQuantileTimestamp = FindQuantileTimestamp(
        *srcData->ObjectsData->GetGroupIds(),
        *srcData->ObjectsData->GetTimestamp(),
        featureEvalOptions.TimeSplitQuantile);
    TVector<NCB::TArraySubsetIndexing<ui32>> trainTestSubsets; // [0, offset + foldCount) -- train, [offset + foldCount] -- test
    if (IsObjectwiseEval(featureEvalOptions)) {
        trainTestSubsets = NCB::QuantileSplitByObjects(
            objectsGrouping,
            *srcData->ObjectsData->GetTimestamp(),
            timesplitQuantileTimestamp,
            foldSize);
    } else {
        trainTestSubsets = NCB::QuantileSplitByGroups(
            objectsGrouping,
            *srcData->ObjectsData->GetTimestamp(),
            timesplitQuantileTimestamp,
            foldSize);
    }
    const ui32 offsetInRange = featureEvalOptions.Offset;
    const ui32 trainSubsetsCount = trainTestSubsets.size() - 1;
    const ui32 foldCount = featureEvalOptions.FoldCount;
    CB_ENSURE_INTERNAL(offsetInRange + foldCount <= trainSubsetsCount, "Dataset permutation logic failed");

    CB_ENSURE(foldsData->empty(), "Need empty vector of folds data");
    foldsData->resize(foldCount);
    if (testFoldsData != nullptr) {
        CB_ENSURE(testFoldsData->empty(), "Need empty vector of test folds data");
        testFoldsData->resize(foldCount);
    } else {
        testFoldsData = foldsData;
    }

    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets(trainTestSubsets.begin(), trainTestSubsets.begin() + trainSubsetsCount);
    TakeMiddleElements(offsetInRange, foldCount, &trainSubsets);

    TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets(foldCount, trainTestSubsets.back());

    CreateFoldData(
        srcData,
        cpuUsedRamLimit,
        trainSubsets,
        testSubsets,
        foldsData,
        testFoldsData,
        localExecutor);
}

static void PrepareFolds(
    typename TTrainingDataProviders::TDataPtr srcData,
    const TCvDataPartitionParams& cvParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    ui64 cpuUsedRamLimit,
    TVector<TTrainingDataProviders>* foldsData,
    TVector<TTrainingDataProviders>* testFoldsData,
    NPar::ILocalExecutor* localExecutor
) {
    const int foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(foldCount > 0, "Fold count must be positive integer");
    const auto& objectsGrouping = *srcData->ObjectsGrouping;
    TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;
    if (cvParams.Initialized()) {
        // group subsets, groups maybe trivial
        testSubsets = NCB::Split(objectsGrouping, foldCount);
        // always inverted
        CB_ENSURE(cvParams.Type == ECrossValidation::Inverted, "Feature evaluation requires inverted cross-validation");
    } else {
        const ui32 foldSize = featureEvalOptions.FoldSize;
        CB_ENSURE(foldSize > 0, "Fold size must be positive integer");
        // group subsets, groups maybe trivial
        const auto isObjectwise = IsObjectwiseEval(featureEvalOptions);
        testSubsets = isObjectwise
            ? NCB::SplitByObjects(objectsGrouping, foldSize)
            : NCB::SplitByGroups(objectsGrouping, foldSize);
        const ui32 offsetInRange = featureEvalOptions.Offset;
        CB_ENSURE_INTERNAL(offsetInRange + foldCount <= testSubsets.size(), "Dataset permutation logic failed");
    }
    const ui32 offsetInRange = !cvParams.Initialized() ? featureEvalOptions.Offset : 0;

    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets
        = CalcTrainSubsetsRange(testSubsets, objectsGrouping.GetGroupCount(), TIndexRange<ui32>(offsetInRange, offsetInRange + foldCount));

    if (!cvParams.Initialized()) {
        TakeMiddleElements(offsetInRange, foldCount, &trainSubsets);
        TakeMiddleElements(offsetInRange, foldCount, &testSubsets);
    }
    testSubsets.swap(trainSubsets);

    CB_ENSURE(foldsData->empty(), "Need empty vector of folds data");
    foldsData->resize(foldCount);
    if (testFoldsData != nullptr) {
        CB_ENSURE(testFoldsData->empty(), "Need empty vector of test folds data");
        testFoldsData->resize(foldCount);
    } else {
        testFoldsData = foldsData;
    }

    CreateFoldData(
        srcData,
        cpuUsedRamLimit,
        trainSubsets,
        testSubsets,
        foldsData,
        testFoldsData,
        localExecutor);
}


enum class ETrainingKind {
    Baseline,
    Testing
};


static TVector<TTrainingDataProviders> UpdateIgnoredFeaturesInLearn(
    const NCatboostOptions::TFeatureEvalOptions& options,
    ETrainingKind trainingKind,
    ui32 testedFeatureSetIdx,
    const TVector<TTrainingDataProviders>& foldsData
) {
    TVector<ui32> ignoredFeatures;
    const auto& testedFeatures = options.FeaturesToEvaluate.Get();
    const auto featureEvalMode = options.FeatureEvalMode;
    if (trainingKind == ETrainingKind::Testing) {
        if (featureEvalMode == NCB::EFeatureEvalMode::OthersVsAll) {
            ignoredFeatures = testedFeatures[testedFeatureSetIdx];
        } else {
            THashSet<ui32> ignoredFeaturesAsSet;
            for (const auto& featureSet : testedFeatures) {
                ignoredFeaturesAsSet.insert(featureSet.begin(), featureSet.end());
            }
            for (ui32 featureIdx : testedFeatures[testedFeatureSetIdx]) {
                ignoredFeaturesAsSet.erase(featureIdx);
            }
            ignoredFeatures.insert(ignoredFeatures.end(), ignoredFeaturesAsSet.begin(), ignoredFeaturesAsSet.end());
        }
    } else if (EqualToOneOf(featureEvalMode, NCB::EFeatureEvalMode::OneVsAll, NCB::EFeatureEvalMode::OthersVsAll)) {
        // no additional ignored features
    } else if (featureEvalMode == NCB::EFeatureEvalMode::OneVsOthers) {
        ignoredFeatures = testedFeatures[testedFeatureSetIdx];
    } else {
        CB_ENSURE(
            featureEvalMode == NCB::EFeatureEvalMode::OneVsNone,
            "Unknown feature evaluation mode " + ToString(featureEvalMode)
        );
        for (const auto& featureSet : testedFeatures) {
            ignoredFeatures.insert(
                ignoredFeatures.end(),
                featureSet.begin(),
                featureSet.end());
        }
    }

    TStringBuilder logMessage;
    logMessage << "Feature set " << testedFeatureSetIdx;
    if (trainingKind == ETrainingKind::Baseline) {
        logMessage << ", baseline";
    } else {
        logMessage << ", testing";
    }
    if (ignoredFeatures.empty()) {
        logMessage << ", no additional ignored features";
    } else {
        std::sort(ignoredFeatures.begin(), ignoredFeatures.end());
        logMessage << ", additional ignored features " << JoinRange(":", ignoredFeatures.begin(), ignoredFeatures.end());
    }
    CATBOOST_INFO_LOG << logMessage << Endl;

    TVector<TTrainingDataProviders> result;
    result.reserve(foldsData.size());

    for (const auto& foldData : foldsData) {
        result.push_back(MakeFeatureSubsetTrainingData(ignoredFeatures, foldData));
    }
    return result;
}


static void LoadOptions(
    const NJson::TJsonValue& plainJsonParams,
    const NCB::TDataMetaInfo& metaInfo,
    NCatboostOptions::TCatBoostOptions* catBoostOptions,
    NCatboostOptions::TOutputFilesOptions* outputFileOptions
) {
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    ConvertParamsToCanonicalFormat(metaInfo, &jsonParams);
    catBoostOptions->Load(jsonParams);
    outputFileOptions->Load(outputJsonParams);

    if (outputFileOptions->IsMetricPeriodSet() && outputFileOptions->GetMetricPeriod() > 1) {
        CATBOOST_WARNING_LOG << "Warning: metric_period is ignored because "
            "feature evaluation needs metric values on each iteration" << Endl;
    }
    outputFileOptions->SetMetricPeriod(1);
}


static void CalcMetricsForTest(
    const TVector<THolder<IMetric>>& metrics,
    ui32 approxDimension,
    TTrainingDataProviders::TTrainingDataProviderTemplatePtr testData,
    TFoldContext* foldContext
) {
    CB_ENSURE_INTERNAL(
        foldContext->FullModel.Defined(), "No model in fold " << foldContext->FoldIdx);
    const auto treeCount = foldContext->FullModel->GetTreeCount();
    const ui32 iterationCount = foldContext->MetricValuesOnTrain.size();
    CB_ENSURE_INTERNAL(
        iterationCount == treeCount,
        "Fold " << foldContext->FoldIdx << ": model size (" << treeCount <<
        ") differs from iteration count (" << iterationCount << ")");

    const auto metricCount = metrics.size();
    auto& metricValuesOnTest = foldContext->MetricValuesOnTest;
    ResizeRank2(treeCount, metricCount, metricValuesOnTest);

    const auto docCount = testData->GetObjectCount();
    TVector<TVector<double>> approx;
    ResizeRank2(approxDimension, docCount, approx);
    TVector<TVector<double>> partialApprox;
    ResizeRank2(approxDimension, docCount, partialApprox);
    TVector<double> flatApproxBuffer;
    flatApproxBuffer.yresize(docCount * approxDimension);

    TModelCalcerOnPool modelCalcer(
        foldContext->FullModel.GetRef(),
        testData->ObjectsData,
        &NPar::LocalExecutor());

    const auto baseline = *testData->TargetData->GetBaseline();
    if (baseline){
        AssignRank2(baseline, &approx);
    }
    for (auto treeIdx : xrange(treeCount)) {
        // TODO(kirillovs):
        //     apply (1) all models to the entire dataset on CPU or (2) GPU,
        // TODO(espetrov):
        //     calculate error for each model,
        //     error on test fold idx = error on entire dataset for model idx - error on learn fold idx
        //     refactor using the Visitor pattern
        modelCalcer.ApplyModelMulti(
            EPredictionType::RawFormulaVal,
            treeIdx,
            treeIdx + 1,
            &flatApproxBuffer,
            &partialApprox);
        for (auto dimensionIdx : xrange(approxDimension)) {
            for (auto docIdx : xrange(docCount)) {
                approx[dimensionIdx][docIdx] += partialApprox[dimensionIdx][docIdx];
            }
        }
        for (auto metricIdx : xrange(metricCount)) {
            metricValuesOnTest[treeIdx][metricIdx] = CalcMetric(
                *metrics[metricIdx],
                testData->TargetData,
                approx,
                &NPar::LocalExecutor()
            );
        }
    }
}


class TFeatureEvaluationCallbacks : public ITrainingCallbacks {
public:
    TFeatureEvaluationCallbacks(
        ui32 iterationCount,
        const NCatboostOptions::TFeatureEvalOptions& evalFeatureOptions,
        TFeatureEvaluationSummary* summary)
    : IterationCount(iterationCount)
    , EvalFeatureOptions(evalFeatureOptions)
    , Summary(summary)
    {
    }

    bool IsContinueTraining(const TMetricsAndTimeLeftHistory& /*history*/) override {
        ++IterationIdx;
        constexpr double HeartbeatSeconds = 600;
        if (TrainTimer.Passed() > HeartbeatSeconds) {
            TSetLogging infomationMode(ELoggingLevel::Info);
            CATBOOST_NOTICE_LOG << "Status after (another) " << HeartbeatSeconds << " seconds: iteration " << IterationIdx << " of " << IterationCount << Endl;
            TrainTimer.Reset();
        }
        return /*continue training*/true;
    }

    void OnSaveSnapshot(const NJson::TJsonValue& processors, IOutputStream* snapshot) override {
        if (processors.IsArray()) {
            Summary->ProcessorsUsage.push_back({float(SnapshotTimer.PassedReset()), IterationIdx, processors});
        }
        Summary->Save(snapshot);
        NJson::TJsonValue options;
        EvalFeatureOptions.Save(&options);
        ::SaveMany(snapshot, FoldRangeBegin, FeatureSetIndex, IsTest, FoldIndex, options);
    }

    bool OnLoadSnapshot(IInputStream* snapshot) override {
        if (!IsNextLoadValid) {
            return false;
        }
        Summary->Load(snapshot);
        NJson::TJsonValue options;
        ::LoadMany(snapshot, FoldRangeBegin, FeatureSetIndex, IsTest, FoldIndex, options);
        NCatboostOptions::TFeatureEvalOptions evalFeatureOptions;
        evalFeatureOptions.Load(options);
        CB_ENSURE(evalFeatureOptions == EvalFeatureOptions, "Current feaure evaluation options differ from options in snapshot");
        EvalFeatureOptions = evalFeatureOptions;
        IsNextLoadValid = false;
        return true;
    }

    void ResetIterationIndex() {
        IterationIdx = 0;
    }

    void LoadSnapshot(ETaskType taskType, const TString& snapshotFile) {
        TProgressHelper progressHelper(ToString(taskType));
        IsNextLoadValid = true;
        progressHelper.CheckedLoad(
            snapshotFile,
            [&](TIFStream* input) {
                OnLoadSnapshot(input);
            });
        IsNextLoadValid = true;
    }

    bool HaveEvalFeatureSummary(ui32 foldRangeBegin, ui32 featureSetIdx, bool isTest, ui32 foldIdx) {
        if (!IsNextLoadValid) {
            return false;
        }
        CB_ENSURE_INTERNAL(
            FoldRangeBegin.Defined() && FeatureSetIndex.Defined() && IsTest.Defined() && FoldIndex.Defined(),
            "No fold range begin, or feature set index, or baseline flag, or fold index in snapshot");
        const std::array<ui32, 4> progress = { foldRangeBegin, featureSetIdx, isTest, foldIdx };
        const std::array<ui32, 4> progressFromSnapshot = { *FoldRangeBegin, *FeatureSetIndex, *IsTest, *FoldIndex };
        return progress < progressFromSnapshot;
    }

    ui32 GetAbsoluteOffset() const {
        return EvalFeatureOptions.Offset;
    }

    TMaybe<ui32> FoldRangeBegin;
    TMaybe<ui32> FeatureSetIndex;
    TMaybe<bool> IsTest;
    TMaybe<ui32> FoldIndex;

private:
    THPTimer TrainTimer;
    THPTimer SnapshotTimer;
    ui32 IterationIdx = 0;
    const ui32 IterationCount;
    NCatboostOptions::TFeatureEvalOptions EvalFeatureOptions;
    TFeatureEvaluationSummary* const Summary;
    bool IsNextLoadValid = false;
};

static bool HaveFeaturesToEvaluate(const TVector<TTrainingDataProviders>& foldsData) {
    for (const auto& foldData : foldsData) {
        if (!foldData.Learn->MetaInfo.FeaturesLayout->HasAvailableAndNotIgnoredFeatures()) {
            return false;
        }
    }
    return true;
}

static ui32 GetTrainingCountPerFold(const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions) {
    const auto useCommonBaseline = featureEvalOptions.FeatureEvalMode != NCB::EFeatureEvalMode::OneVsOthers;
    const ui32 featureSetCount = featureEvalOptions.FeaturesToEvaluate->size();
    return useCommonBaseline ? featureSetCount + 1 : 2 * featureSetCount;
}


static ui32 GetTrainingCount(const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions) {
    const ui32 foldCount = featureEvalOptions.FoldCount;
    return foldCount * GetTrainingCountPerFold(featureEvalOptions);
}

static void EvaluateFeaturesImpl(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    ui32 foldRangeBegin,
    const TCvDataPartitionParams& cvParams,
    TDataProviderPtr data,
    ui32 processedFoldCount,
    TFeatureEvaluationCallbacks* callbacks,
    TFeatureEvaluationSummary* results
) {
    const ui32 foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(data->ObjectsData->GetObjectCount() > foldCount, "Pool is too small to be split into folds");
    CB_ENSURE(data->ObjectsData->GetObjectCount() > featureEvalOptions.FoldSize.Get(), "Pool is too small to be split into folds");

    const ui64 cpuUsedRamLimit
        = ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get());

    TRestorableFastRng64 rand(catBoostOptions.RandomSeed);

    if (cvParams.Shuffle) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, cpuUsedRamLimit, &NPar::LocalExecutor());
    }

    TLabelConverter labelConverter;
    TMaybe<float> targetBorder = catBoostOptions.DataProcessingOptions->TargetBorder;
    NCatboostOptions::TCatBoostOptions dataSpecificOptions(catBoostOptions);

    TString tmpDir;
    if (outputFileOptions.AllowWriteFiles()) {
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
    }

    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*dataCanBeEmpty*/ false,
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRam*/ outputFileOptions.AllowWriteFiles(),
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
        tmpDir,
        /*quantizedFeaturesInfo*/ nullptr,
        &dataSpecificOptions,
        &labelConverter,
        &targetBorder,
        &NPar::LocalExecutor(),
        &rand);

    CB_ENSURE(
        dynamic_cast<TQuantizedObjectsDataProvider*>(trainingData->ObjectsData.Get()),
        "Unable to quantize dataset (probably because it contains categorical features)"
    );

    UpdateYetiRankEvalMetric(trainingData->MetaInfo.TargetStats, Nothing(), &dataSpecificOptions);

    // If eval metric is not set, we assign it to objective metric
    InitializeEvalMetricIfNotSet(dataSpecificOptions.MetricOptions->ObjectiveMetric,
                                 &dataSpecificOptions.MetricOptions->EvalMetric);

    const auto overfittingDetectorOptions = dataSpecificOptions.BoostingOptions->OverfittingDetector;
    dataSpecificOptions.BoostingOptions->OverfittingDetector->OverfittingDetectorType = EOverfittingDetectorType::None;

    // internal training output shouldn't interfere with main stdout
    const auto loggingLevel = dataSpecificOptions.LoggingLevel;
    dataSpecificOptions.LoggingLevel = ELoggingLevel::Silent;

    const auto taskType = catBoostOptions.GetTaskType();
    THolder<IModelTrainer> modelTrainerHolder(TTrainerFactory::Construct(taskType));

    TSetLogging inThisScope(loggingLevel);

    TVector<TTrainingDataProviders> foldsData;
    TVector<TTrainingDataProviders> testFoldsData;
    constexpr bool isFixedMlTools3185 = false;
    if (!trainingData->MetaInfo.HasTimestamp) {
        PrepareFolds(
            trainingData,
            cvParams,
            featureEvalOptions,
            cpuUsedRamLimit,
            &foldsData,
            isFixedMlTools3185 ? &testFoldsData : nullptr,
            &NPar::LocalExecutor());
    } else {
        PrepareTimeSplitFolds(
            trainingData,
            featureEvalOptions,
            cpuUsedRamLimit,
            &foldsData,
            isFixedMlTools3185 ? &testFoldsData : nullptr,
            &NPar::LocalExecutor());
    }

    UpdatePermutationBlockSize(taskType, foldsData, &dataSpecificOptions);

    const ui32 approxDimension = GetApproxDimension(
        dataSpecificOptions,
        labelConverter,
        trainingData->TargetData->GetTargetDimension());
    const auto& metrics = CreateMetrics(
        dataSpecificOptions.MetricOptions,
        evalMetricDescriptor,
        approxDimension,
        trainingData->MetaInfo.HasWeights);
    CheckMetrics(metrics, dataSpecificOptions.LossFunctionDescription.Get().GetLossFunction());

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);

    if (!results->HasHeaderInfo()) {
        results->SetHeaderInfo(metrics, featureEvalOptions.FeaturesToEvaluate);
    }

    ui32 trainingIdx = processedFoldCount * GetTrainingCountPerFold(featureEvalOptions);

    const ui32 offsetInRange = cvParams.Initialized() ? 0 : featureEvalOptions.Offset.Get();
    const auto trainFullModels = [&] (
        bool isTest,
        ui32 featureSetIdx,
        TVector<TTrainingDataProviders>* foldsData) {

        const auto topLevelTrainDir = outputFileOptions.GetTrainDir();
        const bool isCalcFstr = !outputFileOptions.CreateFstrIternalFullPath().empty();
        const bool isCalcRegularFstr = !outputFileOptions.CreateFstrRegularFullPath().empty();
        for (auto foldIdx : xrange(foldCount)) {
            ++trainingIdx;
            CATBOOST_NOTICE_LOG << "Training model number " << trainingIdx << Endl;

            const bool haveSummary = callbacks->HaveEvalFeatureSummary(
                foldRangeBegin,
                featureSetIdx,
                isTest,
                offsetInRange + foldIdx);

            if (haveSummary) {
                continue;
            }

            THPTimer timer;

            TFoldContext foldContext(
                foldRangeBegin + offsetInRange + foldIdx,
                taskType,
                outputFileOptions,
                std::move((*foldsData)[foldIdx]),
                rand.GenRand(),
                /*hasFullModel*/true);
            const auto foldDir = MakeFoldDirName(featureEvalOptions, isTest, featureSetIdx, foldContext.FoldIdx);
            callbacks->FoldRangeBegin = foldRangeBegin;
            callbacks->FeatureSetIndex = featureSetIdx;
            callbacks->IsTest = isTest;
            callbacks->FoldIndex = offsetInRange + foldIdx;
            callbacks->ResetIterationIndex();
            foldContext.OutputOptions.SetSaveSnapshotFlag(outputFileOptions.SaveSnapshot());
            CATBOOST_NOTICE_LOG << "Learn dataset: " << foldContext.TrainingData.Learn->ObjectsGrouping->GetObjectCount() << " objects, "
                << foldContext.TrainingData.Learn->ObjectsGrouping->GetGroupCount() << " groups" << Endl;
            CATBOOST_NOTICE_LOG << "Test dataset: " << foldContext.TrainingData.Test[0]->ObjectsGrouping->GetObjectCount() << " objects, "
                << foldContext.TrainingData.Test[0]->ObjectsGrouping->GetGroupCount() << " groups" << Endl;
            Train(
                dataSpecificOptions,
                JoinFsPaths(topLevelTrainDir, foldDir),
                objectiveDescriptor,
                evalMetricDescriptor,
                labelConverter,
                metrics,
                /*isErrorTrackerActive*/false,
                callbacks,
                &foldContext,
                modelTrainerHolder.Get(),
                &NPar::LocalExecutor());

            if (testFoldsData) {
                CalcMetricsForTest(metrics, approxDimension, testFoldsData[foldIdx].Test[0], &foldContext);
            }

            results->MetricsHistory[isTest][featureSetIdx].emplace_back(foldContext.MetricValuesOnTest);
            results->AppendFeatureSetMetrics(isTest, featureSetIdx, foldContext.MetricValuesOnTest);
            results->Models[isTest][featureSetIdx].emplace_back(foldContext.FullModel.GetRef());

            CATBOOST_INFO_LOG << "Fold " << foldContext.FoldIdx << ": model built in " <<
                FloatToString(timer.Passed(), PREC_NDIGITS, 2) << " sec" << Endl;

            if (isCalcFstr || isCalcRegularFstr) {
                const auto& model = foldContext.FullModel.GetRef();
                const NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);
                const auto fstrType = outputFileOptions.GetFstrType();
                const auto effect = CalcFeatureEffect(model, /*dataset*/nullptr, fstrType, &NPar::LocalExecutor());
                results->FeatureStrengths[isTest][featureSetIdx].emplace_back(ExpandFeatureDescriptions(layout, effect));
                if (isCalcRegularFstr) {
                    const auto regularEffect = CalcRegularFeatureEffect(
                        effect,
                        model);
                    results->RegularFeatureStrengths[isTest][featureSetIdx].emplace_back(
                        ExpandFeatureDescriptions(layout, regularEffect));
                }
            }

            (*foldsData)[foldIdx] = std::move(foldContext.TrainingData);
        }
    };

    if (featureEvalOptions.FeaturesToEvaluate->empty()) {
        trainFullModels(/*isTest*/false, /*featureSetIdx*/0, &foldsData);
        results->CreateLogs(
            outputFileOptions,
            featureEvalOptions,
            metrics,
            catBoostOptions.BoostingOptions->IterationCount,
            /*isTest*/false,
            foldRangeBegin,
            callbacks->GetAbsoluteOffset());
        return;
    }
    const auto useCommonBaseline = featureEvalOptions.FeatureEvalMode != NCB::EFeatureEvalMode::OneVsOthers;
    for (ui32 featureSetIdx : xrange(featureEvalOptions.FeaturesToEvaluate->size())) {
        const auto haveBaseline = featureSetIdx > 0 && useCommonBaseline;
        if (!haveBaseline) {
            auto newFoldsData = UpdateIgnoredFeaturesInLearn(
                featureEvalOptions,
                ETrainingKind::Baseline,
                featureSetIdx,
                foldsData);
            CB_ENSURE(
                HaveFeaturesToEvaluate(newFoldsData),
                "All features in baseline for feature set " << featureSetIdx << " are ignored or constant");
            trainFullModels(/*isTest*/false, featureSetIdx, &newFoldsData);
        } else {
            results->BestMetrics[/*isTest*/0][featureSetIdx] = results->BestMetrics[/*isTest*/0][0];
            results->BestBaselineIterations[featureSetIdx] = results->BestBaselineIterations[0];
        }

        auto newFoldsData = UpdateIgnoredFeaturesInLearn(
            featureEvalOptions,
            ETrainingKind::Testing,
            featureSetIdx,
            foldsData);
        if (HaveFeaturesToEvaluate(newFoldsData)) {
            trainFullModels(/*isTest*/true, featureSetIdx, &newFoldsData);
        } else {
            CATBOOST_WARNING_LOG << "Feature set " << featureSetIdx
                << " consists of ignored or constant features; eval feature assumes baseline data = testing data for this feature set" << Endl;
            const auto baselineIdx = useCommonBaseline ? 0 : featureSetIdx;
            results->MetricsHistory[/*isTest*/1][featureSetIdx] = results->MetricsHistory[/*isTest*/0][baselineIdx];
            results->FeatureStrengths[/*isTest*/1][featureSetIdx] = results->FeatureStrengths[/*isTest*/0][baselineIdx];
            results->RegularFeatureStrengths[/*isTest*/1][featureSetIdx] = results->RegularFeatureStrengths[/*isTest*/0][baselineIdx];
            results->Models[/*isTest*/1][featureSetIdx] = results->Models[/*isTest*/0][baselineIdx];
            results->BestMetrics[/*isTest*/1][featureSetIdx] = results->BestMetrics[/*isTest*/0][baselineIdx];
        }
    }
    for (auto isTest : {false, true}) {
        results->CreateLogs(
            outputFileOptions,
            featureEvalOptions,
            metrics,
            catBoostOptions.BoostingOptions->IterationCount,
            isTest,
            foldRangeBegin,
            callbacks->GetAbsoluteOffset());
    }
}

static ui32 GetSamplingUnitCount(const NCB::TObjectsGrouping& objectsGrouping, bool isObjectwise) {
    return isObjectwise ? objectsGrouping.GetObjectCount() : objectsGrouping.GetGroupCount();
}

static void CountDisjointFolds(
    TDataProviderPtr data,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    ui32* absoluteFoldSize,
    ui32* disjointFoldCount
) {
    const auto isObjectwise = IsObjectwiseEval(featureEvalOptions);
    const auto& objectsGrouping = *data->ObjectsGrouping;

    ui32 samplingUnitsCount = 0;
    if (!data->MetaInfo.HasTimestamp) {
        samplingUnitsCount = GetSamplingUnitCount(objectsGrouping, isObjectwise);
    } else {
        const auto timestamps = *data->ObjectsData->GetTimestamp();
        CB_ENSURE(
            data->ObjectsData->GetGroupIds(),
            "Timestamps require group ids");
        const auto timesplitQuantileTimestamp = FindQuantileTimestamp(
            *data->ObjectsData->GetGroupIds(),
            timestamps,
            featureEvalOptions.TimeSplitQuantile);

        samplingUnitsCount = 0;
        for (ui32 groupIdx : xrange(objectsGrouping.GetGroupCount())) {
            const auto group = objectsGrouping.GetGroup(groupIdx);
            const auto groupTimestamp = timestamps[group.Begin];
            if (groupTimestamp <= timesplitQuantileTimestamp) {
                if (isObjectwise) {
                    samplingUnitsCount += group.GetSize();
                } else {
                    ++samplingUnitsCount;
                }
            }
        }
    }
    CB_ENSURE(
        featureEvalOptions.FoldSize.Get() > 0 || featureEvalOptions.RelativeFoldSize.Get() > 0,
        "Please specify positive fold size or positive relative fold size");
    if (featureEvalOptions.FoldSize.Get() > 0) {
        *absoluteFoldSize = featureEvalOptions.FoldSize.Get();
    } else {
        *absoluteFoldSize = featureEvalOptions.RelativeFoldSize.Get() * samplingUnitsCount;
        CB_ENSURE(
            *absoluteFoldSize > 0,
            "Relative fold size must be greater than " << 1.0f / samplingUnitsCount << " so that size of each fold is non-zero";
        );
    }
    *disjointFoldCount = samplingUnitsCount / *absoluteFoldSize;
    if (*disjointFoldCount < 2) {
        CATBOOST_WARNING_LOG << "Fold size (" << *absoluteFoldSize << " units) excceds 50% of dataset size (" << samplingUnitsCount << " units). "
            << "Fold size is decreased to 50% of dataset size." << Endl;
        *disjointFoldCount = 2;
        *absoluteFoldSize = samplingUnitsCount / 2;
    }
}

TFeatureEvaluationSummary EvaluateFeatures(
    const NJson::TJsonValue& plainJsonParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    TDataProviderPtr data
) {
    const auto taskType = NCatboostOptions::GetTaskType(plainJsonParams);
    if (taskType == ETaskType::GPU) {
        CB_ENSURE(
            TTrainerFactory::Has(ETaskType::GPU),
            "Can't load GPU learning library. "
            "Module was not compiled or driver  is incompatible with package. "
            "Please install latest NVDIA driver and check again");
    }
    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    LoadOptions(plainJsonParams, data.Get()->MetaInfo, &catBoostOptions, &outputFileOptions);
    const auto& absoluteSnapshotPath = MakeAbsolutePath(outputFileOptions.GetSnapshotFilename());
    outputFileOptions.SetSnapshotFilename(absoluteSnapshotPath);

    const ui32 foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(foldCount > 0, "Fold count must be positive integer");
    const ui32 offset = featureEvalOptions.Offset;

    ui32 absoluteFoldSize;
    ui32 disjointFoldCount;
    CountDisjointFolds(data, featureEvalOptions, &absoluteFoldSize, &disjointFoldCount);

    if (disjointFoldCount < offset + foldCount) {
        const auto samplingUnitsCount = GetSamplingUnitCount(*data->ObjectsGrouping, IsObjectwiseEval(featureEvalOptions));
        CB_ENSURE(
            cvParams.Shuffle,
            "Dataset contains too few objects or groups to evaluate features without shuffling. "
            "Please decrease fold size to at most " << samplingUnitsCount / (offset + foldCount) << ", or "
            "enable dataset shuffling in cross-validation "
            "(specify cv_no_suffle=False in Python or remove --cv-no-shuffle from command line).");
    }

    const auto foldRangeRandomSeeds = GenRandUI64Vector(CeilDiv(offset + foldCount, disjointFoldCount), catBoostOptions.RandomSeed);
    auto foldRangeRandomSeed = catBoostOptions;

    TFeatureEvaluationSummary summary;

    const auto callbacks = MakeHolder<TFeatureEvaluationCallbacks>(
        catBoostOptions.BoostingOptions->IterationCount,
        featureEvalOptions,
        &summary);

    if (outputFileOptions.SaveSnapshot() && NFs::Exists(absoluteSnapshotPath)) {
        callbacks->LoadSnapshot(taskType, absoluteSnapshotPath);
    }

    const ui32 trainingCount = GetTrainingCount(featureEvalOptions);
    CATBOOST_NOTICE_LOG << "Feature evaluation requires training " << trainingCount << " model(s); "
        "if training takes more than 10 minutes to complete, progress is printed every 10 minutes" << Endl;

    auto foldRangePart = featureEvalOptions;
    foldRangePart.FoldSize = absoluteFoldSize;
    foldRangePart.Offset = offset % disjointFoldCount;
    foldRangePart.FoldCount = Min(disjointFoldCount - offset % disjointFoldCount, foldCount);
    ui32 foldRangeIdx = offset / disjointFoldCount;
    ui32 processedFoldCount = 0;
    while (processedFoldCount < foldCount) {
        foldRangeRandomSeed.RandomSeed = foldRangeRandomSeeds[foldRangeIdx];
        EvaluateFeaturesImpl(
            foldRangeRandomSeed,
            outputFileOptions,
            foldRangePart,
            objectiveDescriptor,
            evalMetricDescriptor,
            /*foldRangeBegin*/ foldRangeIdx * disjointFoldCount,
            cvParams,
            data,
            processedFoldCount,
            callbacks.Get(),
            &summary
        );
        ++foldRangeIdx;
        processedFoldCount += foldRangePart.FoldCount.Get();
        foldRangePart.Offset = 0;
        foldRangePart.FoldCount = Min(disjointFoldCount, foldCount - processedFoldCount);
    }
    summary.CalcWxTestAndAverageDelta();
    return summary;
}
