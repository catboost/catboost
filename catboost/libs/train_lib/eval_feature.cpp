#include "eval_feature.h"
#include "train_model.h"
#include "options_helper.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/calc_score_cache.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo/roc_curve.h>
#include <catboost/private/libs/algo/train.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/parallel_tasks.h>
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
#include <catboost/private/libs/options/plain_options_helper.h>

#include <util/folder/tempdir.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/mapfindptr.h>
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
    for (ui32 featureSetIdx : xrange(summary.FeatureSets.size())) {
        featureEvalStream << summary.WxTest[featureSetIdx] << '\t';
        const auto& bestIterations = summary.BestBaselineIterations[featureSetIdx];
        featureEvalStream << JoinRange(",", bestIterations.begin(), bestIterations.end());
        featureEvalStream << '\t';
        for (double delta : summary.AverageMetricDelta[featureSetIdx]) {
            featureEvalStream << delta << '\t';
        }
        const auto& featureSet = summary.FeatureSets[featureSetIdx];
        featureEvalStream << JoinRange(",", featureSet.begin(), featureSet.end());
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


static TVector<ui32> GetBestIterations(
    const TVector<EMetricBestValue>& bestValueType,
    const TVector<TFoldContext>& folds
) {
    TVector<ui32> bestIterations; // [foldIdx]
    for (const auto& fold : folds) {
        bestIterations.push_back(GetBestIterationInFold(
            bestValueType,
            fold.MetricValuesOnTest));
    }
    return bestIterations;
}

static TVector<double> GetMetricValues(
    ui32 metricIdx,
    const TVector<ui32>& bestIterations,
    const TVector<TFoldContext>& folds
) {
    Y_ASSERT(bestIterations.size() == folds.size());
    TVector<double> metricValues;
    for (ui32 foldIdx : xrange(folds.size())) {
        metricValues.push_back(
            folds[foldIdx].MetricValuesOnTest[bestIterations[foldIdx]][metricIdx]
        );
    }
    return metricValues;
}

void TFeatureEvaluationSummary::AppendFeatureSetMetrics(
    ui32 featureSetIdx,
    const TVector<TFoldContext>& baselineFolds,
    const TVector<TFoldContext>& testedFolds
) {
    const auto featureSetCount = FeatureSets.size();
    CB_ENSURE_INTERNAL(featureSetIdx < featureSetCount, "Feature set index is too large");

    BestBaselineIterations.resize(featureSetCount);
    BestBaselineMetrics.resize(featureSetCount);
    BestTestedMetrics.resize(featureSetCount);

    CB_ENSURE_INTERNAL(baselineFolds.size() == testedFolds.size(), "Different number of baseline and tested folds");

    const auto bestBaselineIterations = GetBestIterations(MetricTypes, baselineFolds);
    BestBaselineIterations[featureSetIdx].insert(
        BestBaselineIterations[featureSetIdx].end(),
        bestBaselineIterations.begin(),
        bestBaselineIterations.end()
    );
    const auto bestTestedIterations = GetBestIterations(MetricTypes, testedFolds);
    const auto metricCount = MetricTypes.size();
    for (auto metricIdx : xrange(metricCount)) {
        const auto& foldsBaselineMetric = GetMetricValues(metricIdx, bestBaselineIterations, baselineFolds);
        auto& baselineMetrics = BestBaselineMetrics[featureSetIdx];
        baselineMetrics.resize(metricCount);
        baselineMetrics[metricIdx].insert(
            baselineMetrics[metricIdx].end(),
            foldsBaselineMetric.begin(),
            foldsBaselineMetric.end()
        );
        const auto& foldsTestedMetric = GetMetricValues(metricIdx, bestTestedIterations, testedFolds);
        auto& testedMetrics = BestTestedMetrics[featureSetIdx];
        testedMetrics.resize(metricCount);
        testedMetrics[metricIdx].insert(
            testedMetrics[metricIdx].end(),
            foldsTestedMetric.begin(),
            foldsTestedMetric.end()
        );
    }
}


void TFeatureEvaluationSummary::CalcWxTestAndAverageDelta() {
    const auto featureSetCount = FeatureSets.size();
    const auto metricCount = MetricTypes.size();
    TVector<double> averageDelta(metricCount);
    WxTest.resize(featureSetCount);
    AverageMetricDelta.resize(featureSetCount);
    constexpr ui32 LossIdx = 0;
    for (auto featureSetIdx : xrange(featureSetCount)) {
        const auto& baselineMetrics = BestBaselineMetrics[featureSetIdx];
        const auto& testedMetrics = BestTestedMetrics[featureSetIdx];
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
}


static bool IsObjectwiseEval(const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions) {
    return featureEvalOptions.FoldSizeUnit.Get() == ESamplingUnit::Object;
}

template <class TDataProvidersTemplate> // TTrainingDataProvidersTemplate<...>
static void PrepareFolds(
    typename TDataProvidersTemplate::TDataPtr srcData,
    const TCvDataPartitionParams& cvParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    ui64 cpuUsedRamLimit,
    TVector<TDataProvidersTemplate>* foldsData,
    TVector<TDataProvidersTemplate>* testFoldsData,
    NPar::TLocalExecutor* localExecutor
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
        const ui32 foldSize = featureEvalOptions.FoldSize.Get();
        CB_ENSURE(foldSize > 0, "Fold size must be positive integer");
        // group subsets, groups maybe trivial
        const auto isObjectwise = IsObjectwiseEval(featureEvalOptions);
        testSubsets = isObjectwise
            ? NCB::SplitByObjects(objectsGrouping, foldSize)
            : NCB::SplitByGroups(objectsGrouping, foldSize);
        const ui32 offset = featureEvalOptions.Offset.Get();
        CB_ENSURE_INTERNAL(offset + foldCount <= testSubsets.size(), "Dataset permutation logic failed");
    }
    // group subsets, maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets
        = CalcTrainSubsets(testSubsets, objectsGrouping.GetGroupCount());

    testSubsets.swap(trainSubsets);

    CB_ENSURE(foldsData->empty(), "Need empty vector of folds data");
    foldsData->resize(trainSubsets.size());
    if (testFoldsData != nullptr) {
        CB_ENSURE(testFoldsData->empty(), "Need empty vector of test folds data");
        testFoldsData->resize(trainSubsets.size());
    } else {
        testFoldsData = foldsData;
    }

    TVector<std::function<void()>> tasks;

    // NCB::Split keeps objects order
    const NCB::EObjectsOrder objectsOrder = NCB::EObjectsOrder::Ordered;

    const ui64 perTaskCpuUsedRamLimit = cpuUsedRamLimit / (2 * trainSubsets.size());

    for (ui32 foldIdx : xrange(trainSubsets.size())) {
        tasks.emplace_back(
            [&, foldIdx]() {
                (*foldsData)[foldIdx].Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        std::move(trainSubsets[foldIdx]),
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
                            std::move(testSubsets[foldIdx]),
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

    if (!cvParams.Initialized()) {
        const ui32 offset = featureEvalOptions.Offset.Get();
        TVector<TDataProvidersTemplate>(foldsData->begin() + offset, foldsData->end()).swap(*foldsData);
        foldsData->resize(foldCount);
        if (testFoldsData != foldsData) {
            TVector<TDataProvidersTemplate>(testFoldsData->begin() + offset, testFoldsData->end()).swap(*testFoldsData);
            testFoldsData->resize(foldCount);
        }
    }
}


enum class ETrainingKind {
    Baseline,
    Testing
};


template<>
void Out<ETrainingKind>(IOutputStream& stream, ETrainingKind kind) {
     if (kind == ETrainingKind::Baseline) {
         stream << "Baseline";
         return;
     }
     if (kind == ETrainingKind::Testing) {
         stream << "Testing";
         return;
     }
     Y_UNREACHABLE();
}


template <typename TObjectsDataProvider> // TQuantizedForCPUObjectsDataProvider or TQuantizedObjectsDataProvider
TIntrusivePtr<TTrainingDataProvider> MakeFeatureSubsetDataProvider(
    const TVector<ui32>& ignoredFeatures,
    NCB::TTrainingDataProviderPtr trainingDataProvider
) {
    TIntrusivePtr<TObjectsDataProvider> newObjects = dynamic_cast<TObjectsDataProvider*>(
        trainingDataProvider->ObjectsData->GetFeaturesSubset(ignoredFeatures, &NPar::LocalExecutor()).Get());
    CB_ENSURE(
        newObjects,
        "Objects data provider must be TQuantizedForCPUObjectsDataProvider or TQuantizedObjectsDataProvider");
    TDataMetaInfo newMetaInfo = trainingDataProvider->MetaInfo;
    newMetaInfo.FeaturesLayout = newObjects->GetFeaturesLayout();
    return MakeIntrusive<TTrainingDataProvider>(
        TDataMetaInfo(newMetaInfo),
        trainingDataProvider->ObjectsGrouping,
        newObjects,
        trainingDataProvider->TargetData);
}


static TVector<TTrainingDataProviders> UpdateIgnoredFeaturesInLearn(
    ETaskType taskType,
    const NCatboostOptions::TFeatureEvalOptions& options,
    ETrainingKind trainingKind,
    ui32 testedFeatureSetIdx,
    const TVector<TTrainingDataProviders>& foldsData
) {
    TVector<ui32> ignoredFeatures;
    const auto& testedFeatures = options.FeaturesToEvaluate.Get();
    const auto featureEvalMode = options.FeatureEvalMode;
    if (trainingKind == ETrainingKind::Testing) {
        for (ui32 featureSetIdx : xrange(testedFeatures.size())) {
            if (featureSetIdx != testedFeatureSetIdx) {
                ignoredFeatures.insert(
                    ignoredFeatures.end(),
                    testedFeatures[featureSetIdx].begin(),
                    testedFeatures[featureSetIdx].end());
            }
        }
    } else if (featureEvalMode == NCB::EFeatureEvalMode::OneVsAll) {
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
    TVector<TTrainingDataProviders> result;
    result.reserve(foldsData.size());
    if (taskType == ETaskType::CPU) {
        for (const auto& foldData : foldsData) {
            TTrainingDataProviders newTrainingData;
            newTrainingData.Learn = MakeFeatureSubsetDataProvider<TQuantizedForCPUObjectsDataProvider>(
                ignoredFeatures,
                foldData.Learn);
            newTrainingData.Test.push_back(
                MakeFeatureSubsetDataProvider<TQuantizedForCPUObjectsDataProvider>(
                    ignoredFeatures,
                    foldData.Test[0])
            );
            result.push_back(newTrainingData);
        }
    } else {
        for (const auto& foldData : foldsData) {
            TTrainingDataProviders newTrainingData;
            newTrainingData.Learn = MakeFeatureSubsetDataProvider<TQuantizedObjectsDataProvider>(
                ignoredFeatures,
                foldData.Learn);
            newTrainingData.Test.push_back(
                MakeFeatureSubsetDataProvider<TQuantizedObjectsDataProvider>(
                    ignoredFeatures,
                    foldData.Test[0])
            );
            result.push_back(newTrainingData);
        }
    }
    return result;
}


static void LoadOptions(
    const NJson::TJsonValue& plainJsonParams,
    NCatboostOptions::TCatBoostOptions* catBoostOptions,
    NCatboostOptions::TOutputFilesOptions* outputFileOptions
) {
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    catBoostOptions->Load(jsonParams);
    outputFileOptions->Load(outputJsonParams);

    // TODO(akhropov): implement snapshots in CV. MLTOOLS-3439.
    CB_ENSURE(!outputFileOptions->SaveSnapshot(), "Saving snapshots in feature evaluation mode is not supported yet");

    if (outputFileOptions->GetMetricPeriod() > 1) {
        CATBOOST_WARNING_LOG << "Warning: metric_period is ignored because "
            "feature evaluation needs metric values on each iteration" << Endl;
        outputFileOptions->SetMetricPeriod(1);
    }
}


static void CalcMetricsForTest(
    const TVector<THolder<IMetric>>& metrics,
    const TVector<TTrainingDataProviders>& quantizedData,
    TVector<TFoldContext>* foldContexts
) {
    const auto metricCount = metrics.size();
    for (auto foldIdx : xrange(foldContexts->size())) {
        THPTimer timer;
        CB_ENSURE(
            quantizedData[foldIdx].Test.size() == 1,
            "Need exactly one test dataset in test fold " << (*foldContexts)[foldIdx].FoldIdx);
        auto& metricValuesOnTest = (*foldContexts)[foldIdx].MetricValuesOnTest;
        CB_ENSURE(
            metricValuesOnTest.empty(),
            "Fold " << (*foldContexts)[foldIdx].FoldIdx << " already has metric values");
        const auto treeCount = (*foldContexts)[foldIdx].FullModel->GetTreeCount();
        ResizeRank2(treeCount, metricCount, metricValuesOnTest);

        const auto& testData = quantizedData[foldIdx].Test[0];
        const auto classCount = testData->TargetData->GetTargetClassCount().GetOrElse(1);
        const auto docCount = testData->GetObjectCount();
        TVector<TVector<double>> approx;
        ResizeRank2(classCount, docCount, approx);
        TVector<TVector<double>> partialApprox;
        ResizeRank2(classCount, docCount, partialApprox);
        TVector<double> flatApproxBuffer;
        flatApproxBuffer.yresize(docCount * classCount);

        TModelCalcerOnPool modelCalcer(
            (*foldContexts)[foldIdx].FullModel.GetRef(),
            testData->ObjectsData,
            &NPar::LocalExecutor());
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
            for (auto classIdx : xrange(classCount)) {
                for (auto docIdx : xrange(docCount)) {
                    approx[classIdx][docIdx] += partialApprox[classIdx][docIdx];
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
        CATBOOST_INFO_LOG << "Fold "
            << (*foldContexts)[foldIdx].FoldIdx << ": metrics calculated in "
            << FloatToString(timer.Passed(), PREC_NDIGITS, 2) << " sec" << Endl;
    }
}


class TEvalFeatureCallbacks : public ITrainingCallbacks {
public:
    explicit TEvalFeatureCallbacks(ui32 iterationCount)
    : IterationCount(iterationCount)
    {
    }

    bool IsContinueTraining(const TMetricsAndTimeLeftHistory& /*unused*/) override {
        ++IterationIdx;
        constexpr double HeartbeatSeconds = 1;
        if (TrainTimer.Passed() > HeartbeatSeconds) {
            TrainTimer.Reset();
            TSetLogging infomationMode(ELoggingLevel::Info);
            CATBOOST_INFO_LOG << "Train iteration " << IterationIdx << " of " << IterationCount << Endl;
        }
        return /*continue training*/true;
    }
private:
    THPTimer TrainTimer;
    ui32 IterationIdx = 0;
    ui32 IterationCount;
};

static void EvaluateFeaturesImpl(
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    TDataProviderPtr data,
    TFeatureEvaluationSummary* results
) {
    const ui32 foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(data->ObjectsData->GetObjectCount() > foldCount, "Pool is too small to be split into folds");
    CB_ENSURE(data->ObjectsData->GetObjectCount() > featureEvalOptions.FoldSize.Get(), "Pool is too small to be split into folds");
    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        data->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Feature evaluation for ordered objects data is not yet implemented"
    );

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
    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRamIfPossible*/ true,
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
        outputFileOptions.AllowWriteFiles(),
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

    // disable overfitting detector on folds training, it will work on average values
    const auto overfittingDetectorOptions = dataSpecificOptions.BoostingOptions->OverfittingDetector;
    dataSpecificOptions.BoostingOptions->OverfittingDetector->OverfittingDetectorType = EOverfittingDetectorType::None;

    // internal training output shouldn't interfere with main stdout
    const auto loggingLevel = dataSpecificOptions.LoggingLevel;
    dataSpecificOptions.LoggingLevel = ELoggingLevel::Silent;

    const auto taskType = catBoostOptions.GetTaskType();
    if (taskType == ETaskType::GPU) {
        CB_ENSURE(
            TTrainerFactory::Has(ETaskType::GPU),
            "Can't load GPU learning library. "
            "Module was not compiled or driver  is incompatible with package. "
            "Please install latest NVDIA driver and check again");
    }
    THolder<IModelTrainer> modelTrainerHolder = TTrainerFactory::Construct(taskType);

    TSetLogging inThisScope(loggingLevel);

    TVector<TTrainingDataProviders> foldsData;
    TVector<TTrainingDataProviders> testFoldsData;
    constexpr bool isFixedMlTools3185 = false;
    PrepareFolds<TTrainingDataProviders>(
        trainingData,
        cvParams,
        featureEvalOptions,
        cpuUsedRamLimit,
        &foldsData,
        isFixedMlTools3185 ? &testFoldsData : nullptr,
        &NPar::LocalExecutor()
    );

    UpdatePermutationBlockSize(taskType, foldsData, &dataSpecificOptions);

    const auto& metrics = CreateMetrics(
        dataSpecificOptions.MetricOptions,
        evalMetricDescriptor,
        GetApproxDimension(dataSpecificOptions, labelConverter),
        trainingData->MetaInfo.HasWeights);
    CheckMetrics(metrics, dataSpecificOptions.LossFunctionDescription.Get().GetLossFunction());

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);

    if (!results->HasHeaderInfo()) {
        results->SetHeaderInfo(metrics, featureEvalOptions.FeaturesToEvaluate);
    }

    const ui32 iterationCount = dataSpecificOptions.BoostingOptions->IterationCount;
    const THolder<ITrainingCallbacks> evalFeatureCallbacks = MakeHolder<TEvalFeatureCallbacks>(iterationCount);

    const auto trainFullModels = [&] (
        const TString& trainDirPrefix,
        TVector<TTrainingDataProviders>* foldsData,
        TVector<TFoldContext>* foldContexts) {
        Y_ASSERT(foldContexts->empty());
        const ui32 offset = cvParams.Initialized() ? 0 : featureEvalOptions.Offset.Get();
        for (auto foldIdx : xrange(foldCount)) {
            foldContexts->emplace_back(
                offset + foldIdx,
                taskType,
                outputFileOptions,
                std::move((*foldsData)[foldIdx]),
                rand.GenRand(),
                /*hasFullModel*/true
            );
        }

        const auto topLevelTrainDir = outputFileOptions.GetTrainDir();
        const bool isCalcFstr = outputFileOptions.CreateFstrRegularFullPath() || outputFileOptions.CreateFstrIternalFullPath();

        for (auto foldIdx : xrange(foldCount)) {
            THPTimer timer;
            TErrorTracker errorTracker = CreateErrorTracker(
                overfittingDetectorOptions,
                bestPossibleValue,
                bestValueType,
                /*hasTest*/(*foldsData)[foldIdx].Test.size());

            const auto foldTrainDir = trainDirPrefix + "fold_" + ToString((*foldContexts)[foldIdx].FoldIdx + results->FoldRangeOffset);
            Train(
                dataSpecificOptions,
                JoinFsPaths(topLevelTrainDir, foldTrainDir),
                objectiveDescriptor,
                evalMetricDescriptor,
                labelConverter,
                metrics,
                errorTracker.IsActive(),
                evalFeatureCallbacks,
                &(*foldContexts)[foldIdx],
                modelTrainerHolder.Get(),
                &NPar::LocalExecutor()
            );
            CB_ENSURE(
                (*foldContexts)[foldIdx].FullModel.Defined(),
                "Fold " << (*foldContexts)[foldIdx].FoldIdx << ": model is missing"
            );
            const auto treeCount = (*foldContexts)[foldIdx].FullModel->GetTreeCount();
            CB_ENSURE(
                iterationCount == treeCount,
                "Fold " << (*foldContexts)[foldIdx].FoldIdx << ": model size (" << treeCount <<
                ") differs from iteration count (" << iterationCount << ")"
            );
            CATBOOST_INFO_LOG << "Fold " << (*foldContexts)[foldIdx].FoldIdx << ": model built in " <<
                FloatToString(timer.Passed(), PREC_NDIGITS, 2) << " sec" << Endl;

            if (isCalcFstr) {
                auto foldOutputOptions = outputFileOptions;
                foldOutputOptions.SetTrainDir(JoinFsPaths(topLevelTrainDir, foldTrainDir));
                const auto foldRegularFstrPath = foldOutputOptions.CreateFstrRegularFullPath();
                const auto foldInternalFstrPath = foldOutputOptions.CreateFstrIternalFullPath();
                const auto& model = (*foldContexts)[foldIdx].FullModel.GetRef();
                CalcAndOutputFstr(
                    model,
                    /*dataset*/nullptr,
                    &NPar::LocalExecutor(),
                    foldRegularFstrPath ? &foldRegularFstrPath : nullptr,
                    foldInternalFstrPath ? &foldInternalFstrPath : nullptr,
                    outputFileOptions.GetFstrType());
            }
        }

        for (auto foldIdx : xrange(foldCount)) {
            (*foldsData)[foldIdx] = std::move((*foldContexts)[foldIdx].TrainingData);
        }

        if (testFoldsData) {
            CalcMetricsForTest(metrics, testFoldsData, foldContexts);
        }
    };

    TVector<TFoldContext> baselineFoldContexts;

    if (featureEvalOptions.FeaturesToEvaluate->empty()) {
        auto baselineDirPrefix = TStringBuilder() << "Baseline_";
        trainFullModels(baselineDirPrefix, &foldsData, &baselineFoldContexts);
        return;
    }

    const auto useCommonBaseline = featureEvalOptions.FeatureEvalMode != NCB::EFeatureEvalMode::OneVsOthers;
    for (ui32 featureSetIdx : xrange(featureEvalOptions.FeaturesToEvaluate->size())) {
        const auto haveBaseline = featureSetIdx > 0 && useCommonBaseline;
        if (!haveBaseline) {
            auto newFoldsData = UpdateIgnoredFeaturesInLearn(
                taskType,
                featureEvalOptions,
                ETrainingKind::Baseline,
                featureSetIdx,
                foldsData);
            baselineFoldContexts.clear();
            auto baselineDirPrefix = TStringBuilder() << "Baseline_";
            if (!useCommonBaseline) {
                baselineDirPrefix << "set_" << featureSetIdx << "_";
            }
            trainFullModels(baselineDirPrefix, &newFoldsData, &baselineFoldContexts);
        }

        TVector<TFoldContext> testedFoldContexts;
        auto newFoldsData = UpdateIgnoredFeaturesInLearn(
            taskType,
            featureEvalOptions,
            ETrainingKind::Testing,
            featureSetIdx,
            foldsData);
        const auto testingDirPrefix = TStringBuilder() << "Testing_set_" << featureSetIdx << "_";
        trainFullModels(testingDirPrefix, &newFoldsData, &testedFoldContexts);

        results->AppendFeatureSetMetrics(featureSetIdx, baselineFoldContexts, testedFoldContexts);
    }
}

void EvaluateFeatures(
    const NJson::TJsonValue& plainJsonParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    TDataProviderPtr data,
    TFeatureEvaluationSummary* summary
) {
    const auto taskType = NCatboostOptions::GetTaskType(plainJsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    LoadOptions(plainJsonParams, &catBoostOptions, &outputFileOptions);

    const ui32 foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(foldCount > 0, "Fold count must be positive integer");

    const auto isObjectwise = IsObjectwiseEval(featureEvalOptions);
    const ui32 foldSize = featureEvalOptions.FoldSize.Get();
    const auto& objectsGrouping = *data->ObjectsGrouping;
    const auto datasetSize = isObjectwise ? objectsGrouping.GetObjectCount() : objectsGrouping.GetGroupCount();
    const ui32 disjointFoldCount = CeilDiv(datasetSize, foldSize);
    const auto offset = featureEvalOptions.Offset.Get();

    if (disjointFoldCount < offset + foldCount) {
        CB_ENSURE(
            cvParams.Shuffle,
            "Dataset contains too few objects or groups to evaluate features without shuffling. "
            "Please decrease fold size to at most " << datasetSize / foldCount << ", or "
            "enable dataset shuffling in cross-validation "
            "(specify cv_no_suffle=False in Python or remove --cv-no-shuffle from command line).");
    }
    auto foldRange = featureEvalOptions;
    foldRange.Offset = offset % disjointFoldCount;
    foldRange.FoldCount = Min(disjointFoldCount - offset % disjointFoldCount, foldCount);

    const auto foldRangeRandomSeeds = GenRandUI64Vector(CeilDiv(offset + foldCount, disjointFoldCount), catBoostOptions.RandomSeed);
    auto foldRangeRandomSeed = catBoostOptions;

    ui32 foldRangeIdx = offset / disjointFoldCount;
    ui32 processedFoldCount = 0;
    while (processedFoldCount < foldCount) {
        foldRangeRandomSeed.RandomSeed = foldRangeRandomSeeds[foldRangeIdx];
        summary->FoldRangeOffset = foldRangeIdx * disjointFoldCount;
        EvaluateFeaturesImpl(
            foldRangeRandomSeed,
            outputFileOptions,
            foldRange,
            objectiveDescriptor,
            evalMetricDescriptor,
            cvParams,
            data,
            summary
        );
        ++foldRangeIdx;
        processedFoldCount += foldRange.FoldCount.Get();
        foldRange.Offset = 0;
        foldRange.FoldCount = Min(disjointFoldCount, foldCount - processedFoldCount);
    }
    summary->CalcWxTestAndAverageDelta();
}
