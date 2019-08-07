#include "eval_feature.h"
#include "train_model.h"
#include "options_helper.h"

#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/approx_dimension.h>
#include <catboost/libs/algo/calc_score_cache.h>
#include <catboost/libs/algo/data.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/preprocess.h>
#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/algo/train.h>
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
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/feature_eval_options.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/options/plain_options_helper.h>

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
    for (const auto& metric : summary.Metrics) {
        featureEvalStream << metric << '\t';
    }
    featureEvalStream << "feature set" << Endl;
    for (ui32 featureSetIdx : xrange(summary.FeatureSets.size())) {
        featureEvalStream << summary.WxTest[featureSetIdx] << '\t';
        const auto& bestIterations = summary.BestBaselineIterations[featureSetIdx];
        featureEvalStream << JoinRange(",", bestIterations.begin(), bestIterations.end());
        featureEvalStream << '\t';
        for (double delta : summary.MetricDelta[featureSetIdx]) {
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

void TFeatureEvaluationSummary::PushBackWxTestAndDelta(
    const TVector<THolder<IMetric>>& metrics,
    const TVector<TFoldContext>& baselineFolds,
    const TVector<TFoldContext>& testedFolds
) {
    const auto bestValueTypes = GetBestValueType(metrics);
    const auto bestBaselineIterations = GetBestIterations(bestValueTypes, baselineFolds);
    const auto bestTestedIterations = GetBestIterations(bestValueTypes, testedFolds);

    BestBaselineIterations.push_back(bestBaselineIterations);
    constexpr ui32 lossIdx = 0;
    WxTest.push_back(
        ::WxTest(
            GetMetricValues(lossIdx, bestBaselineIterations, baselineFolds),
            GetMetricValues(lossIdx, bestTestedIterations, testedFolds)
        ).PValue);
    const auto metricCount = metrics.size();
    MetricDelta.push_back(TVector<double>(metricCount));
    const auto foldCount = baselineFolds.size();
    for (auto metricIdx : xrange(metricCount)) {
        const auto baselineAverage = Accumulate(
            GetMetricValues(metricIdx, bestBaselineIterations, baselineFolds),
            0.0) / foldCount;
        const auto testedAverage = Accumulate(
            GetMetricValues(metricIdx, bestTestedIterations, testedFolds),
            0.0) / foldCount;
        if (bestValueTypes[metricIdx] == EMetricBestValue::Min) {
            MetricDelta.back()[metricIdx] = -testedAverage + baselineAverage;
        } else {
            MetricDelta.back()[metricIdx] = + testedAverage - baselineAverage;
        }
    }
}


template <class TDataProvidersTemplate> // TTrainingDataProvidersTemplate<...>
static void PrepareFolds(
    typename TDataProvidersTemplate::TDataPtr srcData,
    const TCvDataPartitionParams& cvParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
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
        const auto foldSizeUnit = featureEvalOptions.FoldSizeUnit.Get();
        testSubsets = foldSizeUnit == ESamplingUnit::Object
            ? NCB::SplitByObjects(objectsGrouping, foldSize)
            : NCB::SplitByGroups(objectsGrouping, foldSize);
        const ui32 offset = featureEvalOptions.Offset.Get();
        CB_ENSURE(offset + foldCount <= testSubsets.size(),
            "Dataset contains only " << testSubsets.size() << " folds of " << foldSize << " units of type " << foldSizeUnit
            << " which are not enough to create folds [" << offset << ", " << offset + foldCount << ")");
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

    for (ui32 foldIdx : xrange(trainSubsets.size())) {
        tasks.emplace_back(
            [&, foldIdx]() {
                (*foldsData)[foldIdx].Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        std::move(trainSubsets[foldIdx]),
                        objectsOrder
                    ),
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

static void SetFeaturesAvailability(
    bool isAvailable,
    const TVector<ui32>& featureSet,
    TTrainingDataProviders* foldData
) {
    for (ui32 featureIdx : featureSet) {
        foldData->Learn->MetaInfo.FeaturesLayout->SetExternalFeatureAvailability(featureIdx, isAvailable);
    }
}


static void UpdateIgnoredFeaturesInLearn(
    const NCatboostOptions::TFeatureEvalOptions& options,
    ETrainingKind trainingKind,
    ui32 testedFeatureSetIdx,
    TVector<TTrainingDataProviders>* foldsData
) {
    const auto& testedFeatures = options.FeaturesToEvaluate.Get();
    const auto featureEvalMode = options.FeatureEvalMode;
    if (trainingKind == ETrainingKind::Testing) {
        for (auto& foldData : *foldsData) {
            for (ui32 featureSetIdx : xrange(testedFeatures.size())) {
                SetFeaturesAvailability(
                    featureSetIdx == testedFeatureSetIdx,
                    testedFeatures[featureSetIdx],
                    &foldData);
            }
        }
    } else if (featureEvalMode == NCB::EFeatureEvalMode::OneVsAll) {
        for (auto& foldData : *foldsData) {
            for (const auto& featureSet : testedFeatures) {
                SetFeaturesAvailability(true, featureSet, &foldData);
            }
        }
    } else if (featureEvalMode == NCB::EFeatureEvalMode::OneVsOthers) {
        for (auto& foldData : *foldsData) {
            for (ui32 featureSetIdx : xrange(testedFeatures.size())) {
                SetFeaturesAvailability(
                    featureSetIdx != testedFeatureSetIdx,
                    testedFeatures[featureSetIdx],
                    &foldData);
            }
        }
    } else {
        CB_ENSURE(
            featureEvalMode == NCB::EFeatureEvalMode::OneVsNone,
            "Unknown feature evaluation mode " + ToString(featureEvalMode)
        );
        for (auto& foldData : *foldsData) {
            for (const auto& featureSet : testedFeatures) {
                SetFeaturesAvailability(false, featureSet, &foldData);
            }
        }
    }
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


void EvaluateFeatures(
    const NJson::TJsonValue& plainJsonParams,
    const NCatboostOptions::TFeatureEvalOptions& featureEvalOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TCvDataPartitionParams& cvParams,
    TDataProviderPtr data,
    TFeatureEvaluationSummary* results
) {
    const auto taskType = NCatboostOptions::GetTaskType(plainJsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    NCatboostOptions::TOutputFilesOptions outputFileOptions;
    LoadOptions(plainJsonParams, &catBoostOptions, &outputFileOptions);

    const ui32 foldCount = cvParams.Initialized() ? cvParams.FoldCount : featureEvalOptions.FoldCount.Get();
    CB_ENSURE(data->ObjectsData->GetObjectCount() > foldCount, "Pool is too small to be split into folds");
    CB_ENSURE(data->ObjectsData->GetObjectCount() > featureEvalOptions.FoldSize.Get(), "Pool is too small to be split into folds");
    // TODO(akhropov): implement ordered split. MLTOOLS-2486.
    CB_ENSURE(
        data->ObjectsData->GetOrder() != EObjectsOrder::Ordered,
        "Feature evaluation for ordered objects data is not yet implemented"
    );

    TRestorableFastRng64 rand(catBoostOptions.RandomSeed);

    if (cvParams.Shuffle) {
        auto objectsGroupingSubset = NCB::Shuffle(data->ObjectsGrouping, 1, &rand);
        data = data->GetSubset(objectsGroupingSubset, &NPar::LocalExecutor());
    }

    TLabelConverter labelConverter;
    TMaybe<float> targetBorder = catBoostOptions.DataProcessingOptions->TargetBorder;
    TTrainingDataProviderPtr trainingData = GetTrainingData(
        std::move(data),
        /*isLearnData*/ true,
        TStringBuf(),
        Nothing(), // TODO(akhropov): allow loading borders and nanModes in CV?
        /*unloadCatFeaturePerfectHashFromRamIfPossible*/ true,
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ false,
        outputFileOptions.AllowWriteFiles(),
        /*quantizedFeaturesInfo*/ nullptr,
        &catBoostOptions,
        &labelConverter,
        &targetBorder,
        &NPar::LocalExecutor(),
        &rand);

    CB_ENSURE(
        dynamic_cast<TQuantizedObjectsDataProvider*>(trainingData->ObjectsData.Get()),
        "Unable to quantize dataset (probably because it contains categorical features)"
    );

    UpdateYetiRankEvalMetric(trainingData->MetaInfo.TargetStats, Nothing(), &catBoostOptions);

    // disable overfitting detector on folds training, it will work on average values
    const auto overfittingDetectorOptions = catBoostOptions.BoostingOptions->OverfittingDetector;
    catBoostOptions.BoostingOptions->OverfittingDetector->OverfittingDetectorType = EOverfittingDetectorType::None;

    // internal training output shouldn't interfere with main stdout
    const auto loggingLevel = catBoostOptions.LoggingLevel;
    catBoostOptions.LoggingLevel = ELoggingLevel::Silent;

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
        &foldsData,
        isFixedMlTools3185 ? &testFoldsData : nullptr,
        &NPar::LocalExecutor()
    );

    UpdatePermutationBlockSize(taskType, foldsData, &catBoostOptions);

    TVector<THolder<IMetric>> metrics = CreateMetrics(
        catBoostOptions.MetricOptions,
        evalMetricDescriptor,
        GetApproxDimension(catBoostOptions, labelConverter)
    );
    CheckMetrics(metrics, catBoostOptions.LossFunctionDescription.Get().GetLossFunction());

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);

    const auto trainFullModels = [&] (const TString& trainDirPrefix, TVector<TFoldContext>* foldContexts) {
        Y_ASSERT(foldContexts->empty());
        const ui32 offset = cvParams.Initialized() ? 0 : featureEvalOptions.Offset.Get();
        for (auto foldIdx : xrange(foldCount)) {
            foldContexts->emplace_back(
                offset + foldIdx,
                taskType,
                outputFileOptions,
                std::move(foldsData[foldIdx]),
                rand.GenRand(),
                /*hasFullModel*/true
            );
        }

        const auto iterationCount = catBoostOptions.BoostingOptions->IterationCount.Get();
        const auto topLevelTrainDir = outputFileOptions.GetTrainDir();
        for (auto foldIdx : xrange(foldCount)) {
            THPTimer timer;
            TErrorTracker errorTracker = CreateErrorTracker(
                overfittingDetectorOptions,
                bestPossibleValue,
                bestValueType,
                /*hasTest*/foldsData[foldIdx].Test.size());

            const auto foldTrainDir = trainDirPrefix + "fold_" + ToString((*foldContexts)[foldIdx].FoldIdx);
            Train(
                catBoostOptions,
                JoinFsPaths(topLevelTrainDir, foldTrainDir),
                objectiveDescriptor,
                evalMetricDescriptor,
                labelConverter,
                metrics,
                errorTracker.IsActive(),
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
        }

        for (auto foldIdx : xrange(foldCount)) {
            foldsData[foldIdx] = std::move((*foldContexts)[foldIdx].TrainingData);
        }

        if (testFoldsData) {
            CalcMetricsForTest(metrics, testFoldsData, foldContexts);
        }
    };

    TVector<TFoldContext> baselineFoldContexts;
    // TODO(espetrov): support eval feature for folds specified by featureEvalOptions
    const auto useCommonBaseline = featureEvalOptions.FeatureEvalMode != NCB::EFeatureEvalMode::OneVsOthers;
    for (ui32 featureSetIdx : xrange(featureEvalOptions.FeaturesToEvaluate->size())) {
        const auto haveBaseline = featureSetIdx > 0 && useCommonBaseline;
        if (!haveBaseline) {
            UpdateIgnoredFeaturesInLearn(featureEvalOptions, ETrainingKind::Baseline, featureSetIdx, &foldsData);
            baselineFoldContexts.clear();
            auto baselineDirPrefix = TStringBuilder() << "Baseline_";
            if (!useCommonBaseline) {
                baselineDirPrefix << "set_" << featureSetIdx << "_";
            }
            trainFullModels(baselineDirPrefix, &baselineFoldContexts);
        }

        TVector<TFoldContext> testedFoldContexts;
        UpdateIgnoredFeaturesInLearn(featureEvalOptions, ETrainingKind::Testing, featureSetIdx, &foldsData);
        const auto testingDirPrefix = TStringBuilder() << "Testing_set_" << featureSetIdx << "_";
        trainFullModels(testingDirPrefix, &testedFoldContexts);

        results->PushBackWxTestAndDelta(metrics, baselineFoldContexts, testedFoldContexts);
    }

    for (const auto& metric : metrics) {
        results->Metrics.push_back(metric->GetDescription());
    }
    results->FeatureSets = featureEvalOptions.FeaturesToEvaluate;
}
