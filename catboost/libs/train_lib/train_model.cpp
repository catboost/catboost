#include "train_model.h"

#include "approx_dimension.h"
#include "cross_validation.h"
#include "data.h"
#include "preprocess.h"

#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/data_new/borders_io.h>
#include <catboost/libs/data_new/load_data.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/labels/label_helper_builder.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/pairs/util.h>
#include <catboost/libs/target/classification_target_helper.h>

#include <library/grid_creator/binarization.h>
#include <library/json/json_prettifier.h>

#include <util/generic/mapfindptr.h>
#include <util/generic/scope.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>
#include <util/system/compiler.h>
#include <util/system/hp_timer.h>


using namespace NCB;


static void ShrinkModel(int itCount, const TCtrHelper& ctrsHelper, TLearnProgress* progress) {
    progress->LeafValues.resize(itCount);
    progress->TreeStruct.resize(itCount);
    progress->TreeStats.resize(itCount);
    progress->UsedCtrSplits.clear();
    for (const auto& tree: progress->TreeStruct) {
        for (const auto& split: tree.GetCtrSplits()) {
            TProjection projection = split.Projection;
            ECtrType ctrType = ctrsHelper.GetCtrInfo(projection)[split.CtrIdx].Type;
            progress->UsedCtrSplits.insert(std::make_pair(ctrType, projection));
        }
    }
}


static TDataProviders LoadPools(
    const NCatboostOptions::TPoolLoadParams& loadOptions,
    EObjectsOrder objectsOrder,
    NPar::TLocalExecutor* const executor,
    TProfileInfo* profile
) {
    const auto& cvParams = loadOptions.CvParams;
    const bool cvMode = cvParams.FoldCount != 0;
    CB_ENSURE(
        !cvMode || loadOptions.TestSetPaths.empty(),
        "Test files are not supported in cross-validation mode"
    );

    auto pools = NCB::ReadTrainDatasets(loadOptions, objectsOrder, !cvMode, executor, profile);

    if (cvMode) {
        if (cvParams.Shuffle && (pools.Learn->ObjectsData->GetOrder() != EObjectsOrder::RandomShuffled)) {
            TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

            auto objectsGroupingSubset = NCB::Shuffle(pools.Learn->ObjectsGrouping, 1, &rand);
            pools.Learn = pools.Learn->GetSubset(objectsGroupingSubset, executor);
        }

        TVector<TDataProviders> foldPools = PrepareCvFolds<TDataProviders>(
            std::move(pools.Learn),
            cvParams,
            cvParams.FoldIdx,
            /* oldCvStyleSplit */ true,
            executor);
        Y_VERIFY(foldPools.size() == 1);

        profile->AddOperation("Build cv pools");

        return foldPools[0];
    } else {
        return pools;
    }
}

static inline bool DivisibleOrLastIteration(int currentIteration, int iterationsCount, int period) {
    return currentIteration % period == 0 || currentIteration == iterationsCount - 1;
}

static bool HasInvalidValues(const TVector<TVector<TVector<double>>>& leafValues) {
    for (const auto& tree : leafValues) {
        for (const TVector<double>& leaf : tree) {
            for (double value : leaf) {
                if (!IsValidFloat(value)) {
                    return true;
                }
            }
        }
    }
    return false;
}

namespace {
struct TMetricsData {
    TVector<THolder<IMetric>> Metrics;
    bool CalcEvalMetricOnEveryIteration;
    TMaybe<TErrorTracker> ErrorTracker;
    TMaybe<TErrorTracker> BestModelMinTreesTracker;
    size_t ErrorTrackerMetricIdx;
};
}

static void InitializeAndCheckMetricData(
    const TTrainingForCPUDataProviders& data,
    bool forceCalcEvalMetricOnEveryIteration,
    const TLearnContext& ctx,
    TMetricsData* metricsData) {

    const int approxDimension = ctx.LearnProgress.ApproxDimension;
    auto& metrics = metricsData->Metrics;
    metrics = CreateMetrics(
        ctx.Params.LossFunctionDescription,
        ctx.Params.MetricOptions,
        ctx.EvalMetricDescriptor,
        approxDimension
    );
    CheckMetrics(metrics, ctx.Params.LossFunctionDescription.Get().GetLossFunction());
    if (!ctx.Params.SystemOptions->IsSingleHost()) {
        if (!AllOf(metrics, [](const auto& metric) { return metric->IsAdditiveMetric(); })) {
            CATBOOST_WARNING_LOG << "In distributed training, non-additive metrics are not evaluated on train dataset" << Endl;
        }
    }
    const bool hasTest = data.GetTestSampleCount() > 0;
    if (!hasTest && !ctx.Params.MetricOptions->CustomMetrics->empty()) {
        CATBOOST_WARNING_LOG << "Warning: Custom metrics will not be evaluated because there are no test datasets" << Endl;
    }

    CB_ENSURE(!metrics.empty(), "Eval metric is not defined");

    const bool lastTestDatasetHasTargetData = (data.Test.size() > 0) && data.Test.back()->MetaInfo.HasTarget;

    if (hasTest && metrics[0]->NeedTarget() && !lastTestDatasetHasTargetData) {
        CATBOOST_WARNING_LOG << "Warning: Eval metric " << metrics[0]->GetDescription() <<
            " needs Target data, but test dataset does not have it so it won't be calculated" << Endl;
    }

    const bool canCalcEvalMetric = hasTest && (!metrics[0]->NeedTarget() || lastTestDatasetHasTargetData);

    if (canCalcEvalMetric) {
        EMetricBestValue bestValueType;
        float bestPossibleValue;

        metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);
        metricsData->ErrorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);
        metricsData->BestModelMinTreesTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);
    }

    auto& errorTracker = metricsData->ErrorTracker;
    metricsData->CalcEvalMetricOnEveryIteration
        = canCalcEvalMetric && (forceCalcEvalMetricOnEveryIteration || errorTracker->IsActive());

    if (ctx.OutputOptions.GetMetricPeriod() > 1 && errorTracker && errorTracker->IsActive() && hasTest) {
        CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is " <<
            "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
    }

    // Use only (last_test, first_metric) for best iteration and overfitting detection
    // In case of changing the order it should be changed in GPU mode also.
    metricsData->ErrorTrackerMetricIdx = 0;
}

namespace {
struct TLoggingData {
    TString LearnToken;
    TVector<const TString> TestTokens;
    TLogger Logger;
};
}

static bool ShouldCalcAllMetrics(int iter, const TLearnContext& ctx) {
    return DivisibleOrLastIteration(
        iter,
        ctx.Params.BoostingOptions->IterationCount,
        ctx.OutputOptions.GetMetricPeriod()
    );
}

static bool ShouldCalcErrorTrackerMetric(int iter, const TMetricsData& metricsData, const TLearnContext& ctx) {
    return ShouldCalcAllMetrics(iter, ctx) || metricsData.CalcEvalMetricOnEveryIteration;
}

// Write history metrics to loggers, error trackers and get info from per iteration metric based callback.
static void ProcessHistoryMetrics(
    const TTrainingForCPUDataProviders& data,
    const TLearnContext& ctx,
    const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
    TMetricsData* metricsData,
    TLoggingData* loggingData,
    bool* continueTraining) {

    loggingData->LearnToken = GetTrainModelLearnToken();
    loggingData->TestTokens = GetTrainModelTestTokens(data.Test.ysize());

    if (ctx.OutputOptions.AllowWriteFiles()) {
        InitializeFileLoggers(
            ctx.Params,
            ctx.Files,
            GetConstPointers(metricsData->Metrics),
            loggingData->LearnToken,
            loggingData->TestTokens,
            ctx.OutputOptions.GetMetricPeriod(),
            &loggingData->Logger);
    }

    const TVector<TTimeInfo>& timeHistory = ctx.LearnProgress.MetricsAndTimeHistory.TimeHistory;
    const TVector<TVector<THashMap<TString, double>>>& testMetricsHistory =
        ctx.LearnProgress.MetricsAndTimeHistory.TestMetricsHistory;

    const bool useBestModel = ctx.OutputOptions.ShrinkModelToBestIteration();
    *continueTraining = true;
    for (int iter : xrange(ctx.LearnProgress.TreeStruct.ysize())) {
        if (iter < testMetricsHistory.ysize() && ShouldCalcErrorTrackerMetric(iter, *metricsData, ctx) && metricsData->ErrorTracker) {
            const TString& errorTrackerMetricDescription = metricsData->Metrics[metricsData->ErrorTrackerMetricIdx]->GetDescription();
            const double error = testMetricsHistory[iter].back().at(errorTrackerMetricDescription);
            metricsData->ErrorTracker->AddError(error, iter);
            if (useBestModel && iter + 1 >= ctx.OutputOptions.BestModelMinTrees) {
                metricsData->BestModelMinTreesTracker->AddError(error, iter);
            }
        }

        Log(iter,
            GetMetricsDescription(metricsData->Metrics),
            ctx.LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory,
            testMetricsHistory,
            metricsData->ErrorTracker ? TMaybe<double>(metricsData->ErrorTracker->GetBestError()) : Nothing(),
            metricsData->ErrorTracker ? TMaybe<int>(metricsData->ErrorTracker->GetBestIteration()) : Nothing(),
            TProfileResults(timeHistory[iter].PassedTime, timeHistory[iter].RemainingTime),
            loggingData->LearnToken,
            loggingData->TestTokens,
            ShouldCalcAllMetrics(iter, ctx),
            &loggingData->Logger
        );

        if (onEndIterationCallback) {
            *continueTraining = (*onEndIterationCallback)(ctx.LearnProgress.MetricsAndTimeHistory);
        }
    }

    AddConsoleLogger(
        loggingData->LearnToken,
        loggingData->TestTokens,
        /*hasTrain=*/true,
        ctx.OutputOptions.GetVerbosePeriod(),
        ctx.Params.BoostingOptions->IterationCount,
        &loggingData->Logger
    );
}

static void InitializeSamplingStructures(
    const TTrainingForCPUDataProviders& data,
    TLearnContext* ctx) {

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    const int defaultCalcStatsObjBlockSize = static_cast<int>(ctx->Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);

    if (ctx->UseTreeLevelCaching()) {
        ctx->SmallestSplitSideDocs.Create(ctx->LearnProgress.Folds, isPairwiseScoring, defaultCalcStatsObjBlockSize);
        ctx->PrevTreeLevelStats.Create(
            ctx->LearnProgress.Folds,
            CountNonCtrBuckets(
                *data.Learn->ObjectsData->GetQuantizedFeaturesInfo(),
                ctx->Params.CatFeatureParams->OneHotMaxSize),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth)
        );
    }
    ctx->SampledDocs.Create(
        ctx->LearnProgress.Folds,
        isPairwiseScoring,
        defaultCalcStatsObjBlockSize,
        GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)
    ); // TODO(espetrov): create only if sample rate < 1
}

static void LogThatStoppingOccured(const TErrorTracker& errorTracker) {
    CATBOOST_NOTICE_LOG << "Stopped by overfitting detector "
        << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
}

static void CalcErrors(
    const TTrainingForCPUDataProviders& data,
    const TMetricsData& metricsData,
    int iter,
    TLearnContext* ctx) {

    CalcErrors(data, metricsData.Metrics, ShouldCalcAllMetrics(iter, *ctx), ShouldCalcErrorTrackerMetric(iter, metricsData, *ctx), ctx);
}

static void Train(
    bool forceCalcEvalMetricOnEveryIteration,
    const TTrainingForCPUDataProviders& data,
    const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* testMultiApprox // [test][dim][docIdx]
) {
    TProfileInfo& profile = ctx->Profile;

    TMetricsData metricsData;
    InitializeAndCheckMetricData(data, forceCalcEvalMetricOnEveryIteration, *ctx, &metricsData);

    if (ctx->TryLoadProgress() && ctx->Params.SystemOptions->IsMaster()) {
        MapRestoreApproxFromTreeStruct(ctx);
    }

    TLoggingData loggingData;
    bool continueTraining;
    ProcessHistoryMetrics(data, *ctx, onEndIterationCallback, &metricsData, &loggingData, &continueTraining);

    if (continueTraining) {
        InitializeSamplingStructures(data, ctx);
    }

    THPTimer timer;

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();
    const bool hasTest = data.GetTestSampleCount() > 0;
    const auto& metrics = metricsData.Metrics;
    auto& errorTracker = metricsData.ErrorTracker;
    for (ui32 iter = ctx->LearnProgress.TreeStruct.ysize();
         continueTraining && (iter < ctx->Params.BoostingOptions->IterationCount);
         ++iter)

    {
        if (errorTracker && errorTracker->GetIsNeedStop()) {
            LogThatStoppingOccured(*errorTracker);
            break;
        }

        profile.StartNextIteration();

        if (timer.Passed() > ctx->OutputOptions.GetSnapshotSaveInterval()) {
            profile.AddOperation("Save snapshot");
            ctx->SaveProgress();
            timer.Reset();
        }

        TrainOneIteration(data, ctx);

        CalcErrors(data, metricsData, iter, ctx);

        profile.AddOperation("Calc errors");

        if (hasTest && ShouldCalcErrorTrackerMetric(iter, metricsData, *ctx) && errorTracker) {
            const auto testErrors = ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.back();
            const TString& errorTrackerMetricDescription = metrics[metricsData.ErrorTrackerMetricIdx]->GetDescription();

            // it is possible that metric has not been calculated because it requires target data
            // that is absent
            if (!testErrors.empty()) {
                const double* error = MapFindPtr(testErrors.back(), errorTrackerMetricDescription);
                if (error) {
                    errorTracker->AddError(*error, iter);
                    if (useBestModel && iter == static_cast<ui32>(errorTracker->GetBestIteration())) {
                        ctx->LearnProgress.BestTestApprox = ctx->LearnProgress.TestApprox.back();
                    }
                    if (useBestModel && static_cast<int>(iter + 1) >= ctx->OutputOptions.BestModelMinTrees) {
                        metricsData.BestModelMinTreesTracker->AddError(*error, iter);
                    }
                }
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress.MetricsAndTimeHistory.TimeHistory.push_back(TTimeInfo(profileResults));

        Log(
            iter,
            GetMetricsDescription(metrics),
            ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory,
            ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory,
            errorTracker ? TMaybe<double>(errorTracker->GetBestError()) : Nothing(),
            errorTracker ? TMaybe<int>(errorTracker->GetBestIteration()) : Nothing(),
            profileResults,
            loggingData.LearnToken,
            loggingData.TestTokens,
            ShouldCalcAllMetrics(iter, *ctx),
            &loggingData.Logger
        );

        if (HasInvalidValues(ctx->LearnProgress.LeafValues)) {
            ctx->LearnProgress.LeafValues.pop_back();
            ctx->LearnProgress.TreeStruct.pop_back();
            CATBOOST_WARNING_LOG << "Training has stopped (degenerate solution on iteration "
                << iter << ", probably too small l2-regularization, try to increase it)" << Endl;
            break;
        }

        if (onEndIterationCallback) {
            continueTraining = (*onEndIterationCallback)(ctx->LearnProgress.MetricsAndTimeHistory);
        }
    }

    ctx->SaveProgress();

    if (hasTest) {
        (*testMultiApprox) = ctx->LearnProgress.TestApprox;
        if (useBestModel) {
            (*testMultiApprox)[0] = ctx->LearnProgress.BestTestApprox;
        }
    }

    ctx->LearnProgress.Folds.clear();

    if (hasTest && errorTracker) {
        CATBOOST_NOTICE_LOG << "\n";
        CATBOOST_NOTICE_LOG << "bestTest = " << errorTracker->GetBestError() << "\n";
        CATBOOST_NOTICE_LOG << "bestIteration = " << errorTracker->GetBestIteration() << "\n";
        CATBOOST_NOTICE_LOG << "\n";
    }

    if (useBestModel && metricsData.BestModelMinTreesTracker && ctx->Params.BoostingOptions->IterationCount > 0) {
        const int bestModelIterations = metricsData.BestModelMinTreesTracker->GetBestIteration() + 1;
        if (0 < bestModelIterations && bestModelIterations < static_cast<int>(ctx->Params.BoostingOptions->IterationCount)) {
            CATBOOST_NOTICE_LOG << "Shrink model to first " << bestModelIterations << " iterations.";
            if (errorTracker->GetBestIteration() + 1 < ctx->OutputOptions.BestModelMinTrees) {
                CATBOOST_NOTICE_LOG << " (min iterations for best model = " << ctx->OutputOptions.BestModelMinTrees << ")";
            }
            CATBOOST_NOTICE_LOG << Endl;
            ShrinkModel(bestModelIterations, ctx->CtrsHelper, &ctx->LearnProgress);
        }
    }
}


namespace {
    class TCPUModelTrainer : public IModelTrainer {

        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NJson::TJsonValue& jsonParams,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
            TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            NPar::TLocalExecutor* localExecutor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* model,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory
        ) const override {
            TTrainingForCPUDataProviders trainingDataForCpu
                = trainingData.Cast<TQuantizedForCPUObjectsDataProvider>();

            if (!internalOptions.CalcMetricsOnly) {
                if (model != nullptr) {
                    CB_ENSURE(
                        !outputOptions.ResultModelPath.IsSet(),
                        "Both modelPtr != nullptr and ResultModelPath is set"
                    );
                } else {
                    CB_ENSURE(
                        !outputOptions.ResultModelPath.Get().empty(),
                        "Both modelPtr == nullptr and ResultModelPath is empty"
                    );
                }
            }

            const auto& quantizedFeaturesInfo
                = *trainingDataForCpu.Learn->ObjectsData->GetQuantizedFeaturesInfo();

            NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(jsonParams));
            NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

            SetDataDependentDefaults(
                trainingDataForCpu.Learn->GetObjectCount(),
                /*hasLearnTarget*/ trainingDataForCpu.Learn->MetaInfo.HasTarget,
                quantizedFeaturesInfo.CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(),
                /*testPoolSize*/ trainingDataForCpu.GetTestSampleCount(),
                /*hasTestLabels*/ trainingDataForCpu.Test.size() > 0 &&
                    trainingDataForCpu.Test[0]->MetaInfo.HasTarget &&
                    IsConst(*trainingDataForCpu.Test[0]->TargetData->GetTarget()),
                /*hasTestPairs*/ trainingDataForCpu.Test.size() > 0 &&
                    trainingDataForCpu.Test[0]->MetaInfo.HasPairs,
                &updatedOutputOptions.UseBestModel,
                &updatedParams
            );

            TLearnContext ctx(
                updatedParams,
                objectiveDescriptor,
                evalMetricDescriptor,
                updatedOutputOptions,
                trainingDataForCpu.Learn->MetaInfo.FeaturesLayout,
                rand,
                localExecutor
            );

            ctx.LearnProgress.ApproxDimension = GetApproxDimension(updatedParams, labelConverter);
            if (ctx.LearnProgress.ApproxDimension > 1) {
                ctx.LearnProgress.LabelConverter = labelConverter;
            }

            ctx.OutputMeta();

            ctx.LearnProgress.FloatFeatures = CreateFloatFeatures(quantizedFeaturesInfo);
            ctx.LearnProgress.CatFeatures = CreateCatFeatures(quantizedFeaturesInfo);

            ctx.InitContext(trainingDataForCpu);

            DumpMemUsage("Before start train");

            const auto& systemOptions = ctx.Params.SystemOptions;
            if (!systemOptions->IsSingleHost()) { // send target, weights, baseline (if present), binarized features to workers and ask them to create plain folds
                InitializeMaster(&ctx);
                CB_ENSURE(IsPlainMode(ctx.Params.BoostingOptions->BoostingType), "Distributed training requires plain boosting");
                CB_ENSURE(!ctx.Layout->GetCatFeatureCount(), "Distributed training doesn't support categorical features");
                MapBuildPlainFold(trainingDataForCpu.Learn, &ctx);
            }
            TVector<TVector<double>> oneRawValues(ctx.LearnProgress.ApproxDimension);
            TVector<TVector<TVector<double>>> rawValues(trainingDataForCpu.Test.size(), oneRawValues);

            Train(
                internalOptions.ForceCalcEvalMetricOnEveryIteration,
                trainingDataForCpu,
                onEndIterationCallback,
                &ctx,
                &rawValues
            );

            for (int testIdx = 0; testIdx < trainingDataForCpu.Test.ysize(); ++testIdx) {
                evalResultPtrs[testIdx]->SetRawValuesByMove(rawValues[testIdx]);
            }

            if (metricsAndTimeHistory) {
                *metricsAndTimeHistory = ctx.LearnProgress.MetricsAndTimeHistory;
            }

            const TString trainingOptionsFileName = ctx.OutputOptions.CreateTrainingOptionsFullPath();
            if (!trainingOptionsFileName.empty()) {
                TOFStream trainingOptionsFile(trainingOptionsFileName);
                trainingOptionsFile.Write(NJson::PrettifyJson(ToString(ctx.Params)));
            }

            if (internalOptions.CalcMetricsOnly) {
                return;
            }

            TPerfectHashedToHashedCatValuesMap perfectHashedToHashedCatValuesMap
                = trainingDataForCpu.Learn->ObjectsData->GetQuantizedFeaturesInfo()
                    ->CalcPerfectHashedToHashedCatValuesMap(localExecutor);

            TObliviousTrees obliviousTrees;
            THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
            {
                TObliviousTreeBuilder builder(ctx.LearnProgress.FloatFeatures, ctx.LearnProgress.CatFeatures, ctx.LearnProgress.ApproxDimension);
                for (size_t treeId = 0; treeId < ctx.LearnProgress.TreeStruct.size(); ++treeId) {
                    TVector<TModelSplit> modelSplits;
                    for (const auto& split : ctx.LearnProgress.TreeStruct[treeId].Splits) {
                        auto modelSplit = split.GetModelSplit(ctx, perfectHashedToHashedCatValuesMap);
                        modelSplits.push_back(modelSplit);
                        if (modelSplit.Type == ESplitType::OnlineCtr) {
                            featureCombinationToProjectionMap[modelSplit.OnlineCtr.Ctr.Base.Projection] = split.Ctr.Projection;
                        }
                    }
                    builder.AddTree(modelSplits, ctx.LearnProgress.LeafValues[treeId], ctx.LearnProgress.TreeStats[treeId].LeafWeightsSum);
                }
                obliviousTrees = builder.Build();
            }


//        TODO(kirillovs,espetrov): return this code after fixing R and Python wrappers
//        for (auto& oheFeature : obliviousTrees.OneHotFeatures) {
//            for (const auto& value : oheFeature.Values) {
//                oheFeature.StringValues.push_back(pools.Learn->CatFeaturesHashToString.at(value));
//            }
//        }
            TClassificationTargetHelper classificationTargetHelper(
                ctx.LearnProgress.LabelConverter,
                ctx.Params.DataProcessingOptions
            );

            TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
            datasetDataForFinalCtrs.Data = trainingDataForCpu;
            datasetDataForFinalCtrs.LearnPermutation = &ctx.LearnProgress.AveragingFold.LearnPermutation->GetObjectsIndexing();
            datasetDataForFinalCtrs.Targets = ctx.LearnProgress.AveragingFold.LearnTarget;
            datasetDataForFinalCtrs.LearnTargetClass = &ctx.LearnProgress.AveragingFold.LearnTargetClass;
            datasetDataForFinalCtrs.TargetClassesCount = &ctx.LearnProgress.AveragingFold.TargetClassesCount;

            {
                NCB::TCoreModelToFullModelConverter coreModelToFullModelConverter(
                    ctx.Params,
                    classificationTargetHelper,
                    ctx.Params.CatFeatureParams->CtrLeafCountLimit,
                    ctx.Params.CatFeatureParams->StoreAllSimpleCtrs,
                    ctx.OutputOptions.GetFinalCtrComputationMode()
                );

                coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
                    std::move(datasetDataForFinalCtrs),
                    std::move(featureCombinationToProjectionMap)
                ).WithPerfectHashedToHashedCatValuesMap(
                    &perfectHashedToHashedCatValuesMap
                ).WithObjectsDataFrom(trainingDataForCpu.Learn->ObjectsData);

                TMaybe<TFullModel> fullModel;
                TFullModel* modelPtr = nullptr;
                if (model) {
                    modelPtr = model;
                } else {
                    fullModel.ConstructInPlace();
                    modelPtr = &*fullModel;
                }

                modelPtr->ObliviousTrees = std::move(obliviousTrees);
                coreModelToFullModelConverter.WithCoreModelFrom(modelPtr);

                if (model) {
                    coreModelToFullModelConverter.Do(true, model);
                } else {
                    coreModelToFullModelConverter.Do(
                        ctx.OutputOptions.CreateResultModelFullPath(),
                        ctx.OutputOptions.GetModelFormats(),
                        ctx.OutputOptions.AddFileFormatExtension()
                    );
                }
            }
        }

        void ModelBasedEval(
            const NJson::TJsonValue& /*params*/,
            const NCatboostOptions::TOutputFilesOptions& /*outputOptions*/,
            TTrainingDataProviders /*trainingData*/,
            const TLabelConverter& /*labelConverter*/,
            NPar::TLocalExecutor* /*localExecutor*/) const override {
            CB_ENSURE(false, "Model based eval is not implemented for CPU");
        }
    };

}

TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ETaskType::CPU);


static void TrainModel(
    const NJson::TJsonValue& trainOptionsJson,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TDataProviders pools,
    const TString& outputModelPath,
    TFullModel* modelPtr,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
    NPar::TLocalExecutor* const executor)
{
    CB_ENSURE(pools.Learn != nullptr, "Train data must be provided");
    CB_ENSURE(pools.Test.size() == evalResultPtrs.size());

    THolder<IModelTrainer> modelTrainerHolder;

    const ETaskType taskType = NCatboostOptions::GetTaskType(trainOptionsJson);

    CB_ENSURE(
        (taskType == ETaskType::CPU) || (pools.Test.size() <= 1),
        "Multiple eval sets not supported for GPU"
    );

    NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;
    if (outputModelPath) {
        updatedOutputOptions.ResultModelPath = outputModelPath;
    }

    NJson::TJsonValue updatedTrainOptionsJson = trainOptionsJson;

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);

        if (outputOptions.SaveSnapshot()) {
            UpdateUndefinedRandomSeed(ETaskType::GPU, updatedOutputOptions, &updatedTrainOptionsJson, [&](IInputStream* in, TString& params) {
                ::Load(in, params);
            });
        }
    } else {
        CB_ENSURE(!isGpuDeviceType, "Can't load GPU learning library. Module was not compiled or driver  is incompatible with package. Please install latest NVDIA driver and check again");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);

        if (outputOptions.SaveSnapshot()) {
            UpdateUndefinedRandomSeed(ETaskType::CPU, updatedOutputOptions, &updatedTrainOptionsJson, [&](IInputStream* in, TString& params) {
                TRestorableFastRng64 unusedRng(0);
                ::LoadMany(in, unusedRng, params);
            });
        }
    }

    const auto learnFeaturesLayout = pools.Learn->MetaInfo.FeaturesLayout;

    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    catBoostOptions.Load(updatedTrainOptionsJson);

    if (!quantizedFeaturesInfo) {
        quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *learnFeaturesLayout,
            catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get(),
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            /*allowNansInTestOnly*/true,
            outputOptions.AllowWriteFiles()
        );
    }

    for (auto testPoolIdx : xrange(pools.Test.size())) {
        const auto& testPool = *pools.Test[testPoolIdx];
        if (testPool.GetObjectCount() == 0) {
            continue;
        }
        CheckCompatibleForApply(
            *learnFeaturesLayout,
            *testPool.MetaInfo.FeaturesLayout,
            TStringBuilder() << "test dataset #" << testPoolIdx);
    }

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);

    auto learnDataOrder = pools.Learn->ObjectsData->GetOrder();
    if (learnDataOrder == EObjectsOrder::Ordered) {
        catBoostOptions.DataProcessingOptions->HasTimeFlag = true;
    }

    pools.Learn = ReorderByTimestampLearnDataIfNeeded(catBoostOptions, pools.Learn, executor);

    TRestorableFastRng64 rand(catBoostOptions.RandomSeed.Get());

    pools.Learn = ShuffleLearnDataIfNeeded(catBoostOptions, pools.Learn, executor, &rand);

    TLabelConverter labelConverter;

    TTrainingDataProviders trainingData = GetTrainingData(
        std::move(pools),
        /* borders */ Nothing(), // borders are already loaded to quantizedFeaturesInfo
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ true,
        outputOptions.AllowWriteFiles(),
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        executor,
        &rand);

    UpdateUndefinedClassNames(catBoostOptions.DataProcessingOptions, &updatedTrainOptionsJson);

    CheckConsistency(trainingData);

    if (outputOptions.NeedSaveBorders()) {
        if (outputOptions.GetTrainDir()) {
            TFsPath dirPath(outputOptions.GetTrainDir());
            if (!dirPath.Exists()) {
                dirPath.MkDirs();
            }
        }
        SaveBordersAndNanModesToFileInMatrixnetFormat(
            outputOptions.CreateOutputBordersFullPath(),
            *trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo());
    }

    modelTrainerHolder->TrainModel(
        TTrainModelInternalOptions(),
        updatedTrainOptionsJson,
        updatedOutputOptions,
        objectiveDescriptor,
        evalMetricDescriptor,
        Nothing(),
        std::move(trainingData),
        labelConverter,
        executor,
        &rand,
        modelPtr,
        evalResultPtrs,
        metricsAndTimeHistory);
}


void TrainModel(
    const NCatboostOptions::TPoolLoadParams& loadOptions,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const NJson::TJsonValue& trainJson
) {
    THPTimer runTimer;
    auto catBoostOptions = NCatboostOptions::LoadOptions(trainJson);

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);

    TProfileInfo profile;

    CB_ENSURE(
        (catBoostOptions.GetTaskType() == ETaskType::CPU) || (loadOptions.TestSetPaths.size() <= 1),
        "Multiple eval sets not supported for GPU"
    );

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(catBoostOptions.SystemOptions.Get().NumThreads.Get() - 1);

    TDataProviders pools = LoadPools(
        loadOptions,
        catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ?
            EObjectsOrder::Ordered : EObjectsOrder::Undefined,
        &executor,
        &profile);

    const auto evalOutputFileName = outputOptions.CreateEvalFullPath();
    TVector<TString> outputColumns;
    if (!evalOutputFileName.empty() && !pools.Test.empty()) {
        outputColumns = outputOptions.GetOutputColumns(pools.Test[0]->MetaInfo.HasTarget);
        for (int testIdx = 0; testIdx < pools.Test.ysize(); ++testIdx) {
            const TDataProvider& testPool = *pools.Test[testIdx];

            CB_ENSURE(
                outputColumns == outputOptions.GetOutputColumns(testPool.MetaInfo.HasTarget),
                "Inconsistent output columns between test sets"
            );

            ValidateColumnOutput(
                outputColumns,
                testPool,
                false,
                loadOptions.CvParams.FoldCount > 0
            );
        }
    }
    TVector<TEvalResult> evalResults(pools.Test.ysize());

    NJson::TJsonValue updatedTrainJson = trainJson;
    UpdateUndefinedClassNames(catBoostOptions.DataProcessingOptions, &updatedTrainJson);

    // create here to possibly load borders
    auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        *pools.Learn->MetaInfo.FeaturesLayout,
        catBoostOptions.DataProcessingOptions->IgnoredFeatures.Get(),
        catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
        /*allowNansInTestOnly*/true,
        outputOptions.AllowWriteFiles()
    );
    if (loadOptions.BordersFile) {
        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
            loadOptions.BordersFile,
            quantizedFeaturesInfo.Get());
    }

    const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
    const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
    const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
    bool needPoolAfterTrain = !evalOutputFileName.empty() || (needFstr && outputOptions.GetFstrType() == EFstrType::LossFunctionChange);
    if (needFstr && outputOptions.GetFstrType() == EFstrType::FeatureImportance && updatedTrainJson.Has("loss_function")) {
        NCatboostOptions::TLossDescription modelLossDescription;
        modelLossDescription.Load(updatedTrainJson["loss_function"]);
        needPoolAfterTrain |= IsGroupwiseMetric(modelLossDescription.LossFunction.Get());
    }
    TrainModel(
        updatedTrainJson,
        outputOptions,
        quantizedFeaturesInfo,
        Nothing(),
        Nothing(),
        needPoolAfterTrain ? pools : std::move(pools),
        "",
        nullptr,
        GetMutablePointers(evalResults),
        nullptr,
        &executor
    );
    auto modelFormat = outputOptions.GetModelFormats()[0];
    const auto fullModelPath = NCatboostOptions::AddExtension(
        modelFormat,
        outputOptions.CreateResultModelFullPath(),
        outputOptions.AddFileFormatExtension());

    TSetLoggingVerbose inThisScope2;
    if (!evalOutputFileName.empty()) {
        TFullModel model = ReadModel(fullModelPath, modelFormat);
        auto visibleLabelsHelper = BuildLabelsHelper<TExternalLabelsHelper>(model);
        if (!loadOptions.CvParams.FoldCount && loadOptions.TestSetPaths.empty() && !outputColumns.empty()) {
            CATBOOST_WARNING_LOG << "No test files, can't output columns\n";
        }
        CATBOOST_INFO_LOG << "Writing test eval to: " << evalOutputFileName << Endl;
        TOFStream fileStream(evalOutputFileName);
        for (int testIdx = 0; testIdx < pools.Test.ysize(); ++testIdx) {
            const TDataProvider& testPool = *pools.Test[testIdx];
            const NCB::TPathWithScheme& testSetPath = testIdx < loadOptions.TestSetPaths.ysize() ? loadOptions.TestSetPaths[testIdx] : NCB::TPathWithScheme();
            OutputEvalResultToFile(
                evalResults[testIdx],
                &executor,
                outputColumns,
                visibleLabelsHelper,
                testPool,
                false,
                &fileStream,
                testSetPath,
                {testIdx, pools.Test.ysize()},
                loadOptions.DsvPoolFormatParams.Format,
                /*writeHeader*/ testIdx < 1);
        }

        if (pools.Test.empty()) {
            CATBOOST_WARNING_LOG << "can't evaluate model (--eval-file) without test set" << Endl;
        }
    } else {
        CATBOOST_INFO_LOG << "Skipping test eval output" << Endl;
    }
    profile.AddOperation("Train model");

    if (catBoostOptions.IsProfile || catBoostOptions.LoggingLevel == ELoggingLevel::Debug) {
        TLogger logger;
        logger.AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TConsoleLoggingBackend(true)));
        TOneInterationLogger oneIterLogger(logger);
        oneIterLogger.OutputProfile(profile.GetProfileResults());
    }

    if (needFstr) {
        TFullModel model = ReadModel(fullModelPath, modelFormat);
        CalcAndOutputFstr(
            model,
            needPoolAfterTrain ? pools.Learn : nullptr,
            &executor,
            &fstrRegularFileName,
            &fstrInternalFileName,
            outputOptions.GetFstrType());
    }

    const TString trainingOptionsFileName = outputOptions.CreateTrainingOptionsFullPath();
    if (!trainingOptionsFileName.empty()) {
        TOFStream trainingOptionsFile(trainingOptionsFileName);
        trainingOptionsFile.Write(NJson::PrettifyJson(ToString(catBoostOptions)));
    }

    CATBOOST_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
}

static void ModelBasedEval(
    const NJson::TJsonValue& trainOptionsJson,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TDataProviders pools,
    NPar::TLocalExecutor* const executor)
{
    CB_ENSURE(pools.Learn != nullptr, "Train data must be provided");

    const ETaskType taskType = NCatboostOptions::GetTaskType(trainOptionsJson);

    CB_ENSURE(taskType == ETaskType::GPU, "Model based eval is not implemented for CPU");

    CB_ENSURE(pools.Test.size() <= 1, "Multiple eval sets not supported for GPU");

    NJson::TJsonValue updatedTrainOptionsJson = trainOptionsJson;

    CB_ENSURE(TTrainerFactory::Has(ETaskType::GPU),
        "Can't load GPU learning library. Module was not compiled or driver is incompatible with package. Please install latest NVDIA driver and check again.");

    THolder<IModelTrainer> modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    if (outputOptions.SaveSnapshot()) {
        UpdateUndefinedRandomSeed(ETaskType::GPU, outputOptions, &updatedTrainOptionsJson, [&](IInputStream* in, TString& params) {
            ::Load(in, params);
        });
    }

    const auto learnFeaturesLayout = pools.Learn->MetaInfo.FeaturesLayout;
    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    catBoostOptions.Load(updatedTrainOptionsJson);

    if (!quantizedFeaturesInfo) {
        quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *learnFeaturesLayout,
            catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get(),
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            /*allowNansInTestOnly*/true,
            outputOptions.AllowWriteFiles()
        );
    }

    for (auto testPoolIdx : xrange(pools.Test.size())) {
        const auto& testPool = *pools.Test[testPoolIdx];
        if (testPool.GetObjectCount() == 0) {
            continue;
        }
        CheckCompatibleForApply(
            *learnFeaturesLayout,
            *testPool.MetaInfo.FeaturesLayout,
            TStringBuilder() << "test dataset #" << testPoolIdx);
    }

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);

    auto learnDataOrder = pools.Learn->ObjectsData->GetOrder();
    if (learnDataOrder == EObjectsOrder::Ordered) {
        catBoostOptions.DataProcessingOptions->HasTimeFlag = true;
    }

    pools.Learn = ReorderByTimestampLearnDataIfNeeded(catBoostOptions, pools.Learn, executor);

    TRestorableFastRng64 rand(catBoostOptions.RandomSeed.Get());

    pools.Learn = ShuffleLearnDataIfNeeded(catBoostOptions, pools.Learn, executor, &rand);

    TLabelConverter labelConverter;

    TTrainingDataProviders trainingData = GetTrainingData(
        std::move(pools),
        /* borders */ Nothing(), // borders are already loaded to quantizedFeaturesInfo
        /*ensureConsecutiveLearnFeaturesDataForCpu*/ true,
        outputOptions.AllowWriteFiles(),
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        executor,
        &rand);

    UpdateUndefinedClassNames(catBoostOptions.DataProcessingOptions, &updatedTrainOptionsJson);

    CheckConsistency(trainingData);

    modelTrainerHolder->ModelBasedEval(
        updatedTrainOptionsJson,
        outputOptions,
        std::move(trainingData),
        labelConverter,
        executor);
}

void ModelBasedEval(
    const NCatboostOptions::TPoolLoadParams& loadOptions,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const NJson::TJsonValue& trainJson
) {
    THPTimer runTimer;
    auto catBoostOptions = NCatboostOptions::LoadOptions(trainJson);

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);

    TProfileInfo profile;

    CB_ENSURE(
        (catBoostOptions.GetTaskType() == ETaskType::CPU) || (loadOptions.TestSetPaths.size() <= 1),
        "Multiple eval sets not supported for GPU"
    );

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(catBoostOptions.SystemOptions.Get().NumThreads.Get() - 1);

    TDataProviders pools = LoadPools(
        loadOptions,
        catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ?
            EObjectsOrder::Ordered : EObjectsOrder::Undefined,
        &executor,
        &profile);

    // create here to possibly load borders
    auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        *pools.Learn->MetaInfo.FeaturesLayout,
        catBoostOptions.DataProcessingOptions->IgnoredFeatures.Get(),
        catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
        /*allowNansInTestOnly*/true,
        outputOptions.AllowWriteFiles()
    );
    if (loadOptions.BordersFile) {
        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
            loadOptions.BordersFile,
            quantizedFeaturesInfo.Get());
    }

    NJson::TJsonValue updatedTrainJson = trainJson;
    UpdateUndefinedClassNames(catBoostOptions.DataProcessingOptions, &updatedTrainJson);

    ModelBasedEval(
        updatedTrainJson,
        outputOptions,
        quantizedFeaturesInfo,
        std::move(pools),
        &executor
    );

    profile.AddOperation("Model based eval");

    if (catBoostOptions.IsProfile || catBoostOptions.LoggingLevel == ELoggingLevel::Debug) {
        TLogger logger;
        logger.AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TConsoleLoggingBackend(true)));
        TOneInterationLogger oneIterLogger(logger);
        oneIterLogger.OutputProfile(profile.GetProfileResults());
    }

    CATBOOST_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
}

void TrainModel(
    const NJson::TJsonValue& plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    NCB::TDataProviders pools, // not rvalue reference because Cython does not support them
    const TString& outputModelPath,
    TFullModel* model,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory
) {
    NJson::TJsonValue trainOptionsJson;
    NJson::TJsonValue outputFilesOptionsJson;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptionsJson, &outputFilesOptionsJson);

    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.Load(outputFilesOptionsJson);

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(
        NCatboostOptions::LoadOptions(trainOptionsJson).SystemOptions.Get().NumThreads.Get() - 1);

    TrainModel(
        trainOptionsJson,
        outputOptions,
        quantizedFeaturesInfo,
        objectiveDescriptor,
        evalMetricDescriptor,
        std::move(pools),
        outputModelPath,
        model,
        evalResultPtrs,
        metricsAndTimeHistory,
        &executor);
}
