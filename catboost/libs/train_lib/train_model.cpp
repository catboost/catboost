#include "train_model.h"
#include "preprocess.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/pairs/util.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/labels/label_helper_builder.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>

#include <library/grid_creator/binarization.h>
#include <library/json/json_prettifier.h>

#include <util/generic/scope.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>
#include <util/system/hp_timer.h>
#include <util/system/info.h>

using NCB::TEvalResult;
using NCB::ValidateColumnOutput;

namespace {
void ShrinkModel(int itCount, TLearnProgress* progress) {
    progress->LeafValues.resize(itCount);
    progress->TreeStruct.resize(itCount);
}

static int GetThreadCount(const NCatboostOptions::TCatBoostOptions& options) {
    return Min<int>(options.SystemOptions->NumThreads, (int)NSystemInfo::CachedNumberOfCpus());
}


static void LoadPools(
    const NCatboostOptions::TPoolLoadParams& loadOptions,
    int threadCount,
    NCB::TTargetConverter* const trainTargetConverter,
    TProfileInfo* profile,
    TTrainPools* pools) {

    NCB::ReadTrainPools(loadOptions, true, threadCount, trainTargetConverter, profile, pools);

    const auto& cvParams = loadOptions.CvParams;
    if (cvParams.FoldCount != 0) {
        CB_ENSURE(loadOptions.TestSetPaths.empty(), "Test files are not supported in cross-validation mode");
        Y_VERIFY(cvParams.FoldIdx != -1);

        pools->Test.resize(1);
        BuildCvPools(
            cvParams.FoldIdx,
            cvParams.FoldCount,
            cvParams.Inverted,
            cvParams.RandSeed,
            threadCount,
            &pools->Learn,
            &(pools->Test[0])
        );
        profile->AddOperation("Build cv pools");
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

static void Train(
    const TDataset& learnData,
    const TDatasetPtrs& testDataPtrs,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* testMultiApprox // [test][dim][docIdx]
) {
    TProfileInfo& profile = ctx->Profile;

    const int approxDimension = ctx->LearnProgress.ApproxDimension;
    const bool hasTest = GetSampleCount(testDataPtrs) > 0;
    auto trainOneIterationFunc = GetOneIterationFunc(ctx->Params.LossFunctionDescription->GetLossFunction());
    TVector<THolder<IMetric>> metrics = CreateMetrics(
        ctx->Params.LossFunctionDescription,
        ctx->Params.MetricOptions,
        ctx->EvalMetricDescriptor,
        approxDimension
    );
    CheckMetrics(metrics, ctx->Params.LossFunctionDescription.Get().GetLossFunction());
    if (!ctx->Params.SystemOptions->IsSingleHost()) {
        if (!AllOf(metrics, [](const auto& metric) { return metric->IsAdditiveMetric(); })) {
            CATBOOST_WARNING_LOG << "In distributed training, non-additive metrics are not evaluated on train dataset" << Endl;
        }
    }

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    CB_ENSURE(!metrics.empty(), "Eval metric is not defined");
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);
    TErrorTracker errorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);
    TErrorTracker bestModelMinTreesTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();

    ctx->TryLoadProgress();

    if (ctx->OutputOptions.GetMetricPeriod() > 1 && errorTracker.IsActive() && hasTest) {
        CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is" <<
            "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
    }

    auto learnToken = GetTrainModelLearnToken();
    auto testTokens = GetTrainModelTestTokens(testDataPtrs.ysize());
    TLogger logger;

    if (ctx->OutputOptions.AllowWriteFiles()) {
        InitializeFileLoggers(
                ctx->Params,
                ctx->Files,
                GetConstPointers(metrics),
                learnToken,
                testTokens,
                ctx->OutputOptions.GetMetricPeriod(),
                &logger);
    }

    // Use only (last_test, first_metric) for best iteration and overfitting detection
    // In case of changing the order it should be changed in GPU mode also.
    const size_t errorTrackerMetricIdx = 0;

    const TVector<TVector<THashMap<TString, double>>>& testMetricsHistory =
        ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory;
    const TVector<TTimeInfo>& timeHistory = ctx->LearnProgress.MetricsAndTimeHistory.TimeHistory;

    for (int iter : xrange(ctx->LearnProgress.TreeStruct.ysize())) {
        bool calcAllMetrics = DivisibleOrLastIteration(
            iter,
            ctx->Params.BoostingOptions->IterationCount,
            ctx->OutputOptions.GetMetricPeriod()
        );
        const bool calcErrorTrackerMetric = calcAllMetrics || errorTracker.IsActive();
        if (iter < testMetricsHistory.ysize() && calcErrorTrackerMetric) {
            const TString& errorTrackerMetricDescription = metrics[errorTrackerMetricIdx]->GetDescription();
            const double error = testMetricsHistory[iter].back().at(errorTrackerMetricDescription);
            errorTracker.AddError(error, iter);
            if (useBestModel && iter + 1 >= ctx->OutputOptions.BestModelMinTrees) {
                bestModelMinTreesTracker.AddError(error, iter);
            }
        }

        Log(
            iter,
            GetMetricsDescription(metrics),
            ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory,
            testMetricsHistory,
            errorTracker.GetBestError(),
            errorTracker.GetBestIteration(),
            TProfileResults(timeHistory[iter].PassedTime, timeHistory[iter].RemainingTime),
            learnToken,
            testTokens,
            calcAllMetrics,
            &logger
        );
    }

    AddConsoleLogger(
            learnToken,
            testTokens,
            /*hasTrain=*/true,
            ctx->OutputOptions.GetVerbosePeriod(),
            ctx->Params.BoostingOptions->IterationCount,
            &logger
    );

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    const int defaultCalcStatsObjBlockSize = static_cast<int>(ctx->Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);
    if (IsSamplingPerTree(ctx->Params.ObliviousTreeOptions.Get())) {
        ctx->SmallestSplitSideDocs.Create(ctx->LearnProgress.Folds, isPairwiseScoring, defaultCalcStatsObjBlockSize);
        ctx->PrevTreeLevelStats.Create(
            ctx->LearnProgress.Folds,
            CountNonCtrBuckets(CountSplits(ctx->LearnProgress.FloatFeatures), learnData.AllFeatures.OneHotValues),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth)
        );
    }
    ctx->SampledDocs.Create(
        ctx->LearnProgress.Folds,
        isPairwiseScoring,
        defaultCalcStatsObjBlockSize,
        GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)
    ); // TODO(espetrov): create only if sample rate < 1

    THPTimer timer;
    for (ui32 iter = ctx->LearnProgress.TreeStruct.ysize(); iter < ctx->Params.BoostingOptions->IterationCount; ++iter) {
        if (errorTracker.GetIsNeedStop()) {
            CATBOOST_NOTICE_LOG << "Stopped by overfitting detector "
                << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            break;
        }

        profile.StartNextIteration();

        if (timer.Passed() > ctx->OutputOptions.GetSnapshotSaveInterval()) {
            profile.AddOperation("Save snapshot");
            ctx->SaveProgress();
            timer.Reset();
        }

        trainOneIterationFunc(learnData, testDataPtrs, ctx);

        bool calcAllMetrics = DivisibleOrLastIteration(
            iter,
            ctx->Params.BoostingOptions->IterationCount,
            ctx->OutputOptions.GetMetricPeriod()
        );
        const bool calcErrorTrackerMetric = calcAllMetrics || errorTracker.IsActive();

        CalcErrors(learnData, testDataPtrs, metrics, calcAllMetrics, calcErrorTrackerMetric, ctx);

        profile.AddOperation("Calc errors");
        if (hasTest && calcErrorTrackerMetric) {
            const auto testErrors = ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.back();
            const TString& errorTrackerMetricDescription = metrics[errorTrackerMetricIdx]->GetDescription();
            const double error = testErrors.back().at(errorTrackerMetricDescription);
            errorTracker.AddError(error, iter);
            if (useBestModel && iter == static_cast<ui32>(errorTracker.GetBestIteration())) {
                ctx->LearnProgress.BestTestApprox = ctx->LearnProgress.TestApprox.back();
            }
            if (useBestModel && static_cast<int>(iter + 1) >= ctx->OutputOptions.BestModelMinTrees) {
                bestModelMinTreesTracker.AddError(error, iter);
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress.MetricsAndTimeHistory.TimeHistory.push_back({profileResults.PassedTime, profileResults.RemainingTime});

        Log(
            iter,
            GetMetricsDescription(metrics),
            ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory,
            ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory,
            errorTracker.GetBestError(),
            errorTracker.GetBestIteration(),
            profileResults,
            learnToken,
            testTokens,
            calcAllMetrics,
            &logger
        );

        if (HasInvalidValues(ctx->LearnProgress.LeafValues)) {
            ctx->LearnProgress.LeafValues.pop_back();
            ctx->LearnProgress.TreeStruct.pop_back();
            CATBOOST_WARNING_LOG << "Training has stopped (degenerate solution on iteration "
                << iter << ", probably too small l2-regularization, try to increase it)" << Endl;
            break;
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

    if (hasTest) {
        CATBOOST_NOTICE_LOG << "\n";
        CATBOOST_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << "\n";
        CATBOOST_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << "\n";
        CATBOOST_NOTICE_LOG << "\n";
    }

    if (useBestModel && ctx->Params.BoostingOptions->IterationCount > 0) {
        const int bestModelIterations = bestModelMinTreesTracker.GetBestIteration() + 1;
        if (0 < bestModelIterations && bestModelIterations < static_cast<int>(ctx->Params.BoostingOptions->IterationCount)) {
            CATBOOST_NOTICE_LOG << "Shrink model to first " << bestModelIterations << " iterations.";
            if (errorTracker.GetBestIteration() + 1 < ctx->OutputOptions.BestModelMinTrees) {
                CATBOOST_NOTICE_LOG << " (min iterations for best model = " << ctx->OutputOptions.BestModelMinTrees << ")";
            }
            CATBOOST_NOTICE_LOG << Endl;
            ShrinkModel(bestModelIterations, &ctx->LearnProgress);
        }
    }
}

}

class TCPUModelTrainer : public IModelTrainer {

    void TrainModel(
        const NJson::TJsonValue& jsonParams,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TClearablePoolPtrs& pools,
        TFullModel* modelPtr,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
    ) const override {
        CB_ENSURE(pools.Learn != nullptr, "Train data must be provided");

        // TODO(akhropov): cast will be removed after switch to new Pool format. MLTOOLS-140.
        auto sortedCatFeatures = ToUnsigned(pools.Learn->CatFeatures);
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());

        for (const TPool* testPoolPtr : pools.Test) {
            const TPool& testPool = *testPoolPtr;
            if (testPool.Docs.GetDocCount() == 0) {
                continue;
            }
            CB_ENSURE(
                testPool.GetFactorCount() == pools.Learn->GetFactorCount(),
                "train pool factors count == " << pools.Learn->GetFactorCount() << " and test pool factors count == " << testPool.GetFactorCount()
            );

            // TODO(akhropov): cast will be removed after switch to new Pool format. MLTOOLS-140.
            auto catFeaturesTest = ToUnsigned(testPool.CatFeatures);
            Sort(catFeaturesTest.begin(), catFeaturesTest.end());
            CB_ENSURE(sortedCatFeatures == catFeaturesTest, "Cat features in train and test should be the same.");
        }

        if ((modelPtr == nullptr) == outputOptions.GetResultModelFilename().empty()) {
            if (modelPtr == nullptr) {
                ythrow TCatboostException() << "Both modelPtr == nullptr and outputModelPath empty";
            } else {
                ythrow TCatboostException() << "Both modelPtr != nullptr and outputModelPath non empty";
            }
        }

        const int featureCount = pools.Learn->GetFactorCount();

        NJson::TJsonValue updatedJsonParams = jsonParams;
        if (outputOptions.SaveSnapshot()) {
            UpdateUndefinedRandomSeed(ETaskType::CPU, outputOptions, &updatedJsonParams, [&](IInputStream* in, TString& params) {
                TRestorableFastRng64 unusedRng(0);
                ::LoadMany(in, unusedRng, params);
            });
        }
        NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(updatedJsonParams));
        NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

        SetDataDependantDefaults(
            pools.Learn->Docs.GetDocCount(),
            /*testPoolSize*/ GetDocCount(pools.Test),
            /*hasTestLabels*/ pools.Test.size() > 0 && IsConst(pools.Test[0]->Docs.Target),
            !pools.Learn->IsTrivialWeights(),
            &updatedOutputOptions.UseBestModel,
            &updatedParams
        );

        TLearnContext ctx(
            updatedParams,
            objectiveDescriptor,
            evalMetricDescriptor,
            updatedOutputOptions,
            featureCount,
            sortedCatFeatures,
            pools.Learn->FeatureId
        );
        SetLogingLevel(ctx.Params.LoggingLevel);

        Y_DEFER {
            SetSilentLogingMode();
        };

        TVector<ui64> indices(pools.Learn->Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);

        ui64 minTimestamp = *MinElement(pools.Learn->Docs.Timestamp.begin(), pools.Learn->Docs.Timestamp.end());
        ui64 maxTimestamp = *MaxElement(pools.Learn->Docs.Timestamp.begin(), pools.Learn->Docs.Timestamp.end());
        if (minTimestamp != maxTimestamp) {
            indices = CreateOrderByKey(pools.Learn->Docs.Timestamp);
            ctx.Params.DataProcessingOptions->HasTimeFlag = true;
        }

        if (!ctx.Params.DataProcessingOptions->HasTimeFlag) {
            Shuffle(pools.Learn->Docs.QueryId, ctx.Rand, &indices);
        }

        ELossFunction lossFunction = ctx.Params.LossFunctionDescription.Get().GetLossFunction();
        if (IsPairLogit(lossFunction) && pools.Learn->Pairs.empty()) {
            CB_ENSURE(
                    !pools.Learn->Docs.Target.empty(),
                    "Pool labels are not provided. Cannot generate pairs."
            );
            CATBOOST_WARNING_LOG << "No pairs provided for learn dataset. "
                                  << "Trying to generate pairs using dataset labels." << Endl;
            pools.Learn->Pairs.clear();
            GeneratePairLogitPairs(
                    pools.Learn->Docs.QueryId,
                    pools.Learn->Docs.Target,
                    NCatboostOptions::GetMaxPairCount(ctx.Params.LossFunctionDescription),
                    &ctx.Rand,
                    &(pools.Learn->Pairs));
            CATBOOST_INFO_LOG << "Generated " << pools.Learn->Pairs.size() << " pairs for learn pool." << Endl;
        }

        ApplyPermutation(InvertPermutation(indices), pools.Learn, &ctx.LocalExecutor);
        Y_DEFER {
            ApplyPermutation(indices, pools.Learn, &ctx.LocalExecutor);
        };

        TDataset learnData = BuildDataset(*pools.Learn);

        TVector<TDataset> testDatasets;
        for (const TPool* testPoolPtr : pools.Test) {
            testDatasets.push_back(BuildDataset(*testPoolPtr));
            auto& pairs = testDatasets.back().Pairs;
            if (IsPairLogit(lossFunction) && pairs.empty()) {
                GeneratePairLogitPairs(
                        testDatasets.back().QueryId,
                        testDatasets.back().Target,
                        NCatboostOptions::GetMaxPairCount(ctx.Params.LossFunctionDescription),
                        &ctx.Rand,
                        &pairs);
                CATBOOST_INFO_LOG << "Generated " << pairs.size()
                    << " pairs for test pool " <<  testDatasets.size() << "." << Endl;
            }
        }

        // Note:
        // `testDatasets` refers to the vector of `TDataset` (this function only)
        // `testDataPtrs` refers to the vector of `const TDataset*` (everywhere)
        // Both vectors are not empty
        const TDatasetPtrs& testDataPtrs = GetConstPointers(testDatasets);

        const bool isMulticlass = IsMultiClass(lossFunction, ctx.Params.MetricOptions);

        if (isMulticlass) {
            int classesCount = GetClassesCount(
                ctx.Params.DataProcessingOptions->ClassesCount,
                ctx.Params.DataProcessingOptions->ClassNames
            );
            ctx.LearnProgress.LabelConverter.Initialize(learnData.Target, classesCount);
            ctx.LearnProgress.ApproxDimension = ctx.LearnProgress.LabelConverter.GetApproxDimension();
        }

        TVector<ELossFunction> metrics;
        if (ctx.Params.MetricOptions->EvalMetric.IsSet()) {
            metrics.push_back(ctx.Params.MetricOptions->EvalMetric->GetLossFunction());
        }
        metrics.push_back(ctx.Params.LossFunctionDescription->GetLossFunction());
        for (const auto& metric : ctx.Params.MetricOptions->CustomMetrics.Get()) {
            metrics.push_back(metric.GetLossFunction());
        }
        bool hasQuerywiseMetric = false;
        for (const auto& metric : metrics) {
            if (IsQuerywiseError(metric)) {
                hasQuerywiseMetric = true;
            }
        }
        if (hasQuerywiseMetric) {
            CB_ENSURE(HaveGoodQueryIds(learnData), "Query ids not provided for querywise metric.");
            CB_ENSURE(HaveGoodQueryIds(testDataPtrs), "Query ids not provided for querywise metric.");
        }

        UpdateQueryInfo(&learnData);
        for (TDataset& testData : testDatasets) {
            UpdateQueryInfo(&testData);
        }

        const TVector<float>& classWeights = ctx.Params.DataProcessingOptions->ClassWeights;
        const auto& labelConverter = ctx.LearnProgress.LabelConverter;
        Preprocess(ctx.Params.LossFunctionDescription, classWeights, labelConverter, learnData);
        CheckLearnConsistency(ctx.Params.LossFunctionDescription, ctx.Params.DataProcessingOptions->AllowConstLabel.Get(), learnData);
        for (TDataset& testData : testDatasets) {
            Preprocess(ctx.Params.LossFunctionDescription, classWeights, labelConverter, testData);
            CheckTestConsistency(ctx.Params.LossFunctionDescription, learnData, testData);
        }

        ctx.OutputMeta();

        const auto& catFeatureParams = ctx.Params.CatFeatureParams.Get();

        if (!pools.Learn->IsQuantized()) {
            GenerateBorders(*pools.Learn, &ctx, &ctx.LearnProgress.FloatFeatures);
        } else {
            ctx.LearnProgress.FloatFeatures = pools.Learn->FloatFeatures;
        }
        QuantizeTrainPools(
            pools,
            ctx.LearnProgress.FloatFeatures,
            Nothing(),
            ctx.Params.DataProcessingOptions->IgnoredFeatures,
            catFeatureParams.OneHotMaxSize,
            ctx.LocalExecutor,
            &learnData,
            &testDatasets
        );
        ctx.InitContext(learnData, testDataPtrs);

        ctx.LearnProgress.CatFeatures.resize(sortedCatFeatures.size());
        for (size_t i = 0; i < sortedCatFeatures.size(); ++i) {
            auto& catFeature = ctx.LearnProgress.CatFeatures[i];
            catFeature.FeatureIndex = i;
            catFeature.FlatFeatureIndex = sortedCatFeatures[i];
            if (catFeature.FlatFeatureIndex < pools.Learn->FeatureId.ysize()) {
                catFeature.FeatureId = pools.Learn->FeatureId[catFeature.FlatFeatureIndex];
            }
        }

        DumpMemUsage("Before start train");

        const auto& systemOptions = ctx.Params.SystemOptions;
        if (!systemOptions->IsSingleHost()) { // send target, weights, baseline (if present), binarized features to workers and ask them to create plain folds
            InitializeMaster(&ctx);
            CB_ENSURE(IsPlainMode(ctx.Params.BoostingOptions->BoostingType), "Distributed training requires plain boosting");
            CB_ENSURE(pools.Learn->CatFeatures.empty(), "Distributed training requires all numeric data");
            MapBuildPlainFold(learnData, &ctx);
        }
        TVector<TVector<double>> oneRawValues(ctx.LearnProgress.ApproxDimension);
        TVector<TVector<TVector<double>>> rawValues(testDataPtrs.size(), oneRawValues);

        Train(learnData, testDataPtrs, &ctx, &rawValues);

        for (int testIdx = 0; testIdx < testDataPtrs.ysize(); ++testIdx) {
            evalResultPtrs[testIdx]->SetRawValuesByMove(rawValues[testIdx]);
        }
        if (testDataPtrs.empty() && evalResultPtrs.ysize() > 0) {
            // need at least one evalResult, maybe empty
            evalResultPtrs[0]->SetRawValuesByMove(oneRawValues);
        }

        TObliviousTrees obliviousTrees;
        THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
        {
            TObliviousTreeBuilder builder(ctx.LearnProgress.FloatFeatures, ctx.LearnProgress.CatFeatures, ctx.LearnProgress.ApproxDimension);
            for (size_t treeId = 0; treeId < ctx.LearnProgress.TreeStruct.size(); ++treeId) {
                TVector<TModelSplit> modelSplits;
                for (const auto& split : ctx.LearnProgress.TreeStruct[treeId].Splits) {
                    auto modelSplit = split.GetModelSplit(ctx, learnData);
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

        NCB::TCoreModelToFullModelConverter coreModelToFullModelConverter(
            (ui32)GetThreadCount(ctx.Params),
            ctx.OutputOptions.GetFinalCtrComputationMode(),
            ParseMemorySizeDescription(ctx.Params.SystemOptions->CpuUsedRamLimit),
            ctx.Params.CatFeatureParams->CtrLeafCountLimit,
            ctx.Params.CatFeatureParams->StoreAllSimpleCtrs,
            catFeatureParams
        );

        TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
        datasetDataForFinalCtrs.LearnData = &learnData;
        datasetDataForFinalCtrs.TestDataPtrs = &testDataPtrs;
        datasetDataForFinalCtrs.LearnPermutation = &ctx.LearnProgress.AveragingFold.LearnPermutation;
        datasetDataForFinalCtrs.Targets = &pools.Learn->Docs.Target;
        datasetDataForFinalCtrs.LearnTargetClass = &ctx.LearnProgress.AveragingFold.LearnTargetClass;
        datasetDataForFinalCtrs.TargetClassesCount = &ctx.LearnProgress.AveragingFold.TargetClassesCount;

        coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
            datasetDataForFinalCtrs,
            featureCombinationToProjectionMap
        );

        if (modelPtr) {
            modelPtr->ObliviousTrees = std::move(obliviousTrees);
            modelPtr->ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            CB_ENSURE(isMulticlass == ctx.LearnProgress.LabelConverter.IsInitialized(),
                      "LabelConverter must be initialized ONLY for multiclass problem");
            if (isMulticlass) {
                modelPtr->ModelInfo["multiclass_params"] = ctx.LearnProgress.LabelConverter.SerializeMulticlassParams(
                    ctx.Params.DataProcessingOptions->ClassesCount,
                    ctx.Params.DataProcessingOptions->ClassNames
                );;
            }
            for (const auto& keyValue: ctx.Params.Metadata.Get().GetMap()) {
                modelPtr->ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
            coreModelToFullModelConverter.WithCoreModelFrom(modelPtr).Do(modelPtr, true);
        } else {
            TFullModel Model;
            Model.ObliviousTrees = std::move(obliviousTrees);
            Model.ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            CB_ENSURE(isMulticlass == ctx.LearnProgress.LabelConverter.IsInitialized(),
                      "LabelConverter must be initialized ONLY for multiclass problem");
            if (isMulticlass) {
                Model.ModelInfo["multiclass_params"] = ctx.LearnProgress.LabelConverter.SerializeMulticlassParams(
                    ctx.Params.DataProcessingOptions->ClassesCount,
                    ctx.Params.DataProcessingOptions->ClassNames
                );;
            }
            for (const auto& keyValue: ctx.Params.Metadata.Get().GetMap()) {
                Model.ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
            if (ctx.OutputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) {
                coreModelToFullModelConverter.WithCoreModelFrom(&Model).Do(
                    &Model,
                    ctx.OutputOptions.ExportRequiresStaticCtrProvider()
                );
            }
            TString outputFile = ctx.OutputOptions.CreateResultModelFullPath();
            for (const auto& format : ctx.OutputOptions.GetModelFormats()) {
                ExportModel(Model, outputFile, format, "", ctx.OutputOptions.AddFileFormatExtension(), &pools.Learn->FeatureId, &pools.Learn->CatFeaturesHashToString);
            }
        }
        const TString trainingOptionsFileName = ctx.OutputOptions.CreateTrainingOptionsFullPath();
        if (!trainingOptionsFileName.empty()) {
            TOFStream trainingOptionsFile(trainingOptionsFileName);
            trainingOptionsFile.Write(NJson::PrettifyJson(ToString(ctx.Params)));
        }

        if (metricsAndTimeHistory) {
            *metricsAndTimeHistory = ctx.LearnProgress.MetricsAndTimeHistory;
        }
    }

    void TrainModel(
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const NJson::TJsonValue& trainJson
    ) const override {
        THPTimer runTimer;
        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::CPU);
        catBoostOptions.Load(trainJson);

        int threadCount = GetThreadCount(catBoostOptions);

        TProfileInfo profile;
        TPool learnPool;
        TVector<TPool> testPools;

        auto targetConverter = NCB::MakeTargetConverter(catBoostOptions);

        TTrainPools pools;
        LoadPools(
            loadOptions,
            threadCount,
            &targetConverter,
            &profile,
            &pools
        );

        const auto evalOutputFileName = outputOptions.CreateEvalFullPath();
        if (!evalOutputFileName.empty() && !loadOptions.TestSetPaths.empty()) {
            ValidateColumnOutput(outputOptions.GetOutputColumns(), pools.Learn, false, loadOptions.CvParams.FoldCount > 0);
        }
        TVector<TEvalResult> evalResults(Max(pools.Test.ysize(), 1)); // need at least one evalResult, maybe empty

        NJson::TJsonValue updatedTrainJson = trainJson;
        UpdateUndefinedClassNames(catBoostOptions.DataProcessingOptions, &updatedTrainJson);

        this->TrainModel(
            updatedTrainJson,
            outputOptions,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(pools, true, evalOutputFileName.empty()),
            nullptr,
            GetMutablePointers(evalResults),
            nullptr
        );
        auto modelPath = outputOptions.CreateResultModelFullPath();
        auto modelFormat = outputOptions.GetModelFormats()[0];
        if (outputOptions.AddFileFormatExtension()) {
            NCatboostOptions::AddExtension(NCatboostOptions::GetModelExtensionFromType(modelFormat), &modelPath);
        }

        SetVerboseLogingMode();
        if (!evalOutputFileName.empty()) {
            TFullModel model = ReadModel(modelPath, modelFormat);
            auto visibleLabelsHelper = BuildLabelsHelper<TExternalLabelsHelper>(model);
            if (!loadOptions.CvParams.FoldCount && loadOptions.TestSetPaths.empty() && !outputOptions.GetOutputColumns().empty()) {
                CATBOOST_WARNING_LOG << "No test files, can't output columns\n";
            }
            CATBOOST_INFO_LOG << "Writing test eval to: " << evalOutputFileName << Endl;
            TOFStream fileStream(evalOutputFileName);
            for (int testIdx = 0; testIdx < pools.Test.ysize(); ++testIdx) {
                const TPool& testPool = pools.Test[testIdx];
                const NCB::TPathWithScheme& testSetPath = testIdx < loadOptions.TestSetPaths.ysize() ? loadOptions.TestSetPaths[testIdx] : NCB::TPathWithScheme();
                OutputEvalResultToFile(
                    evalResults[testIdx],
                    threadCount,
                    outputOptions.GetOutputColumns(),
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

        const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
        const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
        const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
        if (needFstr) {
            TFullModel model = ReadModel(modelPath, modelFormat);
            // no need to pass pool data because we always have LeafWeights stored in model now
            CalcAndOutputFstr(model, nullptr, &fstrRegularFileName, &fstrInternalFileName);
        }

        CATBOOST_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
        SetSilentLogingMode();
    }
};

TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ETaskType::CPU);

void TrainModel(const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TClearablePoolPtrs& pools,
    const TString& outputModelPath,
    TFullModel* modelPtr,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory)
{
    CB_ENSURE(pools.Test.size() == evalResultPtrs.size());

    THolder<IModelTrainer> modelTrainerHolder;
    NJson::TJsonValue trainOptions;
    NJson::TJsonValue outputFilesOptionsJson;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptions, &outputFilesOptionsJson);
    const ETaskType taskType = NCatboostOptions::GetTaskType(trainOptions);

    NCatboostOptions::TOutputFilesOptions outputOptions(taskType);
    outputFilesOptionsJson["result_model_file"] = outputModelPath;
    outputOptions.Load(outputFilesOptionsJson);

    NCatboostOptions::TCatBoostOptions catBoostOptions(taskType);
    catBoostOptions.Load(trainOptions);

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "Can't load GPU learning library. Module was not compiled or driver  is incompatible with package. Please install latest NVDIA driver and check again");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(trainOptions, outputOptions, objectiveDescriptor, evalMetricDescriptor, pools, modelPtr, evalResultPtrs, metricsAndTimeHistory);
}

/// Used by cross validation, hence one test dataset.
void TrainOneIteration(const TDataset& trainData, const TDataset* testDataPtr, TLearnContext* ctx) {
    SetLogingLevel(ctx->Params.LoggingLevel);

    TTrainOneIterationFunc trainFunc;
    ELossFunction lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();

    Y_VERIFY(testDataPtr);
    GetOneIterationFunc(lossFunction)(trainData, {testDataPtr}, ctx);
}
