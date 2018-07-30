#include "train_model.h"
#include "preprocess.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/helpers/pairs_util.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>

#include <library/grid_creator/binarization.h>
#include <library/json/json_prettifier.h>
#include <util/random/shuffle.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/info.h>


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
    const TVector<TString>& classNames,
    TProfileInfo* profile,
    TTrainPools* pools) {

    NCB::ReadTrainPools(loadOptions, true, threadCount, classNames, profile, pools);

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
    TVector<TVector<TVector<double>>>* testMultiApprox // [test][dim][metric]
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

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    CB_ENSURE(!metrics.empty(), "Eval metric is not defined");
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);
    TErrorTracker overfittingDetectorErrorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);
    TErrorTracker bestModelErrorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();

    ctx->TryLoadProgress();

    if (ctx->OutputOptions.GetMetricPeriod() > 1 && useBestModel && hasTest) {
        MATRIXNET_WARNING_LOG << "Warning: Parameter 'use_best_model' is true, thus evaluation metric is" <<
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

    WriteHistory(
            GetMetricsDescription(metrics),
            ctx->LearnProgress.MetricsAndTimeHistory,
            learnToken,
            testTokens,
            &logger
    );

    AddConsoleLogger(
            learnToken,
            testTokens,
            /*hasTrain=*/true,
            ctx->OutputOptions.GetVerbosePeriod(),
            ctx->Params.BoostingOptions->IterationCount,
            &logger
    );

    const size_t overfittingDetectorMetricIdx =
        ctx->Params.MetricOptions->EvalMetric.IsSet() ? 0 : (metrics.size() - 1);

    TVector<TVector<TVector<double>>> errorsHistory = ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory;
    for (int iter = 0; iter < errorsHistory.ysize(); ++iter) {
        const bool calcMetrics = DivisibleOrLastIteration(iter, errorsHistory.ysize(), ctx->OutputOptions.GetMetricPeriod());
        const int testIdxToLog = errorsHistory[iter].size() - 1;
        const int metricIdxToLog = calcMetrics ? overfittingDetectorMetricIdx : 0;
        overfittingDetectorErrorTracker.AddError(errorsHistory[iter][testIdxToLog][metricIdxToLog], iter);
        if (calcMetrics) {
            bestModelErrorTracker.AddError(errorsHistory[iter][testIdxToLog][metricIdxToLog], iter);
        }
    }

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    if (IsSamplingPerTree(ctx->Params.ObliviousTreeOptions.Get())) {
        ctx->SmallestSplitSideDocs.Create(ctx->LearnProgress.Folds, isPairwiseScoring);
        ctx->PrevTreeLevelStats.Create(
            ctx->LearnProgress.Folds,
            CountNonCtrBuckets(CountSplits(ctx->LearnProgress.FloatFeatures), learnData.AllFeatures.OneHotValues),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth)
        );
    }
    ctx->SampledDocs.Create(
        ctx->LearnProgress.Folds,
        isPairwiseScoring,
        GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)
    ); // TODO(espetrov): create only if sample rate < 1

    for (ui32 iter = ctx->LearnProgress.TreeStruct.ysize(); iter < ctx->Params.BoostingOptions->IterationCount; ++iter) {
        profile.StartNextIteration();

        trainOneIterationFunc(learnData, testDataPtrs, ctx);

        bool calcMetrics = DivisibleOrLastIteration(
            iter,
            ctx->Params.BoostingOptions->IterationCount,
            ctx->OutputOptions.GetMetricPeriod()
        );

        CalcErrors(learnData, testDataPtrs, metrics, calcMetrics, overfittingDetectorMetricIdx, ctx);

        profile.AddOperation("Calc errors");
        if (hasTest) {
            const auto testErrors = ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory.back();
            // Use only (last_test, last_metric) for overfitting detection
            const int testIdxToLog = testErrors.size() - 1;
            const int metricIdxToLog = calcMetrics ? overfittingDetectorMetricIdx : 0;

            overfittingDetectorErrorTracker.AddError(testErrors[testIdxToLog][metricIdxToLog], iter);
            if (calcMetrics) {
                bestModelErrorTracker.AddError(testErrors[testIdxToLog][metricIdxToLog], iter);
                if (useBestModel && iter == static_cast<ui32>(bestModelErrorTracker.GetBestIteration())) {
                    ctx->LearnProgress.BestTestApprox = ctx->LearnProgress.TestApprox[0];
                }
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress.MetricsAndTimeHistory.TimeHistory.push_back({profileResults.PassedTime, profileResults.RemainingTime});

        Log(
            GetMetricsDescription(metrics),
            GetSkipMetricOnTrain(metrics),
            ctx->LearnProgress.MetricsAndTimeHistory.LearnMetricsHistory,
            ctx->LearnProgress.MetricsAndTimeHistory.TestMetricsHistory,
            bestModelErrorTracker.GetBestError(),
            bestModelErrorTracker.GetBestIteration(),
            profileResults,
            learnToken,
            testTokens,
            calcMetrics,
            &logger
        );

        ctx->SaveProgress();

        if (HasInvalidValues(ctx->LearnProgress.LeafValues)) {
            ctx->LearnProgress.LeafValues.pop_back();
            ctx->LearnProgress.TreeStruct.pop_back();
            MATRIXNET_WARNING_LOG << "Training has stopped (degenerate solution on iteration "
                << iter << ", probably too small l2-regularization, try to increase it)" << Endl;
            break;
        }

        if (overfittingDetectorErrorTracker.GetIsNeedStop()) {
            MATRIXNET_NOTICE_LOG << "Stopped by overfitting detector "
                << " (" << overfittingDetectorErrorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            break;
        }
    }

    if (hasTest) {
        (*testMultiApprox) = ctx->LearnProgress.TestApprox;
        if (useBestModel) {
            (*testMultiApprox)[0] = ctx->LearnProgress.BestTestApprox;
        }
    }

    ctx->LearnProgress.Folds.clear();

    if (hasTest) {
        MATRIXNET_NOTICE_LOG << "\n";
        MATRIXNET_NOTICE_LOG << "bestTest = " << bestModelErrorTracker.GetBestError() << "\n";
        MATRIXNET_NOTICE_LOG << "bestIteration = " << bestModelErrorTracker.GetBestIteration() << "\n";
        MATRIXNET_NOTICE_LOG << "\n";
    }

    if (useBestModel && ctx->Params.BoostingOptions->IterationCount > 0) {
        const int itCount = bestModelErrorTracker.GetBestIteration() + 1;
        MATRIXNET_NOTICE_LOG << "Shrink model to first " << itCount << " iterations." << Endl;
        ShrinkModel(itCount, &ctx->LearnProgress);
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
        const TVector<TEvalResult*>& evalResultPtrs
    ) const override {

        auto sortedCatFeatures = pools.Learn->CatFeatures;
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());

        for (const TPool* testPoolPtr : pools.Test) {
            const TPool& testPool = *testPoolPtr;
            if (testPool.Docs.GetDocCount() == 0) {
                continue;
            }
            CB_ENSURE(
                testPool.Docs.GetEffectiveFactorCount() == pools.Learn->Docs.GetEffectiveFactorCount(),
                "train pool factors count == " << pools.Learn->Docs.GetEffectiveFactorCount() << " and test pool factors count == " << testPool.Docs.GetEffectiveFactorCount()
            );
            auto catFeaturesTest = testPool.CatFeatures;
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
            UpdateUndefinedRandomSeed(outputOptions, &updatedJsonParams);
        }
        NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(updatedJsonParams));
        NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

        SetDataDependantDefaults(
            pools.Learn->Docs.GetDocCount(),
            /*testPoolSize*/ GetDocCount(pools.Test),
            /*hasTestLabels*/ pools.Test.size() > 0 && IsConst(pools.Test[0]->Docs.Target),
            pools.Learn->MetaInfo.HasWeights,
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

        auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

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
            MATRIXNET_WARNING_LOG << "No pairs provided for learn dataset. "
                                  << "Trying to generate pairs using dataset labels." << Endl;
            pools.Learn->Pairs.clear();
            GeneratePairLogitPairs(
                    pools.Learn->Docs.QueryId,
                    pools.Learn->Docs.Target,
                    NCatboostOptions::GetMaxPairCount(ctx.Params.LossFunctionDescription),
                    &ctx.Rand,
                    &(pools.Learn->Pairs));
            MATRIXNET_INFO_LOG << "Generated " << pools.Learn->Pairs.size() << " pairs for learn pool." << Endl;
        }

        ApplyPermutation(InvertPermutation(indices), pools.Learn, &ctx.LocalExecutor);
        auto permutationGuard = Finally([&] { ApplyPermutation(indices, pools.Learn, &ctx.LocalExecutor); });


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
                MATRIXNET_INFO_LOG << "Generated " << pairs.size()
                    << " pairs for test pool " <<  testDatasets.size() << "." << Endl;
            }
        }

        // Note:
        // `testDatasets` refers to the vector of `TDataset` (this function only)
        // `testDataPtrs` refers to the vector of `const TDataset*` (everywhere)
        // Both vectors are not empty
        const TDatasetPtrs& testDataPtrs = GetConstPointers(testDatasets);

        bool isMulticlass = IsMultiClassError(lossFunction) || (lossFunction == ELossFunction::Custom &&
            ctx.Params.MetricOptions->EvalMetric.IsSet() &&
            IsMultiClassError(ctx.Params.MetricOptions->EvalMetric->GetLossFunction()));

        if (isMulticlass) {
            int classesCount = GetClassesCount(
                ctx.Params.DataProcessingOptions->ClassesCount,
                ctx.Params.DataProcessingOptions->ClassNames
            );
            ctx.LearnProgress.LabelConverter.Initialize(learnData.Target, classesCount);
            ctx.LearnProgress.ApproxDimension = ctx.LearnProgress.LabelConverter.GetApproxDimension();
        }

        TVector<ELossFunction> metrics = {ctx.Params.LossFunctionDescription->GetLossFunction()};
        if (ctx.Params.MetricOptions->EvalMetric.IsSet()) {
            metrics.push_back(ctx.Params.MetricOptions->EvalMetric->GetLossFunction());
        }
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

        if (pools.Learn->QuantizedFeatures.FloatHistograms.empty() && pools.Learn->QuantizedFeatures.CatFeaturesRemapped.empty()) {
            GenerateBorders(*pools.Learn, &ctx, &ctx.LearnProgress.FloatFeatures);
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
        } else {
            learnData.AllFeatures = pools.Learn->QuantizedFeatures;
            ctx.LearnProgress.FloatFeatures = pools.Learn->FloatFeatures;
            for (size_t testIdx = 0; testIdx < testDataPtrs.size(); ++testIdx) {
                auto& testPool = *pools.Test[testIdx];
                auto& testData = testDatasets[testIdx];
                PrepareAllFeaturesTest(
                    ctx.CatFeatures,
                    ctx.LearnProgress.FloatFeatures,
                    learnData.AllFeatures,
                    /*allowNansOnlyInTest=*/false,
                    /*clearPoolAfterBinarization=*/pools.AllowClearTest,
                    ctx.LocalExecutor,
                    /*select=*/{},
                    &testPool.Docs,
                    &testData.AllFeatures
                );
            }
        }
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
                bool exportRequiresStaticCtrProvider = AnyOf(
                    updatedOutputOptions.GetModelFormats().cbegin(),
                    updatedOutputOptions.GetModelFormats().cend(),
                    [](EModelType format) {
                        return format == EModelType::Python || format == EModelType::CPP;
                    }
                );

                coreModelToFullModelConverter.WithCoreModelFrom(&Model).Do(
                    &Model,
                    exportRequiresStaticCtrProvider
                );
            }
            bool addFileFormatExtension =
                    updatedOutputOptions.GetModelFormats().size() > 1 || !updatedOutputOptions.ResultModelPath.IsSet();
            TString outputFile = updatedOutputOptions.CreateResultModelFullPath();
            if (addFileFormatExtension && outputFile.EndsWith(".bin")) {
                outputFile = outputFile.substr(0, outputFile.length() - 4);
            }
            for (const auto& format : updatedOutputOptions.GetModelFormats()) {
                ExportModel(Model, outputFile, format, "", addFileFormatExtension);
            }
        }
        const TString trainingOptionsFileName = outputOptions.CreateTrainingOptionsFullPath();
        if (!trainingOptionsFileName.empty()) {
            TOFStream trainingOptionsFile(trainingOptionsFileName);
            trainingOptionsFile.Write(NJson::PrettifyJson(ToString(ctx.Params)));
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

        if (catBoostOptions.SystemOptions->IsWorker()) {
            RunWorker(threadCount, catBoostOptions.SystemOptions->NodePort);
            return;
        }

        TProfileInfo profile;

        TTrainPools pools;
        LoadPools(
            loadOptions,
            threadCount,
            catBoostOptions.DataProcessingOptions->ClassNames,
            &profile,
            &pools
        );

        const auto evalOutputFileName = outputOptions.CreateEvalFullPath();
        if (!evalOutputFileName.empty() && !loadOptions.TestSetPaths.empty()) {
            ValidateColumnOutput(outputOptions.GetOutputColumns(), pools.Learn, false, loadOptions.CvParams.FoldCount > 0);
        }

        const auto modelPath = outputOptions.CreateResultModelFullPath();

        TVector<TEvalResult> evalResults(Max(pools.Test.ysize(), 1)); // need at least one evalResult, maybe empty

        this->TrainModel(
            trainJson,
            outputOptions,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(pools, true, evalOutputFileName.empty()),
            nullptr,
            GetMutablePointers(evalResults)
        );

        SetVerboseLogingMode();
        if (!evalOutputFileName.empty()) {
            TFullModel model = ReadModel(modelPath);
            TVisibleLabelsHelper visibleLabelsHelper;
            if (model.ObliviousTrees.ApproxDimension > 1) {  // is multiclass?
                if(model.ModelInfo.has("multiclass_params")) {
                    visibleLabelsHelper.Initialize(model.ModelInfo.at("multiclass_params"));
                } else {
                    visibleLabelsHelper.Initialize(model.ObliviousTrees.ApproxDimension);
                }

            }
            if (!loadOptions.CvParams.FoldCount && loadOptions.TestSetPaths.empty() && !outputOptions.GetOutputColumns().empty()) {
                MATRIXNET_WARNING_LOG << "No test files, can't output columns\n";
            }
            MATRIXNET_INFO_LOG << "Writing test eval to: " << evalOutputFileName << Endl;
            TOFStream fileStream(evalOutputFileName);
            for (int testIdx = 0; testIdx < pools.Test.ysize(); ++testIdx) {
                const TPool& testPool = pools.Test[testIdx];
                const NCB::TPathWithScheme& testSetPath = testIdx < loadOptions.TestSetPaths.ysize() ? loadOptions.TestSetPaths[testIdx] : NCB::TPathWithScheme();
                evalResults[testIdx].OutputToFile(threadCount,
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
                // Make sure to emit header to fileStream
                evalResults[0].OutputToFile(threadCount,
                                            outputOptions.GetOutputColumns(),
                                            visibleLabelsHelper,
                                            TPool(),
                                            false,
                                            &fileStream,
                                            NCB::TPathWithScheme(),
                                            {0, 1},
                                            loadOptions.DsvPoolFormatParams.Format,
                                            /*writeHeader*/ true);
            }
        } else {
            MATRIXNET_INFO_LOG << "Skipping test eval output" << Endl;
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
            TFullModel model = ReadModel(modelPath);
            // no need to pass pool data because we always have LeafWeights stored in model now
            CalcAndOutputFstr(model, nullptr, &fstrRegularFileName, &fstrInternalFileName);
        }

        MATRIXNET_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
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
    const TVector<TEvalResult*>& evalResultPtrs)
{
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

    if (taskType == ETaskType::CPU && catBoostOptions.SystemOptions->IsWorker()) {
        RunWorker(catBoostOptions.SystemOptions->NumThreads, catBoostOptions.SystemOptions->NodePort);
        return;
    }

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "Can't load GPU learning library. Module was not compiled or CUDA version/driver  is incompatible with package");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(trainOptions, outputOptions, objectiveDescriptor, evalMetricDescriptor, pools, modelPtr, evalResultPtrs);
}

/// Used by cross validation, hence one test dataset.
void TrainOneIteration(const TDataset& trainData, const TDataset* testDataPtr, TLearnContext* ctx) {
    SetLogingLevel(ctx->Params.LoggingLevel);

    TTrainOneIterationFunc trainFunc;
    ELossFunction lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();

    Y_VERIFY(testDataPtr);
    GetOneIterationFunc(lossFunction)(trainData, {testDataPtr}, ctx);
}
