#include "train_model.h"

#include "preprocess.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/model/model_build_helper.h>
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

#include <library/grid_creator/binarization.h>

#include <util/random/shuffle.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/info.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>


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
    TPool* learnPool,
    TVector<TPool>* testPools) {

    loadOptions.Validate();

    const bool verbose = false;
    if (loadOptions.LearnSetPath.Inited()) {
        NCB::ReadPool(loadOptions.LearnSetPath,
                      loadOptions.PairsFilePath,
                      loadOptions.DsvPoolFormatParams,
                      loadOptions.IgnoredFeatures,
                      threadCount,
                      verbose,
                      classNames,
                      learnPool);

        profile->AddOperation("Build learn pool");
    }

    for (int testIdx = 0; testIdx < loadOptions.TestSetPaths.ysize(); ++testIdx) {
        const NCB::TPathWithScheme& testSetPath = loadOptions.TestSetPaths[testIdx];
        const NCB::TPathWithScheme& testPairsFilePath =
                testIdx == 0 ? loadOptions.TestPairsFilePath : NCB::TPathWithScheme();

        TPool testPool;
        NCB::ReadPool(testSetPath,
                      testPairsFilePath,
                      loadOptions.DsvPoolFormatParams,
                      loadOptions.IgnoredFeatures,
                      threadCount,
                      verbose,
                      classNames,
                      &testPool);
        testPools->push_back(std::move(testPool));
        if (testIdx + 1 == loadOptions.TestSetPaths.ysize()) {
            profile->AddOperation("Build test pool");
        }
    }

    const auto& cvParams = loadOptions.CvParams;
    if (cvParams.FoldCount != 0) {
        CB_ENSURE(loadOptions.TestSetPaths.empty(), "Test files are not supported in cross-validation mode");
        Y_VERIFY(cvParams.FoldIdx != -1);

        testPools->resize(1);
        BuildCvPools(
            cvParams.FoldIdx,
            cvParams.FoldCount,
            cvParams.Inverted,
            cvParams.RandSeed,
            threadCount,
            learnPool,
            &(*testPools)[0]
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
        TPool& learnPool,
        bool allowClearPool,
        const TVector<const TPool*>& testPoolPtrs,
        TFullModel* modelPtr,
        const TVector<TEvalResult*>& evalResultPtrs
    ) const override {

        auto sortedCatFeatures = learnPool.CatFeatures;
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());

        for (const TPool* testPoolPtr : testPoolPtrs) {
            const TPool& testPool = *testPoolPtr;
            if (testPool.Docs.GetDocCount() == 0) {
                continue;
            }
            CB_ENSURE(
                testPool.Docs.GetEffectiveFactorCount() == learnPool.Docs.GetEffectiveFactorCount(),
                "train pool factors count == " << learnPool.Docs.GetEffectiveFactorCount() << " and test pool factors count == " << testPool.Docs.GetEffectiveFactorCount()
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
        const int featureCount = learnPool.Docs.GetEffectiveFactorCount();

        NJson::TJsonValue updatedJsonParams = jsonParams;
        if (outputOptions.SaveSnapshot()) {
            UpdateUndefinedRandomSeed(outputOptions, &updatedJsonParams);
        }
        NCatboostOptions::TCatBoostOptions updatedParams(NCatboostOptions::LoadOptions(updatedJsonParams));
        NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

        SetDataDependantDefaults(
            learnPool.Docs.GetDocCount(),
            /*testPoolSize*/ GetDocCount(testPoolPtrs),
            /*hasTestLabels*/ testPoolPtrs.size() > 0 && IsConst(testPoolPtrs[0]->Docs.Target),
            learnPool.MetaInfo.HasWeights,
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
            learnPool.FeatureId
        );
        SetLogingLevel(ctx.Params.LoggingLevel);

        auto loggingGuard = Finally([&] { SetSilentLogingMode(); });

        TVector<ui64> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);

        ui64 minTimestamp = *MinElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());
        ui64 maxTimestamp = *MaxElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());
        if (minTimestamp != maxTimestamp) {
            indices = CreateOrderByKey(learnPool.Docs.Timestamp);
            ctx.Params.DataProcessingOptions->HasTimeFlag = true;
        }

        if (!ctx.Params.DataProcessingOptions->HasTimeFlag) {
            Shuffle(learnPool.Docs.QueryId, ctx.Rand, &indices);
        }

        ApplyPermutation(InvertPermutation(indices), &learnPool, &ctx.LocalExecutor);
        auto permutationGuard = Finally([&] { ApplyPermutation(indices, &learnPool, &ctx.LocalExecutor); });

        ELossFunction lossFunction = ctx.Params.LossFunctionDescription.Get().GetLossFunction();
        TDataset learnData = BuildDataset(learnPool);

        TVector<TDataset> testDatasets;
        for (const TPool* testPoolPtr : testPoolPtrs) {
            testDatasets.push_back(BuildDataset(*testPoolPtr));
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

        GenerateBorders(learnPool, &ctx, &ctx.LearnProgress.FloatFeatures);

        const auto& catFeatureParams = ctx.Params.CatFeatureParams.Get();

        PrepareAllFeaturesLearn(
            ctx.CatFeatures,
            ctx.LearnProgress.FloatFeatures,
            ctx.Params.DataProcessingOptions->IgnoredFeatures,
            /*ignoreRedundantCatFeatures=*/true,
            catFeatureParams.OneHotMaxSize,
            ctx.Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            /*clearPoolAfterBinarization=*/allowClearPool,
            ctx.LocalExecutor,
            /*select=*/{},
            &learnPool.Docs,
            &learnData.AllFeatures
        );

        for (size_t testIdx = 0; testIdx < testDataPtrs.size(); ++testIdx) {
            auto& testPool = *testPoolPtrs[testIdx];
            auto& testData = testDatasets[testIdx];
            PrepareAllFeaturesTest(
                ctx.CatFeatures,
                ctx.LearnProgress.FloatFeatures,
                learnData.AllFeatures,
                /*allowNansOnlyInTest=*/false,
                ctx.Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
                /*clearPoolAfterBinarization=*/allowClearPool,
                ctx.LocalExecutor,
                /*select=*/{},
                &testPool.Docs,
                &testData.AllFeatures
            );
        }

        ctx.InitContext(learnData, testDataPtrs);

        if (allowClearPool) {
            learnPool.Docs.Clear();
        }

        ctx.LearnProgress.CatFeatures.resize(sortedCatFeatures.size());
        for (size_t i = 0; i < sortedCatFeatures.size(); ++i) {
            auto& catFeature = ctx.LearnProgress.CatFeatures[i];
            catFeature.FeatureIndex = i;
            catFeature.FlatFeatureIndex = sortedCatFeatures[i];
            if (catFeature.FlatFeatureIndex < learnPool.FeatureId.ysize()) {
                catFeature.FeatureId = learnPool.FeatureId[catFeature.FlatFeatureIndex];
            }
        }

        DumpMemUsage("Before start train");

        const auto& systemOptions = ctx.Params.SystemOptions;
        if (!systemOptions->IsSingleHost()) { // send target, weights, baseline (if present), binarized features to workers and ask them to create plain folds
            InitializeMaster(&ctx);
            CB_ENSURE(IsPlainMode(ctx.Params.BoostingOptions->BoostingType), "Distributed training requires plain boosting");
            CB_ENSURE(learnPool.CatFeatures.empty(), "Distributed training requires all numeric data");
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
//                oheFeature.StringValues.push_back(learnPool.CatFeaturesHashToString.at(value));
//            }
//        }
        auto ctrTableGenerator = [&] (const TModelCtrBase& ctr) -> TCtrValueTable {
            TCtrValueTable resTable;
            CalcFinalCtrs(
                ctr.CtrType,
                featureCombinationToProjectionMap.at(ctr.Projection),
                learnData,
                testDataPtrs,
                ctx.LearnProgress.AveragingFold.LearnPermutation,
                ctx.LearnProgress.AveragingFold.LearnTargetClass[ctr.TargetBorderClassifierIdx],
                ctx.LearnProgress.AveragingFold.TargetClassesCount[ctr.TargetBorderClassifierIdx],
                catFeatureParams.CtrLeafCountLimit,
                catFeatureParams.StoreAllSimpleCtrs,
                catFeatureParams.CounterCalcMethod,
                &resTable
            );
            resTable.ModelCtrBase = ctr;
            MATRIXNET_DEBUG_LOG << "Finished CTR: " << ctr.CtrType << " " << BuildDescription(ctx.Layout, ctr.Projection) << Endl;
            return resTable;
        };

        auto ctrParallelGenerator = [&] (TFullModel* modelPtr, TVector<TModelCtrBase>& usedCtrBases) {
            TMutex lock;
            MATRIXNET_DEBUG_LOG << "Started parallel calculation of " << usedCtrBases.size() << " unique ctrs" << Endl;
            ctx.LocalExecutor.ExecRange([&](int i) {
                auto& ctr = usedCtrBases[i];
                auto table = ctrTableGenerator(ctr);
                with_lock(lock) {
                    modelPtr->CtrProvider->AddCtrCalcerData(std::move(table));
                }
            }, 0, usedCtrBases.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
            MATRIXNET_DEBUG_LOG << "CTR calculation finished" << Endl;
            modelPtr->UpdateDynamicData();
        };
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
            if (ctx.OutputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) {
                TVector<TModelCtrBase> usedCtrBases = modelPtr->ObliviousTrees.GetUsedModelCtrBases();
                modelPtr->CtrProvider = new TStaticCtrProvider;
                ctrParallelGenerator(modelPtr, usedCtrBases);
            }
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
                TVector<TModelCtrBase> usedCtrBases = Model.ObliviousTrees.GetUsedModelCtrBases();

                bool exportRequiresStaticCtrProvider = AnyOf(
                        updatedOutputOptions.GetModelFormats().cbegin(),
                        updatedOutputOptions.GetModelFormats().cend(),
                        [](EModelType format) {
                            return format == EModelType::Python || format == EModelType::CPP;
                        });
                if (!exportRequiresStaticCtrProvider) {
                    Model.CtrProvider = new TStaticCtrOnFlightSerializationProvider(usedCtrBases, ctrTableGenerator,
                                                                                    ctx.LocalExecutor);
                    MATRIXNET_DEBUG_LOG
                    << "Async calculation and writing of " << usedCtrBases.size() << " unique ctrs started" << Endl;
                } else {
                    Model.CtrProvider = new TStaticCtrProvider;
                    ctrParallelGenerator(&Model, usedCtrBases);
                }
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
        TPool learnPool;
        TVector<TPool> testPools;
        LoadPools(loadOptions, threadCount, catBoostOptions.DataProcessingOptions->ClassNames, &profile, &learnPool, &testPools);

        const auto evalFileName = outputOptions.CreateEvalFullPath();
        if (!evalFileName.empty() && !loadOptions.TestSetPaths.empty()) {
            ValidateColumnOutput(outputOptions.GetOutputColumns(), learnPool, false, loadOptions.CvParams.FoldCount > 0);
        }

        const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
        const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
        const auto modelPath = outputOptions.CreateResultModelFullPath();

        const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
        const bool allowClearPool = !needFstr;

        TVector<TEvalResult> evalResults(Max(testPools.ysize(), 1)); // need at least one evalResult, maybe empty

        this->TrainModel(trainJson, outputOptions, Nothing(), Nothing(), learnPool, allowClearPool, GetConstPointers(testPools),
                         nullptr, GetMutablePointers(evalResults));

        SetVerboseLogingMode();
        if (!evalFileName.empty()) {
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
            MATRIXNET_INFO_LOG << "Writing test eval to: " << evalFileName << Endl;
            TOFStream fileStream(evalFileName);
            for (int testIdx = 0; testIdx < testPools.ysize(); ++testIdx) {
                const TPool& testPool = testPools[testIdx];
                const NCB::TPathWithScheme& testSetPath = testIdx < loadOptions.TestSetPaths.ysize() ? loadOptions.TestSetPaths[testIdx] : NCB::TPathWithScheme();
                evalResults[testIdx].OutputToFile(threadCount,
                                                  outputOptions.GetOutputColumns(),
                                                  visibleLabelsHelper,
                                                  testPool,
                                                  false,
                                                  &fileStream,
                                                  testSetPath,
                                                  {testIdx, testPools.ysize()},
                                                  loadOptions.DsvPoolFormatParams.Format,
                                                  /*writeHeader*/ testIdx < 1);
            }
            if (testPools.empty()) {
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

        if (needFstr) {
            TFullModel model = ReadModel(modelPath);
            CalcAndOutputFstr(model, &learnPool, &fstrRegularFileName, &fstrInternalFileName);
        }

        MATRIXNET_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
        SetSilentLogingMode();
    }
};

TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ETaskType::CPU);

void TrainModel(const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* modelPtr,
    TEvalResult* evalResult)
{
    TrainModel(plainJsonParams, objectiveDescriptor, evalMetricDescriptor, learnPool, allowClearPool, {&testPool}, outputModelPath, modelPtr, {evalResult});
}

void TrainModel(const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TVector<const TPool*>& testPoolPtrs,
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
    modelTrainerHolder->TrainModel(trainOptions, outputOptions, objectiveDescriptor, evalMetricDescriptor, learnPool, allowClearPool, testPoolPtrs, modelPtr, evalResultPtrs);
}

/// Used by cross validation, hence one test dataset.
void TrainOneIteration(const TDataset& trainData, const TDataset* testDataPtr, TLearnContext* ctx) {
    SetLogingLevel(ctx->Params.LoggingLevel);

    TTrainOneIterationFunc trainFunc;
    ELossFunction lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();

    Y_VERIFY(testDataPtr);
    GetOneIterationFunc(lossFunction)(trainData, {testDataPtr}, ctx);
}
