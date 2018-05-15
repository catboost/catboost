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
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/app/output_fstr.h> // TODO(annaveronika): files from app/ should not be used here.

#include <library/grid_creator/binarization.h>

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
    TPool* learnPool,
    TVector<TPool>* testPools) {

    loadOptions.Validate();

    const bool verbose = false;
    if (!loadOptions.LearnFile.empty()) {
        ReadPool(
            loadOptions.CdFile,
            loadOptions.LearnFile,
            loadOptions.PairsFile,
            loadOptions.IgnoredFeatures,
            threadCount,
            verbose,
            loadOptions.Delimiter,
            loadOptions.HasHeader,
            classNames,
            learnPool
        );
        profile->AddOperation("Build learn pool");
    }

    for (int testIdx = 0; testIdx < loadOptions.TestFiles.ysize(); ++testIdx) {
        const TString& testFile = loadOptions.TestFiles[testIdx];
        const TString& testPairsFile = testIdx == 0 ? loadOptions.TestPairsFile : "";
        TPool testPool;
        ReadPool(
            loadOptions.CdFile,
            testFile,
            testPairsFile,
            loadOptions.IgnoredFeatures,
            threadCount,
            verbose,
            loadOptions.Delimiter,
            loadOptions.HasHeader,
            classNames,
            &testPool
        );
        testPools->push_back(std::move(testPool));
        if (testIdx + 1 == loadOptions.TestFiles.ysize()) {
            profile->AddOperation("Build test pool");
        }
    }

    const auto& cvParams = loadOptions.CvParams;
    if (cvParams.FoldCount != 0) {
        CB_ENSURE(loadOptions.TestFiles.empty(), "Test files are not supported in cross-validation mode");
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

    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metrics.front()->GetBestValue(&bestValueType, &bestPossibleValue);
    TErrorTracker errorTracker = BuildErrorTracker(bestValueType, bestPossibleValue, hasTest, ctx);

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();

    ctx->TryLoadProgress();

    TLogger logger;
    TString learnToken = "learn";
    TString testTokenPrefix = "test";
    TVector<const TString> testTokens;
    for (int testIdx = 0; testIdx < testDataPtrs.ysize(); ++testIdx) {
        TString testToken = testTokenPrefix + (testIdx > 0 ? ToString(testIdx) : "");
        testTokens.push_back(testToken);
    }
    if (ctx->OutputOptions.AllowWriteFiles()) {
        TVector<TString> learnSetNames = {ctx->Files.NamesPrefix + learnToken};
        TVector<TString> testSetNames;
        for (int testIdx = 0; testIdx < testDataPtrs.ysize(); ++testIdx) {
            testSetNames.push_back({ctx->Files.NamesPrefix + testTokens[testIdx]});
        }
        auto losses = CreateMetrics(
            ctx->Params.LossFunctionDescription,
            ctx->Params.MetricOptions,
            ctx->EvalMetricDescriptor,
            ctx->LearnProgress.ApproxDimension
        );

        AddFileLoggers(
            ctx->Params.IsProfile,
            ctx->Files.LearnErrorLogFile,
            ctx->Files.TestErrorLogFile,
            ctx->Files.TimeLeftLogFile,
            ctx->Files.JsonLogFile,
            ctx->Files.ProfileLogFile,
            ctx->OutputOptions.GetTrainDir(),
            GetJsonMeta(
                ctx->Params.BoostingOptions->IterationCount.Get(),
                ctx->OutputOptions.GetName(),
                GetConstPointers(losses),
                learnSetNames,
                testSetNames,
                ELaunchMode::Train),
            ctx->OutputOptions.GetMetricPeriod(),
            &logger
        );
    }

    WriteHistory(
        GetMetricsDescription(metrics),
        ctx->LearnProgress.LearnErrorsHistory,
        ctx->LearnProgress.TestErrorsHistory,
        ctx->LearnProgress.TimeHistory,
        learnToken,
        testTokens,
        &logger
    );

    AddConsoleLogger(
        learnToken,
        testTokens,
        /*hasTrain=*/true,
        ctx->OutputOptions.GetMetricPeriod(),
        ctx->Params.BoostingOptions->IterationCount,
        &logger
    );

    TVector<double> valuesToLog;
    TVector<TVector<TVector<double>>> errorsHistory = ctx->LearnProgress.TestErrorsHistory;
    for (int iter = 0; iter < errorsHistory.ysize(); ++iter) {
        const int testIdxToLog = 0;
        const int metricIdxToLog = 0;
        errorTracker.AddError(errorsHistory[iter][testIdxToLog][metricIdxToLog], iter, &valuesToLog);
    }

    if (IsSamplingPerTree(ctx->Params.ObliviousTreeOptions.Get())) {
        ctx->SmallestSplitSideDocs.Create(ctx->LearnProgress.Folds);
        ctx->PrevTreeLevelStats.Create(
            ctx->LearnProgress.Folds,
            CountNonCtrBuckets(CountSplits(ctx->LearnProgress.FloatFeatures), learnData.AllFeatures.OneHotValues),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth)
        );
    }
    ctx->SampledDocs.Create(
        ctx->LearnProgress.Folds,
        GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)
    ); // TODO(espetrov): create only if sample rate < 1

    for (ui32 iter = ctx->LearnProgress.TreeStruct.ysize(); iter < ctx->Params.BoostingOptions->IterationCount; ++iter) {
        profile.StartNextIteration();

        trainOneIterationFunc(learnData, testDataPtrs, ctx);

        CalcErrors(learnData, testDataPtrs, metrics, ctx);

        profile.AddOperation("Calc errors");

        if (hasTest) {
            // Use only (test0, metric0) for overfitting detection
            const int testIdxToLog = 0;
            const int metricIdxToLog = 0;
            errorTracker.AddError(ctx->LearnProgress.TestErrorsHistory.back()[testIdxToLog][metricIdxToLog], iter, &valuesToLog);

            if (useBestModel && iter == static_cast<ui32>(errorTracker.GetBestIteration())) {
                ctx->LearnProgress.BestTestApprox = ctx->LearnProgress.TestApprox[0];
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress.TimeHistory.push_back({profileResults.PassedTime, profileResults.RemainingTime});

        Log(
            GetMetricsDescription(metrics),
            ctx->LearnProgress.LearnErrorsHistory,
            ctx->LearnProgress.TestErrorsHistory,
            errorTracker.GetBestError(),
            errorTracker.GetBestIteration(),
            profileResults,
            learnToken,
            testTokens,
            &logger
        );

        ctx->SaveProgress();

        const auto& errors = ctx->LearnProgress.LearnErrorsHistory.back();
        int objectiveIndex = 0;
        if (ctx->Params.MetricOptions.Get().EvalMetric.IsSet() &&
            ctx->Params.MetricOptions.Get().EvalMetric->GetLossFunction() != ctx->Params.LossFunctionDescription->GetLossFunction()) {
            objectiveIndex = 1;
        }
        if (IsNan(errors[objectiveIndex])) {
            ctx->LearnProgress.LeafValues.pop_back();
            ctx->LearnProgress.TreeStruct.pop_back();
            MATRIXNET_WARNING_LOG << "Training has stopped (degenerate solution on iteration "
                << iter << ", probably too small l2-regularization, try to increase it)" << Endl;
            break;
        }

        if (errorTracker.GetIsNeedStop()) {
            MATRIXNET_NOTICE_LOG << "Stopped by overfitting detector "
                << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
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
        MATRIXNET_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << "\n";
        MATRIXNET_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << "\n";
        MATRIXNET_NOTICE_LOG << "\n";
    }

    if (useBestModel && ctx->Params.BoostingOptions->IterationCount > 0) {
        const int itCount = errorTracker.GetBestIteration() + 1;
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
            CB_ENSURE(testPool.Docs.GetFactorsCount() == learnPool.Docs.GetFactorsCount(), "train pool factors count == " << learnPool.Docs.GetFactorsCount() << " and test pool factors count == " << testPool.Docs.GetFactorsCount());
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
        const int featureCount = learnPool.Docs.GetFactorsCount();

        TLearnContext ctx(
            jsonParams,
            objectiveDescriptor,
            evalMetricDescriptor,
            outputOptions,
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
        SetDataDependantDefaults(
            learnPool.Docs.GetDocCount(),
            /*testPoolSize*/ GetDocCount(testPoolPtrs),
            /*hasTestLabels*/ testPoolPtrs.size() > 0 && IsConst(testPoolPtrs[0]->Docs.Target),
            learnPool.MetaInfo.HasWeights,
            &ctx.OutputOptions.UseBestModel,
            &ctx.Params
        );
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

        if (IsMultiClassError(lossFunction)) {
             ctx.LearnProgress.ApproxDimension = GetClassesCount(learnData.Target, ctx.Params.DataProcessingOptions->ClassesCount);
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
        Preprocess(ctx.Params.LossFunctionDescription, classWeights, learnData);
        for (TDataset& testData : testDatasets) {
            Preprocess(ctx.Params.LossFunctionDescription, classWeights, testData);
            CheckConsistency(ctx.Params.LossFunctionDescription, learnData, testData);
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
            CB_ENSURE(ctx.LearnProgress.ApproxDimension == 1, "Distributed training requires 1D approxes");
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
            for (const auto& keyValue: ctx.Params.Metadata.Get().GetMap()) {
                Model.ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
            if (ctx.OutputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) {
                TVector<TModelCtrBase> usedCtrBases = Model.ObliviousTrees.GetUsedModelCtrBases();

                bool exportRequiresStaticCtrProvider = AnyOf(
                        outputOptions.GetModelFormats().cbegin(),
                        outputOptions.GetModelFormats().cend(),
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
                    outputOptions.GetModelFormats().size() > 1 || !outputOptions.ResultModelPath.IsSet();
            TString outputFile = outputOptions.CreateResultModelFullPath();
            if (addFileFormatExtension && outputFile.EndsWith(".bin")) {
                outputFile = outputFile.substr(0, outputFile.length() - 4);
            }
            for (const auto& format : outputOptions.GetModelFormats()) {
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
            TLearnContext ctx(
                trainJson,
                /*objectiveDescriptor*/ Nothing(),
                /*evalMetricDescriptor*/ Nothing(),
                outputOptions,
                /*featureCount*/ 0,
                /*sortedCatFeatures*/ {},
                /*featuresId*/ {}
            );
            SetWorkerParams(ctx.Params);
            SetWorkerCustomObjective(ctx.ObjectiveDescriptor);
            RunWorker(threadCount, catBoostOptions.SystemOptions->NodePort);
            return;
        }

        TProfileInfo profile;
        TPool learnPool;
        TVector<TPool> testPools;
        LoadPools(loadOptions, threadCount, catBoostOptions.DataProcessingOptions->ClassNames, &profile, &learnPool, &testPools);

        const auto evalFileName = outputOptions.CreateEvalFullPath();
        if (!evalFileName.empty() && !loadOptions.TestFiles.empty()) {
            ValidateColumnOutput(outputOptions.GetOutputColumns(), learnPool, loadOptions.CvParams.FoldCount > 0);
        }

        const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
        const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
        const auto modelPath = outputOptions.CreateResultModelFullPath();

        const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
        const bool allowClearPool = !needFstr;

        TVector<TEvalResult> evalResults(Max(testPools.ysize(), 1)); // need at least one evalResult, maybe empty
        this->TrainModel(trainJson, outputOptions, Nothing(), Nothing(), learnPool, allowClearPool, GetConstPointers(testPools), nullptr, GetMutablePointers(evalResults));

        SetVerboseLogingMode();
        if (!evalFileName.empty()) {
            if (!loadOptions.CvParams.FoldCount && loadOptions.TestFiles.empty() && !outputOptions.GetOutputColumns().empty()) {
                MATRIXNET_WARNING_LOG << "No test files, can't output columns\n";
            }
            MATRIXNET_INFO_LOG << "Writing test eval to: " << evalFileName << Endl;
            TOFStream fileStream(evalFileName);
            for (int testIdx = 0; testIdx < testPools.ysize(); ++testIdx) {
                const TPool& testPool = testPools[testIdx];
                const TString& testFile = testIdx < loadOptions.TestFiles.ysize() ? loadOptions.TestFiles[testIdx] : TString{""};
                evalResults[testIdx].OutputToFile(threadCount, outputOptions.GetOutputColumns(), testPool, &fileStream,
                                                  testFile, {testIdx, testPools.ysize()},
                                                  loadOptions.Delimiter, loadOptions.HasHeader,
                                                  /*writeHeader*/ testIdx < 1);
            }
            if (testPools.empty()) {
                // Make sure to emit header to fileStream
                evalResults[0].OutputToFile(threadCount, outputOptions.GetOutputColumns(), TPool(), &fileStream,
                                            "", {0, 1},
                                            loadOptions.Delimiter, loadOptions.HasHeader,
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
            CalcAndOutputFstr(model, learnPool, &fstrRegularFileName, &fstrInternalFileName, threadCount);
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
        TLearnContext ctx(
            trainOptions,
            objectiveDescriptor,
            evalMetricDescriptor,
            outputOptions,
            /*featureCount*/ 0,
            /*sortedCatFeatures*/ {},
            /*featuresId*/ {}
        );
        SetWorkerParams(ctx.Params);
        SetWorkerCustomObjective(ctx.ObjectiveDescriptor);
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
