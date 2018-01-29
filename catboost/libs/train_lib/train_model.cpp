#include "train_model.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/algo/train.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/algo/model_build_helper.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/loggers/logger.h>

#include <library/grid_creator/binarization.h>

#include <util/random/shuffle.h>
#include <util/generic/vector.h>
#include <catboost/app/output_fstr.h>
#include <util/system/info.h>

static void PrepareTargetBinary(float border, TVector<float>* target) {
    for (int i = 0; i < (*target).ysize(); ++i) {
        (*target)[i] = ((*target)[i] > border);
    }
}

// TODO(nikitxskv): Is this a bottleneck? switch to vector+unique vs vector+sort+unique?
static bool IsCorrectQueryIdsFormat(const TVector<ui32>& queryIds) {
    THashSet<ui32> queryGroupIds;
    ui32 lastId = queryIds.empty() ? 0 : queryIds[0];
    for (ui32 id : queryIds) {
        if (id != lastId) {
            if (queryGroupIds.has(id)) {
                return false;
            }
            queryGroupIds.insert(lastId);
            lastId = id;
        }
    }
    return true;
}

static ui32 CalcFeaturesCheckSum(const TAllFeatures& allFeatures) {
    ui32 checkSum = 0;
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.FloatHistograms);
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.CatFeatures);
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.CatFeaturesRemapped);
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.OneHotValues);
    return checkSum;
}

static THashMap<ui32, ui32> CalcQueriesSize(const TVector<ui32>& queriesId) {
    THashMap<ui32, ui32> queriesSize;
    for (int docId = 0; docId < queriesId.ysize(); ++docId) {
        ++queriesSize[queriesId[docId]];
    }
    return queriesSize;
}

namespace {
void ShrinkModel(int itCount, TLearnProgress* progress) {
    progress->LeafValues.resize(itCount);
    progress->TreeStruct.resize(itCount);
}

static double EvalErrors(
    const TVector<TVector<double>>& avrgApprox,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<ui32>& queryId,
    const THashMap<ui32, ui32>& queriesSize,
    const TVector<TPair>& pairs,
    const THolder<IMetric>& error,
    int learnSampleCount,
    int sampleCount,
    NPar::TLocalExecutor* localExecutor
) {
    return error->GetFinalError(
        error->GetErrorType() == EErrorType::PerObjectError ?
            error->Eval(avrgApprox, target, weight, learnSampleCount, sampleCount, *localExecutor) :
            error->GetErrorType() == EErrorType::PairwiseError ?
                error->EvalPairwise(avrgApprox, pairs, learnSampleCount, sampleCount):
                error->EvalQuerywise(avrgApprox, target, weight, queryId, queriesSize, learnSampleCount, sampleCount));
}

static void CalcErrors(
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<ui32>& queryId,
    const THashMap<ui32, ui32>& queriesSize,
    const TVector<TPair>& pairs,
    const TVector<THolder<IMetric>>& errors,
    int learnSampleCount,
    int sampleCount,
    bool hasTest,
    TLearnProgress* learnProgress,
    NPar::TLocalExecutor* localExecutor
) {
    learnProgress->LearnErrorsHistory.emplace_back();
    if (hasTest) {
        learnProgress->TestErrorsHistory.emplace_back();
    }

    for (int i = 0; i < errors.ysize(); ++i) {
        double learnErr = EvalErrors(
            learnProgress->AvrgApprox,
            target,
            weight,
            queryId,
            queriesSize,
            pairs,
            errors[i],
            0,
            learnSampleCount,
            localExecutor
        );
        learnProgress->LearnErrorsHistory.back().push_back(learnErr);

        if (hasTest) {
            double testErr = EvalErrors(
                learnProgress->AvrgApprox,
                target,
                weight,
                queryId,
                queriesSize,
                pairs,
                errors[i],
                learnSampleCount,
                sampleCount,
                localExecutor
            );
            learnProgress->TestErrorsHistory.back().push_back(testErr);
        }
    }
}

void Train(const TTrainData& data, TLearnContext* ctx, TVector<TVector<double>>* testMultiApprox) {
    TProfileInfo& profile = ctx->Profile;

    const int sampleCount = data.GetSampleCount();
    const int approxDimension = testMultiApprox->ysize();
    const bool hasTest = sampleCount > data.LearnSampleCount;
    auto trainOneIterationFunc = GetOneIterationFunc(ctx->Params.LossFunctionDescription->GetLossFunction());
    TVector<THolder<IMetric>> metrics = CreateMetrics(
        ctx->Params.LossFunctionDescription,
        ctx->Params.MetricOptions,
        ctx->EvalMetricDescriptor,
        approxDimension
    );

    TErrorTracker errorTracker = BuildErrorTracker(metrics.front()->IsMaxOptimal(), hasTest, ctx);

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();
    CB_ENSURE(hasTest || !useBestModel, "cannot select best model, no test provided");

    if (ctx->TryLoadProgress()) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*testMultiApprox)[dim].assign(
                ctx->LearnProgress.AvrgApprox[dim].begin() + data.LearnSampleCount, ctx->LearnProgress.AvrgApprox[dim].end());
        }
    }

    TLogger logger;
    TString learnToken = "learn", testToken = "test";
    if (ctx->OutputOptions.AllowWriteFiles()) {
        AddFileLoggers(
            ctx->Files.LearnErrorLogFile,
            ctx->Files.TestErrorLogFile,
            ctx->Files.TimeLeftLogFile,
            ctx->Files.JsonLogFile,
            ctx->OutputOptions.GetTrainDir(),
            GetJsonMeta(
                {ctx},
                learnToken,
                testToken,
                /*hasTrain=*/true,
                hasTest
            ),
            &logger
        );
    }

    WriteHistory(
        GetMetricsDescription(metrics),
        ctx->LearnProgress.LearnErrorsHistory,
        ctx->LearnProgress.TestErrorsHistory,
        ctx->LearnProgress.TimeHistory,
        learnToken,
        testToken,
        &logger
    );

    AddConsoleLogger(
        ctx->Params.IsProfile,
        learnToken,
        testToken,
        /*hasTrain=*/true,
        hasTest,
        ctx->OutputOptions.GetMetricPeriod(),
        &logger
    );

    TVector<TVector<double>> errorsHistory = ctx->LearnProgress.TestErrorsHistory;
    TVector<double> valuesToLog;
    for (int i = 0; i < errorsHistory.ysize(); ++i) {
        errorTracker.AddError(errorsHistory[i][0], i, &valuesToLog);
    }

    TFold* fold;
    if (!ctx->LearnProgress.Folds.empty()) {
        fold = &ctx->LearnProgress.Folds[0]; // assume that all folds have the same shape
    } else {
        fold = &ctx->LearnProgress.AveragingFold;
    }

    if (IsSamplingPerTree(ctx->Params.ObliviousTreeOptions.Get())) {
        ctx->SmallestSplitSideDocs.Create(*fold);
        const int approxDimension = fold->GetApproxDimension();
        const int bodyTailCount = fold->BodyTailArr.ysize();
        ctx->PrevTreeLevelStats.Create(
            CountNonCtrBuckets(CountSplits(ctx->LearnProgress.FloatFeatures), data.AllFeatures.OneHotValues),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth),
            approxDimension,
            bodyTailCount
        );
    }
    ctx->SampledDocs.Create(*fold, GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)); // TODO(espetrov): create only if sample rate < 1

    for (ui32 iter = ctx->LearnProgress.TreeStruct.ysize(); iter < ctx->Params.BoostingOptions->IterationCount; ++iter) {
        profile.StartNextIteration();

        trainOneIterationFunc(data, ctx);

        CalcErrors(
            data.Target,
            data.Weights,
            data.QueryId,
            data.QuerySize,
            data.Pairs,
            metrics,
            data.LearnSampleCount,
            sampleCount,
            hasTest,
            &ctx->LearnProgress,
            &ctx->LocalExecutor
        );

        profile.AddOperation("Calc errors");

        if (hasTest) {
            TVector<double> valuesToLog;
            errorTracker.AddError(ctx->LearnProgress.TestErrorsHistory.back()[0], iter, &valuesToLog);

            if ((useBestModel && iter == static_cast<ui32>(errorTracker.GetBestIteration())) || !useBestModel) {
                for (int dim = 0; dim < approxDimension; ++dim) {
                    (*testMultiApprox)[dim].assign(
                        ctx->LearnProgress.AvrgApprox[dim].begin() + data.LearnSampleCount, ctx->LearnProgress.AvrgApprox[dim].end()
                    );
                }
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress.TimeHistory.push_back({profileResults.PassedTime,profileResults.RemainingTime});

        Log(
            GetMetricsDescription(metrics),
            ctx->LearnProgress.LearnErrorsHistory,
            ctx->LearnProgress.TestErrorsHistory,
            errorTracker.GetBestError(),
            errorTracker.GetBestIteration(),
            profileResults,
            learnToken,
            testToken,
            &logger
        );

        ctx->SaveProgress();

        if (IsNan(ctx->LearnProgress.LearnErrorsHistory.back()[0])) {
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

    ctx->LearnProgress.Folds.clear();

    if (hasTest) {
        MATRIXNET_NOTICE_LOG << "\n";
        MATRIXNET_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << "\n";
        MATRIXNET_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << "\n";
        MATRIXNET_NOTICE_LOG << "\n";
    }

    if (ctx->Params.IsProfile || ctx->Params.LoggingLevel == ELoggingLevel::Debug) {
        LogAverages(profile.GetProfileResults());
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
        const TPool& testPool,
        TFullModel* modelPtr,
        TEvalResult* evalResult
    ) const override {
        CB_ENSURE(learnPool.Docs.GetDocCount() != 0, "Train dataset is empty");

        auto sortedCatFeatures = learnPool.CatFeatures;
        Sort(sortedCatFeatures.begin(), sortedCatFeatures.end());
        if (testPool.Docs.GetDocCount() != 0) {
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

        const NCatboostOptions::TLossDescription& lossDescription = ctx.Params.LossFunctionDescription;
        ELossFunction lossFunction = lossDescription.GetLossFunction();
        if (!ctx.Params.DataProcessingOptions->HasTimeFlag) {
            Shuffle(learnPool.Docs.QueryId, ctx.Rand, &indices);
        }

        ApplyPermutation(InvertPermutation(indices), &learnPool, &ctx.LocalExecutor);

        auto permutationGuard = Finally([&] { ApplyPermutation(indices, &learnPool, &ctx.LocalExecutor); });

        TTrainData trainData;
        trainData.LearnSampleCount = learnPool.Docs.GetDocCount();

        UpdateBoostingTypeOption(learnPool.Docs.GetDocCount(), &ctx.Params.BoostingOptions->BoostingType);

        trainData.Target.reserve(learnPool.Docs.GetDocCount() + testPool.Docs.GetDocCount());

        trainData.Pairs.reserve(learnPool.Pairs.size() + testPool.Pairs.size());
        trainData.Pairs.insert(trainData.Pairs.end(), learnPool.Pairs.begin(), learnPool.Pairs.end());
        trainData.Pairs.insert(trainData.Pairs.end(), testPool.Pairs.begin(), testPool.Pairs.end());
        for (int pairInd = learnPool.Pairs.ysize(); pairInd < trainData.Pairs.ysize(); ++pairInd) {
            trainData.Pairs[pairInd].WinnerId += trainData.LearnSampleCount;
            trainData.Pairs[pairInd].LoserId += trainData.LearnSampleCount;
        }

        bool trainHasBaseline = learnPool.Docs.GetBaselineDimension() != 0;
        bool testHasBaseline = trainHasBaseline;
        if (testPool.Docs.GetDocCount() != 0) {
            testHasBaseline = testPool.Docs.GetBaselineDimension() != 0;
        }
        if (trainHasBaseline && !testHasBaseline) {
            CB_ENSURE(false, "Baseline for test is not provided");
        }
        if (testHasBaseline && !trainHasBaseline) {
            CB_ENSURE(false, "Baseline for train is not provided");
        }
        if (trainHasBaseline && testHasBaseline && testPool.Docs.GetDocCount() != 0) {
            CB_ENSURE(learnPool.Docs.GetBaselineDimension() == testPool.Docs.GetBaselineDimension(), "Baseline dimensions differ.");
        }

        bool nonZero = !AllOf(learnPool.Docs.Weight, [] (float weight) { return weight == 0; });
        CB_ENSURE(nonZero, "All documents have zero weights");

        trainData.Target = learnPool.Docs.Target;
        trainData.Weights = learnPool.Docs.Weight;
        trainData.QueryId = learnPool.Docs.QueryId;
        trainData.Baseline = learnPool.Docs.Baseline;

        float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.end());
        float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.end());
        CB_ENSURE(minTarget != maxTarget || IsPairwiseError(lossFunction), "All targets are equal");

        trainData.Target.insert(trainData.Target.end(), testPool.Docs.Target.begin(), testPool.Docs.Target.end());
        trainData.Weights.insert(trainData.Weights.end(), testPool.Docs.Weight.begin(), testPool.Docs.Weight.end());
        trainData.QueryId.insert(trainData.QueryId.end(), testPool.Docs.QueryId.begin(), testPool.Docs.QueryId.end());
        for (int dim = 0; dim < testPool.Docs.GetBaselineDimension(); ++dim) {
            trainData.Baseline[dim].insert(trainData.Baseline[dim].end(), testPool.Docs.Baseline[dim].begin(), testPool.Docs.Baseline[dim].end());
        }

        TVector<THolder<IMetric>> metrics = CreateMetrics(
            ctx.Params.LossFunctionDescription,
            ctx.Params.MetricOptions,
            ctx.EvalMetricDescriptor,
            1
        );
        bool hasQuerywiseMetric = false;
        for (const auto& metric : metrics) {
            if (metric.Get()->GetErrorType() == EErrorType::QuerywiseError) {
                hasQuerywiseMetric = true;
            }
        }
        if (hasQuerywiseMetric) {
            CB_ENSURE(trainData.QueryId.size() == trainData.Target.size(), "Query ids not provided for querywise metric.");
            bool isDataCorrect = IsCorrectQueryIdsFormat(trainData.QueryId);
            if (testPool.Docs.GetDocCount() != 0) {
                isDataCorrect &= learnPool.Docs.QueryId.back() != testPool.Docs.QueryId.front();
            }
            CB_ENSURE(isDataCorrect, "Train Pool & Test Pool should be grouped by QueryId and should have different QueryId");
            //TODO(annaveronika): Allow no grouping by query id. Warning
            //when same query id in train+test - no error.
            trainData.QuerySize = CalcQueriesSize(trainData.QueryId);
        }

        if (lossFunction == ELossFunction::Logloss) {
            PrepareTargetBinary(NCatboostOptions::GetLogLossBorder(lossDescription), &trainData.Target);
            float minTarget = *MinElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
            float maxTarget = *MaxElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
            CB_ENSURE(minTarget == 0, "All targets are greater than border");
            CB_ENSURE(maxTarget == 1, "All targets are smaller than border");
        }

        if (trainHasBaseline) {
            CB_ENSURE((trainData.Baseline.ysize() > 1) == IsMultiClassError(lossFunction), "Loss-function is MultiClass iff baseline dimension > 1");
        }
        if (IsMultiClassError(lossFunction)) {
            CB_ENSURE(AllOf(trainData.Target, [](float x) { return floor(x) == x && x >= 0; }), "if loss-function is MultiClass then each target label should be nonnegative integer");
            ctx.LearnProgress.ApproxDimension = GetClassesCount(trainData.Target, ctx.Params.DataProcessingOptions->ClassesCount);
            CB_ENSURE(ctx.LearnProgress.ApproxDimension > 1, "All targets are equal");
        }
        ctx.LearnProgress.PoolCheckSum = CalcFeaturesCheckSum(trainData.AllFeatures);

        ctx.OutputMeta();

        const TVector<float>& classesWeights = ctx.Params.DataProcessingOptions->ClassWeights;
        if (!classesWeights.empty()) {
            int dataSize = trainData.Target.ysize();
            for (int i = 0; i < dataSize; ++i) {
                CB_ENSURE(trainData.Target[i] < classesWeights.ysize(), "class " + ToString(trainData.Target[i]) + " is missing in class weights");
                trainData.Weights[i] *= classesWeights[trainData.Target[i]];
            }
        }

        ctx.InitData(trainData);

        GenerateBorders(learnPool, &ctx, &ctx.LearnProgress.FloatFeatures);

        if (testPool.Docs.GetDocCount() != 0) {
            CB_ENSURE(testPool.Docs.GetFactorsCount() == learnPool.Docs.GetFactorsCount(), "train pool factors count == " << learnPool.Docs.GetFactorsCount() << " and test pool factors count == " << testPool.Docs.GetFactorsCount());
            CB_ENSURE(testPool.Docs.GetBaselineDimension() == learnPool.Docs.GetBaselineDimension(), "train pool baseline dimension == " << learnPool.Docs.GetBaselineDimension() << " and test pool baseline dimension == " << testPool.Docs.GetBaselineDimension());

            if (!IsPairwiseError(lossFunction)) {
                float minTestTarget = *MinElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
                float maxTestTarget = *MaxElement(trainData.Target.begin(), trainData.Target.begin() + trainData.LearnSampleCount);
                CB_ENSURE(minTestTarget != maxTestTarget, "All targets in test are equal.");
            }
        }
        learnPool.Docs.Append(testPool.Docs);
        const int factorsCount = learnPool.Docs.GetFactorsCount();
        const int approxDim = learnPool.Docs.GetBaselineDimension();
        bool hasQueryId = !learnPool.Docs.QueryId.empty();

        auto learnPoolGuard = Finally([&] {
            if (!allowClearPool) {
                learnPool.Docs.Resize(trainData.LearnSampleCount, factorsCount, approxDim, hasQueryId);
            }
        });

        const auto& catFeatureParams = ctx.Params.CatFeatureParams.Get();
        PrepareAllFeatures(
            ctx.CatFeatures,
            ctx.LearnProgress.FloatFeatures,
            ctx.Params.DataProcessingOptions->IgnoredFeatures,
            trainData.LearnSampleCount,
            catFeatureParams.OneHotMaxSize,
            ctx.Params.DataProcessingOptions->FloatFeaturesBinarization->NanMode,
            allowClearPool,
            ctx.LocalExecutor,
            &learnPool.Docs,
            &trainData.AllFeatures
        );

        if (allowClearPool) {
            learnPool.Docs.Clear();
        }

        float minWeight = *MinElement(trainData.Weights.begin(), trainData.Weights.begin() + trainData.LearnSampleCount);
        float maxWeight = *MaxElement(trainData.Weights.begin(), trainData.Weights.begin() + trainData.LearnSampleCount);

        if (minWeight == maxWeight) {
            trainData.Weights.clear();
            trainData.Weights.shrink_to_fit();
        } else { // Learn weight sum should be equal to learn sample count
            CB_ENSURE(minWeight > 0, "weights should be positive");
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

        evalResult->GetRawValuesRef().resize(ctx.LearnProgress.ApproxDimension);
        DumpMemUsage("Before start train");

        Train(trainData, &ctx, &(evalResult->GetRawValuesRef()));
        evalResult->SetPredictionTypes(ctx.OutputOptions.GetPredictionTypes());

        TObliviousTrees obliviousTrees;
        THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
        {
            TObliviousTreeBuilder builder(ctx.LearnProgress.FloatFeatures, ctx.LearnProgress.CatFeatures);
            for (size_t treeId = 0; treeId < ctx.LearnProgress.TreeStruct.size(); ++treeId) {
                TVector<TModelSplit> modelSplits;
                for (const auto& split : ctx.LearnProgress.TreeStruct[treeId].Splits) {
                    auto modelSplit = split.GetModelSplit(ctx);
                    modelSplits.push_back(modelSplit);
                    if (modelSplit.Type == ESplitType::OnlineCtr) {
                        featureCombinationToProjectionMap[modelSplit.OnlineCtr.Ctr.Base.Projection] = split.Ctr.Projection;
                    }
                }
                builder.AddTree(modelSplits, ctx.LearnProgress.LeafValues[treeId]);
            }
            obliviousTrees = builder.Build();
        }

        for (auto& oheFeature : obliviousTrees.OneHotFeatures) {
            for (const auto& value : oheFeature.Values) {
                oheFeature.StringValues.push_back(learnPool.CatFeaturesHashToString.at(value));
            }
        }
        auto ctrTableGenerator = [&] (const TModelCtrBase& ctr) -> TCtrValueTable {
            TCtrValueTable resTable;
            CalcFinalCtrs(
                ctr.CtrType,
                featureCombinationToProjectionMap.at(ctr.Projection),
                trainData,
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
        if (modelPtr) {
            modelPtr->ObliviousTrees = std::move(obliviousTrees);
            modelPtr->ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            TVector<TModelCtrBase> usedCtrBases = modelPtr->ObliviousTrees.GetUsedModelCtrBases();
            modelPtr->CtrProvider = new TStaticCtrProvider;
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
        } else {
            TFullModel Model;
            Model.ObliviousTrees = std::move(obliviousTrees);
            Model.ModelInfo["params"] = ctx.LearnProgress.SerializedTrainParams;
            TVector<TModelCtrBase> usedCtrBases = Model.ObliviousTrees.GetUsedModelCtrBases();

            Model.CtrProvider = new TStaticCtrOnFlightSerializationProvider(usedCtrBases, ctrTableGenerator, ctx.LocalExecutor);
            MATRIXNET_DEBUG_LOG << "Async calculation and writing of " << usedCtrBases.size() << " unique ctrs started" << Endl;
            OutputModel(Model, outputOptions.CreateResultModelFullPath());
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
        int threadCount = Min<int>(catBoostOptions.SystemOptions->NumThreads, (int)NSystemInfo::CachedNumberOfCpus());

        bool verbose = false;
        TVector<TString> classNames;

        TProfileInfo profile(true);

        TPool learnPool;
        if (!loadOptions.LearnFile.empty()) {
            ReadPool(
                loadOptions.CdFile,
                loadOptions.LearnFile,
                loadOptions.PairsFile,
                threadCount,
                verbose,
                loadOptions.Delimiter,
                loadOptions.HasHeader,
                catBoostOptions.DataProcessingOptions->ClassNames,
                &learnPool
            );
            profile.AddOperation("Build learn pool");
        }

        TPool testPool;
        if (!loadOptions.TestFile.empty()) {
            ReadPool(
                loadOptions.CdFile,
                loadOptions.TestFile,
                loadOptions.TestPairsFile,
                threadCount,
                verbose,
                loadOptions.Delimiter,
                loadOptions.HasHeader,
                catBoostOptions.DataProcessingOptions->ClassNames,
                &testPool
            );
            profile.AddOperation("Build test pool");
        }

        const auto& cvParams = loadOptions.CvParams;
        if (cvParams.FoldCount != 0) {
            CB_ENSURE(loadOptions.TestFile.empty(), "Test file is not supported in cross-validation mode");
            CB_ENSURE(loadOptions.PairsFile.empty() && loadOptions.TestPairsFile.empty(), "Pairs are not supported in cross-validation mode");
            Y_VERIFY(cvParams.FoldIdx != -1);

            BuildCvPools(
                cvParams.FoldIdx,
                cvParams.FoldCount,
                cvParams.Inverted,
                cvParams.RandSeed,
                threadCount,
                &learnPool,
                &testPool
            );
            profile.AddOperation("Build cv pools");
        }

        TEvalResult evalResult;
        const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
        const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
        const auto modelPath = outputOptions.CreateResultModelFullPath();

        const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
        const bool allowClearPool = !needFstr;

        TrainModel(trainJson, outputOptions, Nothing(), Nothing(), learnPool, allowClearPool, testPool, nullptr, &evalResult);

        SetVerboseLogingMode();
        const auto evalFileName = outputOptions.CreateEvalFullPath();
        if (!evalFileName.empty()) {
            MATRIXNET_INFO_LOG << "Writing test eval to: " << evalFileName << Endl;
            TOFStream fileStream(evalFileName);
            evalResult.PostProcess(threadCount);
            evalResult.OutputToFile(testPool.Docs.Id, &fileStream, true, &testPool.Docs.Target);
        } else {
            MATRIXNET_INFO_LOG << "Skipping test eval output" << Endl;
        }
        profile.AddOperation("Train model");

        if (catBoostOptions.IsProfile || catBoostOptions.LoggingLevel == ELoggingLevel::Debug) {
            TLogger logger;
            logger.AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TConsoleLoggingBackend(true, outputOptions.GetMetricPeriod())));
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
    THolder<IModelTrainer> modelTrainerHolder;
    NJson::TJsonValue trainOptions;
    NJson::TJsonValue outputFilesOptionsJson;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptions, &outputFilesOptionsJson);
    const ETaskType taskType = NCatboostOptions::GetTaskType(trainOptions);

    NCatboostOptions::TOutputFilesOptions outputOptions(taskType);
    outputFilesOptionsJson["result_model_file"] = outputModelPath;
    outputOptions.Load(outputFilesOptionsJson);

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "GPU Device not found.");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(trainOptions, outputOptions, objectiveDescriptor, evalMetricDescriptor, learnPool, allowClearPool, testPool, modelPtr, evalResult);
}

void TrainOneIteration(const TTrainData& trainData, TLearnContext* ctx) {
    SetLogingLevel(ctx->Params.LoggingLevel);

    TTrainOneIterationFunc trainFunc;
    ELossFunction lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();

    GetOneIterationFunc(lossFunction)(trainData, ctx);
}

void CheckFitParams(
    const NJson::TJsonValue& plainOptions,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor
) {
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputJsonOptions;
    NCatboostOptions::PlainJsonToOptions(plainOptions, &catBoostJsonOptions, &outputJsonOptions);
    auto options = NCatboostOptions::LoadOptions(catBoostJsonOptions);

    if (options.LossFunctionDescription->GetLossFunction() == ELossFunction::Custom) {
        CB_ENSURE(objectiveDescriptor.Defined(), "Error: provide objective descriptor for custom loss");
    }

    if (options.MetricOptions->EvalMetric.IsSet() && options.MetricOptions->EvalMetric->GetLossFunction() == ELossFunction::Custom) {
        CB_ENSURE(evalMetricDescriptor.Defined(), "Error: provide eval metric descriptor for custom eval metric");
    }
}
