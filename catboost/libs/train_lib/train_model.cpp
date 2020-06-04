#include "train_model.h"
#include "options_helper.h"
#include "cross_validation.h"
#include "dir_helper.h"

#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo/train.h>
#include <catboost/private/libs/algo/tree_print.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/distributed/worker.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/libs/model/ctr_data.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/monotone_constraints.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/pairs/util.h>
#include <catboost/private/libs/target/classification_target_helper.h>

#include <library/cpp/grid_creator/binarization.h>
#include <library/cpp/json/json_prettifier.h>

#include <util/generic/cast.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/scope.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>
#include <util/system/compiler.h>
#include <util/system/hp_timer.h>

#include <functional>

using namespace NCB;


static void ShrinkModel(int itCount, const TCtrHelper& ctrsHelper, TLearnProgress* progress) {
    itCount += SafeIntegerCast<int>(progress->InitTreesSize);
    progress->LeafValues.resize(itCount);
    progress->TreeStruct.resize(itCount);
    progress->TreeStats.resize(itCount);
    if (!progress->ModelShrinkHistory.empty()) {
        Y_ASSERT(SafeIntegerCast<int>(progress->ModelShrinkHistory.size()) >= itCount);
        progress->ModelShrinkHistory.resize(itCount);
    }
    progress->UsedCtrSplits.clear();
    for (const auto& tree: progress->TreeStruct) {
        for (const auto& ctr : GetUsedCtrs(tree)) {
            TProjection projection = ctr.Projection;
            ECtrType ctrType = ctrsHelper.GetCtrInfo(projection)[ctr.CtrIdx].Type;
            progress->UsedCtrSplits.insert(std::make_pair(ctrType, projection));
        }
    }
    progress->IsFoldsAndApproxDataValid = false;
}


static TDataProviders LoadPools(
    const NCatboostOptions::TPoolLoadParams& loadOptions,
    ETaskType taskType,
    ui64 cpuRamLimit,
    EObjectsOrder objectsOrder,
    TDatasetSubset trainDatasetSubset,
    TVector<NJson::TJsonValue>* classLabels,
    NPar::TLocalExecutor* const executor,
    TProfileInfo* profile
) {
    const auto& cvParams = loadOptions.CvParams;
    const bool cvMode = cvParams.FoldCount != 0;
    CB_ENSURE(
        !cvMode || loadOptions.TestSetPaths.empty(),
        "Test files are not supported in cross-validation mode"
    );

    auto pools = NCB::ReadTrainDatasets(taskType, loadOptions, objectsOrder, !cvMode, trainDatasetSubset, classLabels, executor, profile);

    if (cvMode) {
        if (cvParams.Shuffle && (pools.Learn->ObjectsData->GetOrder() != EObjectsOrder::RandomShuffled)) {
            TRestorableFastRng64 rand(cvParams.PartitionRandSeed);

            auto objectsGroupingSubset = NCB::Shuffle(pools.Learn->ObjectsGrouping, 1, &rand);
            pools.Learn = pools.Learn->GetSubset(objectsGroupingSubset, cpuRamLimit, executor);
        }

        TVector<TDataProviders> foldPools = PrepareCvFolds<TDataProviders>(
            std::move(pools.Learn),
            cvParams,
            cvParams.FoldIdx,
            /* oldCvStyleSplit */ true,
            cpuRamLimit,
            executor);
        Y_VERIFY(foldPools.size() == 1);

        profile->AddOperation("Build cv pools");

        return foldPools[0];
    } else {
        return pools;
    }
}

static bool HasInvalidValues(const TVector<TVector<double>>& treeLeafValues) {
    for (const auto& leafValuesDimension : treeLeafValues) {
        for (double leafValueCoord : leafValuesDimension) {
            if (!IsValidFloat(leafValueCoord)) {
                return true;
            }
        }
    }
    return false;
}

namespace {
struct TMetricsData {
    TVector<THolder<IMetric>> Metrics;
    bool CalcEvalMetricOnEveryIteration;
    TMaybe<ui32> MetricPeriodOffset; // shift metric period calculations by this value if defined
    TMaybe<TErrorTracker> ErrorTracker;
    TMaybe<TErrorTracker> BestModelMinTreesTracker;
    size_t ErrorTrackerMetricIdx;
};
}

static void InitializeAndCheckMetricData(
    const TTrainModelInternalOptions& internalOptions,
    const TTrainingDataProviders& data,
    const TLearnContext& ctx,
    TMetricsData* metricsData) {

    const int approxDimension = ctx.LearnProgress->ApproxDimension;
    auto& metrics = metricsData->Metrics;
    metrics = CreateMetrics(
        ctx.Params.MetricOptions,
        ctx.EvalMetricDescriptor,
        approxDimension,
        ctx.GetHasWeights()
    );
    CheckMetrics(metrics, ctx.Params.LossFunctionDescription.Get().GetLossFunction());
    if (!ctx.Params.SystemOptions->IsSingleHost()) {
        if (!AllOf(metrics, [](const auto& metric) { return metric->IsAdditiveMetric(); })) {
            CATBOOST_WARNING_LOG << "In distributed training, non-additive metrics are not evaluated on train dataset" << Endl;
        }
    }
    if (ctx.Params.LossFunctionDescription->GetLossFunction() == ELossFunction::MultiRMSE) {
        CB_ENSURE(!ctx.Layout->GetCatFeatureCount(), "Training with MultiRMSE loss function doesn't support categorical features");
    }

    CB_ENSURE(!metrics.empty(), "Eval metric is not defined");

    const bool lastTestDatasetHasTargetData = (data.Test.size() > 0) && data.Test.back()->MetaInfo.TargetCount > 0;

    const bool hasTest = data.GetTestSampleCount() > 0;
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
        = canCalcEvalMetric && (internalOptions.ForceCalcEvalMetricOnEveryIteration || errorTracker->IsActive());

    if (ctx.OutputOptions.GetMetricPeriod() > 1 && errorTracker && errorTracker->IsActive() && hasTest) {
        CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is " <<
            "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
    }

    // Use only (last_test, first_metric) for best iteration and overfitting detection
    // In case of changing the order it should be changed in GPU mode also.
    metricsData->ErrorTrackerMetricIdx = 0;

    if (internalOptions.OffsetMetricPeriodByInitModelSize) {
        metricsData->MetricPeriodOffset = ctx.LearnProgress->GetInitModelTreesSize();
    } else {
        metricsData->MetricPeriodOffset = Nothing();
    }
}

namespace {
struct TLoggingData {
    TString LearnToken;
    TVector<const TString> TestTokens;
    TLogger Logger;
};
}

static bool ShouldCalcAllMetrics(ui32 iter, const TMetricsData& metricsData, const TLearnContext& ctx) {
    const ui32 iterWithOffset = iter + metricsData.MetricPeriodOffset.GetOrElse(0);
    return ((iterWithOffset + 1) == ctx.Params.BoostingOptions->IterationCount) || // last iteration
        !(iterWithOffset % SafeIntegerCast<ui32>(ctx.OutputOptions.GetMetricPeriod()));
}

static bool ShouldCalcErrorTrackerMetric(ui32 iter, const TMetricsData& metricsData, const TLearnContext& ctx) {
    return ShouldCalcAllMetrics(iter, metricsData, ctx) || metricsData.CalcEvalMetricOnEveryIteration;
}

// Write history metrics to loggers, error trackers and get info from per iteration metric based callback.
static void ProcessHistoryMetrics(
    const TTrainingDataProviders& data,
    const TLearnContext& ctx,
    ITrainingCallbacks* trainingCallbacks,
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

    const TVector<TTimeInfo>& timeHistory = ctx.LearnProgress->MetricsAndTimeHistory.TimeHistory;
    const TVector<TVector<THashMap<TString, double>>>& testMetricsHistory =
        ctx.LearnProgress->MetricsAndTimeHistory.TestMetricsHistory;

    const bool useBestModel = ctx.OutputOptions.ShrinkModelToBestIteration();
    *continueTraining = true;
    for (int iter : xrange(ctx.LearnProgress->GetCurrentTrainingIterationCount())) {
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
            ctx.LearnProgress->MetricsAndTimeHistory.LearnMetricsHistory,
            testMetricsHistory,
            metricsData->ErrorTracker ? TMaybe<double>(metricsData->ErrorTracker->GetBestError()) : Nothing(),
            metricsData->ErrorTracker ? TMaybe<int>(metricsData->ErrorTracker->GetBestIteration()) : Nothing(),
            TProfileResults(timeHistory[iter].PassedTime, timeHistory[iter].RemainingTime),
            loggingData->LearnToken,
            loggingData->TestTokens,
            ShouldCalcAllMetrics(iter, *metricsData, ctx),
            &loggingData->Logger
        );

        *continueTraining = trainingCallbacks->IsContinueTraining(ctx.LearnProgress->MetricsAndTimeHistory);
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
    const TTrainingDataProviders& data,
    TLearnContext* ctx) {

    const bool isPairwiseScoring = IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction());
    const int defaultCalcStatsObjBlockSize = static_cast<int>(ctx->Params.ObliviousTreeOptions->DevScoreCalcObjBlockSize);

    if (ctx->UseTreeLevelCaching()) {
        ctx->SmallestSplitSideDocs.Create(
            ctx->LearnProgress->Folds,
            isPairwiseScoring,
            data.EstimatedObjectsData.GetFeatureCount() != 0,
            defaultCalcStatsObjBlockSize
        );
        ctx->PrevTreeLevelStats.Create(
            ctx->LearnProgress->Folds,
            CountNonCtrBuckets(
                *data.Learn->ObjectsData->GetFeaturesLayout(),
                *data.Learn->ObjectsData->GetQuantizedFeaturesInfo(),
                ctx->Params.CatFeatureParams->OneHotMaxSize),
            static_cast<int>(ctx->Params.ObliviousTreeOptions->MaxDepth)
        );
    }
    ctx->SampledDocs.Create(
        ctx->LearnProgress->Folds,
        isPairwiseScoring,
        data.EstimatedObjectsData.GetFeatureCount() != 0,
        defaultCalcStatsObjBlockSize,
        GetBernoulliSampleRate(ctx->Params.ObliviousTreeOptions->BootstrapConfig)
    ); // TODO(espetrov): create only if sample rate < 1
}

static void LogThatStoppingOccured(const TErrorTracker& errorTracker) {
    CATBOOST_NOTICE_LOG << "Stopped by overfitting detector "
        << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
}

static void CalcErrors(
    const TTrainingDataProviders& data,
    const TMetricsData& metricsData,
    int iter,
    TLearnContext* ctx) {

    CalcErrors(data, metricsData.Metrics, ShouldCalcAllMetrics(iter, metricsData, *ctx), ShouldCalcErrorTrackerMetric(iter, metricsData, *ctx), ctx);
}

static void Train(
    const TTrainModelInternalOptions& internalOptions,
    const TTrainingDataProviders& data,
    ITrainingCallbacks* trainingCallbacks,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* testMultiApprox // [test][dim][docIdx]
) {
    TProfileInfo& profile = ctx->Profile;

    TMetricsData metricsData;
    InitializeAndCheckMetricData(internalOptions, data, *ctx, &metricsData);

    const auto onLoadSnapshotCallback = [&] (IInputStream* in) {
        return trainingCallbacks->OnLoadSnapshot(in);
    };

    if (ctx->TryLoadProgress(onLoadSnapshotCallback) && ctx->Params.SystemOptions->IsMaster()) {
        MapRestoreApproxFromTreeStruct(ctx);
    }

    TLoggingData loggingData;
    bool continueTraining;
    ProcessHistoryMetrics(data, *ctx, trainingCallbacks, &metricsData, &loggingData, &continueTraining);

    if (continueTraining) {
        InitializeSamplingStructures(data, ctx);
    }

    THPTimer timer;

    const bool useBestModel = ctx->OutputOptions.ShrinkModelToBestIteration();
    const bool hasTest = data.GetTestSampleCount() > 0;
    const auto& metrics = metricsData.Metrics;
    auto& errorTracker = metricsData.ErrorTracker;
    const auto onSaveSnapshotCallback = [&] (IOutputStream* out) {
        trainingCallbacks->OnSaveSnapshot(out);
    };

    for (ui32 iter = ctx->LearnProgress->GetCurrentTrainingIterationCount();
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
            ctx->SaveProgress(onSaveSnapshotCallback);
            timer.Reset();
        }

        TrainOneIteration(data, ctx);

        CalcErrors(data, metricsData, iter, ctx);

        profile.AddOperation("Calc errors");

        if (hasTest && ShouldCalcErrorTrackerMetric(iter, metricsData, *ctx) && errorTracker) {
            const auto testErrors = ctx->LearnProgress->MetricsAndTimeHistory.TestMetricsHistory.back();
            const TString& errorTrackerMetricDescription = metrics[metricsData.ErrorTrackerMetricIdx]->GetDescription();
            // it is possible that metric has not been calculated because it requires target data
            // that is absent
            if (!testErrors.empty()) {
                const double* error = MapFindPtr(testErrors.back(), errorTrackerMetricDescription);
                if (error) {
                    errorTracker->AddError(*error, iter);
                    if (useBestModel && iter == static_cast<ui32>(errorTracker->GetBestIteration())) {
                        ctx->LearnProgress->BestTestApprox = ctx->LearnProgress->TestApprox.back();
                    }
                    if (useBestModel && static_cast<int>(iter + 1) >= ctx->OutputOptions.BestModelMinTrees) {
                        metricsData.BestModelMinTreesTracker->AddError(*error, iter);
                    }
                }
            }
        }

        profile.FinishIteration();

        TProfileResults profileResults = profile.GetProfileResults();
        ctx->LearnProgress->MetricsAndTimeHistory.TimeHistory.push_back(TTimeInfo(profileResults));

        Log(
            iter,
            GetMetricsDescription(metrics),
            ctx->LearnProgress->MetricsAndTimeHistory.LearnMetricsHistory,
            ctx->LearnProgress->MetricsAndTimeHistory.TestMetricsHistory,
            errorTracker ? TMaybe<double>(errorTracker->GetBestError()) : Nothing(),
            errorTracker ? TMaybe<int>(errorTracker->GetBestIteration()) : Nothing(),
            profileResults,
            loggingData.LearnToken,
            loggingData.TestTokens,
            ShouldCalcAllMetrics(iter, metricsData, *ctx),
            &loggingData.Logger
        );

        if (HasInvalidValues(ctx->LearnProgress->LeafValues.back())) {
            ctx->LearnProgress->LeafValues.pop_back();
            ctx->LearnProgress->TreeStruct.pop_back();
            if (!ctx->LearnProgress->ModelShrinkHistory.empty()) {
                ctx->LearnProgress->ModelShrinkHistory.pop_back();
            }
            CATBOOST_WARNING_LOG << "Training has stopped (degenerate solution on iteration "
                << iter << ", probably too small l2-regularization, try to increase it)" << Endl;
            break;
        }

        continueTraining = trainingCallbacks->IsContinueTraining(ctx->LearnProgress->MetricsAndTimeHistory);
    }

    ctx->SaveProgress(onSaveSnapshotCallback);

    if (hasTest) {
        (*testMultiApprox) = ctx->LearnProgress->TestApprox;
        if (useBestModel) {
            (*testMultiApprox)[0] = ctx->LearnProgress->BestTestApprox;
        }
    }

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
            ShrinkModel(bestModelIterations, ctx->CtrsHelper, ctx->LearnProgress.Get());
        }
    }

    if (!ctx->LearnProgress->ModelShrinkHistory.empty()) {
        const int treeCount = ctx->LearnProgress->ModelShrinkHistory.size();
        Y_ASSERT(SafeIntegerCast<int>(ctx->LearnProgress->LeafValues.size()) == treeCount);
        double accumulatedTreeShrinkage = 1.0;
        for (int treeIndex = treeCount - 1; treeIndex >= 0; --treeIndex) {
            auto& treeLeafValues = ctx->LearnProgress->LeafValues[treeIndex];
            treeLeafValues = ScaleElementwise(accumulatedTreeShrinkage, treeLeafValues);
            accumulatedTreeShrinkage *= ctx->LearnProgress->ModelShrinkHistory[treeIndex];
        }
    }
}


static THolder<TNonSymmetricTreeNode> BuildTree(
    int nodeIdx,
    TConstArrayRef<TSplitNode> nodes,
    TConstArrayRef<TVector<double>> leafValues,
    TConstArrayRef<double> leafWeights,
    std::function<TModelSplit(TSplit)> getModelSplit
) {
    auto finalNode = MakeHolder<TNonSymmetricTreeNode>();
    if (nodeIdx < 0) {
        int leafIdx = ~nodeIdx;
        if (leafValues.size() == 1) {
            finalNode->Value = leafValues[0][leafIdx];
        } else {
            TVector<double> value(leafValues.size());
            for (auto dim : xrange(leafValues.size())) {
                value[dim] = leafValues[dim][leafIdx];
            }
            finalNode->Value = std::move(value);
        }
        finalNode->NodeWeight = leafWeights[leafIdx];
    } else {
        const auto& node = nodes[nodeIdx];
        finalNode->SplitCondition = getModelSplit(node.Split);
        finalNode->Left = BuildTree(node.Left, nodes, leafValues, leafWeights, getModelSplit);
        finalNode->Right = BuildTree(node.Right, nodes, leafValues, leafWeights, getModelSplit);
    }
    return finalNode;
}


static void SaveModel(
    const TTrainingDataProviders& trainingDataForCpu,
    const TLearnContext& ctx,
    TMaybe<TFullModel*> initModel,
    TMaybe<ui32> initLearnProgressLearnAndTestQuantizedFeaturesCheckSum,
    TFullModel* dstModel
) {
    const auto& target = ctx.LearnProgress->AveragingFold.LearnTarget;
    const TQuantizedFeaturesInfo& quantizedFeaturesInfo
        = *(trainingDataForCpu.Learn->ObjectsData->GetQuantizedFeaturesInfo());

    TPerfectHashedToHashedCatValuesMap perfectHashedToHashedCatValuesMap
        = quantizedFeaturesInfo.CalcPerfectHashedToHashedCatValuesMap(ctx.LocalExecutor);
    if (ctx.OutputOptions.AllowWriteFiles()) {
        TString tmpDir;
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(ctx.OutputOptions.GetTrainDir(), &tmpDir);
        quantizedFeaturesInfo.UnloadCatFeaturePerfectHashFromRam(tmpDir);
    }

    const TQuantizedEstimatedFeaturesInfo onlineQuantizedEstimatedFeaturesInfo
        = ctx.LearnProgress->GetOnlineEstimatedFeaturesInfo();

    TModelTrees modelTrees;
    THashMap<TFeatureCombination, TProjection> featureCombinationToProjectionMap;
    const std::function<TModelSplit(TSplit)> getModelSplit = [&] (const TSplit& split) {
        auto modelSplit = split.GetModelSplit(
            ctx,
            perfectHashedToHashedCatValuesMap,
            *trainingDataForCpu.FeatureEstimators,
            trainingDataForCpu.EstimatedObjectsData.QuantizedEstimatedFeaturesInfo,
            onlineQuantizedEstimatedFeaturesInfo
        );
        if (modelSplit.Type == ESplitType::OnlineCtr) {
            featureCombinationToProjectionMap[modelSplit.OnlineCtr.Ctr.Base.Projection] = split.Ctr.Projection;
        }
        return modelSplit;
    };
    if (ctx.Params.ObliviousTreeOptions->GrowPolicy == EGrowPolicy::SymmetricTree) {
        TObliviousTreeBuilder builder(
            ctx.LearnProgress->FloatFeatures,
            ctx.LearnProgress->CatFeatures,
            ctx.LearnProgress->TextFeatures,
            ctx.LearnProgress->ApproxDimension);
        for (size_t treeId = 0; treeId < ctx.LearnProgress->TreeStruct.size(); ++treeId) {
            TVector<TModelSplit> modelSplits;
            Y_ASSERT(HoldsAlternative<TSplitTree>(ctx.LearnProgress->TreeStruct[treeId]));
            TVector<TSplit> splits = Get<TSplitTree>(ctx.LearnProgress->TreeStruct[treeId]).Splits;
            for (const auto& split : splits) {
                modelSplits.push_back(getModelSplit(split));
            }
            builder.AddTree(modelSplits, ctx.LearnProgress->LeafValues[treeId], ctx.LearnProgress->TreeStats[treeId].LeafWeightsSum);
        }
        builder.Build(&modelTrees);
    } else {
        TNonSymmetricTreeModelBuilder builder(
            ctx.LearnProgress->FloatFeatures,
            ctx.LearnProgress->CatFeatures,
            ctx.LearnProgress->TextFeatures,
            ctx.LearnProgress->ApproxDimension);
        for (size_t treeId = 0; treeId < ctx.LearnProgress->TreeStruct.size(); ++treeId) {
            Y_ASSERT(HoldsAlternative<TNonSymmetricTreeStructure>(ctx.LearnProgress->TreeStruct[treeId]));
            const auto& structure = Get<TNonSymmetricTreeStructure>(ctx.LearnProgress->TreeStruct[treeId]);
            auto tree = BuildTree(
                structure.GetRoot(),
                structure.GetNodes(),
                ctx.LearnProgress->LeafValues[treeId],
                ctx.LearnProgress->TreeStats[treeId].LeafWeightsSum,
                getModelSplit);
            builder.AddTree(std::move(tree));
        }
        builder.Build(&modelTrees);
    }


//    TODO(kirillovs,espetrov): return this code after fixing R and Python wrappers
//    for (auto& oheFeature : modelTrees.OneHotFeatures) {
//        for (const auto& value : oheFeature.Values) {
//            oheFeature.StringValues.push_back(pools.Learn->CatFeaturesHashToString.at(value));
//        }
//    }
    TClassificationTargetHelper classificationTargetHelper(
        ctx.LearnProgress->LabelConverter,
        ctx.Params.DataProcessingOptions
    );

    TDatasetDataForFinalCtrs datasetDataForFinalCtrs;
    datasetDataForFinalCtrs.Data = trainingDataForCpu;
    datasetDataForFinalCtrs.LearnPermutation = &ctx.LearnProgress->AveragingFold.LearnPermutation->GetObjectsIndexing();
    if (target.size() == 1) { // since counters are not implemented for multi-dimensional target
        datasetDataForFinalCtrs.Targets = target[0];
    }
    datasetDataForFinalCtrs.LearnTargetClass = &ctx.LearnProgress->AveragingFold.LearnTargetClass;
    datasetDataForFinalCtrs.TargetClassesCount = &ctx.LearnProgress->AveragingFold.TargetClassesCount;

    {
        const bool addResultModelToInitModel = ctx.LearnProgress->SeparateInitModelTreesSize != 0;

        TMaybe<TFullModel> fullModel;
        TFullModel* modelPtr = nullptr;
        if (dstModel && !addResultModelToInitModel) {
            modelPtr = dstModel;
        } else {
            fullModel.ConstructInPlace();
            modelPtr = &*fullModel;
        }

        *modelPtr->ModelTrees.GetMutable() = std::move(modelTrees);
        modelPtr->SetScaleAndBias({1, ctx.LearnProgress->StartingApprox.GetOrElse(0)});
        modelPtr->UpdateDynamicData();

        EFinalFeatureCalcersComputationMode featureCalcerComputationMode
            = ctx.OutputOptions.GetFinalFeatureCalcerComputationMode();
        if (modelPtr->ModelTrees->GetTextFeatures().empty() ||
            modelPtr->ModelTrees->GetEstimatedFeatures().empty()
        ) {
            featureCalcerComputationMode = EFinalFeatureCalcersComputationMode::Skip;
        }

        NCB::TCoreModelToFullModelConverter coreModelToFullModelConverter(
            ctx.Params,
            ctx.OutputOptions,
            classificationTargetHelper,
            ctx.Params.CatFeatureParams->CtrLeafCountLimit,
            ctx.Params.CatFeatureParams->StoreAllSimpleCtrs,
            ctx.OutputOptions.GetFinalCtrComputationMode(),
            featureCalcerComputationMode
        );

        coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
            std::move(datasetDataForFinalCtrs),
            std::move(featureCombinationToProjectionMap)
        ).WithPerfectHashedToHashedCatValuesMap(
            &perfectHashedToHashedCatValuesMap
        ).WithCoreModelFrom(
            modelPtr
        ).WithObjectsDataFrom(
            trainingDataForCpu.Learn->ObjectsData
        ).WithFeatureEstimators(
            trainingDataForCpu.FeatureEstimators
        );

        if (dstModel || addResultModelToInitModel) {
            coreModelToFullModelConverter.Do(true, modelPtr, ctx.LocalExecutor);
            if (addResultModelToInitModel) {
                TVector<const TFullModel*> models = {*initModel, modelPtr};
                TVector<double> weights = {1.0, 1.0};
                (dstModel ? *dstModel : *modelPtr) = SumModels(models, weights);

                if (!dstModel) {
                    const bool allLearnObjectsDataIsAvailable
                        = initLearnProgressLearnAndTestQuantizedFeaturesCheckSum &&
                            (*initLearnProgressLearnAndTestQuantizedFeaturesCheckSum ==
                             ctx.LearnProgress->LearnAndTestQuantizedFeaturesCheckSum);

                    ExportFullModel(
                        *modelPtr,
                        ctx.OutputOptions.CreateResultModelFullPath(),
                        allLearnObjectsDataIsAvailable ?
                            TMaybe<TObjectsDataProvider*>(trainingDataForCpu.Learn->ObjectsData.Get()) :
                            Nothing(),
                        ctx.OutputOptions.GetModelFormats(),
                        ctx.OutputOptions.AddFileFormatExtension()
                    );
                }
            }
        } else if (!dstModel) {
            coreModelToFullModelConverter.Do(
                ctx.OutputOptions.CreateResultModelFullPath(),
                ctx.OutputOptions.GetModelFormats(),
                ctx.OutputOptions.AddFileFormatExtension(),
                ctx.LocalExecutor
            );
        }
    }
}


namespace {
    class TCPUModelTrainer : public IModelTrainer {

        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            ITrainingCallbacks* trainingCallbacks,
            TMaybe<TFullModel*> initModel,
            THolder<TLearnProgress> initLearnProgress,
            TDataProviders initModelApplyCompatiblePools,
            NPar::TLocalExecutor* localExecutor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* dstModel,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
            THolder<TLearnProgress>* dstLearnProgress
        ) const override {
            if (!internalOptions.CalcMetricsOnly) {
                if (dstModel != nullptr) {
                    CB_ENSURE(
                        !outputOptions.ResultModelPath.IsSet(),
                        "Both dstModel != nullptr and ResultModelPath is set"
                    );
                } else {
                    CB_ENSURE(
                        !outputOptions.ResultModelPath.Get().empty(),
                        "Both dstModel == nullptr and ResultModelPath is empty"
                    );
                }
            }

            trainingData.Learn->ObjectsData->CheckCPUTrainCompatibility();
            for (auto& test : trainingData.Test) {
                test->ObjectsData->CheckCPUTrainCompatibility();
            }

            const TString trainingOptionsFileName = outputOptions.CreateTrainingOptionsFullPath();
            if (!trainingOptionsFileName.empty()) {
                TOFStream trainingOptionsFile(trainingOptionsFileName);
                trainingOptionsFile.Write(NJson::PrettifyJson(ToString(catboostOptions)));
            }

            // need to save it because initLearnProgress is moved to TLearnContext
            TMaybe<ui32> initLearnProgressLearnAndTestQuantizedFeaturesCheckSum;
            if (initLearnProgress) {
                initLearnProgressLearnAndTestQuantizedFeaturesCheckSum = initLearnProgress->LearnAndTestQuantizedFeaturesCheckSum;
            }

            if (catboostOptions.BoostingOptions->BoostFromAverage.Get()) {
                CB_ENSURE(!initModel, "You can't use boost_from_average with initial model now.");
                CB_ENSURE(!trainingData.Learn->TargetData->GetBaseline(), "You can't use boost_from_average with baseline now.");
                for (ui32 testIdx = 0; testIdx < trainingData.Test.size(); ++testIdx) {
                    CB_ENSURE(!trainingData.Test[testIdx]->TargetData->GetBaseline(), "You can't use boost_from_average with baseline now.");
                }
            }

            if (catboostOptions.BoostingOptions->ModelShrinkRate.Get() != 0.0f) {
                CB_ENSURE(!initModel,
                    "Usage of model_shrink_rate option in combination with learning continuation is unimplemented yet."
                );
                auto errMessage = "Usage of model_shrink_rate option in combination with baseline is unimplemented yet.";
                CB_ENSURE(!trainingData.Learn->TargetData->GetBaseline(), errMessage);
                for (ui32 testIdx = 0; testIdx < trainingData.Test.size(); ++testIdx) {
                    CB_ENSURE(!trainingData.Test[testIdx]->TargetData->GetBaseline(), errMessage);
                }
            }

            TMaybe<double> startingApprox;
            if (catboostOptions.BoostingOptions->BoostFromAverage.Get()) {
                // TODO(fedorlebed): add boost from average support for multiregression
                CB_ENSURE(trainingData.Learn->TargetData->GetTargetDimension() != 0, "Target is required for boosting from average");
                CB_ENSURE(trainingData.Learn->TargetData->GetTargetDimension() == 1, "Multi-dimensional target boosting from average is unimplemented yet");

                startingApprox = CalcOptimumConstApprox(
                    catboostOptions.LossFunctionDescription,
                    *trainingData.Learn->TargetData->GetOneDimensionalTarget(),
                    GetWeights(*trainingData.Learn->TargetData)
                );
            }
            TLearnContext ctx(
                catboostOptions,
                objectiveDescriptor,
                evalMetricDescriptor,
                outputOptions,
                trainingData,
                labelConverter,
                startingApprox,
                rand,
                std::move(initModel),
                std::move(initLearnProgress),
                std::move(initModelApplyCompatiblePools),
                localExecutor
            );

            DumpMemUsage("Before start train");

            const auto& systemOptions = ctx.Params.SystemOptions;
            if (!systemOptions->IsSingleHost()) { // send target, weights, baseline (if present), binarized features to workers and ask them to create plain folds
                CB_ENSURE(IsPlainMode(ctx.Params.BoostingOptions->BoostingType), "Distributed training requires plain boosting");
                CB_ENSURE(!ctx.Layout->GetCatFeatureCount(), "Distributed training doesn't support categorical features");
                MapBuildPlainFold(&ctx);
            }
            TVector<TVector<double>> oneRawValues(ctx.LearnProgress->ApproxDimension);
            TVector<TVector<TVector<double>>> rawValues(trainingData.Test.size(), oneRawValues);

            Train(internalOptions, trainingData, trainingCallbacks, &ctx, &rawValues);

            if (!dstLearnProgress) {
                // Save memory as it is no longer needed
                ctx.LearnProgress->Folds.clear();
            }

            for (int testIdx = 0; testIdx < trainingData.Test.ysize(); ++testIdx) {
                evalResultPtrs[testIdx]->SetRawValuesByMove(rawValues[testIdx]);
            }

            if (metricsAndTimeHistory) {
                *metricsAndTimeHistory = ctx.LearnProgress->MetricsAndTimeHistory;
            }

            if (!internalOptions.CalcMetricsOnly) {
                SaveModel(
                    trainingData,
                    ctx,
                    initModel,
                    initLearnProgressLearnAndTestQuantizedFeaturesCheckSum,
                    dstModel);
            }
            if (dstLearnProgress) {
                ctx.LearnProgress->PrepareForContinuation();
                *dstLearnProgress = std::move(ctx.LearnProgress);
            }
        }

        void ModelBasedEval(
            const NCatboostOptions::TCatBoostOptions& /*catboostOptions*/,
            const NCatboostOptions::TOutputFilesOptions& /*outputOptions*/,
            TTrainingDataProviders /*trainingData*/,
            const TLabelConverter& /*labelConverter*/,
            NPar::TLocalExecutor* /*localExecutor*/) const override {
            CB_ENSURE(false, "Model based eval is not implemented for CPU");
        }
    };

}

TTrainerFactory::TRegistrator<TCPUModelTrainer> CPURegistrator(ETaskType::CPU);

static bool HaveLearnFeaturesInMemory(
    const NCatboostOptions::TPoolLoadParams* loadOptions,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions
) {
    #if defined(USE_MPI)
    const bool isGpuDistributed = catBoostOptions.GetTaskType() == ETaskType::GPU;
    #else
    const bool isGpuDistributed = false;
    #endif
    const bool isCpuDistributed = catBoostOptions.SystemOptions->IsMaster();
    if (!isCpuDistributed && !isGpuDistributed) {
        return true;
    }
    if (loadOptions == nullptr) {
        return true;
    }
    const auto& learnSetPath = loadOptions->LearnSetPath;
    const bool isQuantized = learnSetPath.Scheme.find("quantized") != std::string::npos;
    return !IsSharedFs(learnSetPath) || !isQuantized;
}

static void TrainModel(
    const NJson::TJsonValue& trainOptionsJson,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TDataProviders pools,
    TMaybe<TFullModel*> initModel,
    THolder<TLearnProgress> initLearnProgress,
    const NCatboostOptions::TPoolLoadParams* poolLoadOptions,
    const TString& outputModelPath,
    TFullModel* dstModel,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
    THolder<TLearnProgress>* dstLearnProgress,
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

    NCatboostOptions::TOutputFilesOptions updatedOutputOptions(outputOptions);
    if (outputModelPath) {
        updatedOutputOptions.ResultModelPath = outputModelPath;
    }

    NJson::TJsonValue updatedTrainOptionsJson = trainOptionsJson;

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "Can't load GPU learning library. Module was not compiled or driver  is incompatible with package. Please install latest NVDIA driver and check again");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }

    if (outputOptions.SaveSnapshot()) {
        UpdateUndefinedRandomSeed(taskType, updatedOutputOptions, &updatedTrainOptionsJson, [&](IInputStream* in, TString& params) {
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
            catBoostOptions.DataProcessingOptions->PerFloatFeatureQuantization.Get(),
            catBoostOptions.DataProcessingOptions->TextProcessingOptions.Get(),
            /*allowNansInTestOnly*/true
        );
        /* TODO(akhropov): reuse float features quantization data from initLearnProgress if data quantization
         * options and raw data is the same
         */
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

    if (updatedOutputOptions.GetVerbosePeriod() == 0 && catBoostOptions.LoggingLevel.NotSet()) {
        catBoostOptions.LoggingLevel.SetDefault(ELoggingLevel::Silent);
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

    const bool needInitModelApplyCompatiblePools = initModel.Defined();

    TString tmpDir;
    if (outputOptions.AllowWriteFiles()) {
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputOptions.GetTrainDir(), &tmpDir);
    }

    const bool haveLearnFeaturesInMemory = HaveLearnFeaturesInMemory(poolLoadOptions, catBoostOptions);
    CB_ENSURE_INTERNAL(
        haveLearnFeaturesInMemory || poolLoadOptions, "Learn dataset is not loaded, and load options are not provided");
    TTrainingDataProviders trainingData = GetTrainingData(
        needInitModelApplyCompatiblePools ? pools : std::move(pools),
        /* borders */ Nothing(), // borders are already loaded to quantizedFeaturesInfo
        /*ensureConsecutiveIfDenseLearnFeaturesDataForCpu*/ haveLearnFeaturesInMemory,
        outputOptions.AllowWriteFiles(),
        tmpDir,
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        executor,
        &rand,
        initModel);
    if (catBoostOptions.SystemOptions->IsMaster()) {
        InitializeMaster(catBoostOptions.SystemOptions);
        if (!haveLearnFeaturesInMemory) {
            SetTrainDataFromQuantizedPool(
                *poolLoadOptions,
                catBoostOptions,
                *trainingData.Learn->ObjectsGrouping,
                *trainingData.Learn->MetaInfo.FeaturesLayout,
                &rand
            );
        } else {
            SetTrainDataFromMaster(
                trainingData,
                ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
                executor
            );
        }
    }

    CheckConsistency(trainingData);

    SetDataDependentDefaults(
        trainingData.Learn->MetaInfo,
        trainingData.Test.size() > 0 ?
            TMaybe<NCB::TDataMetaInfo>(trainingData.Test[0]->MetaInfo) :
            Nothing(),
        initModel.Defined(),
        !!initLearnProgress,
        &updatedOutputOptions.UseBestModel,
        &catBoostOptions
    );

    CB_ENSURE(
        haveLearnFeaturesInMemory || catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain,
        "Only plain boosting is supported in distributed training for schema " << poolLoadOptions->LearnSetPath.Scheme);

    // Eval metric may not be set. If that's the case, we assign it to objective metric
    InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

    if (outputOptions.NeedSaveBorders()) {
        SaveBordersAndNanModesToFileInMatrixnetFormat(
            outputOptions.CreateOutputBordersFullPath(),
            *trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo());
    }

    const auto defaultTrainingCallbacks = MakeHolder<ITrainingCallbacks>();
    modelTrainerHolder->TrainModel(
        TTrainModelInternalOptions(),
        catBoostOptions,
        updatedOutputOptions,
        objectiveDescriptor,
        evalMetricDescriptor,
        std::move(trainingData),
        labelConverter,
        defaultTrainingCallbacks.Get(),
        std::move(initModel),
        std::move(initLearnProgress),
        needInitModelApplyCompatiblePools ? std::move(pools) : TDataProviders(),
        executor,
        &rand,
        dstModel,
        evalResultPtrs,
        metricsAndTimeHistory,
        dstLearnProgress);
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

    CB_ENSURE(
        !(loadOptions.CvParams.FoldCount != 0 && catBoostOptions.GetTaskType() == ETaskType::CPU && catBoostOptions.SystemOptions->IsMaster()),
        "Distributed training on CPU does not support test and train datasests specified by cross validation options"
    );

    const auto evalOutputFileName = outputOptions.CreateEvalFullPath();

    const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
    const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
    bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();

    auto modelFormat = outputOptions.GetModelFormats()[0];
    for (int formatIdx = 1; !IsDeserializableModelFormat(modelFormat) && formatIdx < outputOptions.GetModelFormats().ysize(); ++formatIdx) {
        modelFormat = outputOptions.GetModelFormats()[formatIdx];
    }
    if (!evalOutputFileName.empty()) {
        CB_ENSURE(
            IsDeserializableModelFormat(modelFormat),
            "All chosen model formats not supported deserialization. Add CatboostBinary model-format to save eval-file."
        );
    }
    if (needFstr) {
        CB_ENSURE(
            IsDeserializableModelFormat(modelFormat),
            "All chosen model formats not supported deserialization. Add CatboostBinary model-format to calc fstr."
        );
    }


    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(catBoostOptions.SystemOptions.Get().NumThreads.Get() - 1);

    TVector<NJson::TJsonValue> classLabels = catBoostOptions.DataProcessingOptions->ClassLabels;
    const auto objectsOrder = catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ?
        EObjectsOrder::Ordered : EObjectsOrder::Undefined;

    auto lossFunction = catBoostOptions.LossFunctionDescription->GetLossFunction();
    auto fstrType = AdjustFeatureImportanceType(outputOptions.GetFstrType(), lossFunction);
    const bool isLossFunctionChangeFstr = needFstr && fstrType == EFstrType::LossFunctionChange;

    const bool haveLearnFeaturesInMemory = HaveLearnFeaturesInMemory(&loadOptions, catBoostOptions);
    if (needFstr && !haveLearnFeaturesInMemory && fstrType != EFstrType::PredictionValuesChange) {
        const auto& pathScheme = loadOptions.LearnSetPath.Scheme;
        CB_ENSURE(
            !outputOptions.IsFstrTypeSet(),
            "Only fstr type " << EFstrType::PredictionValuesChange << " is supported in distributed training for schema " << pathScheme);
        CATBOOST_WARNING_LOG << "Recommended fstr type " << fstrType << " is not supported in distributed training for schema "
            << pathScheme << ";" << " fstr type is set to " << EFstrType::PredictionValuesChange << Endl;
        fstrType = EFstrType::PredictionValuesChange;
    }

    TDataProviders pools = LoadPools(
        loadOptions,
        catBoostOptions.GetTaskType(),
        ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
        objectsOrder,
        TDatasetSubset::MakeColumns(haveLearnFeaturesInMemory),
        &classLabels,
        &executor,
        &profile);

    const bool hasTextFeatures = pools.Learn->MetaInfo.FeaturesLayout->GetTextFeatureCount() > 0;
    if (hasTextFeatures) {
        needFstr = false;
    }

    TVector<TString> outputColumns;
    if (!evalOutputFileName.empty() && !pools.Test.empty()) {
        outputColumns = outputOptions.GetOutputColumns(pools.Test[0]->MetaInfo.TargetCount);
        for (int testIdx = 0; testIdx < pools.Test.ysize(); ++testIdx) {
            const TDataProvider& testPool = *pools.Test[testIdx];

            CB_ENSURE(
                outputColumns == outputOptions.GetOutputColumns(testPool.MetaInfo.TargetCount),
                "Inconsistent output columns between test sets"
            );

            ValidateColumnOutput(
                outputColumns,
                testPool,
                loadOptions.CvParams.FoldCount > 0
            );
        }
    }
    TVector<TEvalResult> evalResults(pools.Test.ysize());

    NJson::TJsonValue updatedTrainJson = trainJson;
    UpdateUndefinedClassLabels(classLabels, &updatedTrainJson);

    // create here to possibly load borders
    auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        *pools.Learn->MetaInfo.FeaturesLayout,
        catBoostOptions.DataProcessingOptions->IgnoredFeatures.Get(),
        catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
        catBoostOptions.DataProcessingOptions->PerFloatFeatureQuantization.Get(),
        catBoostOptions.DataProcessingOptions->TextProcessingOptions.Get(),
        /*allowNansInTestOnly*/true
    );
    if (loadOptions.BordersFile) {
        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
            loadOptions.BordersFile,
            quantizedFeaturesInfo.Get());
    }

    bool needPoolAfterTrain = !evalOutputFileName.empty() || isLossFunctionChangeFstr;
    TrainModel(
        updatedTrainJson,
        outputOptions,
        quantizedFeaturesInfo,
        /*objectiveDescriptor*/ Nothing(),
        /*evalMetricDescriptor*/ Nothing(),
        needPoolAfterTrain ? pools : std::move(pools),
        /*initModel*/ Nothing(),
        /*initLearnProgress*/ nullptr,
        &loadOptions,
        /*outputModelPath*/ "",
        /*dstModel*/ nullptr,
        GetMutablePointers(evalResults),
        /*metricsAndTimeHistory*/ nullptr,
        /*dstLearnProgress*/ nullptr,
        &executor
    );

    const auto fullModelPath = NCatboostOptions::AddExtension(
        modelFormat,
        outputOptions.CreateResultModelFullPath(),
        outputOptions.AddFileFormatExtension());

    TSetLoggingVerbose inThisScope2;
    if (!evalOutputFileName.empty()) {
        TFullModel model = ReadModel(fullModelPath, modelFormat);
        const TExternalLabelsHelper visibleLabelsHelper(model);
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
                model.GetLossFunctionName(),
                visibleLabelsHelper,
                testPool,
                &fileStream,
                testSetPath,
                {testIdx, pools.Test.ysize()},
                loadOptions.ColumnarPoolFormatParams.DsvFormat,
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
        const bool useLearnToCalcFstr = haveLearnFeaturesInMemory && isLossFunctionChangeFstr;
        CalcAndOutputFstr(
            model,
            useLearnToCalcFstr ? pools.Learn : nullptr,
            &executor,
            &fstrRegularFileName,
            &fstrInternalFileName,
            fstrType);
    }

    CATBOOST_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
}

static void ValidateFeaturesToEvaluate(const NJson::TJsonValue& trainOptionsJson, ui32 featureCount) {
    const auto maxFeatureToEvaluateIdx = GetOptionFeaturesToEvaluate(trainOptionsJson).back();
    CB_ENSURE(
        maxFeatureToEvaluateIdx < featureCount,
        "Feature index " << maxFeatureToEvaluateIdx << " is too large; dataset has only "
        << featureCount << " features");
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

    ValidateFeaturesToEvaluate(trainOptionsJson, pools.Learn->MetaInfo.GetFeatureCount());

    if (!quantizedFeaturesInfo) {
        quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *learnFeaturesLayout,
            catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get(),
            catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
            catBoostOptions.DataProcessingOptions->PerFloatFeatureQuantization.Get(),
            catBoostOptions.DataProcessingOptions->TextProcessingOptions.Get(),
            /*allowNansInTestOnly*/true
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

    TString tmpDir;
    if (outputOptions.AllowWriteFiles()) {
        NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputOptions.GetTrainDir(), &tmpDir);
    }

    TTrainingDataProviders trainingData = GetTrainingData(
        std::move(pools),
        /* borders */ Nothing(), // borders are already loaded to quantizedFeaturesInfo
        /*ensureConsecutiveIfDenseLearnFeaturesDataForCpu*/ true,
        outputOptions.AllowWriteFiles(),
        tmpDir,
        quantizedFeaturesInfo,
        &catBoostOptions,
        &labelConverter,
        executor,
        &rand);

    CheckConsistency(trainingData);

    NCatboostOptions::TOutputFilesOptions updatedOutputOptions(outputOptions);
    SetDataDependentDefaults(
        trainingData.Learn->MetaInfo,
        trainingData.Test.size() > 0 ?
            TMaybe<NCB::TDataMetaInfo>(trainingData.Test[0]->MetaInfo) :
            Nothing(),
        /*continueFromModel*/ false,
        /*continueFromProgress*/ false,
        &updatedOutputOptions.UseBestModel,
        &catBoostOptions
    );
    InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric, &catBoostOptions.MetricOptions->EvalMetric);

    modelTrainerHolder->ModelBasedEval(
        catBoostOptions,
        updatedOutputOptions,
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

    TVector<NJson::TJsonValue> classLabels = catBoostOptions.DataProcessingOptions->ClassLabels;

    TDataProviders pools = LoadPools(
        loadOptions,
        catBoostOptions.GetTaskType(),
        ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
        catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ?
            EObjectsOrder::Ordered : EObjectsOrder::Undefined,
        TDatasetSubset::MakeColumns(),
        &classLabels,
        &executor,
        &profile);

    ValidateFeaturesToEvaluate(trainJson, pools.Learn->MetaInfo.GetFeatureCount());

    // create here to possibly load borders
    auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        *pools.Learn->MetaInfo.FeaturesLayout,
        catBoostOptions.DataProcessingOptions->IgnoredFeatures.Get(),
        catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization.Get(),
        catBoostOptions.DataProcessingOptions->PerFloatFeatureQuantization.Get(),
        catBoostOptions.DataProcessingOptions->TextProcessingOptions.Get(),
        /*allowNansInTestOnly*/true
    );
    if (loadOptions.BordersFile) {
        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
            loadOptions.BordersFile,
            quantizedFeaturesInfo.Get());
    }

    NJson::TJsonValue updatedTrainJson = trainJson;
    UpdateUndefinedClassLabels(classLabels, &updatedTrainJson);

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
    NJson::TJsonValue plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    NCB::TDataProviders pools, // not rvalue reference because Cython does not support them
    TMaybe<TFullModel*> initModel,
    THolder<TLearnProgress>* initLearnProgress,
    const TString& outputModelPath,
    TFullModel* dstModel,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
    THolder<TLearnProgress>* dstLearnProgress
) {
    NJson::TJsonValue trainOptionsJson;
    NJson::TJsonValue outputFilesOptionsJson;
    ConvertIgnoredFeaturesFromStringToIndices(pools.Learn.Get()->MetaInfo, &plainJsonParams);
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptionsJson, &outputFilesOptionsJson);
    ConvertParamsToCanonicalFormat(pools.Learn.Get()->MetaInfo, &trainOptionsJson);

    CB_ENSURE(!plainJsonParams.Has("node_type") || plainJsonParams["node_type"] == "SingleHost", "CatBoost Python module does not support distributed training");

    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.Load(outputFilesOptionsJson);

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(
        NCatboostOptions::GetThreadCount(trainOptionsJson) - 1);

    TrainModel(
        trainOptionsJson,
        outputOptions,
        quantizedFeaturesInfo,
        objectiveDescriptor,
        evalMetricDescriptor,
        std::move(pools),
        std::move(initModel),
        initLearnProgress ? std::move(*initLearnProgress) : THolder<TLearnProgress>(),
        /*poolLoadOptions*/nullptr,
        outputModelPath,
        dstModel,
        evalResultPtrs,
        metricsAndTimeHistory,
        dstLearnProgress,
        &executor);
}
