#pragma once

#include <util/random/shuffle.h>
#include <util/random/normal.h>

#include "learn_context.h"
#include "params.h"
#include "target_classifier.h"
#include "full_features.h"
#include "error_functions.h"
#include "online_predictor.h"
#include "bin_tracker.h"
#include "rand_score.h"
#include "fold.h"
#include "online_ctr.h"
#include "score_calcer.h"
#include "approx_calcer.h"
#include "index_hash_calcer.h"
#include "greedy_tensor_search.h"
#include "metric.h"
#include "logger.h"

#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/model/hash.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/projection.h>
#include <catboost/libs/model/tensor_struct.h>

#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/logging/profile_info.h>

#include <library/fast_exp/fast_exp.h>
#include <library/fast_log/fast_log.h>

#include <util/string/vector.h>
#include <util/string/iterator.h>
#include <util/stream/file.h>
#include <util/generic/ymath.h>

void ShrinkModel(int itCount, TCoreModel* model);

yvector<int> CountSplits(const yvector<yvector<float>>& borders);

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx);

template <typename TError>
TError BuildError(const TFitParams& params) {
    return TError(params.StoreExpApprox);
}

template <>
inline TQuantileError BuildError<TQuantileError>(const TFitParams& params) {
    auto lossParams = GetLossParams(params.Objective);
    if (lossParams.empty()) {
        return TQuantileError(params.StoreExpApprox);
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << params.Objective);
        return TQuantileError(lossParams["alpha"], params.StoreExpApprox);
    }
}

template <>
inline TLogLinQuantileError BuildError<TLogLinQuantileError>(const TFitParams& params) {
    auto lossParams = GetLossParams(params.Objective);
    if (lossParams.empty()) {
        return TLogLinQuantileError(params.StoreExpApprox);
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << params.Objective);
        return TLogLinQuantileError(lossParams["alpha"], params.StoreExpApprox);
    }
}

template <>
inline TCustomError BuildError<TCustomError>(const TFitParams& params) {
    Y_ASSERT(params.ObjectiveDescriptor.Defined());
    return TCustomError(params);
}

using TTrainFunc = std::function<void(const TTrainData& data,
                                      TLearnContext* ctx,
                                      yvector<yvector<double>>* testMultiApprox)>;

using TTrainOneIterationFunc = std::function<void(const TTrainData& data,
                                                  TLearnContext* ctx)>;

template <typename TError>
inline void CalcWeightedDerivatives(const yvector<yvector<double>>& approx,
                                    const yvector<float>& target,
                                    const yvector<float>& weight,
                                    const yvector<ui32>& queriesId,
                                    const yhash<ui32, ui32>& queriesSize,
                                    const yvector<yvector<TCompetitor>>& competitors,
                                    const TError& error,
                                    int tailFinish,
                                    TLearnContext* ctx,
                                    yvector<yvector<double>>* derivatives) {
    if (error.GetErrorType() == EErrorType::QuerywiseError) {
        yvector<TDer1Der2> ders((*derivatives)[0].ysize());
        error.CalcDersForQueries(0, tailFinish, approx[0], target, weight, queriesId, queriesSize, &ders);
        for (int docId = 0; docId < ders.ysize(); ++docId) {
            (*derivatives)[0][docId] = ders[docId].Der1;
        }
    } else if (error.GetErrorType() == EErrorType::PairwiseError) {
        error.CalcDersPairs(approx[0], competitors, 0, tailFinish, &(*derivatives)[0]);
    } else {
        int approxDimension = approx.ysize();
        NPar::TLocalExecutor::TBlockParams blockParams(0, tailFinish);
        blockParams.SetBlockSize(1000);

        Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
        if (approxDimension == 1) {
            ctx->LocalExecutor.ExecRange([&](int blockId) {
                const int blockOffset = blockId * blockParams.GetBlockSize();
                error.CalcFirstDerRange(blockOffset, Min<int>(blockParams.GetBlockSize(), tailFinish - blockOffset),
                    approx[0].data(),
                    nullptr, // no approx deltas
                    target.data(),
                    weight.data(),
                    (*derivatives)[0].data());
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            ctx->LocalExecutor.ExecRange([&](int blockId) {
                yvector<double> curApprox(approxDimension);
                yvector<double> curDelta(approxDimension);
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int z) {
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        curApprox[dim] = approx[dim][z];
                    }
                    error.CalcDersMulti(curApprox, target[z], weight.empty() ? 1 : weight[z], &curDelta, nullptr);
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        (*derivatives)[dim][z] = curDelta[dim];
                    }
                })(blockId);
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}

inline void CalcAndLogLearnErrors(const yvector<yvector<double>>& avrgApprox,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  const yvector<ui32>& queryId,
                                  const yhash<ui32, ui32>& queriesSize,
                                  const yvector<TPair>& pairs,
                                  const yvector<THolder<IMetric>>& errors,
                                  int learnSampleCount,
                                  int iteration,
                                  yvector<yvector<double>>* learnErrorsHistory,
                                  NPar::TLocalExecutor* localExecutor,
                                  TLogger* logger) {
    learnErrorsHistory->emplace_back();
    for (int i = 0; i < errors.ysize(); ++i) {
        double learnErr = errors[i]->GetFinalError(
            errors[i]->GetErrorType() == EErrorType::PerObjectError ?
                errors[i]->Eval(avrgApprox, target, weight, 0, learnSampleCount, *localExecutor) :
                errors[i]->GetErrorType() == EErrorType::PairwiseError ?
                    errors[i]->EvalPairwise(avrgApprox, pairs, 0, learnSampleCount):
                    errors[i]->EvalQuerywise(avrgApprox, target, weight, queryId, queriesSize, 0, learnSampleCount));
        if (i == 0) {
            MATRIXNET_NOTICE_LOG << "learn: " << Prec(learnErr, 7);
        }
        learnErrorsHistory->back().push_back(learnErr);
    }

    if (logger != nullptr) {
        Log(iteration, learnErrorsHistory->back(), errors, logger, EPhase::Learn);
    }
}

inline void CalcAndLogTestErrors(const yvector<yvector<double>>& avrgApprox,
                                 const yvector<float>& target,
                                 const yvector<float>& weight,
                                 const yvector<ui32>& queryId,
                                 const yhash<ui32, ui32>& queriesSize,
                                 const yvector<TPair>& pairs,
                                 const yvector<THolder<IMetric>>& errors,
                                 int learnSampleCount,
                                 int sampleCount,
                                 int iteration,
                                 TErrorTracker& errorTracker,
                                 yvector<yvector<double>>* testErrorsHistory,
                                 NPar::TLocalExecutor* localExecutor,
                                 TLogger* logger) {
    yvector<double> valuesToLog;

    testErrorsHistory->emplace_back();
    for (int i = 0; i < errors.ysize(); ++i) {
        double testErr = errors[i]->GetFinalError(
            errors[i]->GetErrorType() == EErrorType::PerObjectError ?
                errors[i]->Eval(avrgApprox, target, weight, learnSampleCount, sampleCount, *localExecutor) :
                errors[i]->GetErrorType() == EErrorType::PairwiseError ?
                    errors[i]->EvalPairwise(avrgApprox, pairs, learnSampleCount, sampleCount):
                    errors[i]->EvalQuerywise(avrgApprox, target, weight, queryId, queriesSize, learnSampleCount, sampleCount));

        if (i == 0) {
            errorTracker.AddError(testErr, iteration, &valuesToLog);
            double bestErr = errorTracker.GetBestError();

            MATRIXNET_NOTICE_LOG << "\ttest: " << Prec(testErr, 7) << "\tbestTest: " << Prec(bestErr, 7)
                                 << " (" << errorTracker.GetBestIteration() << ")";
        }
        testErrorsHistory->back().push_back(testErr);
    }

    if (logger != nullptr) {
        Log(iteration, testErrorsHistory->back(), errors, logger, EPhase::Test);
    }
}

template <typename TError>
void TrainOneIter(const TTrainData& data,
                  TLearnContext* ctx) {
    TError error = BuildError<TError>(ctx->Params);
    TProfileInfo& profile = ctx->Profile;

    const auto sampleCount = data.GetSampleCount();
    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();

    yvector<THolder<IMetric>> errors = CreateMetrics(ctx->Params, approxDimension);

    auto splitCounts = CountSplits(ctx->LearnProgress.Model.Borders);

    float l2LeafRegularizer = ctx->Params.L2LeafRegularizer;
    if (l2LeafRegularizer == 0) {
        l2LeafRegularizer += 1e-20;
    }

    const int foldCount = ctx->LearnProgress.Folds.ysize();
    const int currentIteration = ctx->LearnProgress.Model.TreeStruct.ysize();

    MATRIXNET_NOTICE_LOG << currentIteration << ": ";
    profile.StartNextIteration();

    CheckInterrupted(); // check after long-lasting operation

    double modelLength = currentIteration * ctx->Params.LearningRate;

    yvector<TSplit> bestSplitTree;
    TTensorStructure3 bestTree;
    {
        TFold* takenFold = &ctx->LearnProgress.Folds[ctx->Rand.GenRand() % foldCount];

        ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
            TFold::TBodyTail& bt = takenFold->BodyTailArr[bodyTailId];
            CalcWeightedDerivatives<TError>(bt.Approx,
                                            takenFold->LearnTarget,
                                            takenFold->LearnWeights,
                                            takenFold->LearnQueryId,
                                            takenFold->LearnQuerySize,
                                            bt.Competitors,
                                            error,
                                            bt.TailFinish,
                                            ctx,
                                            &bt.Derivatives);
        }, 0, takenFold->BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        profile.AddOperation("Calc derivatives");

        GreedyTensorSearch(
            data,
            splitCounts,
            modelLength,
            l2LeafRegularizer,
            ctx->Params.RandomStrength,
            profile,
            takenFold,
            ctx,
            &bestTree,
            &bestSplitTree);
    }
    CheckInterrupted(); // check after long-lasting operation
    {
        yvector<TFold*> trainFolds;
        for (int foldId = 0; foldId < foldCount; ++foldId) {
            trainFolds.push_back(&ctx->LearnProgress.Folds[foldId]);
        }

        TrimOnlineCTRcache(trainFolds);
        TrimOnlineCTRcache({&ctx->LearnProgress.AveragingFold});
        {
            yvector<TFold*> allFolds = trainFolds;
            allFolds.push_back(&ctx->LearnProgress.AveragingFold);

            yvector<TCalcOnlineCTRsBatchTask> parallelJobsData;
            for (const auto& split : bestTree.SelectedSplits) {
                if (split.Type == ESplitType::FloatFeature) {
                    continue;
                }

                auto& proj = split.OnlineCtr.Ctr.Projection;
                for (auto* foldPtr : allFolds) {
                    if (!foldPtr->GetCtrs(proj).has(proj)) {
                        parallelJobsData.emplace_back(TCalcOnlineCTRsBatchTask{proj, foldPtr, &foldPtr->GetCtrRef(proj)});
                    }
                }
            }
            CalcOnlineCTRsBatch(parallelJobsData, data, ctx);
        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        yvector<yvector<yvector<yvector<double>>>> approxDelta;
        CalcApproxForLeafStruct(
            data,
            error,
            ctx->Params.GradientIterations,
            trainFolds,
            bestSplitTree,
            ctx->Params.LeafEstimationMethod,
            l2LeafRegularizer,
            ctx,
            &approxDelta);
        profile.AddOperation("CalcApprox tree struct");
        CheckInterrupted(); // check after long-lasting operation

        const double learningRate = ctx->Params.LearningRate;

        for (int foldId = 0; foldId < foldCount; ++foldId) {
            TFold& ff = ctx->LearnProgress.Folds[foldId];

            for (int bodyTailId = 0; bodyTailId < ff.BodyTailArr.ysize(); ++bodyTailId) {
                TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];
                for (int dim = 0; dim < approxDimension; ++dim) {
                    double* approxDeltaData = approxDelta[foldId][bodyTailId][dim].data();
                    double* approxData = bt.Approx[dim].data();
                    ctx->LocalExecutor.ExecRange([=](int z) {
                        approxData[z] = UpdateApprox<TError::StoreExpApprox>(approxData[z], ApplyLearningRate<TError::StoreExpApprox>(approxDeltaData[z], learningRate));
                    }, NPar::TLocalExecutor::TBlockParams(0, bt.TailFinish).SetBlockSize(10000).WaitCompletion());
                }
            }
        }
        profile.AddOperation("Update tree structure approx");
        CheckInterrupted(); // check after long-lasting operation

        yvector<TIndexType> indices;
        yvector<yvector<double>> treeValues;
        CalcLeafValues(data,
                       ctx->LearnProgress.AveragingFold,
                       bestSplitTree,
                       error,
                       ctx->Params.GradientIterations,
                       ctx->Params.LeafEstimationMethod,
                       l2LeafRegularizer,
                       ctx,
                       &treeValues,
                       &indices);

        yvector<yvector<double>> expTreeValues;
        expTreeValues.yresize(approxDimension);
        for (int dim = 0; dim < approxDimension; ++dim) {
            for (auto& leafVal : treeValues[dim]) {
                leafVal *= learningRate;
            }
            expTreeValues[dim] = treeValues[dim];
            ExpApproxIf(TError::StoreExpApprox, &expTreeValues[dim]);
        }

        profile.AddOperation("CalcApprox result leafs");
        CheckInterrupted(); // check after long-lasting operation

        Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
        TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];

        const int tailFinish = bt.TailFinish;
        const int learnSampleCount = data.LearnSampleCount;
        const int* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();
        const TIndexType* indicesData = indices.data();
        for (int dim = 0; dim < approxDimension; ++dim) {
            const double* expTreeValuesData = expTreeValues[dim].data();
            const double* treeValuesData = treeValues[dim].data();
            double* approxData = bt.Approx[dim].data();
            double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
            ctx->LocalExecutor.ExecRange([=](int docIdx) {
                const int permutedDocIdx = docIdx < learnSampleCount ? learnPermutationData[docIdx] : docIdx;
                if (docIdx < tailFinish) {
                    approxData[docIdx] = UpdateApprox<TError::StoreExpApprox>(approxData[docIdx], expTreeValuesData[indicesData[docIdx]]);
                }
                avrgApproxData[permutedDocIdx] += treeValuesData[indicesData[docIdx]];
            }, NPar::TLocalExecutor::TBlockParams(0, sampleCount).SetBlockSize(10000).WaitCompletion());
        }

        ctx->LearnProgress.Model.LeafValues.push_back(treeValues);
        ctx->LearnProgress.Model.TreeStruct.push_back(bestTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}

template <typename TError>
void Train(const TTrainData& data, TLearnContext* ctx, yvector<yvector<double>>* testMultiApprox) {
    TProfileInfo& profile = ctx->Profile;

    const int sampleCount = data.GetSampleCount();
    const int approxDimension = testMultiApprox->ysize();
    const bool hasTest = sampleCount > data.LearnSampleCount;

    yvector<THolder<IMetric>> metrics = CreateMetrics(ctx->Params, approxDimension);

    // TODO(asaitgalin): Should we have multiple error trackers?
    TErrorTracker errorTracker = BuildErrorTracker(metrics.front()->IsMaxOptimal(), hasTest, ctx);

    CB_ENSURE(hasTest || !ctx->Params.UseBestModel, "cannot select best model, no test provided");

    THolder<TLogger> logger;

    if (ctx->TryLoadProgress()) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*testMultiApprox)[dim].assign(
                    ctx->LearnProgress.AvrgApprox[dim].begin() + data.LearnSampleCount, ctx->LearnProgress.AvrgApprox[dim].end());
        }
    }
    if (ctx->Params.AllowWritingFiles) {
        logger = CreateLogger(metrics, *ctx, hasTest);
    }
    yvector<yvector<double>> errorsHistory = ctx->LearnProgress.TestErrorsHistory;
    yvector<double> valuesToLog;
    for (int i = 0; i < errorsHistory.ysize(); ++i) {
        errorTracker.AddError(errorsHistory[i][0], i, &valuesToLog);
    }

    yvector<TFold*> folds;
    for (auto& fold : ctx->LearnProgress.Folds) {
        folds.push_back(&fold);
    }

    for (int iter = ctx->LearnProgress.Model.TreeStruct.ysize(); iter < ctx->Params.Iterations; ++iter) {
        TrainOneIter<TError>(data, ctx);

        CalcAndLogLearnErrors(
            ctx->LearnProgress.AvrgApprox,
            data.Target,
            data.Weights,
            data.QueryId,
            data.QuerySize,
            data.Pairs,
            metrics,
            data.LearnSampleCount,
            iter,
            &ctx->LearnProgress.LearnErrorsHistory,
            &ctx->LocalExecutor,
            logger.Get());

        profile.AddOperation("Calc learn errors");

        if (hasTest) {
            CalcAndLogTestErrors(
                ctx->LearnProgress.AvrgApprox,
                data.Target,
                data.Weights,
                data.QueryId,
                data.QuerySize,
                data.Pairs,
                metrics,
                data.LearnSampleCount,
                sampleCount,
                iter,
                errorTracker,
                &ctx->LearnProgress.TestErrorsHistory,
                &ctx->LocalExecutor,
                logger.Get());

            profile.AddOperation("Calc test errors");

            if (ctx->Params.UseBestModel && iter == errorTracker.GetBestIteration() || !ctx->Params.UseBestModel) {
                for (int dim = 0; dim < approxDimension; ++dim) {
                    (*testMultiApprox)[dim].assign(
                        ctx->LearnProgress.AvrgApprox[dim].begin() + data.LearnSampleCount, ctx->LearnProgress.AvrgApprox[dim].end());
                }
            }
        }

        profile.FinishIteration();
        ctx->SaveProgress();

        if (IsNan(ctx->LearnProgress.LearnErrorsHistory.back()[0])) {
            ctx->LearnProgress.Model.LeafValues.pop_back();
            ctx->LearnProgress.Model.TreeStruct.pop_back();
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

    if (ctx->Params.DetailedProfile || ctx->Params.DeveloperMode) {
        profile.PrintAverages();
    }

    if (ctx->Params.UseBestModel && ctx->Params.Iterations > 0) {
        const int itCount = errorTracker.GetBestIteration() + 1;
        MATRIXNET_NOTICE_LOG << "Shrink model to first " << itCount << " iterations." << Endl;
        ShrinkModel(itCount, &ctx->LearnProgress.Model);
    }
}
