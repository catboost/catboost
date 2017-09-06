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
#include "interrupt.h"

#include <catboost/libs/model/hash.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/projection.h>
#include <catboost/libs/model/tensor_struct.h>

#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/logging/profile_info.h>

#include <util/string/vector.h>
#include <util/string/iterator.h>
#include <util/stream/file.h>
#include <util/generic/ymath.h>

void ShrinkModel(int itCount, TCoreModel* model);

yvector<int> CountSplits(const yvector<yvector<float>>& borders);

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx);

template <typename TError>
TError BuildError(const TFitParams&) {
    return TError();
}

template <>
inline TQuantileError BuildError<TQuantileError>(const TFitParams& params) {
    auto lossParams = GetLossParams(params.Objective);
    if (lossParams.empty()) {
        return TQuantileError();
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description");
        return TQuantileError(lossParams["alpha"]);
    }
}

template <>
inline TLogLinearQuantileError BuildError<TLogLinearQuantileError>(const TFitParams& params) {
    auto lossParams = GetLossParams(params.Objective);
    if (lossParams.empty()) {
        return TLogLinearQuantileError();
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description");
        return TLogLinearQuantileError(lossParams["alpha"]);
    }
}

template <>
inline TCustomError BuildError<TCustomError>(const TFitParams& params) {
    Y_ASSERT(params.ObjectiveDescriptor.Defined());
    return TCustomError(*params.ObjectiveDescriptor);
}

inline THolder<TOFStream> InitErrLog(const yvector<THolder<IMetric>>& errors, const yvector<yvector<double>>& history, const TString& logName) {
    THolder<TOFStream> errLog = new TOFStream(logName);
    *errLog << "iter";
    for (const auto& error : errors) {
        *errLog << "\t" << error->GetDescription();
    }
    *errLog << Endl;
    for (const auto& errors : history) {
        for (const auto& error : errors) {
            *errLog << "\t" << error;
        }
        *errLog << Endl;
    }
    return errLog;
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
                                    const TError& error,
                                    int tailFinish,
                                    TLearnContext* ctx,
                                    yvector<yvector<double>>* derivatives) {
    int approxDimension = approx.ysize();
    NPar::TLocalExecutor::TBlockParams blockParams(0, tailFinish);
    blockParams.SetBlockSize(1000);
    if (approxDimension == 1) {
        ctx->LocalExecutor.ExecRange([&](int blockId) {
            const int blockOffset = blockId * blockParams.GetBlockSize();
            const double* approxData = approx[0].data() + blockOffset;
            const float* targetArrData = target.data() + blockOffset;
            double* dersPtr = (*derivatives)[0].data() + blockOffset;
            error.CalcFirstDerRange(Min<int>(blockParams.GetBlockSize(), tailFinish - blockOffset), approxData, targetArrData,
                                    weight.empty() ? nullptr : weight.data() + blockOffset, dersPtr);
        },
                                     0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
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
        },
                                     0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

inline void CalcAndLogLearnErrors(const yvector<yvector<double>>& avrgApprox,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  const yvector<THolder<IMetric>>& errors,
                                  int learnSampleCount,
                                  int iteration,
                                  yvector<yvector<double>>* learnErrorsHistory,
                                  NPar::TLocalExecutor* localExecutor,
                                  IOutputStream* learnErrLog) {
    learnErrorsHistory->emplace_back();
    for (int i = 0; i < errors.ysize(); ++i) {
        double learnErr = errors[i]->GetFinalError(
            errors[i]->Eval(avrgApprox, target, weight, 0, learnSampleCount, *localExecutor));
        if (i == 0) {
            MATRIXNET_INFO_LOG << "learn " << learnErr;
        }
        learnErrorsHistory->back().push_back(learnErr);
    }

    if (learnErrLog != nullptr) {
        *learnErrLog << iteration;
        for (int i = 0; i < errors.ysize(); ++i) {
            *learnErrLog << "\t" << learnErrorsHistory->back()[i];
        }
        *learnErrLog << Endl;
    }
}

inline void CalcAndLogTestErrors(const yvector<yvector<double>>& avrgApprox,
                                 const yvector<float>& target,
                                 const yvector<float>& weight,
                                 const yvector<THolder<IMetric>>& errors,
                                 int learnSampleCount,
                                 int sampleCount,
                                 int iteration,
                                 TErrorTracker& errorTracker,
                                 yvector<yvector<double>>* testErrorsHistory,
                                 NPar::TLocalExecutor* localExecutor,
                                 IOutputStream* testErrLog) {
    yvector<double> valuesToLog;

    testErrorsHistory->emplace_back();
    for (int i = 0; i < errors.ysize(); ++i) {
        double testErr = errors[i]->GetFinalError(
            errors[i]->Eval(avrgApprox, target, weight, learnSampleCount, sampleCount, *localExecutor));

        if (i == 0) {
            errorTracker.AddError(testErr, iteration, &valuesToLog);
            double bestErr = errorTracker.GetBestError();

            MATRIXNET_INFO_LOG << "\ttest " << testErr << "\tbestTest " << bestErr << "\t";
        }
        testErrorsHistory->back().push_back(testErr);
    }

    if (testErrLog != nullptr) {
        *testErrLog << iteration;
        for (int i = 0; i < errors.ysize(); ++i) {
            *testErrLog << "\t" << testErrorsHistory->back()[i];
        }
        *testErrLog << Endl;
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

    yvector<yvector<yvector<yvector<double>>>> approxDelta;
    yvector<yvector<double>> approxDeltaAvrg;

    auto splitCounts = CountSplits(ctx->LearnProgress.Model.Borders);

    float l2LeafRegularizer = ctx->Params.L2LeafRegularizer;
    if (l2LeafRegularizer == 0) {
        l2LeafRegularizer += 1e-20;
    }

    const int foldCount = ctx->LearnProgress.Folds.ysize();
    const int currentIteration = ctx->LearnProgress.Model.TreeStruct.ysize();

    MATRIXNET_INFO_LOG << currentIteration << ":\t";
    profile.StartNextIteration();

    CheckInterrupted(); // check after long-lasting operation

    double modelLength = currentIteration * ctx->Params.LearningRate;

    yvector<TSplit> bestSplitTree;
    TTensorStructure3 bestTree;
    {
        TFold* takenFold = &ctx->LearnProgress.Folds[ctx->Rand.GenRand() % foldCount];

        ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
            TFold::TBodyTail& bt = takenFold->BodyTailArr[bodyTailId];
            CalcWeightedDerivatives(bt.Approx, takenFold->LearnTarget, takenFold->LearnWeights, error,
                                    bt.TailFinish, ctx, &bt.Derivatives);
        },
                                     0, takenFold->BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
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

        yvector<yvector<double>> treeValues;
        CalcApprox(
            data,
            error,
            ctx->Params.GradientIterations,
            {&ctx->LearnProgress.AveragingFold},
            bestSplitTree,
            ctx->Params.LeafEstimationMethod,
            l2LeafRegularizer,
            ctx,
            &approxDeltaAvrg,
            &treeValues);
        profile.AddOperation("CalcApprox result leafs");
        CheckInterrupted(); // check after long-lasting operation

        for (int dim = 0; dim < approxDimension; ++dim) {
            for (auto& leafVal : treeValues[dim]) {
                leafVal *= ctx->Params.LearningRate;
            }
        }

        ctx->LearnProgress.Model.LeafValues.push_back(treeValues);
        ctx->LearnProgress.Model.TreeStruct.push_back(bestTree);

        for (int foldId = 0; foldId < foldCount; ++foldId) {
            TFold& ff = ctx->LearnProgress.Folds[foldId];

            for (int bodyTailId = 0; bodyTailId < ff.BodyTailArr.ysize(); ++bodyTailId) {
                TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];
                for (int dim = 0; dim < approxDimension; ++dim) {
                    const double* approxDeltaData = approxDelta[foldId][bodyTailId][dim].data();
                    const double learningRate = ctx->Params.LearningRate;
                    double* approxData = bt.Approx[dim].data();
                    ctx->LocalExecutor.ExecRange([&](int z) {
                        approxData[z] += approxDeltaData[z] * learningRate;
                    },
                                                 NPar::TLocalExecutor::TBlockParams(0, bt.TailFinish).SetBlockSize(10000).WaitCompletion());
                }
            }
        }
        profile.AddOperation("Update tree structure approx");
        CheckInterrupted(); // check after long-lasting operation

        Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
        TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];

        for (int dim = 0; dim < approxDimension; ++dim) {
            for (int z = 0; z < bt.TailFinish; ++z) {
                bt.Approx[dim][z] += approxDeltaAvrg[dim][z] * ctx->Params.LearningRate;
            }
            const int* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();
            double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
            const double* approxDeltaAvrgData = approxDeltaAvrg[dim].data();
            const double learningRate = ctx->Params.LearningRate;
            ctx->LocalExecutor.ExecRange([&](int z) {
                int i = learnPermutationData[z];
                avrgApproxData[i] += approxDeltaAvrgData[z] * learningRate;
            },
                                         NPar::TLocalExecutor::TBlockParams(0, data.LearnSampleCount).SetBlockSize(10000).WaitCompletion());

            for (int i = data.LearnSampleCount; i < sampleCount; ++i) {
                ctx->LearnProgress.AvrgApprox[dim][i] += approxDeltaAvrg[dim][i] * ctx->Params.LearningRate;
            }
        }

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

    THolder<TOFStream> learnErrLog;
    THolder<TOFStream> testErrLog;

    if (ctx->Params.AllowWritingFiles) {
        learnErrLog = InitErrLog(metrics, ctx->LearnProgress.LearnErrorsHistory, ctx->Files.LearnErrorLogFile);
        if (hasTest) {
            testErrLog = InitErrLog(metrics, ctx->LearnProgress.TestErrorsHistory, ctx->Files.TestErrorLogFile);
        }
    }

    ctx->LoadProgress();

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
        TrainOneIter<TError>(
            data,
            ctx);

        CalcAndLogLearnErrors(
            ctx->LearnProgress.AvrgApprox,
            data.Target,
            data.Weights,
            metrics,
            data.LearnSampleCount,
            iter,
            &ctx->LearnProgress.LearnErrorsHistory,
            &ctx->LocalExecutor,
            learnErrLog.Get());

        profile.AddOperation("Calc learn errors");

        if (hasTest) {
            CalcAndLogTestErrors(
                ctx->LearnProgress.AvrgApprox,
                data.Target,
                data.Weights,
                metrics,
                data.LearnSampleCount,
                sampleCount,
                iter,
                errorTracker,
                &ctx->LearnProgress.TestErrorsHistory,
                &ctx->LocalExecutor,
                testErrLog.Get());

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
            MATRIXNET_INFO_LOG << "Stopped by overfitting detector "
                               << " (" << errorTracker.GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            break;
        }
    }

    ctx->LearnProgress.Folds.clear();

    if (hasTest) {
        MATRIXNET_INFO_LOG << "\n";
        MATRIXNET_INFO_LOG << "bestTest = " << errorTracker.GetBestError() << "\n";
        MATRIXNET_INFO_LOG << "bestIteration = " << errorTracker.GetBestIteration() << "\n";
        MATRIXNET_INFO_LOG << "\n";
    }

    if (ctx->Params.DetailedProfile || ctx->Params.DeveloperMode) {
        profile.PrintAverages();
    }

    if (ctx->Params.UseBestModel && ctx->Params.Iterations > 0) {
        const int itCount = errorTracker.GetBestIteration() + 1;
        MATRIXNET_INFO_LOG << "Shrink model to first " << itCount << " iterations." << Endl;
        ShrinkModel(itCount, &ctx->LearnProgress.Model);
    }
}
