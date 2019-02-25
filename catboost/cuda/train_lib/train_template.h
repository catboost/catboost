#pragma once

#include "train.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/cuda/methods/boosting_progress_tracker.h>

#include <library/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    template <class TBoosting>
    inline THolder<TAdditiveModel<typename TBoosting::TWeakModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                                         const TTrainModelInternalOptions& internalOptions,
                                                                         const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                         const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                         const NCB::TTrainingDataProvider& learn,
                                                                         const NCB::TTrainingDataProvider* test,
                                                                         TGpuAwareRandom& random,
                                                                         ui32 approxDimension,
                                                                         const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
                                                                         NPar::TLocalExecutor* localExecutor,
                                                                         TVector<TVector<double>>* testMultiApprox, // [dim][docIdx]
                                                                         TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        using TWeakLearner = typename TBoosting::TWeakLearner;

        const bool zeroAverage = catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::PairLogit;
        TWeakLearner weak(featureManager,
                          catBoostOptions,
                          zeroAverage);

        const auto& boostingOptions = catBoostOptions.BoostingOptions.Get();
        TBoosting boosting(featureManager,
                           boostingOptions,
                           catBoostOptions.LossFunctionDescription,
                           catBoostOptions.DataProcessingOptions->GpuCatFeaturesStorage,
                           random,
                           weak,
                           localExecutor);

        boosting.SetDataProvider(learn,
                                 test);

        TBoostingProgressTracker progressTracker(catBoostOptions,
                                                 outputOptions,
                                                 internalOptions.ForceCalcEvalMetricOnEveryIteration,
                                                 test != nullptr,
                                                 /*testHasTarget*/ (test != nullptr) && test->MetaInfo.HasTarget,
                                                 approxDimension,
                                                 onEndIterationCallback);

        boosting.SetBoostingProgressTracker(&progressTracker);

        auto model = boosting.Run();

        if (progressTracker.EvalMetricWasCalculated()) {
            const auto& errorTracker = progressTracker.GetErrorTracker();
            CATBOOST_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << Endl;
            CATBOOST_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << Endl;

            *testMultiApprox = progressTracker.GetBestTestCursor();
        }

        if (outputOptions.ShrinkModelToBestIteration()) {
            if (test == nullptr) {
                CATBOOST_INFO_LOG << "Warning: can't use-best-model without test set. Will skip model shrinking";
            } else if (!progressTracker.EvalMetricWasCalculated()) {
                CATBOOST_INFO_LOG << "Warning: can't use-best-model because eval metric was not calculated "
                                     "due to the absence of target data in test set. Will skip model shrinking";
            } else {
                const auto& errorTracker = progressTracker.GetErrorTracker();
                const auto& bestModelTracker = progressTracker.GetBestModelMinTreesTracker();
                const ui32 bestIter = static_cast<const ui32>(bestModelTracker.GetBestIteration());
                if (0 < bestIter + 1 && bestIter + 1 < progressTracker.GetCurrentIteration()) {
                    CATBOOST_NOTICE_LOG << "Shrink model to first " << bestIter + 1 << " iterations.";
                    if (bestIter > static_cast<const ui32>(errorTracker.GetBestIteration())) {
                        CATBOOST_NOTICE_LOG << " (min iterations for best model = " << outputOptions.BestModelMinTrees << ")";
                    }
                    CATBOOST_NOTICE_LOG << Endl;
                    model->Shrink(bestIter + 1);
                }
            }
        }

        if (metricsAndTimeHistory) {
            *metricsAndTimeHistory = progressTracker.GetMetricsAndTimeLeftHistory();
        }

        return model;
    }

}
