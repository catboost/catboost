#pragma once

#include "train.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/cuda/methods/boosting_progress_tracker.h>
#include <catboost/libs/eval_result/eval_result.h>

namespace NCatboostCuda {
    template <class TBoosting>
    inline THolder<TAdditiveModel<typename TBoosting::TWeakModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                                         const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                         const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                         const TDataProvider& learn,
                                                                         const TDataProvider* test,
                                                                         TGpuAwareRandom& random,
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
                           weak);

        boosting.SetDataProvider(learn,
                                 test);

        ui32 approxDim = 1;
        if (learn.IsMulticlassificationPool()) {
            approxDim = learn.GetTargetHelper().GetNumClasses();
        }
        TBoostingProgressTracker progressTracker(catBoostOptions,
                                                 outputOptions,
                                                 test != nullptr,
                                                 approxDim
                                                 );

        boosting.SetBoostingProgressTracker(&progressTracker);

        auto model = boosting.Run();

        const auto evalOutputFileName = outputOptions.CreateEvalFullPath();
        if (test && !evalOutputFileName.empty()) {
            const auto& errorTracker = progressTracker.GetErrorTracker();
            CATBOOST_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << Endl;
            CATBOOST_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << Endl;

            NCB::OutputGpuEvalResultToFile(
                progressTracker.GetBestTestCursor(),
                catBoostOptions.SystemOptions->NumThreads,
                outputOptions.GetOutputColumns(),
                test ? test->GetPoolPath() : NCB::TPathWithScheme(),
                test ? test->GetDsvPoolFormatOptions() : NCB::TDsvFormatOptions(),
                test ? test->GetPoolMetaInfo() : TPoolMetaInfo(),
                (test && test->IsMulticlassificationPool()) ? test->GetTargetHelper().Serialize() : "",
                evalOutputFileName
            );
        } else if (!test && !evalOutputFileName.empty()) {
            // TODO(yazevnul): this should be an error in option validation;
            //
            // But right now `--eval-file` is always added to `fit` options in Nirvana cubes, thus,
            // we first must make `--eval-file` conditional (based on presence of `--test-set`),
            // then roll out new version of Nirvana cube and only after this we can make it an
            // error.
            CATBOOST_WARNING_LOG << "can't evaluate model (--eval-file) without test set" << Endl;
        }

        if (outputOptions.ShrinkModelToBestIteration()) {
            if (test == nullptr) {
                CATBOOST_INFO_LOG << "Warning: can't use-best-model without test set. Will skip model shrinking";
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
