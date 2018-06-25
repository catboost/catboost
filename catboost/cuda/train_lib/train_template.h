#pragma once


#include "train.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/cuda/methods/boosting_progress_tracker.h>

namespace NCatboostCuda {
    template <class TBoosting>
    inline THolder<TAdditiveModel<typename TBoosting::TWeakModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                                         const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                         const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                         const TDataProvider& learn,
                                                                         const TDataProvider* test,
                                                                         TGpuAwareRandom& random) {
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

        TBoostingProgressTracker progressTracker(catBoostOptions,
                                                 outputOptions,
                                                 test != nullptr);

        boosting.SetBoostingProgressTracker(&progressTracker);

        auto model = boosting.Run();

        if (test) {
            const auto& errorTracker = progressTracker.GetErrorTracker();
            MATRIXNET_NOTICE_LOG << "bestTest = " << errorTracker.GetBestError() << Endl;
            MATRIXNET_NOTICE_LOG << "bestIteration = " << errorTracker.GetBestIteration() << Endl;
        }

        if (outputOptions.ShrinkModelToBestIteration()) {
            if (test == nullptr) {
                MATRIXNET_INFO_LOG << "Warning: can't use-best-model without test set. Will skip model shrinking";
            } else {
                const auto& errorTracker = progressTracker.GetErrorTracker();
                const ui32 bestIter = static_cast<const ui32>(errorTracker.GetBestIteration());
                model->Shrink(bestIter + 1);
            }
        }

        return model;
    }

}
