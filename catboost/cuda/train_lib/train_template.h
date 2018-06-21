#pragma once

#include "train.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/methods/dynamic_boosting.h>
#include <catboost/cuda/methods/feature_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/methods/doc_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/methods/doc_parallel_boosting.h>
#include <catboost/cuda/methods/pairwise_oblivious_trees/pairwise_oblivious_tree.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>

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
                           random,
                           weak);


        boosting.SetDataProvider(learn,
                                 test);


        TBoostingProgressTracker progressTracker(catBoostOptions,
                                                 outputOptions,
                                                 test != nullptr
        );

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

    template <template <class TMapping, class> class TTargetTemplate>
    THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                       const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                       const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                       const TDataProvider& learn,
                                                       const TDataProvider* test,
                                                       TGpuAwareRandom& random,
                                                       bool storeCatFeaturesInPinnedMemory) {
        if (catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel) {
            using TFeatureParallelWeakLearner = TFeatureParallelPointwiseObliviousTree;
#define TRAIN_FEATURE_PARALLEL(PtrType)                                                        \
    using TBoosting = TDynamicBoosting<TTargetTemplate, TFeatureParallelWeakLearner, PtrType>; \
    return Train<TBoosting>(featureManager, catBoostOptions, outputOptions, learn, test, random);

            if (storeCatFeaturesInPinnedMemory) {
                TRAIN_FEATURE_PARALLEL(NCudaLib::EPtrType::CudaHost)
            } else {
                TRAIN_FEATURE_PARALLEL(NCudaLib::EPtrType::CudaDevice)
            }
#undef TRAIN_FEATURE_PARALLEL

        } else {
            using TDocParallelBoosting = TBoosting<TTargetTemplate, TDocParallelObliviousTree>;
            return Train<TDocParallelBoosting>(featureManager, catBoostOptions, outputOptions,
                                               learn, test, random);
        }
    };

    template <template <class TMapping, class> class TTargetTemplate>
    THolder<TAdditiveModel<TObliviousTreeModel>> TrainPairwise(TBinarizedFeaturesManager& featureManager,
                                                               const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                               const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                               const TDataProvider& learn,
                                                               const TDataProvider* test,
                                                               TGpuAwareRandom& random) {
        CB_ENSURE(catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel, "NonDiag learning works with doc-parallel learning");
        CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain, "Boosting scheme should be plain for nonDiag targets");

        using TDocParallelBoosting = TBoosting<TTargetTemplate, TPairwiseObliviousTree>;
        return Train<TDocParallelBoosting>(featureManager,
                                           catBoostOptions,
                                           outputOptions,
                                           learn,
                                           test,
                                           random);
    };

    template <template <class, class> class TTargetTemplate>
    class TGpuTrainer: public IGpuTrainer {
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(TBinarizedFeaturesManager& featuresManager,
                                                                        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                        const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                        const TDataProvider& learn,
                                                                        const TDataProvider* test,
                                                                        TGpuAwareRandom& random,
                                                                        bool storeInPinnedMemory) const {
            return Train<TTargetTemplate>(featuresManager,
                                          catBoostOptions,
                                          outputOptions,
                                          learn,
                                          test,
                                          random,
                                          storeInPinnedMemory);
        };
    };

    template <template <class, class> class TTargetTemplate>
    class TPairwiseGpuTrainer: public IGpuTrainer {
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(TBinarizedFeaturesManager& featuresManager,
                                                                        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                        const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                        const TDataProvider& learn,
                                                                        const TDataProvider* test,
                                                                        TGpuAwareRandom& random,
                                                                        bool) const {
            return TrainPairwise<TTargetTemplate>(featuresManager,
                                                  catBoostOptions,
                                                  outputOptions,
                                                  learn,
                                                  test,
                                                  random);
        };
    };

}
