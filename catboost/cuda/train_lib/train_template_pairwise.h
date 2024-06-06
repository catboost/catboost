#pragma once

#include "train_template.h"
#include <catboost/cuda/methods/doc_parallel_boosting.h>
#include <catboost/cuda/methods/pairwise_oblivious_trees/pairwise_oblivious_tree.h>

namespace NCatboostCuda {
    template <template <class TMapping> class TTargetTemplate>
    TGpuTrainResult TrainPairwise(TBinarizedFeaturesManager& featureManager,
                                                               const TTrainModelInternalOptions& internalOptions,
                                                               const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                               const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                               const NCB::TTrainingDataProvider& learn,
                                                               const NCB::TTrainingDataProvider* test,
                                                               const NCB::TFeatureEstimators& featureEstimators,
                                                               const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                                                               const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                                                               TGpuAwareRandom& random,
                                                               ui32 approxDimension,
                                                               ITrainingCallbacks* trainingCallbacks,
                                                               NPar::ILocalExecutor* localExecutor,
                                                               TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                                               TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        CB_ENSURE(catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
                  "NonDiag learning works with doc-parallel learning");
        CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain,
                  "Boosting scheme should be plain for nonDiag targets");

        using TDocParallelBoosting = TBoosting<TTargetTemplate, TPairwiseObliviousTree>;
        return Train<TDocParallelBoosting>(featureManager,
                                           internalOptions,
                                           catBoostOptions,
                                           outputOptions,
                                           learn,
                                           test,
                                           featureEstimators,
                                           objectiveDescriptor,
                                           evalMetricDescriptor,
                                           random,
                                           approxDimension,
                                           trainingCallbacks,
                                           localExecutor,
                                           testMultiApprox,
                                           metricsAndTimeHistory);
    };

    template <template <class TMapping> class TTargetTemplate>
    void ModelBasedEvalPairwise(TBinarizedFeaturesManager& featureManager,
                                const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                const NCB::TTrainingDataProvider& learn,
                                const NCB::TTrainingDataProvider& test,
                                TGpuAwareRandom& random,
                                ui32 approxDimension,
                                NPar::ILocalExecutor* localExecutor) {
        CB_ENSURE(catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
                  "NonDiag learning works with doc-parallel learning");
        CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain,
                  "Boosting scheme should be plain for nonDiag targets");
        CB_ENSURE(!catBoostOptions.ModelBasedEvalOptions->FeaturesToEvaluate->empty(),
            "Model based evaluation requires features to evaluate"
        );

        using TDocParallelBoosting = TBoosting<TTargetTemplate, TPairwiseObliviousTree>;
        ModelBasedEval<TDocParallelBoosting>(featureManager,
            catBoostOptions,
            outputOptions,
            learn,
            test,
            random,
            approxDimension,
            localExecutor);
    }

    template <template <class> class TTargetTemplate>
    class TPairwiseGpuTrainer: public IGpuTrainer {
        virtual TGpuTrainResult TrainModel(TBinarizedFeaturesManager& featuresManager,
                                           const TTrainModelInternalOptions& internalOptions,
                                           const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                           const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                           const NCB::TTrainingDataProvider& learn,
                                           const NCB::TTrainingDataProvider* test,
                                           const NCB::TFeatureEstimators& featureEstimators,
                                           const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                                           const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                                           TGpuAwareRandom& random,
                                           ui32 approxDimension,
                                           ITrainingCallbacks* trainingCallbacks,
                                           NPar::ILocalExecutor* localExecutor,
                                           TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                           TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const {
            return TrainPairwise<TTargetTemplate>(featuresManager,
                                                  internalOptions,
                                                  catBoostOptions,
                                                  outputOptions,
                                                  learn,
                                                  test,
                                                  featureEstimators,
                                                  objectiveDescriptor,
                                                  evalMetricDescriptor,
                                                  random,
                                                  approxDimension,
                                                  trainingCallbacks,
                                                  localExecutor,
                                                  testMultiApprox,
                                                  metricsAndTimeHistory);
        };

        virtual void ModelBasedEval(TBinarizedFeaturesManager& featuresManager,
                                    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                    const NCB::TTrainingDataProvider& learn,
                                    const NCB::TTrainingDataProvider& test,
                                    TGpuAwareRandom& random,
                                    ui32 approxDimension,
                                    NPar::ILocalExecutor* localExecutor) const {
            ModelBasedEvalPairwise<TTargetTemplate>(featuresManager,
                catBoostOptions,
                outputOptions,
                learn,
                test,
                random,
                approxDimension,
                localExecutor);
        }
    };
}
