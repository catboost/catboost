#pragma once

#include "train_template.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/methods/dynamic_boosting.h>
#include <catboost/cuda/methods/feature_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/methods/doc_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/methods/doc_parallel_boosting.h>

namespace NCatboostCuda {
    template <template <class TMapping> class TTargetTemplate>
    TGpuTrainResult Train(TBinarizedFeaturesManager& featureManager,
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
        if (catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel) {
            using TFeatureParallelWeakLearner = TFeatureParallelPointwiseObliviousTree;
            using TBoosting = TDynamicBoosting<TTargetTemplate, TFeatureParallelWeakLearner>;
            CB_ENSURE(
                !IsMultiTargetObjective(catBoostOptions.LossFunctionDescription->LossFunction),
                "Catboost does not support ordered boosting with multitarget on GPU yet");
            return Train<TBoosting>(featureManager,
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

        } else {
            using TDocParallelBoosting = TBoosting<TTargetTemplate, TDocParallelObliviousTree>;
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
        }
    };

    template <template <class TMapping> class TTargetTemplate>
    void ModelBasedEval(TBinarizedFeaturesManager& featureManager,
                        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                        const NCatboostOptions::TOutputFilesOptions& outputOptions,
                        const NCB::TTrainingDataProvider& learn,
                        const NCB::TTrainingDataProvider& test,
                        TGpuAwareRandom& random,
                        ui32 approxDimension,
                        NPar::ILocalExecutor* localExecutor) {
        CB_ENSURE(catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
            "Model based evaluation is supported only for DocParallel partition type");
        using TDocParallelBoosting = TBoosting<TTargetTemplate, TDocParallelObliviousTree>;
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
    class TGpuTrainer: public IGpuTrainer {
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
            return Train<TTargetTemplate>(featuresManager,
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
            ::NCatboostCuda::ModelBasedEval<TTargetTemplate>(featuresManager,
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
