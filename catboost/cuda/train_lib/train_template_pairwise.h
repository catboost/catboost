#pragma once

#include "train_template.h"
#include <catboost/cuda/methods/doc_parallel_boosting.h>
#include <catboost/cuda/methods/pairwise_oblivious_trees/pairwise_oblivious_tree.h>

namespace NCatboostCuda {

    namespace {

        template <template <class TMapping> class TTargetTemplate>
        THolder<TAdditiveModel<TObliviousTreeModel>> TrainPairwise(TBinarizedFeaturesManager& featureManager,
                                                                   const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                   const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                   const TDataProvider& learn,
                                                                   const TDataProvider* test,
                                                                   TGpuAwareRandom& random,
                                                                   TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
            CB_ENSURE(catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::DocParallel,
                      "NonDiag learning works with doc-parallel learning");
            CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain,
                      "Boosting scheme should be plain for nonDiag targets");

            using TDocParallelBoosting = TBoosting<TTargetTemplate, TPairwiseObliviousTree>;
            return Train<TDocParallelBoosting>(featureManager,
                                               catBoostOptions,
                                               outputOptions,
                                               learn,
                                               test,
                                               random,
                                               metricsAndTimeHistory);
        };


        template <template <class> class TTargetTemplate>
        class TPairwiseGpuTrainer: public IGpuTrainer {
            virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(TBinarizedFeaturesManager& featuresManager,
                                                                            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                            const TDataProvider& learn,
                                                                            const TDataProvider* test,
                                                                            TGpuAwareRandom& random,
                                                                            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const {
                return TrainPairwise<TTargetTemplate>(featuresManager,
                                                      catBoostOptions,
                                                      outputOptions,
                                                      learn,
                                                      test,
                                                      random,
                                                      metricsAndTimeHistory);
            };
        };
    }

}
