#pragma once

#include "train_template.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/methods/leaves_estimation/pointwise_oracle.h>
#include <catboost/cuda/methods/doc_parallel_boosting.h>
#include <catboost/cuda/methods/greedy_subsets_searcher.h>
#include <catboost/cuda/models/model_converter.h>

namespace NCatboostCuda {
    namespace {
        /*
        * New implementation of doc-parallel training with support for any type of trees and multiclassification
        * But no ordered boosting
        */
        template <template <class TMapping> class TTargetTemplate, class TModel = TObliviousTreeModel>
        class TGpuTrainer: public IGpuTrainer {
            virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(TBinarizedFeaturesManager& featuresManager,
                                                                            const TTrainModelInternalOptions& internalOptions,
                                                                            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                            const NCB::TTrainingDataProvider& learn,
                                                                            const NCB::TTrainingDataProvider* test,
                                                                            TGpuAwareRandom& random,
                                                                            ui32 approxDimension,
                                                                            const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
                                                                            NPar::TLocalExecutor* localExecutor,
                                                                            TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                                                            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const {
                CB_ENSURE(catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain, "Only plain boosting is supported in current mode");
                using TWeakLearner = TGreedySubsetsSearcher<TModel>;
                using TBoostingImpl = TBoosting<TTargetTemplate, TWeakLearner>;
                auto resultModel = Train<TBoostingImpl>(featuresManager,
                                                        internalOptions,
                                                        catBoostOptions,
                                                        outputOptions,
                                                        learn,
                                                        test,
                                                        random,
                                                        approxDimension,
                                                        onEndIterationCallback,
                                                        localExecutor,
                                                        testMultiApprox,
                                                        metricsAndTimeHistory);
                return MakeObliviousModel<TModel>(std::move(resultModel), localExecutor);
            };
        };
    }

    inline TGpuTrainerFactoryKey GetTrainerFactoryKeyForRegion(ELossFunction loss) {
        return GetTrainerFactoryKey(loss, EGrowingPolicy::Region);
    }

}
