#pragma once

#include "ctr_helper.h"
#include "feature_parallel_dataset.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/libs/data/feature_estimators.h>

namespace NCatboostCuda {

    class TEstimatorsExecutor {
    public:
        using TBinarizedFeatureVisitor =  std::function<void(TConstArrayRef<ui8>, //binarizedFeature
                                                             NCB::TEstimatedFeatureId,
                                                             ui8)>; //binCount

        TEstimatorsExecutor(TBinarizedFeaturesManager& featuresManager,
                            const NCB::TFeatureEstimators& featureEstimators,
                            const TDataPermutation& permutation,
                            NPar::ILocalExecutor* localExecutor)
            : FeaturesManager(featuresManager)
            , Estimators(featureEstimators)
            , Permutation(permutation)
            , LocalExecutor(localExecutor) {
                Permutation.FillOrder(PermutationIndices);
        }

        void ExecEstimators(TConstArrayRef<NCB::TEstimatorId> estimatorIds,
                            TBinarizedFeatureVisitor visitor,
                            TMaybe<TBinarizedFeatureVisitor> testVisitor
                            );

        void ExecBinaryFeaturesEstimators(TConstArrayRef<NCB::TEstimatorId> estimatorIds,
                                          TBinarizedFeatureVisitor visitor,
                                          TMaybe<TBinarizedFeatureVisitor> testVisitor
                                          );

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TFeatureEstimators& Estimators;
        const TDataPermutation& Permutation;
        TVector<ui32> PermutationIndices;
        NPar::ILocalExecutor* LocalExecutor;
    };
}
