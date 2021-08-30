#pragma once

#include "weak_target_helpers.h"
#include "stripe_target_wrapper.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    class TDocParallelObliviousTreeSearcher {
    public:
        using TDataSet = TDocParallelDataSet;
        using TVec = TStripeBuffer<float>;
        using TSampelsMapping = NCudaLib::TStripeMapping;
        using TWeakTarget = TL2Target<NCudaLib::TStripeMapping>;

        TDocParallelObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                          const NCatboostOptions::TBoostingOptions& boostingOptions,
                                          const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
                                          TBootstrap<NCudaLib::TStripeMapping>& bootstrap,
                                          double randomStrengthMult,
                                          TGpuAwareRandom& random)
            : FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , TreeConfig(learnerOptions)
            , Bootstrap(bootstrap)
            , ModelLengthMultiplier(randomStrengthMult)
            , Random(random)
        {
        }

        template <class TTargetFunc>
        TObliviousTreeModel Fit(const TDataSet& dataSet,
                                const TTargetFunc& objective) {
            return FitImpl(dataSet, TStripeTargetWrapper<TTargetFunc>(objective));
        }

    private:
        TObliviousTreeModel FitImpl(const TDataSet& dataSet,
                                    const IStripeTargetWrapper& objective);

        TVector<float> ReadAndEstimateLeaves(const TCudaBuffer<TPartitionStatistics, NCudaLib::TMirrorMapping>& parts);

        TVector<float> EstimateLeaves(const TVector<TPartitionStatistics>& statCpu);

        TVector<double> ExtractWeights(const TVector<TPartitionStatistics>& statCpu);

        void ComputeWeakTarget(const IStripeTargetWrapper& objective,
                               double* scoreStdDev,
                               TWeakTarget* target,
                               TStripeBuffer<ui32>* indices);

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TBoostingOptions& BoostingOptions;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        TBootstrap<NCudaLib::TStripeMapping>& Bootstrap;
        double ModelLengthMultiplier = 0.0;
        TGpuAwareRandom& Random;
    };
}
