#pragma once

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/targets/non_diag_target_der.h>
#include <catboost/cuda/cuda_util/gpu_random.h>

namespace NCatboostCuda {
    class IPairwiseTargetWrapper: public TNonCopyable {
    public:
        virtual ~IPairwiseTargetWrapper() {
        }

        virtual void ComputeStochasticDerivatives(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                                  bool isGradient,
                                                  TNonDiagQuerywiseTargetDers* result) const = 0;

        virtual TGpuAwareRandom& GetRandom() const = 0;
    };

    template <class TTargetFunc>
    class TPairwiseTargetWrapper: public IPairwiseTargetWrapper {
    public:
        TPairwiseTargetWrapper(const TTargetFunc& target)
            : Target(target)
        {
        }

        TGpuAwareRandom& GetRandom() const final {
            return Target.GetRandom();
        }

        void ComputeStochasticDerivatives(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                          bool isGradient,
                                          TNonDiagQuerywiseTargetDers* result) const final {
            Target.ComputeStochasticDerivatives(bootstrapConfig, isGradient, result);
        }

    private:
        const TTargetFunc& Target;
    };

    class TPairwiseObliviousTreeSearcher {
    public:
        using TVec = TStripeBuffer<float>;
        using TDataSet = TDocParallelDataSet;
        using TSampelsMapping = NCudaLib::TStripeMapping;

        TPairwiseObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                       const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions)
            : FeaturesManager(featuresManager)
            , TreeConfig(learnerOptions)
        {
        }

        template <class TTargetFunc>
        TObliviousTreeModel Fit(const TDataSet& dataSet,
                                const TTargetFunc& objective) {
            return FitImpl(dataSet, TPairwiseTargetWrapper<TTargetFunc>(objective));
        }

    private:
        TObliviousTreeModel FitImpl(const TDataSet& dataSet,
                                    const IPairwiseTargetWrapper& objective);

        void FixSolutionLeavesValuesLayout(const TVector<TBinarySplit>& splits, TVector<float>* leavesPtr,
                                           TVector<double>* weightsPtr);

        TNonDiagQuerywiseTargetDers ComputeWeakTarget(const IPairwiseTargetWrapper& objective);

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
    };

}
