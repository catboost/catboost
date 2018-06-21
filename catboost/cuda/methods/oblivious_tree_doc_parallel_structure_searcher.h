#pragma once

#include "weak_target_helpers.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/libs/options/oblivious_tree_options.h>

namespace NCatboostCuda {

    class IStripeTargetWrapper : public TNonCopyable {
    public:
        virtual ~IStripeTargetWrapper() {
        }

        virtual void GradientAtZero(TStripeBuffer<float>& weightedDer,
                                    TStripeBuffer<float>& weights,
                                    ui32 stream = 0) const = 0;

        virtual void NewtonAtZero(TStripeBuffer<float>& weightedDer,
                                  TStripeBuffer<float>& weightedDer2,
                                  ui32 stream = 0) const = 0;

        virtual const TTarget<NCudaLib::TStripeMapping>& GetTarget() const = 0;
        virtual TGpuAwareRandom& GetRandom() const = 0;
    };

    template <class TTargetFunc>
    class TStripeTargetWrapper : public IStripeTargetWrapper {
    public:

        TStripeTargetWrapper(const TTargetFunc& target)
                : Target(target) {

        }

        const TTarget<NCudaLib::TStripeMapping>& GetTarget() const final {
            return Target.GetTarget();
        }

        TGpuAwareRandom& GetRandom() const final {
            return Target.GetRandom();
        }

        void GradientAtZero(TStripeBuffer<float>& weightedDer,
                            TStripeBuffer<float>& weights,
                            ui32 stream = 0) const final {
            Target.GradientAtZero(weightedDer, weights, stream);
        }

        void NewtonAtZero(TStripeBuffer<float>& weightedDer,
                          TStripeBuffer<float>& weightedDer2,
                          ui32 stream = 0) const final {
            Target.NewtonAtZero(weightedDer, weightedDer2, stream);
        };
    private:
        const TTargetFunc& Target;
    };

    class TDocParallelObliviousTreeSearcher {
    public:
        using TDataSet = TDocParallelDataSet;
        using TVec = TStripeBuffer<float>;
        using TSampelsMapping = NCudaLib::TStripeMapping;
        using TWeakTarget = TL2Target<NCudaLib::TStripeMapping>;

        TDocParallelObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                          const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
                                          TBootstrap<NCudaLib::TStripeMapping>& bootstrap,
                                          double randomStrengthMult)
            : FeaturesManager(featuresManager)
            , TreeConfig(learnerOptions)
            , Bootstrap(bootstrap)
            , ModelLengthMultiplier(randomStrengthMult)
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

        TVector<float> ExtractWeights(const TVector<TPartitionStatistics>& statCpu);

        void ComputeWeakTarget(const IStripeTargetWrapper& objective,
                               double* scoreStdDev,
                               TWeakTarget* target,
                               TStripeBuffer<ui32>* indices);

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        TBootstrap<NCudaLib::TStripeMapping>& Bootstrap;
        double ModelLengthMultiplier = 0.0;
    };
}
