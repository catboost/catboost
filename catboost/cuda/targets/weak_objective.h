#pragma once

#include <util/generic/noncopyable.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/private/libs/options/bootstrap_options.h>

namespace NCatboostCuda {
    struct TOptimizationTarget {
        TStripeBuffer<float> StatsToAggregate; //zero column is weight, other are targets
        TStripeBuffer<ui32> Indices;
        /* for continuous we don't need to gather on first level */
        bool IsContinuousIndices = true;
        /* for multiclass there is one interesting thing: with l2-leaf-reg it's better to use approxDim equal to
         * number of classes cause convergence will be slightly faster and for weak learners this gives benefits
         * but for performance reasons we zero last approx after obtaining each tree and don't want to compute histograms
         * for them, so there are some math "magic" formulas to rewrite everything
        */
        bool MultiLogitOptimization = false;
    };

    class IWeakObjective: public TNonCopyable {
    public:
        virtual ~IWeakObjective() {
        }

        virtual void StochasticDer(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                   bool secondDerAsWeights,
                                   TOptimizationTarget* target) const = 0;

        virtual ui32 GetDim() const = 0;

        virtual TGpuAwareRandom& GetRandom() const = 0;
    };

}
