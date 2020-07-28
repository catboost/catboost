#pragma once

#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/targets/weak_objective.h>

namespace NCatboostCuda {
    template <class TTargetFunc>
    class TWeakObjective: public IWeakObjective, public TMoveOnly {
    public:
        using TMapping = typename TTargetFunc::TMapping;
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;
        using TConstVec = TBuffer<const float>;

        TWeakObjective(const TTargetFunc& target)
            : Target(target)
        {
        }

        void StochasticDer(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                           bool secondDerAsWeights,
                           TOptimizationTarget* targetDer) const final {
            TGpuAwareRandom& random = Target.GetRandom();

            auto samplesMapping = Target.GetTarget().GetSamplesMapping();
            TStripeBuffer<float> sampledWeights;
            TStripeBuffer<ui32> sampledIndices;

            Y_ASSERT(bootstrapConfig.GetBootstrapType() != EBootstrapType::MVS);
            const bool isContinuousIndices = TBootstrap<NCudaLib::TStripeMapping>::BootstrapAndFilter(
                bootstrapConfig,
                random,
                samplesMapping,
                &sampledWeights,
                &sampledIndices);

            CATBOOST_DEBUG_LOG << "Sampled docs count " << sampledIndices.GetObjectsSlice().Size() << Endl;

            Target.StochasticDer(std::move(sampledWeights),
                                 std::move(sampledIndices),
                                 secondDerAsWeights,
                                 targetDer);

            targetDer->IsContinuousIndices = isContinuousIndices;
        }

        ui32 GetDim() const final {
            return Target.GetDim();
        }

        TGpuAwareRandom& GetRandom() const {
            return Target.GetRandom();
        }

    private:
        const TTargetFunc& Target;
    };

}
