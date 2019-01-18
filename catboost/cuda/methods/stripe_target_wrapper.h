#pragma once

#include <catboost/cuda/gpu_data/dataset_base.h>
#include <catboost/cuda/cuda_util/gpu_random.h>

namespace NCatboostCuda {
    class IStripeTargetWrapper: public TNonCopyable {
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
    class TStripeTargetWrapper: public IStripeTargetWrapper {
    public:
        TStripeTargetWrapper(const TTargetFunc& target)
            : Target(target)
        {
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

}
