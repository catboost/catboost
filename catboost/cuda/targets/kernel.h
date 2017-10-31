#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/targets/kernel/pointwise_targets.cuh>

namespace NKernelHost {
    class TCrossEntropyTargetKernelKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TargetClasses;
        TCudaBufferPtr<const float> TargetWeights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;
        float Border;
        bool UseBorder;

    public:
        TCrossEntropyTargetKernelKernel() = default;

        TCrossEntropyTargetKernelKernel(TCudaBufferPtr<const float> targetClasses, TCudaBufferPtr<const float> targetWeights, TCudaBufferPtr<const float> predictions, TCudaBufferPtr<float> functionValue, TCudaBufferPtr<float> der, TCudaBufferPtr<float> der2, float border, bool useBorder)
            : TargetClasses(targetClasses)
            , TargetWeights(targetWeights)
            , Predictions(predictions)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
            , Border(border)
            , UseBorder(useBorder)
        {
        }

        SAVELOAD(TargetClasses, TargetWeights, Predictions, FunctionValue, Der, Der2, Border, UseBorder);

        void Run(const TCudaStream& stream) const {
            NKernel::CrossEntropyTargetKernel(TargetClasses.Get(), TargetWeights.Get(), TargetClasses.Size(), Predictions.Get(), FunctionValue.Get(), Der.Get(), Der2.Get(), Border, UseBorder, stream.GetStream());
        }
    };

    class TMseTargetKernelKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Relevs;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<float> FunctionValue;
        TCudaBufferPtr<float> Der;
        TCudaBufferPtr<float> Der2;

    public:
        TMseTargetKernelKernel() = default;

        TMseTargetKernelKernel(TCudaBufferPtr<const float> relevs, TCudaBufferPtr<const float> weights, TCudaBufferPtr<const float> predictions, TCudaBufferPtr<float> functionValue, TCudaBufferPtr<float> der, TCudaBufferPtr<float> der2)
            : Relevs(relevs)
            , Weights(weights)
            , Predictions(predictions)
            , FunctionValue(functionValue)
            , Der(der)
            , Der2(der2)
        {
        }

        SAVELOAD(Relevs, Weights, Predictions, FunctionValue, Der, Der2);

        void Run(const TCudaStream& stream) const {
            NKernel::MseTargetKernel(Relevs.Get(), Weights.Get(), static_cast<ui32>(Relevs.Size()),
                                     Predictions.Get(),
                                     FunctionValue.Get(), Der.Get(), Der2.Get(),
                                     stream.GetStream());
        }
    };
}

template <class TMapping>
inline void ApproximateMse(const TCudaBuffer<const float, TMapping>& target,
                           const TCudaBuffer<const float, TMapping>& weights,
                           const TCudaBuffer<const float, TMapping>& point,
                           TCudaBuffer<float, TMapping>* score,
                           TCudaBuffer<float, TMapping>* weightedDer,
                           TCudaBuffer<float, TMapping>* weightedDer2,
                           ui32 stream = 0) {
    using TKernel = NKernelHost::TMseTargetKernelKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, score, weightedDer, weightedDer2);
}

template <class TMapping>
inline void ApproximateCrossEntropy(const TCudaBuffer<const float, TMapping>& target,
                                    const TCudaBuffer<const float, TMapping>& weights,
                                    const TCudaBuffer<const float, TMapping>& point,
                                    TCudaBuffer<float, TMapping>* score,
                                    TCudaBuffer<float, TMapping>* weightedDer,
                                    TCudaBuffer<float, TMapping>* weightedDer2,
                                    bool useBorder,
                                    float border,
                                    ui32 stream = 0) {
    using TKernel = NKernelHost::TCrossEntropyTargetKernelKernel;
    LaunchKernels<TKernel>(target.NonEmptyDevices(), stream, target, weights, point, score, weightedDer, weightedDer2, border, useBorder);
}
