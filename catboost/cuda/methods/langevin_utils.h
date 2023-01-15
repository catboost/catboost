#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/methods/kernel/langevin_utils.cuh>
#include <catboost/private/libs/algo_helpers/langevin_utils.h>

namespace NKernelHost {

    class TAddLangevinNoiseKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Values;
        float Coefficient;

    public:
        TAddLangevinNoiseKernel() = default;

        TAddLangevinNoiseKernel(TCudaBufferPtr<ui64> seeds,
                                TCudaBufferPtr<float> values,
                                float coefficient)
                : Seeds(seeds)
                , Values(values)
                , Coefficient(coefficient)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Values, Coefficient);

        void Run(const TCudaStream& stream) const {
            NKernel::AddLangevinNoise(Seeds.Get(),
                                      Values.Get(),
                                      Values.Size(),
                                      Coefficient,
                                      stream.GetStream());
        }
    };
}

template <class TMapping>
inline void AddLangevinNoise(TCudaBuffer<ui64, TMapping>& seeds,
                             TCudaBuffer<float, TMapping>* values,
                             float diffusionTemperature,
                             float learningRate,
                             ui32 stream = 0) {
    if (diffusionTemperature == 0.0f) {
        return;
    }

    const float coefficient = CalcLangevinNoiseRate(diffusionTemperature, learningRate);

    using TKernel = NKernelHost::TAddLangevinNoiseKernel;
    LaunchKernels<TKernel>(seeds.NonEmptyDevices(),
                           stream,
                           seeds,
                           *values,
                           coefficient);
}
