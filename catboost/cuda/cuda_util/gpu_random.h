#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/kernel/random.cuh>

namespace NKernelHost {
    class TPoissonKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ulong> Seeds;
        TCudaBufferPtr<const float> Alphas;
        TCudaBufferPtr<int> Result;

    public:
        TPoissonKernel() = default;

        TPoissonKernel(TCudaBufferPtr<ulong> seeds,
                       TCudaBufferPtr<const float> alphas,
                       TCudaBufferPtr<int> result)
            : Seeds(seeds)
            , Alphas(alphas)
            , Result(result)
        {
        }

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (1L << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::PoissonRand(Seeds.Get(), Result.Size(), Alphas.Get(), Result.Get(), stream.GetStream());
        }

        SAVELOAD(Seeds, Alphas, Result)
    };

    class TGaussianRandKernel {
    private:
        TCudaBufferPtr<ulong> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TGaussianRandKernel() = default;

        TGaussianRandKernel(TCudaBufferPtr<ulong> seeds,
                            TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        SAVELOAD(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (1L << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::GaussianRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };

    class TUniformRandKernel {
    private:
        TCudaBufferPtr<ulong> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TUniformRandKernel() = default;

        TUniformRandKernel(TCudaBufferPtr<ulong> seeds,
                           TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        SAVELOAD(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (1L << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::UniformRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };
}

template <class TMapping>
inline void PoissonRand(TCudaBuffer<ulong, TMapping>& seeds, const TCudaBuffer<float, TMapping>& alphas,
                        TCudaBuffer<int, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TPoissonKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, alphas, result);
}

template <class TMapping>
inline void GaussianRand(TCudaBuffer<ulong, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TGaussianRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

template <class TMapping>
inline void UniformRand(TCudaBuffer<ulong, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TUniformRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}
