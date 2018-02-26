#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/kernel/random.cuh>

namespace NKernelHost {
    class TPoissonKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<const float> Alphas;
        TCudaBufferPtr<int> Result;

    public:
        TPoissonKernel() = default;

        TPoissonKernel(TCudaBufferPtr<ui64> seeds,
                       TCudaBufferPtr<const float> alphas,
                       TCudaBufferPtr<int> result)
            : Seeds(seeds)
            , Alphas(alphas)
            , Result(result)
        {
        }

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::PoissonRand(Seeds.Get(), Result.Size(), Alphas.Get(), Result.Get(), stream.GetStream());
        }

        Y_SAVELOAD_DEFINE(Seeds, Alphas, Result)
    };

    class TGaussianRandKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TGaussianRandKernel() = default;

        TGaussianRandKernel(TCudaBufferPtr<ui64> seeds,
                            TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::GaussianRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };

    class TUniformRandKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TUniformRandKernel() = default;

        TUniformRandKernel(TCudaBufferPtr<ui64> seeds,
                           TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::UniformRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };

    class TGenerateSeeds: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        ui64 BaseSeed;

    public:
        TGenerateSeeds() = default;

        TGenerateSeeds(TCudaBufferPtr<ui64> seeds,
                       ui64 baseSeed)
            : Seeds(seeds)
            , BaseSeed(baseSeed)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, BaseSeed);

        void Run(const TCudaStream& stream) const {
            NKernel::GenerateSeeds(BaseSeed, Seeds.Get(), Seeds.Size(), stream.GetStream());
        }
    };
}

template <class TMapping>
inline void PoissonRand(TCudaBuffer<ui64, TMapping>& seeds, const TCudaBuffer<float, TMapping>& alphas,
                        TCudaBuffer<int, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TPoissonKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, alphas, result);
}

template <class TMapping>
inline void GaussianRand(TCudaBuffer<ui64, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TGaussianRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

template <class TMapping>
inline void UniformRand(TCudaBuffer<ui64, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TUniformRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

template <class TMapping>
inline void GenerateSeeds(const NCudaLib::TDistributedObject<ui64>& baseSeed,
                          TCudaBuffer<ui64, TMapping>& seeds, ui64 streamId = 0) {
    using TKernel = NKernelHost::TGenerateSeeds;
    LaunchKernels<TKernel>(seeds.NonEmptyDevices(), streamId, seeds, baseSeed);
}
