#include "random.cuh"
#include "random_gen.cuh"
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

namespace NKernel {

    __global__ void PoissonRandImpl(ui64* seeds, ui32 seedSize,
                                    const float* alpha, int* result)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < seedSize) {
            ui64 s = seeds[i];
            result[i] = NextPoisson(&s, alpha[i]);
            seeds[i] = s;
            i += gridDim.x * blockDim.x;
        }
    }

    void PoissonRand(ui64* seeds, ui32 size, const float* alphas, int* result, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   TArchProps::MaxBlockCount());
        PoissonRandImpl<<<numBlocks,blockSize, 0, stream>>>(seeds, size, alphas, result);
    }

    __global__ void GaussianRandImpl(ui64* seeds, ui32 seedSize, float* result)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < seedSize) {
            ui64 s = seeds[i];
            result[i] = NextNormal(&s);
            seeds[i] = s;
            i += gridDim.x * blockDim.x;
        }
    }

    void GaussianRand(ui64* seeds, ui32 size, float* result, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   TArchProps::MaxBlockCount());
        GaussianRandImpl<<<numBlocks,blockSize, 0, stream>>>(seeds, size, result);
    }

    __global__ void UniformRandImpl(ui64* seeds, ui32 seedSize, float* result)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < seedSize) {
            ui64 s = seeds[i];
            result[i] = NextUniform(&s);
            seeds[i] = s;
            i += gridDim.x * blockDim.x;
        }
    }

    void UniformRand(ui64* seeds, ui32 size, float* result, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   TArchProps::MaxBlockCount());
        UniformRandImpl<<<numBlocks, blockSize, 0, stream>>>(seeds, size, result);
    }

    __global__ void GammaRandImpl(ui64* seeds, const float* alphas,
                                  const float* scale, ui32 seedSize, float* result)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < seedSize) {
            ui64 s = seeds[i];
            result[i] = NextGamma(&s, alphas[i], scale[i]);
            seeds[i] = s;
            i += gridDim.x * blockDim.x;
        }
    }

    void GammaRand(ui64* seeds, const float* alphas, const float* scale,
                   ui32 size, float* result, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   TArchProps::MaxBlockCount());
        GammaRandImpl<<<numBlocks, blockSize, 0, stream>>>(seeds, alphas, scale, size, result);
    }

    __global__ void BetaRandImpl(ui64* seeds, const float* alphas,
                                 const float* betas, ui32 seedSize, float* result)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < seedSize) {
            ui64 s = seeds[i];
            result[i] = NextBeta(&s, alphas[i], betas[i]);
            seeds[i] = s;
            i += gridDim.x * blockDim.x;
        }
    }

    void BetaRand(ui64* seeds, const float* alphas, const float* betas,
                  ui32 size, float* result, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   TArchProps::MaxBlockCount());
        BetaRandImpl<<<numBlocks, blockSize, 0, stream>>>(seeds, alphas, betas, size, result);
    }

}
