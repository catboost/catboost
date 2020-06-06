#include "random.cuh"
#include "random_gen.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

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
            ui64 s = __ldg(seeds + i);
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


    __global__ void GenerateSeedsImpl(ui64 baseSeed, ui64* seeds, ui64 size) {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            ui32 baseSeed1 = (baseSeed >> 32);
            ui32 baseSeed2 = (baseSeed & 0xFFFFFF);
            ui32 tmp1 = 134775813 * i + 1664525 * baseSeed1 + 69069 *  baseSeed2 + 1013904225;
            ui32 tmp2 = 1664525 * (baseSeed1 + 134775813  * baseSeed2 + i + 1) + 1013904223;
            for (int j = 0; j < 4 + (threadIdx.x % 8); ++j) {
                tmp1 = AdvanceSeed32(&tmp1);
                tmp2 = AdvanceSeed32(&tmp2);
            }
            //no math here, just stupid heuristics
            ui64 s = (((ui64)tmp1) << 32) | tmp2;
            seeds[i] = AdvanceSeed(&s, blockIdx.x);
            i += gridDim.x * blockDim.x;
        }
    }

    void GenerateSeeds(ui64 baseSeed, ui64* seeds, ui64 size, TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        GenerateSeedsImpl<<<numBlocks, blockSize, 0, stream>>>(baseSeed, seeds, size);

    }


}
