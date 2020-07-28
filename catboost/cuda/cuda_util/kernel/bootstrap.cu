#include "bootstrap.cuh"
#include "random_gen.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

namespace NKernel {

__global__ void PoissonBootstrapImpl(const float lambda, ui64* seeds, ui32 seedSize, float* weights, ui32 size) {

    seeds += blockIdx.x * blockDim.x + threadIdx.x;
    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    ui64 s = seeds[0];
    while (i < size) {
        float w = weights[i];
        weights[i] = w * NextPoisson(&s, lambda);
        i += gridDim.x * blockDim.x;
    }
    seeds[0] = s;

}

__global__ void GammaBootstrapImpl(const float scale, const float shape, ui64* seeds, ui32 seedSize, float* weights, ui32 size) {

    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    seeds += i;
    ui64 s = seeds[0];

    while (i < size) {
        float w = weights[i];
        weights[i] = w * NextGamma(&s, scale, shape);
        i += gridDim.x * blockDim.x;
    }
    seeds[0] = s;
}

__global__ void BayesianBootstrapImpl(ui64* seeds, ui32 seedSize, float* weights, ui32 size, float temperature) {

    seeds += blockIdx.x * blockDim.x + threadIdx.x;
    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

    ui64 s = seeds[0];

    while (i < size) {
        float w = weights[i];
        const float tmp = (-log(NextUniform(&s) + 1e-20f));
        weights[i] = w * (temperature != 1.0f ? powf(tmp, temperature) : tmp);
        i += gridDim.x * blockDim.x;
    }
    seeds[0] = s;
}

__global__ void UniformBootstrapImpl(const float sampleRate, ui64* seeds, ui32 seedSize, float* weights, ui32 size) {

    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    seeds += i;
    ui64 s = seeds[0];
    while (i < size) {
        const float w = weights[i];
        const float flag = (NextUniform(&s) < sampleRate) ? 1.0f : 0.0f;
        weights[i] = w * flag;
        i += gridDim.x * blockDim.x;
    }
    seeds[0] = s;
}

void PoissonBootstrap(const float lambda, ui64* seeds, ui32 seedsSize, float* weights, ui32 weighsSize, TCudaStream stream) {
    const ui32 blockSize = 256;
    const ui32 numBlocks = min(CeilDivide(seedsSize, blockSize), CeilDivide(weighsSize, blockSize));
    PoissonBootstrapImpl<<<numBlocks, blockSize, 0, stream>>>(lambda, seeds, seedsSize, weights, weighsSize);
}

void UniformBootstrap(const float sampleRate, ui64* seeds, ui32 seedSize, float* weights, ui32 size, TCudaStream stream) {
    const ui32 blockSize = 256;
    const ui32 numBlocks =  min(CeilDivide(seedSize, blockSize), CeilDivide(size, blockSize));
    UniformBootstrapImpl<<<numBlocks, blockSize, 0 , stream>>>(sampleRate, seeds, seedSize, weights, size);
}

void BayesianBootstrap(ui64* seeds, ui32 seedSize, float* weights, ui32 size, float temperature, TCudaStream stream) {
    const ui32 blockSize = 256;
    const ui32 numBlocks =  min(CeilDivide(seedSize, blockSize), CeilDivide(size, blockSize));
//    GammaBootstrapImpl<<<numBlocks, blockSize, 0 , stream>>>(1.0f, 1.0f, seeds, seedSize, weights, size);
    BayesianBootstrapImpl<<<numBlocks, blockSize, 0 , stream>>>(seeds, seedSize, weights, size, temperature);
}
}

