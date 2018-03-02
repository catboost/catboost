#include "pair_logit.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>


namespace NKernel {


    template <int BLOCK_SIZE>
    __global__ void PairLogitPointwiseTargetImpl(const float* point,
                                                 const uint2* pairs, const float* pairWeights,
                                                 const ui32* writeMap,
                                                 ui32 pairCount, int pairShift,
                                                 float* functionValue,
                                                 float* der,
                                                 float* der2)  {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float tmpScores[BLOCK_SIZE];

        uint2 pair;
        if (i < pairCount) {
            pair = __ldg(pairs + i);
        } else {
            pair.x = pairShift;
            pair.y = pairShift;
        }
        pair.x -= pairShift;
        pair.y -= pairShift;
        const float w = pairWeights && (i < pairCount) ? pairWeights[i] : 1.0f;
        const float diff = i < pairCount ? __ldg(point + pair.x) - __ldg(point + pair.y) : 0;
        const float expDiff = __expf(diff);
        const float p = max(min(isfinite(expDiff) ? expDiff / (1.0f + expDiff) : 1.0f, 1.0f - 1e-40f), 1e-40f);

        const float direction = (1.0f - p);

        const ui32 firstDst = writeMap ? writeMap[pair.x] : pair.x;
        const ui32 secondDst = writeMap ? writeMap[pair.y] : pair.y;

        if (der && i < pairCount) {
            atomicAdd(der + firstDst, w * direction);
            atomicAdd(der + secondDst, -w * direction);
        }

        if (der2 && i < pairCount) {
            const float scale = p * (1.0f - p);
            atomicAdd(der2 + firstDst, w * scale);
            atomicAdd(der2 + secondDst, w * scale);
        }

        if (functionValue) {
            const float logExpValPlusOne = isfinite(expDiff) ? __logf(1.0f + expDiff) : expDiff;
            tmpScores[threadIdx.x] = (i < pairCount) ? w * (diff - logExpValPlusOne) : 0;

            __syncthreads();
            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }


    __global__ void MakePairWeightsImpl(const uint2* pairs, const float* pairWeights, ui32 pairCount,
                                        float* weights)  {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < pairCount) {
            uint2 pair = __ldg(pairs + i);
            const float w = pairWeights ? pairWeights[i] : 1.0f;
            atomicAdd(weights + pair.x, w);
            atomicAdd(weights + pair.y, w);
        }
    }

    void MakePairWeights(const uint2* pairs, const float* pairWeights, ui32 pairCount,
                         float* weights, TCudaStream stream) {
        const int blockSize = 512;
        const int numBlocks = (pairCount + blockSize - 1) / blockSize;
        MakePairWeightsImpl<<<numBlocks, blockSize, 0, stream>>>(pairs, pairWeights, pairCount, weights);

    }

    void PairLogitPointwiseTarget(const float* point,
                                  const uint2* pairs, const float* pairWeights,
                                  const ui32* writeMap,
                                  ui32 pairCount, int pairShift,
                                  float* functionValue,
                                  float* der,
                                  float* der2,
                                  ui32 docCount,
                                  TCudaStream stream) {

        const int blockSize = 1024;
        const int numBlocks = (pairCount + blockSize - 1) / blockSize;
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        if (der) {
            FillBuffer(der, 0.0f, docCount, stream);
        }
        if (der2) {
            FillBuffer(der2, 0.0f, docCount, stream);
        }
        if (numBlocks)
        {
            PairLogitPointwiseTargetImpl<blockSize> << <numBlocks, blockSize, 0, stream >> > (point, pairs, pairWeights, writeMap, pairCount, pairShift, functionValue, der, der2);
        }

    }
}
