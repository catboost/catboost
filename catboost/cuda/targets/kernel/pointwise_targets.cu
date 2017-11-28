#include "pointwise_targets.cuh"

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>


namespace NKernel {

    template <int BLOCK_SIZE>
    __global__ void MseImpl(const float* relevs, const float* weights, ui32 size,
                            const float* predictions,
                            float* functionValue,
                            float* der,
                            float* der2) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float val = i < size ? predictions[i] : 0;
        const float relev = i < size ? relevs[i] : 0;
        const float direction = relev - val;
        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;

        if (i < size) {
            if (der) {
                der[i] = weight * direction;
            }
            if (der2) {
                der2[i] = weight;
            }
        }

        if (functionValue) {
            tmpScores[threadIdx.x] = (i < size) ? -weight * (val - relev) * (val - relev)  : 0;
            __syncthreads();
        }

        if (functionValue) {
            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }


    template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD, bool HAS_BORDER>
    __launch_bounds__(BLOCK_SIZE, 2048 / BLOCK_SIZE)
    __global__ void CrossEntropyImpl(const float* targetClasses, const float* targetWeights, ui32 size,
                                     const float* predictions,
                                     float* functionValue, float* der, float* der2,
                                     float border) {
        ui32 tid = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x;

        float tmpScore = 0;

        float direction[ELEMENTS_PER_THREAD];
        float weight[ELEMENTS_PER_THREAD];
        float scale[ELEMENTS_PER_THREAD];

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;
            direction[j] = idx < size ? predictions[idx] : 0;
            weight[j] = (targetWeights && (idx < size)) ? targetWeights[idx] : 1.0f;
            scale[j] = (idx < size) ? targetClasses[idx] : 1.0f;
        }

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;
            const float val = direction[j];
            const float targetClass = scale[j];

            const float expVal = idx < size ? __expf(val) : 0;
            const float p = max(min(isfinite(expVal) ? expVal / (1.0f + expVal) : 1.0f, 1.0f - 1e-40f), 1e-40f);
            const float c = HAS_BORDER ? targetClass > border : targetClass;

            direction[j] = c - p; //c * (1 - p) - (1-c) * p;
            scale[j] = p * (1.0f - p);

            if (functionValue) {
                const float logExpValPlusOne = isfinite(expVal) ? __logf(1 + expVal) : val;
                tmpScore += (idx < size) ? weight[j] * (c * val - logExpValPlusOne) : 0;
            }
        }

        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = tid + j * BLOCK_SIZE;

            //we already classify this observations
            if (der && (idx < size)) {
                der[idx] = weight[j] * direction[j];
            }
            if (der2  && (idx < size)) {
               der2[idx] = weight[j] * scale[j];
            }
        }

        if (functionValue) {
            __shared__ float tmpScores[BLOCK_SIZE];
            tmpScores[threadIdx.x] = tmpScore;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }



    void CrossEntropyTargetKernel(const float* targetClasses, const float* targetWeights, ui32 size,
                                  const float* predictions,
                                  float* functionValue, float* der, float* der2,
                                  float border, bool useBorder, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 elementsPerThreads = 2;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        if (useBorder)
        {
            CrossEntropyImpl < blockSize, elementsPerThreads, true ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
        } else {
            CrossEntropyImpl < blockSize, elementsPerThreads, false ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
        }
    }


    void MseTargetKernel(const float* relevs, const float* weights, ui32 size,
                         const float* predictions,
                         float* functionValue, float* der, float* der2,
                         TCudaStream stream) {
        const ui32 blockSize = 1024;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        MseImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(relevs, weights, size, predictions, functionValue, der, der2);
    }


}
