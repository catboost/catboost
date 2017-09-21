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


    template <int BLOCK_SIZE, bool HAS_BORDER>
    __global__ void CrossEntropyImpl(const float* targetClasses, const float* targetWeights, ui32 size,
                                     const float* predictions,
                                     float* functionValue, float* der, float* der2,
                                     float border) {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float val = i < size ? predictions[i] : 0;
        const float expVal = i < size ? __expf(val) : 0;

        const float p = max(min(isfinite(expVal) ? expVal / (1.0f + expVal) : 1.0f, 1.0f - 1e-40f), 1e-40f);
        const float targetClass = (i < size) ? targetClasses[i] : 1.0f;
        const float c = HAS_BORDER ? targetClass > border : targetClass;
        const float direction = c - p; //c * (1 - p) - (1-c) * p;
        const float weight =  (targetWeights && (i < size)) ? targetWeights[i] : 1.0f;

        if (i < size) {
            const float scale = p * (1.0f - p);
            //we already classify this observations
            if (der) {
                der[i] = weight * direction;
            }
            if (der2) {
               der2[i] = weight * scale;
            }
        }

        if (functionValue) {
            const float logExpValPlusOne = isfinite(expVal) ? __logf(1 + expVal) : val;
            tmpScores[threadIdx.x] = (i < size) ? weight * (c * val - logExpValPlusOne) : 0;
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
        const ui32 blockSize = 1024;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        if (useBorder)
        {
            CrossEntropyImpl < blockSize, true ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
        } else {
            CrossEntropyImpl < blockSize, false ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, targetWeights, size, predictions, functionValue, der, der2, border);
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
