#include "query_rmse.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernel {




    template <int BLOCK_SIZE>
    __global__ void QueryRmseImpl(const float* diffs, const float* weights,
                                  const ui32* qids, ui32 size,
                                  const float* queryMeans,
                                  const ui32* writeMap,
                                  float* functionValue,
                                  float* der,
                                  float* der2) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float val = i < size ?  diffs[i] : 0;
        const float queryMean = i < size ? __ldg(queryMeans + __ldg(qids + i)) : 0;
        const float direction = val - queryMean;
        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;

        if (i < size) {
            const ui32 dstIdx = writeMap != nullptr ? writeMap[i] : i;

            if (der) {
                der[dstIdx] = weight * direction;
            }
            if (der2) {
                der2[dstIdx] = weight;
            }
        }


        if (functionValue) {
            tmpScores[threadIdx.x] = (i < size) ? -weight * (val - queryMean) * (val - queryMean)  : 0;
            __syncthreads();
        }

        if (functionValue) {
            const float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    void ApproximateQueryRmse(const float* diffs, const float* weights,
                              const ui32* qids, ui32 size,
                              const float* queryMeans,
                              const ui32* writeMap,
                              float* functionValue,
                              float* der,
                              float* der2,
                              TCudaStream stream) {
        const ui32 blockSize = 1024;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        QueryRmseImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(diffs, weights, qids, size, queryMeans, writeMap, functionValue, der, der2);
    }

}
