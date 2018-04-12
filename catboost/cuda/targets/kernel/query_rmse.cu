#include "query_rmse.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernel {

    template<int BLOCK_SIZE>
    __global__ void ComputeGroupIdsImpl(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst)
    {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;
        ui32 writeOffset = qid < qCount ? (qOffsets[qid] - offsetsBias) : 0;
        dst += writeOffset;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qSizes[qid] : 0;

        for (int i = x; i < querySize; i += 32) {
            dst[i] = qid;
        }
    }


    void ComputeGroupIds(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst, TCudaStream stream) {
        const int blockSize = 128;
        const int numBlocks = (qCount * 32 + 127) / blockSize;
        if (numBlocks > 0)
        {
            ComputeGroupIdsImpl<blockSize><<< numBlocks, blockSize, 0, stream >>>(qSizes, qOffsets, offsetsBias, qCount, dst);
        }
    }

    template<int BLOCK_SIZE>
    __global__ void ComputeGroupMeansImpl(const float* target, const float* weights,
                                          const ui32* qOffsets, int offsetsBias,
                                          const ui32* qSizes, int qCount,
                                          float* queryMeans)
    {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ volatile float line[BLOCK_SIZE];
        __shared__ float result[queriesPerBlock];
        ui32 readOffset = qid < qCount ? (qOffsets[qid] - offsetsBias) : 0;
        weights += (weights != nullptr) ? readOffset : 0;
        target += readOffset;
        queryMeans += blockIdx.x * queriesPerBlock;
        line[threadIdx.x] = 0;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qSizes[qid] : 0;

        float sumTarget = 0;
        float sumWeight = 0;

        for (int i = x; i < querySize; i += 32) {
            const float t = __ldg(target + i);
            const float w = weights != nullptr ? __ldg(weights + i) : 1.0f;
            sumTarget += t;
            sumWeight += w;
        }

        line[threadIdx.x] = sumTarget;
        const float totalSum = WarpReduce(x, line + localQid * 32, 32);
        line[threadIdx.x] = sumWeight;
        const float totalWeight = WarpReduce(x, line + localQid * 32, 32);

        if (x == 0) {
            result[localQid] = totalWeight != 0 ? totalSum / totalWeight : 0;
        }
        __syncthreads();

        if (x == 0 && (qid < qCount)) {
            queryMeans[localQid] = result[localQid];
        }
    }

    void ComputeGroupMeans(const float* target, const float* weights,
                           const ui32* qOffsets, ui32 qOffsetsBias,
                           const ui32* qSizes, ui32 qCount,
                           float* result, TCudaStream stream) {
        const int blockSize = 128;
        const int numBlocks = (qCount * 32 + 127) / blockSize;
        if (numBlocks > 0)
        {
            ComputeGroupMeansImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>> (target, weights, qOffsets, qOffsetsBias, qSizes, qCount, result);
        }
    }

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
