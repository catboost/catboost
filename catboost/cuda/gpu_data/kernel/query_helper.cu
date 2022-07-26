#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <util/generic/cast.h>

namespace NKernel {

    template <ui32 BLOCK_SIZE>
    __global__ void ComputeGroupIdsImpl(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst) {
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

    __device__ __forceinline__ ui32 SampledQuerySize(float sampleRate, ui32 qSize)  {
        const ui32 sampledSize = ceil(sampleRate * qSize);
        if (sampledSize < 2) {
            return min(2, qSize);
        }
        return sampledSize;
    }

    void ComputeGroupIds(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst, TCudaStream stream) {
        const ui64 blockSize = 128;
        const ui64 numBlocks = CeilDivide(static_cast<ui64>(qCount) * 32, blockSize);
        if (numBlocks > 0) {
            ComputeGroupIdsImpl<blockSize><<< numBlocks, blockSize, 0, stream >>>(qSizes, qOffsets, offsetsBias, qCount, dst);
        }
    }



    __global__ void FillQueryEndMaskImpl(const ui32* qids, const ui32* docs, ui32 docCount, ui32* masks) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < docCount) {
            ui32 idx = docs[i];
            const ui32 qid = qids[idx];
            const ui32 nextDoc = i + 1 < docCount ? docs[i + 1] : static_cast<ui32>(-1);
            const ui32 isEnd = i + 1 < docCount ? qid != qids[nextDoc] : 1;
            masks[i] = isEnd;
        }
    }

    __global__ void CreateSortKeysImpl(ui64* seeds, const ui32* qids,  ui32 docCount, ui64* keys) {
            ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            seeds += i;
            ui64 s = seeds[0];

            while (i < docCount) {
                const ui64 highBits = ((ui64)qids[i]) << 32;
                const ui64 lowBits = AdvanceSeed(&s) >> 32;
                keys[i] = lowBits | highBits;
                i += gridDim.x * blockDim.x;
            }
            seeds[0] = s;
    }

    void CreateSortKeys(ui64* seeds, ui32 seedSize, const ui32* qids,  ui32 docCount, ui64* keys, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks =  min(CeilDivide(seedSize, blockSize),
                                    CeilDivide(docCount, blockSize));
        if (numBlocks) {
            CreateSortKeysImpl<<<numBlocks, blockSize, 0, stream>>>(seeds, qids, docCount, keys);
        }
    }


    void FillQueryEndMask(const ui32* qids, const ui32* docs, ui32 docCount, ui32* masks, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = (docCount + blockSize - 1) / blockSize;
        if (numBlocks) {
            FillQueryEndMaskImpl<<<numBlocks, blockSize, 0, stream>>>(qids, docs, docCount, masks);

        }
    }


    __global__ void FillTakenDocsMaskImpl(const float* takenQueryMasks,
                                          const ui32* qids,
                                          const ui32* docs, ui32 docCount,
                                          const ui32* queryOffsets,
                                          const ui32 queryOffsetsBias,
                                          const ui32* querySizes,
                                          const float docwiseSampleRate,
                                          const ui32 maxQuerySize,
                                          float* takenMask) {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        while (i < docCount) {
            const ui32 doc = docs[i];
            const ui32 queryId = __ldg(qids + doc);
            const ui32 queryOffset = __ldg(queryOffsets + queryId) - queryOffsetsBias;
            const ui32 querySize = __ldg(querySizes + queryId);
            const ui32 sampledQuerySize = min(maxQuerySize, SampledQuerySize(docwiseSampleRate, querySize));
            float mask = __ldg(takenQueryMasks + queryId) * ((i - queryOffset) < sampledQuerySize);
            takenMask[i] = mask;
            i += gridDim.x * blockDim.x;
        }
    }

    void FillTakenDocsMask(const float* takenQueryMasks,
                           const ui32* qids,
                           const ui32* docs, ui32 docCount,
                           const ui32* queryOffsets,
                           const ui32 queryOffsetsBias,
                           const ui32* querySizes,
                           const float docwiseSampleRate,
                           const ui32 maxQuerySize,
                           float* takenMask,
                           TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = (docCount + blockSize - 1) / blockSize;
        if (numBlocks) {
            FillTakenDocsMaskImpl<<<numBlocks, blockSize, 0, stream>>>(takenQueryMasks, qids, docs, docCount, queryOffsets, queryOffsetsBias, querySizes, docwiseSampleRate, maxQuerySize, takenMask);
        }

    }


    template <int BLOCK_SIZE>
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
            sumTarget += t * w;
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
    __global__ void ComputeGroupMeansImpl(const float* target, const float* weights,
                                          const ui32* qOffsets, int qCount,
                                          float* queryMeans)
    {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ volatile float line[BLOCK_SIZE];
        __shared__ float result[queriesPerBlock];
        ui32 queryOffset = qid < qCount ? qOffsets[qid] : 0;
        weights += (weights != nullptr) ? queryOffset : 0;
        target += queryOffset;
        queryMeans += blockIdx.x * queriesPerBlock;
        line[threadIdx.x] = 0;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qOffsets[qid + 1] - queryOffset : 0; // qCount == QidsOffsets.Size() - 1

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
                           const ui32* qOffsets,  ui32 qCount,
                           float* result, TCudaStream stream) {
        const ui32 blockSize = 128;
        const ui32 numBlocks = SafeIntegerCast<ui32>(((ui64)(qCount + 1) * 32 + blockSize - 1) / blockSize); // qOffsets points at qCount+1 ui32's
        if (numBlocks > 0) {
            ComputeGroupMeansImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>> (target, weights, qOffsets,  qCount, result);
        }
    }


    template <int BLOCK_SIZE>
    __global__ void ComputeGroupMaxImpl(const float* target,
                                        const ui32* qOffsets, int qCount,
                                        float* result) {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ volatile float line[BLOCK_SIZE];
        ui32 queryOffset = qid < qCount ? qOffsets[qid] : 0;
        target += queryOffset;
        result += blockIdx.x * queriesPerBlock;
        line[threadIdx.x] = 0;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qOffsets[qid + 1] - queryOffset : 0; // qCount == QidsOffsets.Size() - 1

        float maxValue = NegativeInfty();

        for (int i = x; i < querySize; i += 32) {
            const float t = __ldg(target + i);
            maxValue = max(t, maxValue);
        }

        line[threadIdx.x] = maxValue;
        const float queryMax = WarpReduce(x, line + localQid * 32, 32, TCudaMax<float>());

        __syncthreads();

        if (x == 0 && (qid < qCount)) {
            result[localQid] = queryMax > NegativeInfty() ? queryMax : 0;
        }
    }

    void ComputeGroupMax(const float* target,
                         const ui32* qOffsets,  ui32 qCount,
                         float* result, TCudaStream stream) {
        const ui32 blockSize = 128;
        const ui32 numBlocks = SafeIntegerCast<ui32>(((ui64)(qCount + 1) * 32 + blockSize - 1) / blockSize); // qOffsets points at qCount+1 ui32's
        if (numBlocks > 0) {
            ComputeGroupMaxImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>> (target, qOffsets,  qCount, result);
        }
    }


    __global__ void RemoveGroupMeansImpl(const float* queryMeans, const ui32* qids, ui32 size, float* dst) {
        const ui32 docId = blockIdx.x * blockDim.x + threadIdx.x;
        if (docId < size) {
            dst[docId] -= __ldg(queryMeans + qids[docId]);
        }
    }

    void RemoveGroupBias(const float *queryMeans, const ui32 *qids, ui32 size, float *dst, TCudaStream stream) {
        const int blockSize = 256;
        const int numBlocks = (size + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            RemoveGroupMeansImpl<<< numBlocks, blockSize, 0, stream >>> (queryMeans, qids, size, dst);
        }
    }
}
