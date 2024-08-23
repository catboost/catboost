#include "query_rmse.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernel {

    template <int BLOCK_SIZE>
    __global__ void ComputeGroupMaximalsImpl(const float* target, const float* weights,
                                             const float* approxExp,
                                             const ui32* qOffsets, int offsetsBias,
                                             const ui32* qSizes, int qCount,
                                             float* maximals, float* sumWeightedTargets)
    {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ volatile float line[BLOCK_SIZE];
        ui32 readOffset = qid < qCount ? (qOffsets[qid] - offsetsBias) : 0;
        weights += (weights != nullptr) ? readOffset : 0;
        target += readOffset;
        approxExp += readOffset;
        maximals += blockIdx.x * queriesPerBlock;
        sumWeightedTargets += blockIdx.x * queriesPerBlock;
        line[threadIdx.x] = 0;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qSizes[qid] : 0;

        float maxApprox = -FLT_MAX;
        float sumWeightedTarget = 0;

        for (int i = x; i < querySize; i += 32) {
            const float t = __ldg(target + i);
            const float w = weights != nullptr ? __ldg(weights + i) : 1.0f;
            const float a = __ldg(approxExp + i);
            maxApprox = (w > 0) ? max(maxApprox, a) : maxApprox;
            sumWeightedTarget += t * w;
        }

        line[threadIdx.x] = maxApprox;
        const float totalMaxApprox = WarpReduce(x, line + localQid * 32, 32, TCudaMax<float>());
        line[threadIdx.x] = sumWeightedTarget;
        const float totalSumWeightedTarget = WarpReduce(x, line + localQid * 32, 32);

        if (x == 0 && (qid < qCount)) {
            maximals[localQid] = totalMaxApprox;
            sumWeightedTargets[localQid] = totalSumWeightedTarget;
        }
    }

    void ComputeGroupMaximals(const float* target, const float* weights,
                              const float* approxExp,
                              const ui32* qOffsets, ui32 qOffsetsBias,
                              const ui32* qSizes, ui32 qCount,
                              float* maximals, float* sumWeightedTargets,
                              TCudaStream stream) {
        const int blockSize = 128;
        const int numBlocks = (qCount * 32 + 127) / blockSize;
        if (numBlocks > 0) {
            ComputeGroupMaximalsImpl<blockSize> <<< numBlocks, blockSize, 0, stream >>> (target, weights, approxExp, qOffsets, qOffsetsBias, qSizes, qCount, maximals, sumWeightedTargets);
        }
    }

    template <int BLOCK_SIZE>
    __global__ void ComputeQueryExponentsImpl(const float* weights,
                                              const ui32* qids, ui32 size,
                                              const float* maximals,
                                              const ui32* writeMap,
                                              float* approxExp,
                                              float beta) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;
        const float approx = i < size ? approxExp[i] : 0;
        const float apprMax = i < size ? __ldg(maximals + __ldg(qids + i)) : 0;
        const float apprExp = __expf(beta * (approx - apprMax)) * weight;

        if (i < size) {
            approxExp[i] = apprExp;
        }
    }

    void ComputeQueryExponents(const float* weights,
                               const ui32* qids, ui32 size,
                               const float* maximals,
                               const ui32* writeMap,
                               float* approxExp,
                               float beta,
                               TCudaStream stream) {
        const ui32 blockSize = 1024;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        ComputeQueryExponentsImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(weights, qids, size, maximals, writeMap, approxExp, beta);
    }

    template <int BLOCK_SIZE>
    __global__ void ComputeGroupSumsImpl(const float* data,
                                         const ui32* qOffsets, int offsetsBias,
                                         const ui32* qSizes, int qCount,
                                         float* groupSums)
    {
        const int queriesPerBlock = BLOCK_SIZE / 32;
        const int localQid = threadIdx.x / 32;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ volatile float line[BLOCK_SIZE];
        __shared__ float result[queriesPerBlock];
        ui32 readOffset = qid < qCount ? (qOffsets[qid] - offsetsBias) : 0;
        data += readOffset;
        groupSums += blockIdx.x * queriesPerBlock;
        line[threadIdx.x] = 0;

        const int x = threadIdx.x & 31;
        const int querySize = qid < qCount ? qSizes[qid] : 0;

        float sumData = 0;

        for (int i = x; i < querySize; i += 32) {
            const float a = __ldg(data + i);
            sumData += a;
        }

        line[threadIdx.x] = sumData;
        const float totalSumData = WarpReduce(x, line + localQid * 32, 32);

        if (x == 0) {
            result[localQid] = totalSumData;
        }
        __syncthreads();

        if (x == 0 && (qid < qCount)) {
            groupSums[localQid] = result[localQid];
        }
    }

    void ComputeGroupSums(const float* approxExp,
                          const ui32* qOffsets, ui32 qOffsetsBias,
                          const ui32* qSizes, ui32 qCount,
                          float* approxExpSum, TCudaStream stream) {
        const int blockSize = 128;
        const int numBlocks = (qCount * 32 + 127) / blockSize;
        if (numBlocks > 0) {
            ComputeGroupSumsImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(approxExp, qOffsets, qOffsetsBias, qSizes, qCount, approxExpSum);
        }
    }

    template <int BLOCK_SIZE>
    __global__ void QuerySoftMaxImpl(const float* target, const float* weights,
                                     const float* approxExp,
                                     const ui32* qids,
                                     float lambdaReg, float beta, ui32 size,
                                     const float* approxExpSum,
                                     const float* sumWeightedTargets,
                                     const ui32* writeMap,
                                     float* functionValue,
                                     float* der,
                                     float* der2) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float tmpScores[BLOCK_SIZE];

        const float targetVal = i < size ?  target[i] : 0;
        const float weight =  (weights && (i < size)) ? weights[i] : 1.0f;
        const float approx =  i < size ? approxExp[i] : 0;
        const ui32 qid = i < size ? __ldg(qids + i) : 0;
        const float approxSum = i < size ? __ldg(approxExpSum + qid) : 0;
        const float sumTargets = i < size ? __ldg(sumWeightedTargets + qid) : 0;

        const float softmax = approx / approxSum;
        const float wt = weight * targetVal;

        if (i < size) {
            const ui32 dstIdx = writeMap != nullptr ? writeMap[i] : i;

            if (der) {
                der[dstIdx] = beta * (((weight > 0 && sumTargets > 0) ? (-sumTargets * softmax) : 0) + wt);
            }
            if (der2) {
                der2[dstIdx] = (weight > 0 && sumTargets > 0) ? beta * sumTargets * (beta * softmax * (1 - softmax) + lambdaReg) : 0;
            }
        }

        if (functionValue) {
            tmpScores[threadIdx.x] = (i < size && weight > 0 && targetVal > 0) ? wt * __logf(softmax) : 0;
            __syncthreads();
        }

        if (functionValue) {
            const float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BLOCK_SIZE);
            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    void ApproximateQuerySoftMax(const float* target, const float* weights,
                                 const float* approxExp,
                                 const ui32* qids,
                                 float lambdaReg, float beta, ui32 size,
                                 const float* approxExpSum,
                                 const float* sumWeightedTargets,
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
        QuerySoftMaxImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(target, weights, approxExp, qids, lambdaReg, beta, size, approxExpSum, sumWeightedTargets, writeMap, functionValue, der, der2);
    }

}
