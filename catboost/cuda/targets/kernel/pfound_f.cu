#include "yeti_rank_pointwise.cuh"
#include "radix_sort_block.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/scan.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <util/generic/cast.h>

namespace NKernel
{

    __global__ void ComputeMatrixSizesImpl(const ui32* queryOffsets,
                                            ui32 qCount,
                                            ui32* matrixSize) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        const ui32 qSize = i < qCount ? queryOffsets[i + 1] - queryOffsets[i] : 0;
        if (i <= qCount) {
            matrixSize[i] = qSize * (qSize - 1) / 2;
        }
    }


    void ComputeMatrixSizes(const ui32* queryOffsets,
                            ui32 qCount,
                            ui32* matrixSize,
                            TCudaStream stream) {
        const ui32 blockSize = 256;
        //matrix count is qCount + 1 (for last index)
        const ui32 numBlocks = (qCount + blockSize) / blockSize;
        ComputeMatrixSizesImpl<<<numBlocks, blockSize, 0, stream>>>(queryOffsets, qCount, matrixSize);
    }


    template <ui32 BLOCK_SIZE, ui32 THREADS_PER_QUERY>
    __global__ void MakePairsImpl(const ui32* queryOffsets,
                                  const ui64* matrixOffsets,
                                  ui32 queryCount,
                                  uint2* pairs) {
        const ui32 queriesPerBlock = BLOCK_SIZE / THREADS_PER_QUERY;
        const ui32 localQid = threadIdx.x / THREADS_PER_QUERY;
        const ui32 qid = blockIdx.x * queriesPerBlock + localQid;


        ui32 queryOffset = qid < queryCount ? queryOffsets[qid] : 0;
        ui32 querySize = qid < queryCount ? queryOffsets[qid + 1] - queryOffset : 0; // queryCount = QOffsets.Size() - 1
        ui64 matrixOffset = qid < queryCount ? matrixOffsets[qid] : 0;

        const ui32 x = threadIdx.x & (THREADS_PER_QUERY - 1);


        const ui32 matrixSize = querySize * (querySize - 1) / 2;
        pairs += matrixOffset;
        for (ui32 i = x; i < matrixSize; i += THREADS_PER_QUERY) {
            uint2 pair = GetPair(i);
            pair.x += queryOffset;
            pair.y += queryOffset;
            pairs[i] = pair;
        }
    }

    void MakePairs(const ui32* qOffsets,
                   const ui64* matrixOffset,
                   ui32 qCount,
                   uint2* pairs,
                   TCudaStream stream) {

        const ui32 blockSize = 128;
        const ui32 threadPerQuery = 32;
        const ui32 numBlocks = SafeIntegerCast<ui32>((((ui64)qCount + 1) * threadPerQuery +  blockSize - 1) / blockSize);
        if (numBlocks > 0) {
            MakePairsImpl<blockSize, threadPerQuery> <<< numBlocks, blockSize, 0, stream >>> (qOffsets, matrixOffset, qCount, pairs);
        }
    }


    template <ui32 BLOCK_SIZE>
    __device__  void PFoundFGradientSingleGroup(ui32 seed,
                                                ui32 bootstrapIter,
                                                const float decaySpeed,
                                                const float* __restrict__ expApprox,
                                                const float* __restrict__ relev,
                                                const ui32* __restrict__ qids,
                                                const ui64* __restrict__ matrixOffsets,
                                                ui32 docCount,
                                                float* sharedApproxes,
                                                volatile float* __restrict__ weightsMatrixDst) {

        const int N = 4;
        ui32 srcIndex[N]; //contains offset and qid of point

        i16 queryBegin[N];
        uchar queryId[N];

        __shared__ float sharedRelev[BLOCK_SIZE * 4]; // 4K

        {
            {
                ui32* blockQueryIds = (ui32*) sharedApproxes;
                const ui32 firstQid = __ldg(qids);
                matrixOffsets += firstQid;

                for (int k = 0; k < N; k++) {
                    ui32 offset = threadIdx.x + k * BLOCK_SIZE;
                    ui32 qid = offset < docCount ? qids[offset] : qids[docCount - 1] + 1;
                    qid -= firstQid;
                    blockQueryIds[offset] = qid;

                    srcIndex[k] = offset;
                    srcIndex[k] |= qid << 10; //first 10 bits — point in group, then local qid
                }


                ui32* queryOffsets = (ui32*) sharedRelev;
                for (int k = 0; k < N; k++) {
                    ui32 offset = threadIdx.x + k * BLOCK_SIZE;
                    queryOffsets[offset] = docCount; // init [0, 4 * BLOCK_SIZE)
                }
                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const ui32 offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    if (!offset || blockQueryIds[offset] != blockQueryIds[offset - 1]) {
                        const ui32 qid = blockQueryIds[offset];
                        queryOffsets[qid] = offset;
                    }
                }

                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const ui32 offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    ui32 qid = blockQueryIds[offset];

                    queryBegin[k] = queryOffsets[qid];
                    queryId[k] = qid;
                }
                __syncthreads();
            }


            for (int k = 0; k < 4; k++) {
                const ui32 offset = threadIdx.x + k * BLOCK_SIZE;
                sharedRelev[offset] = offset < docCount ? relev[offset] : 1000.0f;
                sharedApproxes[offset] = offset < docCount ? expApprox[offset] : 1000.0f;
            }
        }
        __syncthreads();

        __shared__ ui32 indices[BLOCK_SIZE * N];

        for (int t = 0; t < bootstrapIter; t++) {
            ui32 key[N];
            ui32 idx[N] = {srcIndex[0], srcIndex[1], srcIndex[2], srcIndex[3]};

            for (int k = 0; k < N; k++) {
                float val = (idx[k] & 1023) < docCount ? sharedApproxes[idx[k] & 1023] : -1000.0f;
                const float uni = NextUniformFloat32(&seed);
                val *= uni / (1.000001f - uni);
                key[k] = __float_as_int(val);
                key[k] ^= (key[k] & 0x80000000) ? 0xffffffff : 0x80000000;
            }

            {
                RadixSortSingleBlock4<BLOCK_SIZE, false, 0, 32>((uint4&)key, (uint4&)idx, indices);
                RadixSortSingleBlock4<BLOCK_SIZE, true, 10, 10>((uint4&)idx, indices);
            }

            //now key[k] is idx of document on position (threadIdx.x + k * BlockSize - queryOffset) in query key[k] >> 10

            for (int k = 0; k < N; k++) {
                const ui32 offset = threadIdx.x + k * BLOCK_SIZE;
                indices[offset] = idx[k] & 1023;
            }
            __syncthreads();


            #pragma unroll
            for (int k = 0; k < N; k++) {
                const ui32 offset = threadIdx.x + k * BLOCK_SIZE;

                const ui32 idx1 =  offset != queryBegin[k] ? indices[offset - 1] : -1u;
                const ui32 idx2 =  indices[offset];

                const float relev1 = idx1 != -1u ? sharedRelev[idx1] : 0;
                const float relev2 = sharedRelev[idx2];

                const float decay =  powf(decaySpeed, offset - queryBegin[k] - 1);

                float pairWeight = 0.15f * decay * fabs(relev1 - relev2) /  bootstrapIter;


                ui64 pairIdx = idx1 < idx2 ? GetPairIndex(idx1 - queryBegin[k], idx2 - queryBegin[k])
                                           : GetPairIndex(idx2 - queryBegin[k], idx1 - queryBegin[k]);

                pairIdx += __ldg(matrixOffsets + queryId[k]);

                //there can't be write conflicts
                if (idx1 != -1u && offset < docCount) {
                    weightsMatrixDst[pairIdx] += pairWeight;
                }
                //without sync we can't be sure that right to weights will be correct
                //otherwise we have gurantee that all pairs participating in write are unique
                __syncthreads();
            }

            __syncthreads();
        }
    };

    template <ui32 BLOCK_SIZE>
    __global__ void PFoundFGradientImpl(int seed, float decaySpeed,
                                        ui32 bootstrapIter,
                                        const ui32* queryOffsets,
                                        volatile ui32* qidCursor,
                                        ui32 qCount,
                                        const ui32* qids,
                                        const ui64* matrixOffsets,
                                        const float* expApprox,
                                        const float* relev,
                                        ui32 size,
                                        float* weightMatrixDst) {

        __shared__ float sharedApproxes[BLOCK_SIZE * 4]; // 4K

        while (true) {
            ui32 taskQid = 0;
            ui32* sharedQid = (ui32*) sharedApproxes;
            ui32 offset = 0;
            ui32 nextTaskOffset = 0;

            if (threadIdx.x == 0) {
                taskQid = qidCursor[0];
                while (true) {
                    if (taskQid >= qCount) {
                        break;
                    }

                    offset = queryOffsets[taskQid];
                    nextTaskOffset = min(offset + 4 * BLOCK_SIZE, size);
                    ui32 nextTaskQid = nextTaskOffset < size ? qids[nextTaskOffset] : qCount;
                    ui32 oldQid = atomicCAS(const_cast<ui32*>(qidCursor), taskQid, nextTaskQid);
                    if (oldQid == taskQid) {
                        nextTaskOffset = queryOffsets[nextTaskQid];
                        break;
                    } else {
                        taskQid = oldQid;
                    }
                }
            }

            if (threadIdx.x == 0) {
                sharedQid[0] = taskQid;
                sharedQid[1] = offset;
                sharedQid[2] = nextTaskOffset;
            }
            __syncthreads();

            taskQid = sharedQid[0];
            offset = sharedQid[1];
            nextTaskOffset = sharedQid[2];
            __syncthreads();

            if (taskQid >= qCount) {
                return;
            }

            //statisticians will complain :) but we don't need high-quality random generators
            ui32 taskSeed = 127 * taskQid + 16807 * threadIdx.x + seed * (1 + taskQid);
            AdvanceSeed32(&taskSeed);
            taskSeed += seed;
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                AdvanceSeed32(&taskSeed);
            }

            PFoundFGradientSingleGroup<BLOCK_SIZE>(taskSeed,
                                                   bootstrapIter,
                                                   decaySpeed,
                                                   expApprox + offset,
                                                   relev + offset,
                                                   qids + offset,
                                                   matrixOffsets,
                                                   nextTaskOffset - offset,
                                                   sharedApproxes,
                                                   weightMatrixDst);
            __syncthreads();
        }

    }

    void PFoundFGradient(ui64 seed,
                         float decaySpeed,
                         ui32 bootstrapIter,
                         const ui32* queryOffsets,
                         ui32* qidCursor,
                         ui32 qCount,
                         const ui32* qids,
                         const ui64* matrixOffsets,
                         const float* expApprox,
                         const float* relev,
                         ui32 size,
                         float* weightMatrixDst, //should contain zeroes
                         TCudaStream stream) {

        const ui32 maxBlocksPerSm = 4;
        const ui32 smCount = TArchProps::SMCount();
        const ui32 blockSize = 256;

        FillBuffer(qidCursor, 0u, 1, stream);

        int cudaSeed = seed + (seed >> 32);

        PFoundFGradientImpl<blockSize> <<<maxBlocksPerSm * smCount, blockSize, 0, stream>>>(cudaSeed,
                decaySpeed,
                bootstrapIter,
                queryOffsets,
                qidCursor,
                qCount,
                qids,
                matrixOffsets,
                expApprox,
                relev,
                size,
                weightMatrixDst);
    }



    __global__ void MakeFinalTargetImpl(const ui32* docIds,
                                         const float* expApprox,
                                         const float* querywiseWeights,
                                         const float* relevs,
                                         float* nzPairWeights,
                                         ui32 nzPairCount,
                                         float* resultDers,
                                         uint2* nzPairs) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        while (i < nzPairCount) {

            uint2 pair = nzPairs[i];

            const float approx1 = __ldg(expApprox + pair.x) + 1e-20; // avoid ll == nan, if approx1 == approx2 == 0
            const float approx2 = __ldg(expApprox + pair.y) + 1e-20;

            const float relev1 = __ldg(relevs + pair.x);
            const float relev2 = __ldg(relevs + pair.y);

            const float queryWeight = __ldg(querywiseWeights + pair.x);
            const float w = nzPairWeights[i] * queryWeight;


            const float ll = w * (relev1 > relev2 ? approx2 : -approx1) / (approx2 + approx1);

            atomicAdd(resultDers + pair.x, ll);
            atomicAdd(resultDers + pair.y, -ll);

            pair.x = docIds[pair.x];
            pair.y = docIds[pair.y];

            nzPairs[i] = pair;
            if (queryWeight != 1.0f) {
                nzPairWeights[i] = w;
            }

            i += blockDim.x * gridDim.x;
        }
    }

    void MakeFinalTarget(const ui32* docIds,
                         const float* expApprox,
                         const float* querywiseWeights,
                         const float* relevs,
                         float* nzPairWeights,
                         ui32 nzPairCount,
                         float* resultDers,
                         uint2* nzPairs,
                         TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = (nzPairCount + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            MakeFinalTargetImpl<<< numBlocks, blockSize, 0, stream >>> (docIds, expApprox, querywiseWeights, relevs, nzPairWeights, nzPairCount, resultDers, nzPairs);
        }
    }



    __global__ void SwapWrongOrderPairsImpl(const float* relevs,
                                            ui32 nzPairCount,
                                            uint2* nzPairs) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        while (i < nzPairCount) {

            uint2 pair = nzPairs[i];

            const float relev1 = __ldg(relevs + pair.x);
            const float relev2 = __ldg(relevs + pair.y);
            if (relev1 < relev2) {
                ui32 tmp = pair.x;
                pair.x = pair.y;
                pair.y = tmp;
                nzPairs[i] = pair;
            }
            i += blockDim.x * gridDim.x;
        }
    }

    void SwapWrongOrderPairs(const float* relevs,
                             ui32 nzPairCount,
                             uint2* nzPairs,
                             TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = (nzPairCount + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            SwapWrongOrderPairsImpl<<< numBlocks, blockSize, 0, stream >>> (relevs, nzPairCount, nzPairs);
        }
    }
}
