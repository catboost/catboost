#include "yeti_rank_pointwise.cuh"
#include "radix_sort_block.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <library/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/scan.cuh>
#include <contrib/libs/cub/cub/block/block_radix_sort.cuh>

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


    template <int BLOCK_SIZE, int THREADS_PER_QUERY>
    __global__ void MakePairsImpl(const ui32* queryOffsets,
                                  const ui32* matrixOffsets,
                                  ui32 queryCount,
                                  uint2* pairs) {
        const int queriesPerBlock = BLOCK_SIZE / THREADS_PER_QUERY;
        const int localQid = threadIdx.x / THREADS_PER_QUERY;
        const int qid = blockIdx.x * queriesPerBlock + localQid;


        ui32 queryOffset = qid < queryCount ? queryOffsets[qid] : 0;
        ui32 querySize = qid < queryCount ? queryOffsets[qid + 1] - queryOffset : 0;
        ui32 matrixOffset = qid < queryCount ? matrixOffsets[qid] : 0;

        const int x = threadIdx.x & (THREADS_PER_QUERY - 1);


        const ui32 matrixSize = querySize * (querySize - 1) / 2;
        pairs += matrixOffset;
        for (int i = x; i < matrixSize; i += THREADS_PER_QUERY) {
            uint2 pair = GetPair(i);
            pair.x += queryOffset;
            pair.y += queryOffset;
            pairs[i] = pair;
        }
    }

    void MakePairs(const ui32* qOffsets,
                   const ui32* matrixOffset,
                   ui32 qCount,
                   uint2* pairs,
                   TCudaStream stream) {

        const int blockSize = 128;
        const int threadPerQuery = 32;
        const int numBlocks = (qCount * threadPerQuery +  blockSize - 1) / blockSize;
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
                                                const int* __restrict__ qids,
                                                const ui32* __restrict__ matrixOffsets,
                                                int docCount,
                                                float* sharedApproxes,
                                                volatile float* __restrict__ weightsMatrixDst) {

        const int N = 4;
        ui32 srcIndex[N]; //contains offset and qid of point

        i16 queryBegin[N];
        uchar queryId[N];

        __shared__ float relevs[BLOCK_SIZE * 4]; // 4K

        {
            {
                int* blockQueryIds = (int*) sharedApproxes;
                const int firstQid = __ldg(qids);
                matrixOffsets += firstQid;

                for (int k = 0; k < N; k++) {
                    int offset = threadIdx.x + k * BLOCK_SIZE;
                    int qid = offset < docCount ? qids[offset] : qids[docCount - 1] + 1;
                    qid -= firstQid;
                    blockQueryIds[offset] = qid;

                    srcIndex[k] = offset;
                    srcIndex[k] |= qid << 10; //first 10 bits — point in group, then local qid
                }


                int* queryOffsets = (int*) relevs;
                queryOffsets[threadIdx.x] = docCount;
                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const int offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    if (!offset || blockQueryIds[offset] != blockQueryIds[offset - 1]) {
                        const int qid = blockQueryIds[offset];
                        queryOffsets[qid] = offset;
                    }
                }

                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const int offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    int qid = blockQueryIds[offset];

                    queryBegin[k] = queryOffsets[qid];
                    queryId[k] = qid;
                }
                __syncthreads();
            }


            for (int k = 0; k < 4; k++) {
                const int offset = threadIdx.x + k * BLOCK_SIZE;
                relevs[offset] = offset < docCount ? relev[offset] : 1000.0f;
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
                const int offset = threadIdx.x + k * BLOCK_SIZE;
                indices[offset] = idx[k] & 1023;
            }
            __syncthreads();


            #pragma unroll
            for (int k = 0; k < N; k++) {
                const int offset = threadIdx.x + k * BLOCK_SIZE;

                const int idx1 =  offset != queryBegin[k] ? (int)indices[offset - 1] : -1;
                const int idx2 =  (int)indices[offset];

                const float relev1 = idx1 != -1 ? relevs[idx1] : 0;
                const float relev2 = relevs[idx2];

                const float decay =  powf(decaySpeed, offset - queryBegin[k] - 1);

                float pairWeight = 0.15f * decay * fabs(relev1 - relev2) /  bootstrapIter;


                ui32 pairIdx = idx1 < idx2 ? GetPairIndex(idx1 - queryBegin[k], idx2 - queryBegin[k])
                                           : GetPairIndex(idx2 - queryBegin[k], idx1 - queryBegin[k]);

                pairIdx += __ldg(matrixOffsets + queryId[k]);

                //there can't be write conflicts
                if (idx1 != -1 && offset < docCount) {
                    weightsMatrixDst[pairIdx] += pairWeight;
                }
                //without sync we can't be sure that right to weights will be correct
                //otherwise we have gurantee that all pairs participating in write are unique
                __syncthreads();
            }

            __syncthreads();
        }
    };

    template <int BLOCK_SIZE>
    __global__ void PFoundFGradientImpl(int seed, float decaySpeed,
                                        ui32 bootstrapIter,
                                        const ui32* queryOffsets,
                                        volatile int* qidCursor,
                                        ui32 qCount,
                                        const int* qids,
                                        const ui32* matrixOffsets,
                                        const float* expApprox,
                                        const float* relev,
                                        ui32 size,
                                        float* weightMatrixDst) {

        __shared__ float approxes[BLOCK_SIZE * 4]; // 4K

        while (true) {
            int taskQid = 0;
            int* sharedQid = (int*) approxes;
            int offset = 0;
            int nextTaskOffset = 0;

            if (threadIdx.x == 0) {
                taskQid = qidCursor[0];
                while (true) {
                    if (taskQid >= qCount) {
                        break;
                    }

                    offset = queryOffsets[taskQid];
                    nextTaskOffset = min(offset + 4 * BLOCK_SIZE, size);
                    int nextTaskQid = nextTaskOffset < size ? qids[nextTaskOffset] : qCount;
                    int oldQid = atomicCAS(const_cast<int*>(qidCursor), taskQid, nextTaskQid);
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
                                                   approxes,
                                                   weightMatrixDst);
            __syncthreads();
        }

    }

    void PFoundFGradient(ui64 seed,
                         float decaySpeed,
                         ui32 bootstrapIter,
                         const ui32* queryOffsets,
                         int* qidCursor,
                         ui32 qCount,
                         const ui32* qids,
                         const ui32* matrixOffsets,
                         const float* expApprox,
                         const float* relev,
                         ui32 size,
                         float* weightMatrixDst, //should contain zeroes
                         TCudaStream stream) {

        const ui32 maxBlocksPerSm = 4;
        const ui32 smCount = TArchProps::SMCount();
        const int blockSize = 256;

        FillBuffer(qidCursor, 0, 1, stream);

        int cudaSeed = seed + (seed >> 32);

        PFoundFGradientImpl<blockSize> <<<maxBlocksPerSm * smCount, blockSize, 0, stream>>>(cudaSeed,
                decaySpeed,
                bootstrapIter,
                queryOffsets,
                qidCursor,
                qCount,
                (const int*)qids,
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

            const float approx1 = __ldg(expApprox + pair.x);
            const float approx2 = __ldg(expApprox + pair.y);

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

        const int blockSize = 256;
        const int numBlocks = (nzPairCount + blockSize - 1) / blockSize;
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

        const int blockSize = 256;
        const int numBlocks = (nzPairCount + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            SwapWrongOrderPairsImpl<<< numBlocks, blockSize, 0, stream >>> (relevs, nzPairCount, nzPairs);
        }
    }
}
