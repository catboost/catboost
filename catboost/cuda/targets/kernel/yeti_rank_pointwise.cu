#include "yeti_rank_pointwise.cuh"
#include "radix_sort_block.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <cub/block/block_radix_sort.cuh>

namespace NKernel
{

    __global__ void RemoveQueryMeansImpl(const int* qids, int size, const float* queryMeans,
                                         float* approx)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size) {
            approx[tid] -= queryMeans[qids[tid]];
        }
    }

    void RemoveQueryMeans(const int* qids, int size, const float* queryMeans,
                          float* approx, TCudaStream stream) {

        const int blockSize = 256;
        const int numBlocks = (size + blockSize - 1) / blockSize;
        if (numBlocks > 0) {
            RemoveQueryMeansImpl<<< numBlocks, blockSize, 0, stream >>> (qids, size, queryMeans, approx);
        }
    }


    template <ui32 BLOCK_SIZE>
    __device__  void YetiRankGradientSingleGroup(ui32 seed, float decaySpeed,
                                                 ui32 bootstrapIter,
                                                 const float* __restrict__ approx,
                                                 const float* __restrict__ relev,
                                                 const float* __restrict__ querywiseWeights,
                                                 const int* __restrict__ qids, int size,
                                                 float* approxes,
                                                 volatile float* __restrict__ targetDst,
                                                 volatile float*  __restrict__ weightDst) {

        const int N = 4;
        ui32 srcIndex[N]; //contains offset and qid of point

        i16 queryBegin[N];

        __shared__ float relevs[BLOCK_SIZE * 4]; // 4K

        {
            {
                int* queryIds = (int*) approxes;
                const int firstQid = __ldg(qids);

                for (int k = 0; k < N; k++) {
                    int offset = threadIdx.x + k * BLOCK_SIZE;
                    int qid = offset < size ? qids[offset] : qids[size - 1] + 1;
                    qid -= firstQid;
                    queryIds[offset] = qid;

                    srcIndex[k] = offset;
                    srcIndex[k] |= qid << 10; //first 10 bits — point in group, then local qid
                }


                int* queryOffsets = (int*) relevs;
                queryOffsets[threadIdx.x] = size;
                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const int offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    if (!offset || queryIds[offset] != queryIds[offset - 1])
                    {
                        const int qid = queryIds[offset];
                        queryOffsets[qid] = offset;
                    }
                }

                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const int offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    int qid = queryIds[offset];

                    queryBegin[k] = queryOffsets[qid];
                }
                __syncthreads();
            }


            for (int k = 0; k < 4; k++) {
                const int offset = threadIdx.x + k * BLOCK_SIZE;
                relevs[offset] = offset < size ? relev[offset] : 1000.0f;
                relevs[offset] *= offset < size ? querywiseWeights[offset] : 1.0f;
                approxes[offset] = offset < size ? __expf(min(approx[offset], 70.0f)) : 1000.0f;
            }
        }
        __syncthreads();

        __shared__ ui32 indices[BLOCK_SIZE * N];

        for (int t = 0; t < bootstrapIter; t++)
        {
            ui32 key[N];
            ui32 idx[N] = {srcIndex[0], srcIndex[1], srcIndex[2], srcIndex[3]};

            for (int k = 0; k < N; k++)
            {
                float val = (idx[k] & 1023) < size ? approxes[idx[k] & 1023] : -1000.0f;
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


            for (int k = 0; k < N; k++) {
                const int offset = threadIdx.x + k * BLOCK_SIZE;

                const int idx1 =  offset != queryBegin[k] ? (int)indices[offset - 1] : -1;
                const int idx2 =  (int)indices[offset];

                const float relev1 = idx1 != -1 ? relevs[idx1] : 0;
                const float relev2 = relevs[idx2];

                const float approx1 = idx1 != -1 ? approxes[idx1] : 0;
                const float approx2 = approxes[idx2];

                const float magicConst = 0.15f; //to make learning rate more comparable with pair classification
                const float decay = magicConst * powf(decaySpeed, offset - queryBegin[k] - 1);
                const float pairWeight = decay * fabs(relev1 - relev2) /  bootstrapIter;
                const float ll = pairWeight * (relev1 > relev2 ? approx2 : -approx1) / (approx2 + approx1); //

                if (idx1 != -1 && idx1 < size) {
                    weightDst[idx1] += pairWeight;
                    targetDst[idx1] += ll;
                }
                __syncthreads();

                if (idx1 != -1 &&  idx2 < size) {
                    weightDst[idx2] += pairWeight;
                    targetDst[idx2] += -ll;
                }
                __syncthreads();
            }

            __syncthreads();
        }
    };

    template <int BLOCK_SIZE>
    __global__ void YetiRankGradientImpl(int seed, float decaySpeed,
                                         ui32 bootstrapIter,
                                         const ui32* queryOffsets,
                                         volatile int* qidCursor,
                                         ui32 qOffsetsBias, ui32 qCount,
                                         const int* qids,
                                         const float* approx,
                                         const float* relev,
                                         const float* querywiseWeights,
                                         ui32 size,
                                         float* targetDst,
                                         float* weightDst) {

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

                    offset = queryOffsets[taskQid] - qOffsetsBias;
                    nextTaskOffset = min(offset + 4 * BLOCK_SIZE, size);
                    int nextTaskQid = nextTaskOffset < size ? qids[nextTaskOffset] : qCount;
                    int oldQid = atomicCAS(const_cast<int*>(qidCursor), taskQid, nextTaskQid);
                    if (oldQid == taskQid) {
                        nextTaskOffset = nextTaskQid < qCount ? queryOffsets[nextTaskQid] - qOffsetsBias : size;
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
            ui32 taskSeed = 127 * taskQid + 16807 * threadIdx.x + 1;

            #pragma unroll 3
            for (int k = 0; k < 3; ++k) {
                AdvanceSeed32(&taskSeed);
            }

            taskSeed += seed;
            #pragma unroll 3
            for (int k = 0; k < 3; ++k) {
                AdvanceSeed32(&taskSeed);
            }

            YetiRankGradientSingleGroup<BLOCK_SIZE>(taskSeed, decaySpeed,
                                                    bootstrapIter,
                                                    approx + offset,
                                                    relev + offset,
                                                    querywiseWeights + offset,
                                                    qids + offset,
                                                    nextTaskOffset - offset,
                                                    approxes,
                                                    targetDst + offset,
                                                    weightDst + offset);
            __syncthreads();
        }

    }

    void YetiRankGradient(ui64 seed, float decaySpeed,
                          ui32 bootstrapIter,
                          const ui32* queryOffsets,
                          int* qidCursor,
                          ui32 qOffsetsBias,
                          ui32 qCount,
                          const int* qids,
                          const float* approx,
                          const float* relev,
                          const float* querywiseWeights,
                          ui32 size,
                          float* targetDst,
                          float* weightDst,
                          TCudaStream stream) {

        const ui32 maxBlocksPerSm = 4;
        const ui32 smCount = TArchProps::SMCount();
         const int blockSize = 256;

        FillBuffer(targetDst, 0.0f, size, stream);
        FillBuffer(weightDst, 0.0f, size, stream);
        FillBuffer(qidCursor, 0, 1, stream);

        int cudaSeed = ((ui32)seed) + ((ui32)(seed >> 32));

        YetiRankGradientImpl<blockSize><<<maxBlocksPerSm * smCount, blockSize, 0, stream>>>(cudaSeed, decaySpeed,
                bootstrapIter, queryOffsets,
                qidCursor, qOffsetsBias, qCount, qids,
                approx, relev, querywiseWeights, size, targetDst, weightDst);
    }

//
}
