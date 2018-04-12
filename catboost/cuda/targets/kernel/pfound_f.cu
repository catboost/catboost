#include "yeti_rank_pointwise.cuh"
#include "radix_sort_block.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <contrib/libs/cub/cub/block/block_radix_sort.cuh>

namespace NKernel
{

    __forceinline__ __device__ bool NotZero(float val) {
        return fabs(val) > 1e-20f;
    }

    __forceinline__ __device__ uint GetPairIndex(int i, int j) {
        return ((j * (j - 1)) >> 1) + i;
    }

    __device__  void LocalToGlobalPairs(const ui32* docIds,
                                        uint2 pairs,
                                        ui32 pairCount) {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < pairCount) {
            uint2 pair = pairs[i];
            pair.x = docIds[pair.x];
            pair.y = docIds[pair.y];
            i += blockDim.x * gridDim.x;
        }
    }


    template <ui32 BLOCK_SIZE>
    __device__  void ComputeMatrixSizes(const ui32* querySizes, ui32 qCount,
                                        ui32* matrixSize) {

        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < qCount) {
            const ui32 qSize = querySizes[i];
            matrixSize[i] = qSize * (qSize + 1) / 2;
        }
    }

    template <ui32 BLOCK_SIZE>
    __device__  void PFoundFGradientSingleGroup(ui32 seed,
                                                ui32 bootstrapIter,
                                                const float* __restrict__ expApprox,
                                                const float* __restrict__ relev,
                                                const int* __restrict__ qids, int size,
                                                const ui32* __restrict__ pairQueryOffsets,
                                                float* approxes,
                                                volatile float* __restrict__ weightDst) {

        const int N = 4;
        ui32 srcIndex[N]; //contains offset and qid of point

        i16 queryBegin[N];
        uchar queryId[N];

        __shared__ float relevs[BLOCK_SIZE * 4]; // 4K

        {
            {
                int* queryIds = (int*) approxes;
                const int firstQid = __ldg(qids);
                pairQueryOffsets += firstQid;

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
                    if (!offset || queryIds[offset] != queryIds[offset - 1]) {
                        const int qid = queryIds[offset];
                        queryOffsets[qid] = offset;
                    }
                }

                __syncthreads();

                for (int k = 0; k < N; k++) {
                    const int offset = threadIdx.x + k * BLOCK_SIZE; //point id
                    int qid = queryIds[offset];

                    queryBegin[k] = queryOffsets[qid];
                    queryId[k] = qid;
                }
                __syncthreads();
            }


            for (int k = 0; k < 4; k++) {
                const int offset = threadIdx.x + k * BLOCK_SIZE;
                relevs[offset] = offset < size ? relev[offset] : 1000.0f;
                approxes[offset] = offset < size ? expApprox[offset] : 1000.0f;
            }
        }
        __syncthreads();

        __shared__ ui32 indices[BLOCK_SIZE * N];

        for (int t = 0; t < bootstrapIter; t++) {
            ui32 key[N];
            ui32 idx[N] = {srcIndex[0], srcIndex[1], srcIndex[2], srcIndex[3]};

            for (int k = 0; k < N; k++) {
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

            //now key[k] is idx of document on position (threadIdx.x + k * BLOCK_SIZE - queryOffset) in query key[k] >> 10

            for (int k = 0; k < N; k++)
            {
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

                const float approx1 = idx1 != -1 ? approxes[idx1] : 0;
                const float approx2 = approxes[idx2];

                const float decaySpeed = 0.99f;
                const float magicConst = 10; //to make learning rate more comparable with pair classification
                const float decay = magicConst * powf(decaySpeed, offset - queryBegin[k] - 1);
                const float pairWeight = decay * fabs(relev1 - relev2) /  bootstrapIter;
                ui32 idx = idx1 < idx2 ? GetPairIndex(idx1 - queryBegin[k], idx2 - queryBegin[k])
                                       : GetPairIndex(idx2 - queryBegin[k], idx1 - queryBegin[k]);

                idx += __ldg(pairQueryOffsets + queryId[k]);

                if (idx1 != -1 && offset < size) {
                    weightDst[idx] += pairWeight;
                }
                //without sync we can't be sure that right to weights will be correct
                //otherwise we have gurantee that all pairs participating in write are unique
                __syncthreads();
            }

            __syncthreads();
        }
    };

    template<int BLOCK_SIZE>
    __global__ void PFoundFGradientImpl(int seed,
                                        ui32 bootstrapIter,
                                        const ui32* queryOffsets,
                                        volatile int* qidCursor,
                                        ui32 qOffsetsBias,
                                        ui32 qCount,
                                        const int* qids,
                                        const float* approx,
                                        const float* relev,
                                        ui32 size,
                                        float* weightDst) {

        __shared__ float approxes[BLOCK_SIZE * 4]; // 4K

        while (true)
        {
            int taskQid = 0;
            int* sharedQid = (int*) approxes;
            int offset = 0;
            int nextTaskOffset = 0;

            if (threadIdx.x == 0) {
                taskQid = qidCursor[0];
                while (true)
                {
                    if (taskQid >= qCount) {
                        break;
                    }

                    offset = queryOffsets[taskQid] - qOffsetsBias;
                    nextTaskOffset = min(offset + 4 * BLOCK_SIZE, size);
                    int nextTaskQid = nextTaskOffset < size ? qids[nextTaskOffset] : qCount;
                    int oldQid = atomicCAS(const_cast<int*>(qidCursor), taskQid, nextTaskQid);
                    if (oldQid == taskQid) {
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
            ui32 taskSeed = 127 * taskQid + 16807 * threadIdx.x;
            AdvanceSeed32(&taskSeed);
            taskSeed += seed;
            AdvanceSeed32(&taskSeed);

            PFoundFGradientSingleGroup<BLOCK_SIZE>(taskSeed,
                                                   bootstrapIter,
                                                   approx + offset,
                                                   relev + offset,
                                                   qids + offset,
                                                   nextTaskOffset - offset,
                                                   approxes,
                                                   targetDst + offset,
                                                   weightDst + offset);
            __syncthreads();
        }

    }

    void PFoundFGradient(ui64 seed,
                         ui32 bootstrapIter,
                         const ui32* queryOffsets,
                         int* qidCursor,
                         ui32 qOffsetsBias,
                         ui32 qCount,
                         const int* qids,
                         const float* approx,
                         const float* relev,
                         ui32 size,
                         float* targetDst,
                         float* weightDst,
                         TCudaStream stream)
    {

        const ui32 maxBlocksPerSm = 4;
        const ui32 smCount = TArchProps::SMCount();
        const int blockSize = 256;

        FillBuffer(targetDst, 0.0f, size, stream);
        FillBuffer(weightDst, 0.0f, size, stream);
        FillBuffer(qidCursor, 0, 1, stream);

        int cudaSeed = seed + (seed >> 32);

        PFoundFGradientImpl<blockSize> <<<maxBlocksPerSm * smCount, blockSize, 0, stream>>>(cudaSeed, bootstrapIter, queryOffsets,
                qidCursor, qOffsetsBias, qCount, qids,
                approx, relev, size, targetDst, weightDst);
    }



    template <int BLOCK_SIZE>
    __global__ void GatherDerImpl(const float* relev,
                                  const float* expApprox,
                                  const uint2* pairs,
                                  const float* pairWeights,
                                  ui32 pairCount,
                                  float* der)  {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= pairCount) {
            return;
        }

        uint2 pair;
        pair = __ldg(pairs + i);

        const float w = pairWeights && (i < pairCount) ? pairWeights[i] : 0.0f;

        const float approx1 = __ldg(expApprox + pair.x);
        const float approx2 = __ldg(expApprox + pair.y);

        const float relev1 = __ldg(relev + pair.x);
        const float relev2 = __ldg(relev + pair.y)

        const float ll = w * (relev1 > relev2 ? approx2 : -approx1) / (approx2 + approx1); //


        atomicAdd(der + pair.x, ll);
        atomicAdd(der + pair.y, -ll);
    }

//
}
