#include "query_cross_entropy.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

#include <util/generic/cast.h>

#include <cub/util_ptx.cuh>


#include <cooperative_groups.h>

#include <cassert>
#include <cstdio>


using namespace cooperative_groups;

namespace NKernel {



    //TODO(noxoomo): multiple docs per thread to reduce sync overhead
    template <int BlockSize, bool IsSingleClassBlock>
    __forceinline__ __device__ void QueryCrossEntropySingleBlockImpl(const float alpha,
                                                                     const float* targets,
                                                                     const float* weights,
                                                                     const float* values,
                                                                     const ui32 offset,
                                                                     const ui32 size,
                                                                     const int* qids,
                                                                     const ui32* qOffsets,
                                                                     const float* approxScale,
                                                                     const float defaultScale,
                                                                     const ui32 approxScaleSize,
                                                                     const bool* isSingleClassFlags,
                                                                     const ui32* trueClassCount,
                                                                     float* functionValue,
                                                                     float* ders,
                                                                     float* ders2llp,
                                                                     float* ders2llmax,
                                                                     float* groupDers2) {
        __shared__ float sharedDer[BlockSize];
        __shared__ float sharedDer2[BlockSize];

        isSingleClassFlags += offset;
        qids += offset;


        if (ders) {
            ders += offset;
        }

        if (ders2llp) {
            ders2llp += offset;
        }

        if (ders2llmax) {
            ders2llmax += offset;
        }

        if (trueClassCount) {
            trueClassCount += offset;
        }

        const float MAX_SHIFT = 20;
        const ui32 tid = threadIdx.x;

        const ui32 loadIdx = tid < size ?  offset + tid : 0;
        const bool isSingleClass = tid < size ? isSingleClassFlags[tid] : true;
        const ui32 trueCount = tid < size && trueClassCount ? trueClassCount[tid] : 0;
        const ui32 tidQid = tid < size ? Ldg(qids + tid) : -1;
        const ui32 queryOffset = tid < size ? Ldg(qOffsets + tidQid) : 0;

        const ui32 querySize = tid < size ? Ldg(qOffsets + tidQid + 1) - queryOffset : 0;
        const ui32 localIdx = tid < size ? offset + tid - queryOffset : 0;

        const float clazz = tid < size ? Ldg(targets + loadIdx) : 0;
        const float scale = querySize < approxScaleSize ? approxScale[querySize * approxScaleSize + trueCount]
            : defaultScale;
        const float cursor = tid < size ? Ldg(values + loadIdx) : 0;
        const float w = tid < size ? Ldg(weights + loadIdx) : 0;


        float left = -MAX_SHIFT;
        float right = MAX_SHIFT;

        float bestShift = (left + right) / 2;

        int reduceSize = 0;

        if (!IsSingleClassBlock) {
            {
                sharedDer[tid] = querySize; // use sharedDer to calc max query size in this block
                __syncthreads();
                for (int s = BlockSize >> 1; s > 0; s >>= 1) {
                    if (tid < s) {
                        sharedDer[tid] = max(sharedDer[tid], sharedDer[tid + s]);
                    }
                    __syncthreads();
                }
                reduceSize = (1 << int(ceil(log2(sharedDer[0])) - 1)); // sharedDer[0] = max query size in this block
                __syncthreads();
            }

            float midDer = 0;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {

                const float tmp = __expf(cursor * scale + bestShift);
                const float p = ClipProb((isfinite(1.0f + tmp) ? (tmp / (1.0f + tmp)) : 1.0f));

                sharedDer[tid] = w * (clazz - p);

                __syncthreads();

                for (int s = reduceSize; s > 0; s >>= 1) {
                    if ((localIdx < s) && ((localIdx + s) < querySize)) {
                        sharedDer[tid] += sharedDer[tid + s];
                    }
                    __syncthreads();
                }

                midDer = sharedDer[tid - localIdx]; // sum of sharedDer in this thread's query

                if (midDer > 0) {
                    left = bestShift;
                } else {
                    right = bestShift;
                }

                bestShift = (left + right) / 2;
                __syncthreads();
            }

            #pragma unroll
            for (int i = 0; i < 5; ++i) {
                const float tmp = __expf(cursor * scale + bestShift);
                const float p = ClipProb(isfinite(1.0f + tmp) ? (tmp / (1.0f + tmp)) : 1.0f);

                __syncthreads();

                sharedDer[tid] = w * (clazz - p);
                sharedDer2[tid] = w * (1.0f - p) * p;

                __syncthreads();

                for (int s = reduceSize; s > 0; s >>= 1) {
                    if ((localIdx < s) && ((localIdx + s) < querySize)) {
                        sharedDer[tid] += sharedDer[tid + s];
                        sharedDer2[tid] += sharedDer2[tid + s];
                    }
                    __syncthreads();
                }

                float currentDer = sharedDer[tid - localIdx];

                if (currentDer > 0) {
                    left = bestShift;
                } else {
                    right = bestShift;
                }

                bestShift += currentDer / (sharedDer2[tid - localIdx] + 1e-9f);

                if (bestShift > right) {
                    bestShift = 0.1f * left + 0.9f * right;
                }
                if (bestShift < left) {
                    bestShift = 0.9f * left + 0.1f * right;
                }

                __syncthreads();
            }
        }

        const float shiftedApprox = cursor * scale + bestShift;
        const float expVal = __expf(cursor);
        const float expShiftedVal = __expf(shiftedApprox);

        if (functionValue) {
            const float logExpValPlusOne = isfinite(expVal) ? __logf(1.0f + expVal) : cursor;
            const float llp = (tid < size) ? (clazz * cursor - logExpValPlusOne) : 0;

            const float logExpValPlusOneShifted = isfinite(expShiftedVal) ? __logf(1.0f + expShiftedVal) : shiftedApprox;
            const float llmax = (tid < size) ? (clazz * shiftedApprox - logExpValPlusOneShifted) : 0;

            const float docScore = (1.0f - alpha) * llp + (isSingleClass ? 0 : alpha * llmax);

            sharedDer[tid] = w * docScore;
            __syncthreads();

            float blockScore = FastInBlockReduce(tid, sharedDer, BlockSize);

            if (tid == 0) {
                atomicAdd(functionValue, blockScore);
            }
        }

        const float prob = ClipProb(isfinite(expVal + 1.0f) ? expVal / (1.0f + expVal) : 1.0f);
        const float shiftedProb = ClipProb(isfinite(expShiftedVal + 1.0f) ? expShiftedVal / (1.0f + expShiftedVal) : 1.0f);

        if (ders && (tid < size)) {
            const float derllp = clazz - prob;
            const float derllmax = (isSingleClass ? 0 : clazz - shiftedProb) * scale;
            ders[tid] = w * ((1.0f - alpha) * derllp + alpha * derllmax);
        }

        if (ders2llp && (tid < size)) {
            ders2llp[tid] = w * (1.0f - alpha) * prob * (1.0f - prob);
        }

        float der2llmax = (isSingleClass ? 0 : w * alpha * shiftedProb * (1.0f - shiftedProb)) * scale * scale;

        if (ders2llmax && (tid < size)) {
            ders2llmax[tid] = der2llmax;
        }

        if (groupDers2) {

            float groupDer2 = 0;

            if (!IsSingleClassBlock) {
                __syncthreads();
                sharedDer2[tid] = der2llmax;
                __syncthreads();

                for (int s = reduceSize; s > 0; s >>= 1) {
                    if ((localIdx < s) && ((localIdx + s) < querySize)) {
                        sharedDer2[tid] += sharedDer2[tid + s];
                    }
                    __syncthreads();
                }
                if (localIdx == 0 && tid < size) {
                    groupDer2 = sharedDer2[tid - localIdx];
                }
            }

            if (localIdx == 0 && tid < size) {
                groupDers2[tidQid] = groupDer2;
            }
        }
    }


    template <int BlockSize>
    __global__ void  QueryCrossEntropyImpl(volatile int* qidCursor,
                                           const ui32 qCount,
                                           const float alpha,
                                           const float* targets,
                                           const float* weights,
                                           const float* values,
                                           const int* qids,
                                           const bool* isSingleClassQueries,
                                           const ui32* qOffsets,
                                           const float* approxScale,
                                           const float defaultScale,
                                           const ui32 approxScaleSize,
                                           const ui32* trueClassCount,
                                           const ui32 size,
                                           float* functionValue,
                                           float* ders,
                                           float* ders2llp,
                                           float* ders2llmax,
                                           float* groupDers2) {

        while (true) {

            ui32 taskQid = 0;
            ui32 offset = 0;
            ui32 nextTaskOffset = 0;
            {
                __shared__ ui32 sharedTaskQid;
                __shared__ ui32 sharedTaskOffset;
                __shared__ ui32 sharedNextTaskOffset;

                if (threadIdx.x == 0) {
                    taskQid = qidCursor[0];

                    while (true) {
                        if (taskQid >= qCount) {
                            break;
                        }

                        offset = qOffsets[taskQid];
                        nextTaskOffset = min(offset + BlockSize, size);
                        ui32 nextTaskQid = nextTaskOffset < size ? qids[nextTaskOffset] : qCount;

                        ui32 oldQid = atomicCAS(/*const_cast<int*>*/(ui32*)(qidCursor), taskQid, nextTaskQid);
                        if (oldQid == taskQid) {
                            nextTaskOffset = qOffsets[nextTaskQid];
                            break;
                        } else {
                            taskQid = oldQid;
                        }
                    }
                }

                if (threadIdx.x == 0) {
                    sharedTaskQid = taskQid;
                    sharedTaskOffset = offset;
                    sharedNextTaskOffset = nextTaskOffset;
                }
                __syncthreads();

                taskQid = sharedTaskQid;
                offset = sharedTaskOffset;
                nextTaskOffset = sharedNextTaskOffset;
                __syncthreads();
            }

            if (taskQid >= qCount) {
                return;
            }

            const ui32 blockSize = nextTaskOffset - offset;
            //we assume, that docs are sorted by isSingleClass mask
            //otherwise will be slower for adv-pools
            //first part - queries with pairs
            //second part - all other queries
            bool isSingleClassBlock = threadIdx.x < blockSize ? Ldg(isSingleClassQueries + offset + threadIdx.x) : true;
            {
                __shared__ float sharedFlags[BlockSize];
                sharedFlags[threadIdx.x] = isSingleClassBlock ? 1.0f : 0.0f;
                using TOp = TCudaMultiply<float>;
                float tmp = FastInBlockReduce<float, TOp>(threadIdx.x, sharedFlags, BlockSize);
                if (threadIdx.x == 0) {
                    sharedFlags[0] = tmp;
                }
                __syncthreads();
                isSingleClassBlock = sharedFlags[0] > 0;
                __syncthreads();
            }

            #define COMPUTE_SINGLE_GROUP(IsSingleClassQuery) \
            QueryCrossEntropySingleBlockImpl<BlockSize, IsSingleClassQuery>(alpha, \
                                             targets, weights, values,\
                                             offset, blockSize,\
                                             qids, qOffsets,\
                                             approxScale, defaultScale, approxScaleSize,\
                                             isSingleClassQueries,\
                                             trueClassCount,\
                                             functionValue,\
                                             ders,\
                                             ders2llp,\
                                             ders2llmax,\
                                             groupDers2);


            if (isSingleClassBlock) {
                COMPUTE_SINGLE_GROUP(true);
            } else {
                COMPUTE_SINGLE_GROUP(false);
            }
            __syncthreads();
        }
    }

    void QueryCrossEntropy(int* qidCursor, const ui32 qCount,
                           const float alpha,
                           const float* targets,
                           const float* weights,
                           const float* values,
                           const ui32* qids,
                           const bool* isSingleClassQueries,
                           const ui32* qOffsets,
                           const float* approxScale,
                           const float defaultScale,
                           const ui32 approxScaleSize,
                           const ui32* trueClassCount,
                           const  ui32 docCount,
                           float* functionValue,
                           float* ders,
                           float* ders2llp,
                           float* ders2llmax,
                           float* groupDers2,
                           TCudaStream stream)
    {

        const ui32 maxBlocksPerSm = 4;
        const ui32 smCount = TArchProps::SMCount();
        const ui32 gridSize = maxBlocksPerSm * smCount;
        const int blockSize = docCount / gridSize + 1;

        FillBuffer(qidCursor, 0, 1, stream);
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        #define RUN_KERNEL(N) QueryCrossEntropyImpl<N> <<<gridSize, N, 0, stream>>>( \
                qidCursor, qCount, alpha, \
                targets, weights, values, \
                (int*)qids, isSingleClassQueries, qOffsets, \
                approxScale, defaultScale, approxScaleSize, \
                trueClassCount, docCount, \
                functionValue, \
                ders, ders2llp, ders2llmax, groupDers2);

        if (blockSize < 256) {
            RUN_KERNEL(256)
        } else if (blockSize < 512) {
            RUN_KERNEL(512)
        } else {
            RUN_KERNEL(1024)
        }
        #undef RUN_KERNEL
    }



    __global__ void ComputeQueryLogitMatrixSizesImpl(const ui32* queryOffsets,
                                                     const bool* isSingleClassQuery,
                                                     ui32 qCount,
                                                     ui32* matrixSizes) {

        //matrix count is qCount + 1 (for last index)
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        const bool isSingleClassFlag = i < qCount ? Ldg(isSingleClassQuery + queryOffsets[i]) : true;
        const ui32 qSize = (i < qCount && !isSingleClassFlag) ? queryOffsets[i + 1] - queryOffsets[i] : 0;
        if (i <= qCount) {
            matrixSizes[i] = qSize * (qSize - 1) / 2;
        }
    }


    void ComputeQueryLogitMatrixSizes(const ui32* queryOffsets,
                                      const bool* isSingleClassQuery,
                                      ui32 qCount,
                                      ui32* matrixSize,
                                      TCudaStream stream) {
        const ui32 blockSize = 256;
        //matrix count is qCount + 1 (for last index)
        const ui32 numBlocks = (qCount + blockSize) / blockSize;
        ComputeQueryLogitMatrixSizesImpl<<<numBlocks, blockSize, 0, stream>>>(queryOffsets, isSingleClassQuery, qCount, matrixSize);
    }


    template <ui32 BlockSize, ui32 ThreadsPerQuery>
    __global__ void MakePairsQueryLogitImpl(const ui32* queryOffsets,
                                            const ui64* matrixOffsets,
                                            const bool* isSingleClassQuery,
                                            ui32 queryCount,
                                            uint2* pairs) {

        const ui32 queriesPerBlock = BlockSize / ThreadsPerQuery;
        const ui32 localQid = threadIdx.x / ThreadsPerQuery;
        const ui32 qid = blockIdx.x * queriesPerBlock + localQid;


        ui32 queryOffset = qid < queryCount ? queryOffsets[qid] : 0;
        const bool singleClassFlag = qid < queryCount ? isSingleClassQuery[queryOffset] : true;

        ui32 querySize = (qid < queryCount && !singleClassFlag) ? queryOffsets[qid + 1] - queryOffset : 0; // queryCount == QueryOffsets.Size() - 1
        ui64 matrixOffset = qid < queryCount ? matrixOffsets[qid] : 0;

        const ui32 x = threadIdx.x & (ThreadsPerQuery - 1);


        const ui32 matrixSize = querySize * (querySize - 1) / 2;
        pairs += matrixOffset;
        for (ui32 i = x; i < matrixSize; i += ThreadsPerQuery) {
            uint2 pair = GetPair(i);
            pair.x += queryOffset;
            pair.y += queryOffset;
            pairs[i] = pair;
        }
    }

    void MakeQueryLogitPairs(const ui32* qOffsets,
                             const ui64* matrixOffset,
                             const bool* isSingleFlags,
                             double meanQuerySize,
                             ui32 qCount,
                             uint2* pairs,
                             TCudaStream stream) {

        const ui32 blockSize = 128;

        #define MAKE_PAIRS(threadsPerQuery) \
        const ui32 numBlocks = SafeIntegerCast<ui32>((((ui64)qCount + 1) * threadsPerQuery +  blockSize - 1) / blockSize); \
        if (numBlocks > 0) { \
            MakePairsQueryLogitImpl<blockSize, threadsPerQuery> <<< numBlocks, blockSize, 0, stream >>> (qOffsets, matrixOffset,  isSingleFlags, qCount, pairs); \
        }

        if (meanQuerySize < 4) {
            MAKE_PAIRS(4)
        } else if (meanQuerySize < 8) {
            MAKE_PAIRS(8)
        } else if (meanQuerySize < 16) {
            MAKE_PAIRS(16)
        } else {
            MAKE_PAIRS(32)
        }
        #undef MAKE_PAIRS
    }



    template <int BlockSize, int ThreadsPerQuery>
    __global__ void MakeIsSingleClassFlagsImpl(const int* queryOffsets, int queryCount,
                                               const ui32* loadIndices, const float* targets,
                                               bool* isSingleClassQuery,
                                               ui32* trueClassCount) {

        int bias = queryCount ? Ldg(queryOffsets) : 0;

        auto workingTile = tiled_partition<ThreadsPerQuery>(this_thread_block());

        const int queriesPerBlock = BlockSize / ThreadsPerQuery;
        const int localQid = threadIdx.x / ThreadsPerQuery;
        const int qid = blockIdx.x * queriesPerBlock + localQid;

        __shared__ ui32 resultsIsSingle[BlockSize];
        __shared__ ui32 resultsCount[BlockSize];

        const int queryOffset = (qid < queryCount) ? (queryOffsets[qid] - bias) : 0;
        const int querySize = (qid < queryCount) ? (queryOffsets[qid + 1] - bias - queryOffset) : 0; // queryCount == QueryOffsets.Size() - 1

        const ui32 firstIdx = qid < queryCount ? loadIndices[queryOffset] : 0;
        float firstTarget = Ldg(targets + firstIdx);
        int isSingleClass = 1;
        ui32 trueCount = 0;

        for (int i = workingTile.thread_rank(); i < querySize; i += ThreadsPerQuery) {
            const ui32 loadIdx = loadIndices[queryOffset + i];
            float docTarget = Ldg(targets + loadIdx);
            if (abs(firstTarget - docTarget) > 1e-5f) {
                isSingleClass = 0;
            }
            trueCount += docTarget > 0.5;
        }

        using TOp = TCudaMultiply<int>;
        isSingleClass = TileReduce<int, ThreadsPerQuery, TOp>(workingTile, isSingleClass);
        using TAdd = TCudaAdd<ui32>;
        trueCount = TileReduce<ui32, ThreadsPerQuery, TAdd>(workingTile, trueCount);

        if (workingTile.thread_rank() == 0) {
            resultsIsSingle[localQid] = isSingleClass;
            resultsCount[localQid] = trueCount;
            workingTile.sync();
        }
        isSingleClass = resultsIsSingle[localQid];
        trueCount = resultsCount[localQid];

        for (int i = workingTile.thread_rank(); i < querySize; i += ThreadsPerQuery) {
            isSingleClassQuery[queryOffset + i] = isSingleClass == 1;
            trueClassCount[queryOffset + i] = trueCount;
        }
    }

    void MakeIsSingleClassFlags(const float* targets, const ui32* loadIndices,
                                const ui32* queryOffsets,
                                ui32 queryCount,
                                double meanQuerySize,
                                bool* isSingleClassQuery,
                                ui32* trueClassCount,
                                TCudaStream stream) {

        const ui32 blockSize = 128;


        #define RUN_KERNEL(threadsPerQuery) \
        const ui32 numBlocks = SafeIntegerCast<ui32>((((ui64)queryCount + 1) * threadsPerQuery +  blockSize - 1) / blockSize); \
        if (numBlocks > 0) { \
            MakeIsSingleClassFlagsImpl<blockSize, threadsPerQuery> <<< numBlocks, blockSize, 0, stream >>> ((int*)queryOffsets,  queryCount, loadIndices, targets, isSingleClassQuery, trueClassCount); \
        }

        if (meanQuerySize < 2) {
            RUN_KERNEL(2)
        } else if (meanQuerySize < 4) {
            RUN_KERNEL(4)
        } else if (meanQuerySize < 8) {
            RUN_KERNEL(8)
        } else if (meanQuerySize < 16) {
            RUN_KERNEL(16)
        } else {
            RUN_KERNEL(32)
        }
        #undef RUN_KERNEL
    }


    //for stochastic gradient
    __global__ void FillPairDer2AndRemapPairDocumentsImpl(const float* ders2,
                                                          const float* groupDers2,
                                                          const ui32* docIds,
                                                          const ui32* qids,
                                                          ui64 pairCount,
                                                          float* pairDer2,
                                                          uint2* pairs) {

        const ui32 tid = threadIdx.x;
        const ui64 i = (ui64)blockIdx.x * blockDim.x + tid;

        if (i < pairCount) {
            uint2 pair = Ldg(pairs + i);

            const float der2x = Ldg(ders2 + pair.x);
            const float der2y = Ldg(ders2 + pair.y);
            const int qid = Ldg(qids + pair.x);
            const float groupDer2 = Ldg(groupDers2 + qid);

            pair.x = Ldg(docIds + pair.x);
            pair.y = Ldg(docIds + pair.y);

            pairDer2[i] = groupDer2 > 1e-20f ? der2x * der2y / (groupDer2 + 1e-20f) : 0;
            pairs[i] = pair;
        }
    }

    void FillPairDer2AndRemapPairDocuments(const float* ders2,
                                           const float* groupDers2,
                                           const ui32* docIds,
                                           const ui32* qids,
                                           ui64 pairCount,
                                           float* pairDer2,
                                           uint2* pairs,
                                           TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>((pairCount + blockSize - 1) / blockSize); // cuda grid sizes are ui32
        if (numBlocks > 0) {
            FillPairDer2AndRemapPairDocumentsImpl<<< numBlocks, blockSize,0, stream >>>(ders2, groupDers2, docIds, qids, pairCount, pairDer2, pairs);
        }
    }






}
