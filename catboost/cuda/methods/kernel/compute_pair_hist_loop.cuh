#pragma once
#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "pair.cuh"

#include <library/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

    template <int BlockSize, int LoadSize, int N, typename THist>
    __forceinline__  __device__ void AlignMemoryAccess(int& offset, int& partSize,
                                                       const ui32*& cindex,
                                                       const uint2*& pairs,
                                                       const float*& weight,
                                                       int blockId, int blockCount,
                                                       THist& hist) {
        const int warpSize = 32;
        int tid = threadIdx.x;

        pairs += offset;
        weight += offset;

        //all operations should be warp-aligned
        const int alignSize = LoadSize * warpSize * N;
        {
            int lastId = min(partSize, alignSize - (offset % alignSize));

            if (blockId == 0) {
                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const uint2 pair = idx < lastId ? Ldg(pairs, idx) : ZeroPair();
                    const ui32 ci1 = idx < lastId ? Ldg(cindex, pair.x) : 0;
                    const ui32 ci2 = idx < lastId ? Ldg(cindex, pair.y) : 0;
                    const float w = idx < lastId ? Ldg(weight, idx) : 0;
                    hist.AddPair(ci1, ci2, w);
                }
            }
            partSize = max(partSize - lastId, 0);

            weight += lastId;
            pairs += lastId;
        }

        //now lets align end
        const int unalignedTail = (partSize % alignSize);

        if (unalignedTail != 0) {
            if (blockId == 0) {
                const int tailOffset = partSize - unalignedTail;
                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const uint2 pair = idx < unalignedTail ? Ldg(pairs, tailOffset + idx) : ZeroPair();
                    const ui32 ci1 = idx < unalignedTail ? Ldg(cindex, pair.x) : 0;
                    const ui32 ci2 = idx < unalignedTail ? Ldg(cindex, pair.y) : 0;
                    const float w = idx < unalignedTail ? Ldg(weight, tailOffset + idx) : 0;
                    hist.AddPair(ci1, ci2, w);
                }
            }
        }
        partSize -= unalignedTail;
    }


    template <int BlockSize, int N, int OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram(int offset, int partSize,
                                                          const ui32* cindex,
                                                          const uint2* pairs,
                                                          const float* weight,
                                                          int blockId, int blockCount,
                                                          THist& hist) {
        const int warpSize = 32;
        const int warpsPerBlock = (BlockSize / 32);
        const int globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        const int loadSize = 1;
        int tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, loadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }

        const int entriesPerWarp = warpSize * N * loadSize;
        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const int stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = max(partSize - globalWarpId * entriesPerWarp, 0);

        int localIdx = (tid & 31) * loadSize;
        const int iterCount = (partSize - localIdx + stripeSize - 1)  / stripeSize;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (int j = 0; j < iterCount; ++j) {

                uint2 localPairs[N];


                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<uint2>(pairs, warpSize * k);
                }
                ui32 localBins1[N];
                ui32 localBins2[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localBins1[k] = Ldg(cindex, localPairs[k].x);
                    localBins2[k] = Ldg(cindex, localPairs[k].y);
                }

                float localWeights[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localWeights[k] = Ldg(weight, warpSize * k);
                }

                pairs += stripeSize;
                weight += stripeSize;

                hist.AddPairs<N>(localBins1, localBins2, localWeights);
            }
        }

        hist.Reduce();
        __syncthreads();
    }

    template <int BlockSize, int N, int OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram2(int offset, int partSize,
                                                           const ui32* cindex,
                                                           const uint2* pairs,
                                                           const float* weight,
                                                           int blockId, int blockCount,
                                                           THist& hist) {
        const int warpSize = 32;
        const int warpsPerBlock = (BlockSize / 32);
        const int globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        const int loadSize = 2;
        int tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, loadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }



        const int entriesPerWarp = warpSize * N * loadSize;

        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const int stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = max(partSize - globalWarpId * entriesPerWarp, 0);

        int localIdx = (tid & 31) * loadSize;
        const int iterCount = (partSize - localIdx + stripeSize - 1)  / stripeSize;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (int j = 0; j < iterCount; ++j) {

                TPair2 localPairs[N];

                const TPair2* pairs2 = (const TPair2*)pairs;

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<TPair2>(pairs2, warpSize * k);
                }

                uint2 localBins1[N];
                uint2 localBins2[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localBins1[k].x = Ldg(cindex, localPairs[k].x.x);
                    localBins1[k].y = Ldg(cindex, localPairs[k].y.x);

                    localBins2[k].x = Ldg(cindex, localPairs[k].x.y);
                    localBins2[k].y = Ldg(cindex, localPairs[k].y.y);
                }

                float2 localWeights[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localWeights[k] = Ldg(((float2*)weight), warpSize * k);
                }

                pairs += stripeSize;
                weight += stripeSize;

                hist.AddPairs2<N>(localBins1, localBins2, localWeights);
            }
        }

        hist.Reduce();
        __syncthreads();
    }



    template <int BlockSize, int N, int OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram4(int offset, int partSize,
                                                           const ui32* cindex,
                                                           const uint2* pairs,
                                                           const float* weight,
                                                           int blockId, int blockCount,
                                                           THist& hist) {
        const int warpSize = 32;
        const int warpsPerBlock = (BlockSize / 32);
        const int globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        const int loadSize = 4;
        int tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, loadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }



        const int entriesPerWarp = warpSize * N * loadSize;

        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const int stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = max(partSize - globalWarpId * entriesPerWarp, 0);

        int localIdx = (tid & 31) * loadSize;
        const int iterCount = (partSize - localIdx + stripeSize - 1)  / stripeSize;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (int j = 0; j < iterCount; ++j) {

                TPair4 localPairs[N];

                const TPair4* pairs4 = (const TPair4*)pairs;

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<TPair4>(pairs4, warpSize * k);
                }

                uint4 localBins1[N];
                uint4 localBins2[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localBins1[k].x = Ldg(cindex, localPairs[k].x.x);
                    localBins1[k].y = Ldg(cindex, localPairs[k].y.x);
                    localBins1[k].z = Ldg(cindex, localPairs[k].z.x);
                    localBins1[k].w = Ldg(cindex, localPairs[k].w.x);

                    localBins2[k].x = Ldg(cindex, localPairs[k].x.y);
                    localBins2[k].y = Ldg(cindex, localPairs[k].y.y);
                    localBins2[k].z = Ldg(cindex, localPairs[k].z.y);
                    localBins2[k].w = Ldg(cindex, localPairs[k].w.y);
                }

                float4 localWeights[N];

                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    localWeights[k] = Ldg(((float4*)weight), warpSize * k);
                }

                pairs += stripeSize;
                weight += stripeSize;

                hist.AddPairs4<N>(localBins1, localBins2, localWeights);
            }
        }

        hist.Reduce();
        __syncthreads();
    }



}
