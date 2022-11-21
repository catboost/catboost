#pragma once
#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "pair.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

    template <ui32 BlockSize, ui32 LoadSize, ui32 N, typename THist>
    __forceinline__  __device__ void AlignMemoryAccess(ui32& offset, ui32& partSize,
                                                       const ui32*& cindex,
                                                       const uint2*& pairs,
                                                       const float*& weight,
                                                       int blockId, int blockCount,
                                                       THist& hist) {
        const ui32 warpSize = 32;
        ui32 tid = threadIdx.x;

        pairs += offset;
        weight += offset;

        //all operations should be warp-aligned
        const ui32 alignSize = LoadSize * warpSize * N;
        {
            ui32 lastId = min(partSize, alignSize - (offset % alignSize));

            if (blockId == 0) {
                for (ui32 idx = tid; idx < alignSize; idx += BlockSize) {
                    const uint2 pair = idx < lastId ? Ldg(pairs, idx) : ZeroPair();
                    const ui32 ci1 = idx < lastId ? Ldg(cindex, pair.x) : 0;
                    const ui32 ci2 = idx < lastId ? Ldg(cindex, pair.y) : 0;
                    const float w = idx < lastId ? Ldg(weight, idx) : 0;
                    hist.AddPair(ci1, ci2, w);
                }
            }
            partSize = partSize > lastId ? partSize - lastId : 0;

            weight += lastId;
            pairs += lastId;
        }

        //now lets align end
        const ui32 unalignedTail = (partSize % alignSize);

        if (unalignedTail != 0) {
            if (blockId == 0) {
                const ui32 tailOffset = partSize - unalignedTail;
                for (ui32 idx = tid; idx < alignSize; idx += BlockSize) {
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


    template <ui32 BlockSize, ui32 N, ui32 OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram(ui32 offset, ui32 partSize,
                                                          const ui32* cindex,
                                                          const uint2* pairs,
                                                          const float* weight,
                                                          int blockId, int blockCount,
                                                          THist& hist) {
        const ui32 warpSize = 32;
        const ui32 warpsPerBlock = (BlockSize / 32);
        const ui32 globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        constexpr ui32 LoadSize = 1;
        ui32 tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, LoadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }

        const ui32 entriesPerWarp = warpSize * N * LoadSize;
        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const ui32 stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = partSize > globalWarpId * entriesPerWarp ? partSize - globalWarpId * entriesPerWarp : 0;

        ui32 localIdx = (tid & 31u) * LoadSize;
        const ui32 iterCount = partSize > localIdx ? (partSize - localIdx + stripeSize - 1)  / stripeSize : 0;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (ui32 j = 0; j < iterCount; ++j) {

                uint2 localPairs[N];


                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<uint2>(pairs, warpSize * k);
                }
                ui32 localBins1[N];
                ui32 localBins2[N];

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    localBins1[k] = Ldg(cindex, localPairs[k].x);
                    localBins2[k] = Ldg(cindex, localPairs[k].y);
                }

                float localWeights[N];

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
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

    // for research -- dead code
    template <ui32 BlockSize, ui32 N, int OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram2(ui32 offset, ui32 partSize,
                                                           const ui32* cindex,
                                                           const uint2* pairs,
                                                           const float* weight,
                                                           ui32 blockId, ui32 blockCount,
                                                           THist& hist) {
        const ui32 warpSize = 32;
        const ui32 warpsPerBlock = (BlockSize / 32);
        const ui32 globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        constexpr ui32 LoadSize = 2;
        ui32 tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, LoadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }



        const ui32 entriesPerWarp = warpSize * N * LoadSize;

        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const ui32 stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = partSize > globalWarpId * entriesPerWarp ? partSize - globalWarpId * entriesPerWarp : 0;

        ui32 localIdx = (tid & 31u) * LoadSize;
        const ui32 iterCount = partSize > localIdx ? (partSize - localIdx + stripeSize - 1)  / stripeSize : 0;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (ui32 j = 0; j < iterCount; ++j) {

                TPair2 localPairs[N];

                const TPair2* pairs2 = (const TPair2*)pairs;

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<TPair2>(pairs2, warpSize * k);
                }

                uint2 localBins1[N];
                uint2 localBins2[N];

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    localBins1[k].x = Ldg(cindex, localPairs[k].x.x);
                    localBins1[k].y = Ldg(cindex, localPairs[k].y.x);

                    localBins2[k].x = Ldg(cindex, localPairs[k].x.y);
                    localBins2[k].y = Ldg(cindex, localPairs[k].y.y);
                }

                float2 localWeights[N];

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
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



    // for research -- dead code
    template <ui32 BlockSize, ui32 N, ui32 OuterUnroll, typename THist>
    __forceinline__  __device__ void ComputePairHistogram4(ui32 offset, ui32 partSize,
                                                           const ui32* cindex,
                                                           const uint2* pairs,
                                                           const float* weight,
                                                           ui32 blockId, ui32 blockCount,
                                                           THist& hist) {
        const ui32 warpSize = 32;
        const ui32 warpsPerBlock = (BlockSize / 32);
        const ui32 globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);
        constexpr ui32 LoadSize = 4;
        ui32 tid = threadIdx.x;

        AlignMemoryAccess<BlockSize, LoadSize, N, THist>(offset, partSize, cindex, pairs, weight, blockId, blockCount, hist);

        if (blockId == 0 && partSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }



        const ui32 entriesPerWarp = warpSize * N * LoadSize;

        weight += globalWarpId * entriesPerWarp;
        pairs  += globalWarpId * entriesPerWarp;

        const ui32 stripeSize = entriesPerWarp * warpsPerBlock * blockCount;
        partSize = partSize > globalWarpId * entriesPerWarp ? partSize - globalWarpId * entriesPerWarp : 0;

        ui32 localIdx = (tid & 31u) * LoadSize;
        const ui32 iterCount = partSize > localIdx ? (partSize - localIdx + stripeSize - 1)  / stripeSize : 0;

        weight += localIdx;
        pairs += localIdx;

        if (partSize) {

            #pragma unroll OuterUnroll
            for (ui32 j = 0; j < iterCount; ++j) {

                TPair4 localPairs[N];

                const TPair4* pairs4 = (const TPair4*)pairs;

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
                    localPairs[k] = Ldg<TPair4>(pairs4, warpSize * k);
                }

                uint4 localBins1[N];
                uint4 localBins2[N];

                #pragma unroll
                for (ui32 k = 0; k < N; ++k) {
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
                for (ui32 k = 0; k < N; ++k) {
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
