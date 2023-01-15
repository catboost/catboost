#pragma once

#include "tuning_policy_enums.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {


    #define ALIGN_MEMORY(LoadSize)\
    const int warpSize = 32;\
    const int warpsPerBlock = (BlockSize / 32);\
    const int globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);\
    const int loadSize = LoadSize;\
    int tid = threadIdx.x;\
    AlignMemoryAccess<BlockSize, loadSize, N, THist>(partOffset, partSize, bins, stats, statsLineSize, blockId, blockCount, hist);\
    if (blockId == 0 && partSize <= 0) {\
        __syncthreads();\
        hist.Reduce();\
        return;\
    }\
    const int entriesPerWarp = warpSize * N * loadSize;\
    stats += globalWarpId * entriesPerWarp;\
    bins  += globalWarpId * entriesPerWarp;\
    const int stripeSize = entriesPerWarp * warpsPerBlock * blockCount;\
    partSize = max(partSize - globalWarpId * entriesPerWarp, 0);\
    const int localIdx = (tid & 31) * loadSize;\
    const int iterCount = (partSize - localIdx + stripeSize - 1)  / stripeSize;\
    stats += localIdx;\
    bins += localIdx;



    #define ALIGN_MEMORY_GATHER(LoadSize)\
    const int warpSize = 32;\
    const int warpsPerBlock = (BlockSize / 32);\
    const int globalWarpId = (blockId * warpsPerBlock) + (threadIdx.x / 32);\
    const int loadSize = LoadSize;\
    int tid = threadIdx.x;\
    AlignMemoryAccess<BlockSize, loadSize, N, THist>(partOffset, partSize, cindex, indices, stats, statsLineSize, blockId, blockCount, hist);\
    if (blockId == 0 && partSize <= 0) {\
        __syncthreads();\
        hist.Reduce();\
        return;\
    }\
    const int entriesPerWarp = warpSize * N * loadSize;\
    stats += globalWarpId * entriesPerWarp;\
    indices  += globalWarpId * entriesPerWarp;\
    const int stripeSize = entriesPerWarp * warpsPerBlock * blockCount;\
    partSize = max(partSize - globalWarpId * entriesPerWarp, 0);\
    const int localIdx = (tid & 31) * loadSize;\
    const int iterCount = (partSize - localIdx + stripeSize - 1)  / stripeSize;\
    stats += localIdx;\
    indices += localIdx;

    template <int BlockSize, int LoadSize, int N, typename THist>
    __forceinline__  __device__ void AlignMemoryAccess(int partOffset,
                                                       int& partSize,
                                                       const ui32*& bins,
                                                       const float*& stats,
                                                       const int statsLineSize,
                                                       int blockId,
                                                       int blockCount,
                                                       THist& hist) {
        bins += partOffset;
        stats += partOffset;

        const int warpSize = 32;
        int tid = threadIdx.x;

        //all operations should be warp-aligned
        const int alignSize = LoadSize * warpSize * N;

        {
            int lastId = min(partSize, alignSize - (partOffset % alignSize));

            if (blockId == 0) {

                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const ui32 bin = idx < lastId ?  Ldg(bins, idx) : 0;
                    const float stat1 = idx < lastId ? Ldg(stats, idx) : 0;
                    const float stat2 = idx < lastId ? Ldg(stats, idx + statsLineSize) : 0;
                    hist.AddPoint(bin, stat1, stat2);
                }
            }

            partSize = max(partSize - lastId, 0);
            stats += lastId;
            bins += lastId;
        }

        //now lets align end
        const int unalignedTail = (partSize % alignSize);

        if (unalignedTail != 0) {
            if (blockId == 0) {
                const int tailOffset = partSize - unalignedTail;
                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const ui32 bin = idx < unalignedTail ? Ldg(bins, tailOffset + idx) : 0;
                    const float stat1 = idx < unalignedTail ? Ldg(stats, tailOffset + idx) : 0;
                    const float stat2 = idx < unalignedTail ? Ldg(stats, tailOffset + idx + statsLineSize) : 0;
                    hist.AddPoint(bin, stat1, stat2);
                }
            }
        }
        partSize -= unalignedTail;
    }

    template <int BlockSize, int LoadSize, int N, typename THist>
    __forceinline__  __device__ void AlignMemoryAccess(int partOffset,
                                                       int& partSize,
                                                       const ui32* cindex,
                                                       const int*& indices,
                                                       const float*& stats,
                                                       const int statsLineSize,
                                                       int blockId,
                                                       int blockCount,
                                                       THist& hist) {
        indices += partOffset;
        stats += partOffset;

        const int warpSize = 32;
        int tid = threadIdx.x;

        //all operations should be warp-aligned
        const int alignSize = LoadSize * warpSize * N;

        {
            int lastId = min(partSize, alignSize - (partOffset % alignSize));

            if (blockId == 0) {
                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const int loadIdx = idx < lastId ? Ldg(indices, idx) : 0;
                    const ui32 bin = idx < lastId ?  Ldg(cindex, loadIdx) : 0;
                    const float stat1 = idx < lastId ? Ldg(stats, idx) : 0;
                    const float stat2 = idx < lastId ? Ldg(stats, idx + statsLineSize) : 0;
                    hist.AddPoint(bin, stat1, stat2);
                }
            }

            partSize = max(partSize - lastId, 0);
            stats += lastId;
            indices += lastId;
        }

        //now lets align end
        const int unalignedTail = (partSize % alignSize);

        if (unalignedTail != 0) {
            if (blockId == 0) {
                const int tailOffset = partSize - unalignedTail;
                for (int idx = tid; idx < alignSize; idx += BlockSize) {
                    const ui32 loadIdx = idx < unalignedTail ? Ldg(indices, tailOffset + idx) :  0;
                    const ui32 bin = idx < unalignedTail ? Ldg(cindex, loadIdx) : 0;
                    const float stat1 = idx < unalignedTail ? Ldg(stats, tailOffset + idx) : 0;
                    const float stat2 = idx < unalignedTail ? Ldg(stats, tailOffset + idx + statsLineSize) : 0;
                    hist.AddPoint(bin, stat1, stat2);
                }
            }
        }
        partSize -= unalignedTail;
    }


    template <ELoadSize, int BlockSize, typename THist>
    struct TComputeHistogramTwoStatsImpl;


    template <int BlockSize,
              typename THist>
    struct TComputeHistogramTwoStatsImpl<ELoadSize::OneElement, BlockSize, THist> {

        __device__ __forceinline__ static  void Compute(const ui32* bins,
                                                        const float* stats,
                                                        const int statsLineSize,
                                                        int partOffset,
                                                        int partSize,
                                                        int blockId,
                                                        int blockCount,
                                                        THist& hist) {

            constexpr int N = THist::Unroll(ECIndexLoadType::Direct);

            ALIGN_MEMORY(1)

            if (partSize) {
                ui32 localBins[N];
                float localStats1[N];
                float localStats2[N];

                for (int j = 0; j < iterCount; ++j) {

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localBins[k] = Ldg(bins, warpSize * k);
                    }


                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg(stats, warpSize * k);
                        localStats2[k] = Ldg(stats + statsLineSize, warpSize * k);
                    }

                    hist.AddPoints<N>(localBins, localStats1, localStats2);

                    bins += stripeSize;
                    stats += stripeSize;

                }
            }

            hist.Reduce();
            __syncthreads();
        }

        __forceinline__  __device__ static  void Compute(const ui32* cindex,
                                                         const int* indices,
                                                         const float* stats,
                                                         const int statsLineSize,
                                                         int partOffset,
                                                         int partSize,
                                                         int blockId,
                                                         int blockCount,
                                                         THist& hist) {

            constexpr int N = THist::Unroll(ECIndexLoadType::Gather);

            ALIGN_MEMORY_GATHER(1)

            if (partSize) {
                int localIndices[N];
                ui32 localBins[N];
                float localStats1[N];
                float localStats2[N];

                for (int j = 0; j < iterCount; ++j) {

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localIndices[k] = Ldg(indices, warpSize * k);
                    }

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localBins[k] = Ldg(cindex, localIndices[k]);
                    }

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg(stats, warpSize * k);
                        localStats2[k] = Ldg(stats + statsLineSize, warpSize * k);
                    }

                    stats += stripeSize;
                    indices += stripeSize;

                    hist.AddPoints<N>(localBins, localStats1, localStats2);
                }
            }

            hist.Reduce();
            __syncthreads();
        }
    };



    template <int BlockSize, typename THist>
    struct TComputeHistogramTwoStatsImpl<ELoadSize::TwoElements, BlockSize, THist>  {

        __device__ __forceinline__ static void Compute(const ui32* bins,
                                                       const float* stats,
                                                       int statsLineSize,
                                                       int partOffset,
                                                       int partSize,
                                                       int blockId,
                                                       int blockCount,
                                                       THist& hist) {

            constexpr int N = THist::Unroll(ECIndexLoadType::Direct);
            ALIGN_MEMORY(2)

            if (partSize) {
                for (int j = 0; j < iterCount; ++j) {
                    uint2 localBins[N];
                    float2 localStats1[N];
                    float2 localStats2[N];

                    #pragma unroll N
                    for (int k = 0; k < N; ++k) {
                        localBins[k] = Ldg((uint2*) bins, warpSize * k);
                    }


                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg((float2*) stats, warpSize * k);
                        localStats2[k] = Ldg((float2*)(stats + statsLineSize), warpSize * k);
                    }

                    stats += stripeSize;
                    bins += stripeSize;

                    hist.AddPoints<loadSize * N>((ui32*)localBins, (float*)localStats1, (float*)localStats2);
                }
            }

            hist.Reduce();
            __syncthreads();
        }

        __forceinline__  __device__  static void Compute(const ui32* cindex,
                                                         const int* indices,
                                                         const float* stats,
                                                         int statsLineSize,
                                                         int partOffset,
                                                         int partSize,
                                                         int blockId,
                                                         int blockCount,
                                                         THist& hist) {
            constexpr int N = THist::Unroll(ECIndexLoadType::Gather);
            ALIGN_MEMORY_GATHER(2)

            if (partSize) {
                for (int j = 0; j < iterCount; ++j) {
                    int2 localIndices[N];

                    #pragma unroll N
                    for (int k = 0; k < N; ++k) {
                        localIndices[k] = Ldg((int2*)indices, warpSize * k);
                    }

                    float2 localStats1[N];
                    float2 localStats2[N];
                    uint2 localBins[N];

                    #pragma unroll N
                    for (int k = 0; k < N; ++k) {
                        localBins[k].x = Ldg(cindex, localIndices[k].x);
                        localBins[k].y = Ldg(cindex, localIndices[k].y);
                    }

                    #pragma unroll N
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg((float2*)stats, warpSize * k);
                        localStats2[k] = Ldg((float2*)(stats + statsLineSize), warpSize * k);
                    }

                    stats += stripeSize;
                    indices += stripeSize;

                    hist.AddPoints<loadSize * N>((ui32*)localBins, (float*)localStats1, (float*)localStats2);
                }
            }

            hist.Reduce();
            __syncthreads();

        }
    };


    template <int BlockSize, typename THist>
    struct TComputeHistogramTwoStatsImpl<ELoadSize::FourElements, BlockSize, THist> {

       __device__ __forceinline__  static void Compute(const ui32* bins,
                                                       const float* stats,
                                                       const int statsLineSize,
                                                       int partOffset,
                                                       int partSize,
                                                       int blockId,
                                                       int blockCount,
                                                       THist& hist) {
            constexpr int N = THist::Unroll(ECIndexLoadType::Direct);

            ALIGN_MEMORY(4)

            if (partSize) {
                for (int j = 0; j < iterCount; ++j) {
                    uint4 localBins[N];
                    float4 localStats1[N];
                    float4 localStats2[N];

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localBins[k] = Ldg((uint4*) bins, warpSize * k);
                    }


                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg((float4*) stats, warpSize * k);
                        localStats2[k] = Ldg((float4*) (stats + statsLineSize), warpSize * k);
                    }

                    stats += stripeSize;
                    bins += stripeSize;

                    hist.AddPoints<loadSize * N>((ui32*)localBins, (float*)localStats1, (float*)localStats2);
                }
            }

            hist.Reduce();
            __syncthreads();
        }

        __forceinline__  __device__ static void Compute(const ui32* cindex,
                                                        const int* indices,
                                                        const float* stats,
                                                        const int statsLineSize,
                                                        int partOffset,
                                                        int partSize,
                                                        int blockId,
                                                        int blockCount,
                                                        THist& hist) {

            constexpr int N = THist::Unroll(ECIndexLoadType::Gather);

            ALIGN_MEMORY_GATHER(4)

            if (partSize) {
                for (int j = 0; j < iterCount; ++j) {
                    int4 localIndices[N];

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localIndices[k] = Ldg((int4*)indices, warpSize * k);
                    }

                    uint4 localBins[N];

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localBins[k].x = Ldg(cindex, localIndices[k].x);
                        localBins[k].y = Ldg(cindex, localIndices[k].y);
                        localBins[k].z = Ldg(cindex, localIndices[k].z);
                        localBins[k].w = Ldg(cindex, localIndices[k].w);
                    }

                    float4 localStats1[N];
                    float4 localStats2[N];

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        localStats1[k] = Ldg((float4*)stats, warpSize * k);
                        localStats2[k] = Ldg((float4*)(stats + statsLineSize), warpSize * k);
                    }

                    stats += stripeSize;
                    indices += stripeSize;

                    hist.AddPoints<loadSize * N>((ui32*)localBins, (float*)localStats1, (float*)localStats2);
                }
            }

            hist.Reduce();
            __syncthreads();
        }
    };

    template <typename THist>
    struct TComputeHistogramTwoStats {
        __device__ __forceinline__  static void Compute(const ui32* bins,
                                                        const float* stats,
                                                        int statsLineSize,
                                                        int partOffset,
                                                        int partSize,
                                                        int blockId,
                                                        int blockCount,
                                                        THist& hist) {

            TComputeHistogramTwoStatsImpl<THist::LoadSize(), THist::GetBlockSize(), THist>::Compute(bins, stats, statsLineSize, partOffset, partSize, blockId, blockCount, hist);

        }

        __forceinline__  __device__ static void Compute(const ui32* cindex,
                                                        const int* indices,
                                                        const float* stats,
                                                        int statsLineSize,
                                                        int partOffset,
                                                        int partSize,
                                                        int blockId,
                                                        int blockCount,
                                                        THist& hist) {
            TComputeHistogramTwoStatsImpl<THist::LoadSize(), THist::GetBlockSize(), THist>::Compute(cindex, indices, stats, statsLineSize, partOffset, partSize, blockId, blockCount, hist);
        }
    };

    #undef ALIGN_MEMORY
    #undef ALIGN_MEMORY_GATHER


    template <class THist, int BlockSize, int GroupSize, bool SkipFirst>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesDirectLoadsTwoStastImpl(
            const TFeatureInBlock* __restrict__ features,
            int fCount,
            const ui32* __restrict__ bins,
            ui32 binsLineSize,
            const float* __restrict__ stats,
            const int statsLineSize,
            const TDataPartition* __restrict__ partitions,
            const ui32* partIds,
            float* __restrict__ binSums) {
        const int partId = partIds[blockIdx.y];
        TDataPartition partition = partitions[partId];


        const int maxBlocksPerPart = gridDim.x / ((fCount + GroupSize - 1) / GroupSize);
        const int featureOffset = (blockIdx.x / maxBlocksPerPart) * GroupSize;
        bins += (binsLineSize * (blockIdx.x / maxBlocksPerPart));
        features += featureOffset;
        fCount = min(fCount - featureOffset, GroupSize);

        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int minDocsPerBlock = THist::BlockLoadSize(ECIndexLoadType::Direct);
        const int activeBlockCount = min((partition.Size + minDocsPerBlock - 1) / minDocsPerBlock,
                                         maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        stats += ((SkipFirst ? 1 : 0) + 2 * blockIdx.z) * statsLineSize;

        constexpr int histSize = THist::GetHistSize();
        __shared__ float smem[histSize];
        THist hist(smem);

        TComputeHistogramTwoStats<THist>::Compute(bins,
                                          stats,
                                          statsLineSize,
                                          partition.Offset,
                                          partition.Size,
                                          localBlockIdx,
                                          activeBlockCount,
                                          hist
        );

        __syncthreads();

        const int statCount = gridDim.z * 2 + (SkipFirst ? 1 : 0);

        hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z, statCount,
                               activeBlockCount,
                               features,
                               fCount,
                               blockIdx.y,
                               gridDim.y,
                               binSums);
    }


    template <class THist, int BlockSize, int GroupSize, bool SkipFirst>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesTwoStatsGatherImpl(
            const TFeatureInBlock* __restrict__ features, int fCount,
            const ui32* __restrict__ cindex,
            const int* __restrict__ indices,
            const float* __restrict__ stats,
            const int statsLineSize,
            const TDataPartition* __restrict__ partitions,
            const ui32* partIds,
            float* __restrict__ binSums) {

        const int partId = partIds[blockIdx.y];
        TDataPartition partition = partitions[partId];


        const int maxBlocksPerPart = gridDim.x / ((fCount + GroupSize - 1) / GroupSize);
        const int featureOffset = (blockIdx.x / maxBlocksPerPart) * GroupSize;

        features += featureOffset;
        cindex += features->CompressedIndexOffset;
        fCount = min(fCount - featureOffset, GroupSize);

        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int minDocsPerBlock = THist::BlockLoadSize(ECIndexLoadType::Gather);
        const int activeBlockCount = min((partition.Size + minDocsPerBlock - 1) / minDocsPerBlock,
                                         maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        stats += ((SkipFirst ? 1 : 0)  + 2 * blockIdx.z) * statsLineSize;

        constexpr int histSize = THist::GetHistSize();
        __shared__ float smem[histSize];
        THist hist(smem);

        TComputeHistogramTwoStats<THist>::Compute(cindex,
                                          indices,
                                          stats,
                                          statsLineSize,
                                          partition.Offset,
                                          partition.Size,
                                          localBlockIdx,
                                          activeBlockCount,
                                          hist
        );

        __syncthreads();

        const int statCount = (SkipFirst ? 1 : 0) + 2 * gridDim.z;
        hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z,
                               statCount,
                               activeBlockCount,
                               features,
                               fCount,
                               blockIdx.y,
                               gridDim.y,
                               binSums);

    }




    /* Single part */


    template <class THist, int BlockSize, int GroupSize, bool SkipFirst>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesDirectLoadsTwoStastImpl(
        const TFeatureInBlock* __restrict__ features,
        int fCount,
        const ui32* __restrict__ bins,
        ui32 binsLineSize,
        const float* __restrict__ stats,
        const int statsLineSize,
        const TDataPartition* __restrict__ partitions,
        const ui32 partId,
        float* __restrict__ binSums) {
        TDataPartition partition = partitions[partId];


        const int maxBlocksPerPart = gridDim.x / ((fCount + GroupSize - 1) / GroupSize);
        const int featureOffset = (blockIdx.x / maxBlocksPerPart) * GroupSize;
        bins += (binsLineSize * (blockIdx.x / maxBlocksPerPart));
        features += featureOffset;
        fCount = min(fCount - featureOffset, GroupSize);

        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int minDocsPerBlock = THist::BlockLoadSize(ECIndexLoadType::Direct);
        const int activeBlockCount = min((partition.Size + minDocsPerBlock - 1) / minDocsPerBlock,
                                         maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        stats += ((SkipFirst ? 1 : 0) + 2 * blockIdx.z) * statsLineSize;

        constexpr int histSize = THist::GetHistSize();
        __shared__ float smem[histSize];
        THist hist(smem);

        TComputeHistogramTwoStats<THist>::Compute(bins,
                                                  stats,
                                                  statsLineSize,
                                                  partition.Offset,
                                                  partition.Size,
                                                  localBlockIdx,
                                                  activeBlockCount,
                                                  hist
        );

        __syncthreads();

        const int statCount = gridDim.z * 2 + (SkipFirst ? 1 : 0);

        hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z, statCount,
                               activeBlockCount,
                               features,
                               fCount,
                               0,
                               1,
                               binSums);
    }


    template <class THist, int BlockSize, int GroupSize, bool SkipFirst>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesTwoStatsGatherImpl(
        const TFeatureInBlock* __restrict__ features, int fCount,
        const ui32* __restrict__ cindex,
        const int* __restrict__ indices,
        const float* __restrict__ stats,
        const int statsLineSize,
        const TDataPartition* __restrict__ partitions,
        const ui32 partId,
        float* __restrict__ binSums) {

        TDataPartition partition = partitions[partId];


        const int maxBlocksPerPart = gridDim.x / ((fCount + GroupSize - 1) / GroupSize);
        const int featureOffset = (blockIdx.x / maxBlocksPerPart) * GroupSize;

        features += featureOffset;
        cindex += features->CompressedIndexOffset;
        fCount = min(fCount - featureOffset, GroupSize);

        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int minDocsPerBlock = THist::BlockLoadSize(ECIndexLoadType::Gather);
        const int activeBlockCount = min((partition.Size + minDocsPerBlock - 1) / minDocsPerBlock,
                                         maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        stats += ((SkipFirst ? 1 : 0)  + 2 * blockIdx.z) * statsLineSize;

        constexpr int histSize = THist::GetHistSize();
        __shared__ float smem[histSize];
        THist hist(smem);

        TComputeHistogramTwoStats<THist>::Compute(cindex,
                                                  indices,
                                                  stats,
                                                  statsLineSize,
                                                  partition.Offset,
                                                  partition.Size,
                                                  localBlockIdx,
                                                  activeBlockCount,
                                                  hist
        );

        __syncthreads();

        const int statCount = (SkipFirst ? 1 : 0) + 2 * gridDim.z;
        hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z,
                               statCount,
                               activeBlockCount,
                               features,
                               fCount,
                               0,
                               1,
                               binSums);

    }


}
