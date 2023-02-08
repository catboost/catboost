/**/#include "gather_bins.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <cooperative_groups.h>


using namespace cooperative_groups;

namespace NKernel {

    /* this routine gathers cindex in a way, that access in compute histograms could be sequential   */
    /* gathered index is groupCount * indicesCount size, because we should not copy target buffers as this
     * could affect memory usage with big number of stats */
    template <int N, int Unroll>
    __global__ void GatherCompressedIndexImpl(const TFeatureInBlock* features,
                                              int featuresPerInt,
                                              int groupCount,
                                              const TDataPartition* parts,
                                              const ui32* partIds,
                                              const ui32* indices,
                                              const ui32* cindex,
                                              const ui32 gatheredIndexLineSize,
                                              ui32* gatheredIndex) {

        const int firstGroup = blockIdx.z;

        features += firstGroup * featuresPerInt;
        gatheredIndex += gatheredIndexLineSize * firstGroup;
        groupCount = Min<int>(groupCount - firstGroup, N);

        const int partId = partIds[blockIdx.y];
        const TDataPartition partition = parts[partId];

        const ui32* cindexPtrs[N];

        for (int k = 0; k < N; ++k) {
            if (k < groupCount) {
                auto feature = features[k * featuresPerInt];
                cindexPtrs[k] = cindex + feature.CompressedIndexOffset;
            }
        }

        int i = partition.Offset + blockIdx.x * blockDim.x + threadIdx.x;
        const ui32 partEnd = partition.Offset + partition.Size;
        //TODO(noxoomo): unrolls
        const int stripe = gridDim.x * blockDim.x;

        #pragma unroll Unroll
        for (; i < partEnd; i += stripe) {
            const ui32 loadIdx = __ldg(indices + i);

            ui32 bins[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                bins[k] = k < groupCount ? __ldg(cindexPtrs[k] + loadIdx) : 0;
            }

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (k < groupCount) {
                    WriteThrough(gatheredIndex + i + gatheredIndexLineSize * k, bins[k]);
                }
            }
        }
    }


    void GatherCompressedIndex(const TFeatureInBlock* feature,
                               int fCount,
                               int featuresPerBlock,
                               const TDataPartition* parts,
                               const ui32* partIds,
                               const int partCount,
                               const ui32* indices,
                               const ui32* cindex,
                               ui32 gatheredIndexLineSize,
                               ui32* gatheredIndex,
                               TCudaStream stream) {

        if (partCount) {
            const int blockSize = 128;
            const int blocksPerSm = 16;

            const int groupCount = CeilDivide(fCount, featuresPerBlock);

            #define RUN_KERNEL(K, Unroll)\
                dim3 numBlocks;\
                numBlocks.y = partCount;\
                numBlocks.z = CeilDivide(groupCount, K);\
                const int maxBlocksPerGpu = blocksPerSm * TArchProps::SMCount();\
                const int mult = partCount > 1 ? 2 : 1;\
                numBlocks.x = CeilDivide(mult * maxBlocksPerGpu, (int) (numBlocks.y * numBlocks.z));\
                if (IsGridEmpty(numBlocks)) {\
                    return;\
                }\
                GatherCompressedIndexImpl<K, Unroll> <<< numBlocks, blockSize, 0, stream >>> (feature, featuresPerBlock, groupCount, parts, partIds, \
                                                                                              indices,  cindex, gatheredIndexLineSize, gatheredIndex);
//            if (groupCount > 4) {
//                RUN_KERNEL(8, 1)
//            } else
//            if (groupCount > 2) {
//                RUN_KERNEL(4, 1)
//            } else if (groupCount == 2) {
//                RUN_KERNEL(2, 8)
//            } else {
                RUN_KERNEL(1, 16)
//            }
            #undef  RUN_KERNEL
        }
    }








    /* this routine gathers cindex in a way, that access in compute histograms could be sequential   */
    /* gathered index is groupCount * indicesCount size, because we should not copy target buffers as this
     * could affect memory usage with big number of stats */
    template <int N, int Unroll>
    __global__ void GatherCompressedIndexSingleLeafImpl(const TFeatureInBlock* features,
                                                       int featuresPerInt,
                                                       int groupCount,
                                                       const TDataPartition* parts,
                                                       const ui32 partId,
                                                       const ui32* indices,
                                                       const ui32* cindex,
                                                       const ui32 gatheredIndexLineSize,
                                                       ui32* gatheredIndex) {

        const int firstGroup = blockIdx.z;

        features += firstGroup * featuresPerInt;
        gatheredIndex += gatheredIndexLineSize * firstGroup;
        groupCount = Min<int>(groupCount - firstGroup, N);

        const TDataPartition partition = parts[partId];

        const ui32* cindexPtrs[N];

        for (int k = 0; k < N; ++k) {
            if (k < groupCount) {
                auto feature = features[k * featuresPerInt];
                cindexPtrs[k] = cindex + feature.CompressedIndexOffset;
            }
        }

        int i = partition.Offset + blockIdx.x * blockDim.x + threadIdx.x;
        const ui32 partEnd = partition.Offset + partition.Size;
        //TODO(noxoomo): unrolls
        const int stripe = gridDim.x * blockDim.x;

        #pragma unroll Unroll
        for (; i < partEnd; i += stripe) {
            const ui32 loadIdx = __ldg(indices + i);

            ui32 bins[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                bins[k] = k < groupCount ? __ldg(cindexPtrs[k] + loadIdx) : 0;
            }

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (k < groupCount) {
                    WriteThrough(gatheredIndex + i + gatheredIndexLineSize * k, bins[k]);
                }
            }
        }
    }


    void GatherCompressedIndex(const TFeatureInBlock* feature,
                               int fCount,
                               int featuresPerBlock,
                               const TDataPartition* parts,
                               const ui32 partId,
                               const ui32* indices,
                               const ui32* cindex,
                               ui32 gatheredIndexLineSize,
                               ui32* gatheredIndex,
                               TCudaStream stream) {

        const int blockSize = 128;
        const int blocksPerSm = 16;

        const int groupCount = CeilDivide(fCount, featuresPerBlock);

        #define RUN_KERNEL(K, Unroll)\
            dim3 numBlocks;\
            numBlocks.y = 1;\
            numBlocks.z = CeilDivide(groupCount, K);\
            const int maxBlocksPerGpu = blocksPerSm * TArchProps::SMCount();\
            const int mult = 1;\
            numBlocks.x = CeilDivide(mult * maxBlocksPerGpu, (int) (numBlocks.y * numBlocks.z));\
            if (IsGridEmpty(numBlocks)) {\
                return;\
            }\
            GatherCompressedIndexSingleLeafImpl<K, Unroll> <<< numBlocks, blockSize, 0, stream >>> (feature, featuresPerBlock, groupCount, parts, partId, \
            indices,  cindex, gatheredIndexLineSize, gatheredIndex);

        RUN_KERNEL(1, 16)

        #undef  RUN_KERNEL

    }


}
