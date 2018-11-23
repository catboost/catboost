#include "split_points.cuh"

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <contrib/libs/cub/cub/device/device_radix_sort.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

namespace NKernel {


    template <typename T>
    __global__ void CopyInLeavesImpl(const ui32* leaves,
                                     const TDataPartition* parts,
                                     const T *src,
                                     T *dst,
                                     ui32 numStats,
                                     ui64 lineSize) {

        const ui32 leafId = leaves[blockIdx.y];

        const ui32 offset = parts[leafId].Offset;
        const ui32 size = parts[leafId].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        src += offset;
        dst += offset;

        while (i < size) {

            #pragma unroll 8
            for (int k = 0; k < numStats; ++k) {
                WriteThrough(dst + i + k * lineSize, __ldg(src + i + k * lineSize));
            }
            i += gridDim.x * blockDim.x;
        }
    }

    /* this should be called before updatePartProps */
    template <typename T>
    __global__ void GatherInLeavesImpl(const ui32* leaves,
                                       const TDataPartition* parts,
                                       const T *src,
                                       const ui32* map,
                                       T *dst,
                                       ui32 numStats,
                                       ui64 lineSize) {

        const ui32 leafId = leaves[blockIdx.y];

        const ui32 offset = parts[leafId].Offset;
        const ui32 size = parts[leafId].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        map += offset;
        src += offset;
        dst += offset;

        while (i < size) {
            const ui32 loadIdx = __ldg(map + i);

            #pragma unroll 8
            for (int k = 0; k < numStats; ++k) {
                WriteThrough(dst + i + k * lineSize, __ldg(src + loadIdx + k * lineSize));
            }
            i += gridDim.x * blockDim.x;
        }
    }

    template <class T>
    void CopyInLeaves(const ui32* leaves, const ui32 leavesCount,
                      const TDataPartition* parts,
                      const T *src,
                      T *dst,
                      ui32 numStats,
                      ui32 lineSize,
                      TCudaStream stream) {

        const ui32 blockSize = 256;

        dim3 numBlocks;
        numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
        numBlocks.y  =  leavesCount;
        numBlocks.z  =  1;

        if (leavesCount) {
            CopyInLeavesImpl<T><<<numBlocks, blockSize, 0, stream>>>(leaves, parts, src, dst, numStats, lineSize);
        }
    }

    template <class T>
    void GatherInLeaves(const ui32* leaves, const ui32 leavesCount,
                        const TDataPartition* parts,
                        const T *src,
                        const ui32* map,
                        T *dst,
                        ui32 numStats,
                        ui32 lineSize,
                        TCudaStream stream) {

        const ui32 blockSize = 256;

        dim3 numBlocks;
        numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
        numBlocks.y  =  leavesCount;
        numBlocks.z  =  1;

        if (leavesCount) {
            GatherInLeavesImpl<<<numBlocks, blockSize, 0, stream>>>(leaves, parts, src,  map, dst, numStats, lineSize);
        }
    }



    __global__ void UpdatePartitionsAfterSplitImpl(const ui32* leftLeaves,
                                                   const ui32* rightLeaves,
                                                   ui32 leafCount,
                                                   const bool* sortedFlags,
                                                   TDataPartition* parts) {

        const ui32 leftLeaf = leftLeaves[blockIdx.y];
        const ui32 rightLeaf = rightLeaves[blockIdx.y];

        sortedFlags += parts[leftLeaf].Offset;
        const ui32 partSize = parts[leftLeaf].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        const ui32 offset = parts[leftLeaf].Offset;
        while (i <= partSize) {
            int flag0 = i < partSize ? sortedFlags[i] : 1;
            int flag1 = i ? sortedFlags[i - 1] : 0;

            if (flag0 != flag1) {
                //we are on border
                parts[leftLeaf].Size = i;
                parts[rightLeaf].Offset = offset + i;
                parts[rightLeaf].Size =  partSize - i;
                break;
            }
            i += blockDim.x * gridDim.x;
        }
    }


    void UpdatePartitionsAfterSplit(const ui32* leftLeafs,
                                    const ui32* rightLeafs,
                                    ui32 leavesCount,
                                    const bool* sortedFlag,
                                    TDataPartition* parts,
                                    TCudaStream stream) {
        const ui32 blockSize = 512;

        dim3 numBlocks;
        numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
        numBlocks.y  =  leavesCount;
        numBlocks.z  =  1;

        if (leavesCount) {
            UpdatePartitionsAfterSplitImpl<<<numBlocks, blockSize, 0, stream>>>(leftLeafs, rightLeafs, leavesCount, sortedFlag, parts);
        }

    }
    /*
     * blockIdx.x * gridDim.x + threadIdx.x is index in leaf
     * blockIdx.y is part number
     * this is not time critical kernel, so we make for several blocks per SM for each leaf and just skip computations if necessary
     */
    __global__ void SplitAndMakeSequenceInLeavesImpl(const ui32* compressedIndex,
                                                     const ui32* loadIndices,
                                                     const TDataPartition* parts,
                                                     const ui32* leafIds,
                                                     const TCFeature* splitFeatures,
                                                     const ui32* splitBins,
                                                     bool* splitFlags,
                                                     ui32* indices) {

        const ui32 leafId = leafIds[blockIdx.y];

        const ui32 size = parts[leafId].Size;
        const ui32 offset = parts[leafId].Offset;

        loadIndices += offset;

        indices += offset;
        splitFlags += offset;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= size) {
            return;
        }

        TCFeature feature = splitFeatures[blockIdx.y];
        const ui32 binIdx = splitBins[blockIdx.y];

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;
        const bool oneHot = feature.OneHotFeature;

        compressedIndex += feature.Offset;

        while (i < size) {
            const ui32 loadIndex = loadIndices ? __ldg(loadIndices + i) : i;
            const ui32 featureVal = __ldg(compressedIndex + loadIndex) & mask;
            const bool split = (oneHot ? (featureVal == value) : featureVal > value);
            splitFlags[i] = split;
            indices[i] = i;
            i += blockDim.x * gridDim.x;
        }
    }


    void SplitAndMakeSequenceInLeaves(const ui32* compressedIndex,
                                      const ui32* loadIndices,
                                      const TDataPartition* parts,
                                      const ui32* leafIds,
                                      ui32 leavesCount,
                                      const TCFeature* splitFeatures,
                                      const ui32* splitBins,
                                      bool* splitFlags,
                                      ui32* indices,
                                      TCudaStream stream) {
        if (leavesCount) {
            const ui32 blockSize = 512;

            dim3 numBlocks;
            numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
            numBlocks.y  =  leavesCount;
            numBlocks.z  =  1;

            SplitAndMakeSequenceInLeavesImpl<<<numBlocks, blockSize, 0, stream>>>(compressedIndex, loadIndices, parts, leafIds, splitFeatures, splitBins, splitFlags, indices);

        }
    }


    //TODO(noxoomo): cub sucks for this, write proper segmented version
    void SortByFlagsInLeaves(const ui32* leavesToSplit, const ui32 leafCount,
                             const TDataPartition* partsCpu,
                             TSplitPointsContext& context,
                             TCudaStream stream) {
        /*
         * Sort leaves by flags
         */
        for (ui32 i = 0; i < leafCount; ++i) {
            const ui32 leafId = leavesToSplit[i];
            TDataPartition part = partsCpu[leafId];

            const bool* flagsSrc = context.TempFlags.Get() + part.Offset;
            bool* flagsDst = context.Flags.Get() + part.Offset;
            const ui32* indicesSrc = context.TempIndices.Get() + part.Offset;
            ui32* indicesDst = context.Indices.Get() + part.Offset;
            cudaError_t error = cub::DeviceRadixSort::SortPairs<bool, ui32>((void*)context.TempStorage.Get(),
                                                                             context.TempStorageSizes[i],
                                                                             flagsSrc,
                                                                             flagsDst,
                                                                             indicesSrc,
                                                                             indicesDst,
                                                                             (int)part.Size,
                                                                             0,
                                                                             1,
                                                                             stream);
            CUDA_SAFE_CALL(error);
        }
    }


    #define TEMPL_INST(Type)\
    template void CopyInLeaves<Type>(const ui32* leaves, const ui32 leavesCount, const TDataPartition* parts, const Type *src, Type *dst, ui32 numCopies, ui32 lineSize, TCudaStream stream);\
    template void GatherInLeaves<Type>(const ui32* leaves, const ui32 leavesCount, const TDataPartition* parts, const Type* src, const ui32* map, Type *dst, ui32 numStats, ui32 lineSize, TCudaStream stream);


    TEMPL_INST(ui32)
    TEMPL_INST(float)

    #undef TEMPL_INST



}


