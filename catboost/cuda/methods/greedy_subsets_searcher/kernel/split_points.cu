#include "split_points.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>

#include <catboost/cuda/cuda_lib/cuda_base.h>

#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>
#include <catboost/cuda/cuda_util/kernel/reorder_one_bit.cuh>
#include <catboost/cuda/cuda_util/kernel/reorder_one_bit_impl.cuh>

#include <cub/device/device_radix_sort.cuh>

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





    template <ui32 Size, ui32 BlockSize = 1024>
    __global__ void GatherInplaceImpl(const ui32* leaf,
                                      const TDataPartition* parts,
                                      const ui32* map,
                                      float* stats,
                                      ui64 lineSize,
                                      ui32* indices) {
        __shared__  char4 tmp[Size];

        char4* data = blockIdx.x == 0 ? (char4*)indices : (char4*)(stats + (blockIdx.x - 1) * lineSize);


        const ui32 leafId = leaf[blockIdx.y];

        TDataPartition part = Ldg(parts + leafId);
        const ui32 offset = part.Offset;
        ui32 size = part.Size;

        //should be always true btw, but may help compiler

        const ui32 tid = threadIdx.x;

        map += offset;
        data += offset;

        #pragma unroll
        for (ui32 i = tid; i < Size; i += BlockSize) {
            if (i < size) {
                const ui32 loadIdx = __ldg(map + i);
                tmp[i] = __ldg(data + loadIdx);
            }
        }

        __syncthreads();

        #pragma unroll
        for (ui32 i = tid; i < Size; i += BlockSize) {
            if (i < size) {
                WriteThrough(data + i, tmp[i]);
            }
        }
    }

    template <int Size>
    void GatherInplaceLeqSize(const ui32* leaf, ui32 leavesCount,
                               const TDataPartition* parts,
                               const ui32* map,
                               float* stats, ui32 statCount,
                               ui64 lineSize,
                               ui32* indices,
                               TCudaStream stream) {
        const ui32 blockSize = 1024;
        dim3 numBlocks;
        numBlocks.x = 1 + statCount;
        numBlocks.y = leavesCount;
        numBlocks.z = 1;
        if (IsGridEmpty(numBlocks)) {
            return;
        }
        GatherInplaceImpl<Size, blockSize> <<<numBlocks, blockSize, 0, stream>>>(leaf, parts, map, stats, lineSize, indices);
    }



    template <ui32 Size, ui32 BlockSize = 1024>
    __global__ void GatherInplaceSingleLeafImpl(const ui32 leafId,
                                                const TDataPartition* parts,
                                                const ui32* map,
                                                float* stats,
                                                ui64 lineSize,
                                                ui32* indices) {
        __shared__  char4 tmp[Size];

        char4* data = blockIdx.x == 0 ? (char4*)indices : (char4*)(stats + (blockIdx.x - 1) * lineSize);



        TDataPartition part = Ldg(parts + leafId);
        const ui32 offset = part.Offset;
        ui32 size = part.Size;

        //should be always true btw, but may help compiler

        const ui32 tid = threadIdx.x;

        data += offset;

        #pragma unroll
        for (ui32 i = tid; i < Size; i += BlockSize) {
            if (i < size) {
                const ui32 loadIdx = __ldg(map + i);
                tmp[i] = __ldg(data + loadIdx);
            }
        }

        __syncthreads();

        #pragma unroll
        for (ui32 i = tid; i < Size; i += BlockSize) {
            if (i < size) {
                WriteThrough(data + i, tmp[i]);
            }
        }
    }

    template <int Size>
    void GatherInplaceSingleLeaf(const ui32 leaf,
                                 const TDataPartition* parts,
                                 const ui32* map,
                                 float* stats, ui32 statCount,
                                 ui64 lineSize,
                                 ui32* indices,
                                 TCudaStream stream) {
        const ui32 blockSize = 1024;
        dim3 numBlocks;
        numBlocks.x = 1 + statCount;
        numBlocks.y = 1;
        numBlocks.z = 1;
        if (IsGridEmpty(numBlocks)) {
            return;
        }
        GatherInplaceSingleLeafImpl<Size, blockSize> <<<numBlocks, blockSize, 0, stream>>>(leaf, parts, map, stats, lineSize, indices);
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


    template <typename T>
    __global__ void CopyLeafImpl(const ui32 leafId,
                                 const TDataPartition* parts,
                                 const T* src,
                                 T* dst,
                                 ui32 numStats,
                                 ui64 lineSize) {

        const ui32 offset = parts[leafId].Offset;
        const ui32 size = parts[leafId].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        src += offset;

        while (i < size) {
            #pragma unroll 8
            for (int k = 0; k < numStats; ++k) {
                WriteThrough(dst + i + k * size, __ldg(src + i + k * lineSize));
            }
            i += gridDim.x * blockDim.x;
        }
    }


    template <class T>
    void CopyLeaf(const ui32 leafId, const ui32 leafSize,
                  const TDataPartition* parts,
                  const T* src,
                  T* dst,
                  ui32 numStats,
                  ui32 lineSize,
                  TCudaStream stream) {
        const ui32 blockSize = 256;

        dim3 numBlocks;
        numBlocks.x  =  (leafSize + blockSize - 1) / blockSize;
        numBlocks.y  =  1;
        numBlocks.z  =  1;

        if (leafSize) {
            CopyLeafImpl<T><<<numBlocks, blockSize, 0, stream>>>(leafId, parts, src, dst, numStats, lineSize);
        }
    }

    /* this should be called before updatePartProps */
    template <typename T>
    __global__ void GatherLeafImpl(const ui32 leafId,
                                   const TDataPartition* parts,
                                   const T* src,
                                   const ui32* map,
                                   T* dst,
                                   ui32 numStats,
                                   ui64 lineSize) {
        const ui32 offset = parts[leafId].Offset;
        const ui32 size = parts[leafId].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        dst += offset;

        while (i < size) {
            const ui32 loadIdx = __ldg(map + i);

            #pragma unroll 8
            for (int k = 0; k < numStats; ++k) {
                WriteThrough(dst + i + k * lineSize, __ldg(src + loadIdx + k * size));
            }
            i += gridDim.x * blockDim.x;
        }
    }

    template <class T>
    void GatherLeaf(const ui32 leafId, const ui32 leafSize,
                    const TDataPartition* parts,
                    const T* src,
                    const ui32* map,
                    T* dst,
                    ui32 numStats,
                    ui32 lineSize,
                    TCudaStream stream) {

        const ui32 blockSize = 256;

        dim3 numBlocks;
        numBlocks.x  = (leafSize + blockSize - 1) / blockSize;
        numBlocks.y  =  1;
        numBlocks.z  =  1;

        if (leafSize) {
            GatherLeafImpl<<<numBlocks, blockSize, 0, stream>>>(leafId, parts, src,  map, dst, numStats, lineSize);
        }
    }


    __global__ void UpdatePartitionsAfterSplitImpl(const ui32* leftLeaves,
                                                   const ui32* rightLeaves,
                                                   ui32 leafCount,
                                                   const bool* sortedFlags,
                                                   TDataPartition* parts,
                                                   TDataPartition* partsCpu
                                                   ) {

        const ui32 leftLeaf = leftLeaves[blockIdx.y];
        const ui32 rightLeaf = rightLeaves[blockIdx.y];

        sortedFlags += parts[leftLeaf].Offset;
        const ui32 partSize = parts[leftLeaf].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        const ui32 offset = parts[leftLeaf].Offset;
        while (i <= partSize) {
            int flag0 = i < partSize ? Ldg(sortedFlags + i) : 1;
            int flag1 = i ? Ldg(sortedFlags + i - 1) : 0;

            if (flag0 != flag1) {
                //we are on border
                TDataPartition leftPart =  parts[leftLeaf];
                leftPart.Size = i;
                parts[leftLeaf] = leftPart;
                partsCpu[leftLeaf] = leftPart;

                TDataPartition rightPart = parts[rightLeaf];
                rightPart.Offset = offset + i;
                rightPart.Size =  partSize - i;

                parts[rightLeaf] = rightPart;
                partsCpu[rightLeaf] = rightPart;
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
                                    TDataPartition* partsCpu,
                                    TCudaStream stream) {
        const ui32 blockSize = 512;

        dim3 numBlocks;
        numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
        numBlocks.y  =  leavesCount;
        numBlocks.z  =  1;

        if (leavesCount) {
            UpdatePartitionsAfterSplitImpl<<<numBlocks, blockSize, 0, stream>>>(leftLeafs, rightLeafs, leavesCount, sortedFlag, parts, partsCpu);
        }
    }



    __global__ void UpdatePartitionAfterSplitImpl(const ui32 leftLeaf,
                                                   const ui32 rightLeaf,
                                                   const bool* sortedFlags,
                                                   TDataPartition* parts,
                                                   TDataPartition* partsCpu
    ) {

        const ui32 partSize = parts[leftLeaf].Size;

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        const ui32 offset = parts[leftLeaf].Offset;
        while (i <= partSize) {
            int flag0 = i < partSize ? Ldg(sortedFlags + i) : 1;
            int flag1 = i ? Ldg(sortedFlags + i - 1) : 0;

            if (flag0 != flag1) {
                //we are on border
                TDataPartition leftPart =  parts[leftLeaf];
                leftPart.Size = i;
                partsCpu[leftLeaf] = leftPart;
                parts[leftLeaf] = leftPart;

                TDataPartition rightPart = parts[rightLeaf];
                rightPart.Offset = offset + i;
                rightPart.Size =  partSize - i;

                partsCpu[rightLeaf] = rightPart;
                parts[rightLeaf] = rightPart;

                break;
            }
            i += blockDim.x * gridDim.x;
        }
    }


    void UpdatePartitionAfterSplit(const ui32 leftLeaf,
                                    const ui32 rightLeaf,
                                    ui32 leafSize,
                                    const bool* sortedFlag,
                                    TDataPartition* parts,
                                    TDataPartition* partsCpu,
                                    TCudaStream stream) {
        const ui32 blockSize = 512;

        dim3 numBlocks;
        numBlocks.x  =  (leafSize + blockSize - 1) / blockSize;
        numBlocks.y  =  1;
        numBlocks.z  =  1;

        if (leafSize) {
            UpdatePartitionAfterSplitImpl<<<numBlocks, blockSize, 0, stream>>>(leftLeaf, rightLeaf,  sortedFlag, parts, partsCpu);
        }
    }


    /*
     * blockIdx.x * gridDim.x + threadIdx.x is index in leaf
     * blockIdx.y is part number
     * this is not time critical kernel, so we make for several blocks per SM for each leaf and just skip computations if necessary
     */
    template <int N, int BlockSize>
    __global__ void SplitAndMakeSequenceInLeavesImpl(const ui32* compressedIndex,
                                                     const ui32* loadIndices,
                                                     const TDataPartition* parts,
                                                     const ui32* leafIds,
                                                     const TCFeature* splitFeatures,
                                                     const ui32* splitBins,
                                                     bool* splitFlags,
                                                     ui32* indices) {

        const ui32 leafId = leafIds[blockIdx.y];

        TDataPartition part = Ldg(parts + leafId);
        const i32 size = part.Size;
        const i32 offset = part.Offset;

        loadIndices += offset;

        indices += offset;
        splitFlags += offset;

        int i = blockIdx.x * BlockSize * N + threadIdx.x;

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
            ui32 loadIndex[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (i + k * BlockSize < size) {
                    loadIndex[k] = loadIndices ? __ldg(loadIndices + i + k * BlockSize) : i + k * BlockSize;
                }
            }
            ui32 featureVal[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (i + k * BlockSize < size) {
                    featureVal[k] = __ldg(compressedIndex + loadIndex[k]) & mask;
                }
            }

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (i + k * BlockSize < size) {
                    WriteThrough(indices + i + k * BlockSize, static_cast<ui32>(i + k * BlockSize));
                }
            }

            bool split[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (i + k * BlockSize < size) {
                    split[k] = (oneHot ? (featureVal[k] == value) : featureVal[k] > value);
                }
            }

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                if (i + k * BlockSize < size) {
                    WriteThrough(splitFlags + i + k * BlockSize, split[k]);
                }
            }
            i += N * BlockSize * gridDim.x;
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
            const int N = 4;

            dim3 numBlocks;
            numBlocks.x  =  (leavesCount > 4 ? 2 : 4) * TArchProps::SMCount();
            numBlocks.y  =  leavesCount;
            numBlocks.z  =  1;

            SplitAndMakeSequenceInLeavesImpl<N, blockSize><<<numBlocks, blockSize, 0, stream>>>(compressedIndex, loadIndices, parts, leafIds, splitFeatures, splitBins, splitFlags, indices);
        }
    }

    template <int N, int BlockSize>
    __global__ void SplitAndMakeSequenceInSingleLeafImpl(const ui32* compressedIndex,
                                                         const ui32* loadIndices,
                                                         const TDataPartition* parts,
                                                         const ui32 leafId,
                                                         const TCFeature feature,
                                                         const ui32 binIdx,
                                                         bool* splitFlags,
                                                         ui32* indices) {

        TDataPartition part = Ldg(parts + leafId);
        const i32 size = part.Size;
        const i32 offset = part.Offset;

        loadIndices += offset;

        const int i = blockIdx.x * BlockSize * N + threadIdx.x;

        const ui32 value = binIdx << feature.Shift;
        const ui32 mask = feature.Mask << feature.Shift;
        const bool oneHot = feature.OneHotFeature;
        compressedIndex += feature.Offset;

        ui32 loadIndex[N];
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            if (i + k * BlockSize < size) {
                loadIndex[k] = __ldg(loadIndices + i + k * BlockSize);
            }
        }
        ui32 featureVal[N];

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            if (i + k * BlockSize < size) {
                featureVal[k] = __ldg(compressedIndex + loadIndex[k]) & mask;
            }
        }

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            if (i + k * BlockSize < size) {
                WriteThrough(indices + i + k * BlockSize, static_cast<ui32>(i + k * BlockSize));
            }
        }

        bool split[N];
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            if (i + k * BlockSize < size) {
                split[k] = (oneHot ? (featureVal[k] == value) : featureVal[k] > value);
            }
        }

        #pragma unroll
        for (int k = 0; k < N; ++k) {
            if (i + k * BlockSize < size) {
                WriteThrough(splitFlags + i + k * BlockSize, split[k]);
            }
        }
    }

    void SplitAndMakeSequenceInLeaf(const ui32* compressedIndex,
                                    const ui32* loadIndices,
                                    const TDataPartition* parts,
                                    ui32 leafId,
                                    ui32 leafSize,
                                    TCFeature splitFeature,
                                    ui32 splitBin,
                                    bool* splitFlags,
                                    ui32* indices,
                                    TCudaStream stream) {
        const ui32 blockSize = 256;
        const int N = 2;

        dim3 numBlocks;
        numBlocks.x  =  (leafSize + blockSize * N -  1) / (blockSize * N);
        numBlocks.y  =  1;
        numBlocks.z  =  1;
        if (numBlocks.x) {
            SplitAndMakeSequenceInSingleLeafImpl<N, blockSize> << < numBlocks, blockSize, 0, stream >>
                > (compressedIndex, loadIndices, parts, leafId, splitFeature, splitBin, splitFlags, indices);
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

            if (part.Size) {
                cudaError_t
                error = cub::DeviceRadixSort::SortPairs < bool, ui32 > ((void*) context.TempStorage.Get(),
                    context.TempStorageSizes[i],
                    flagsSrc,
                    flagsDst,
                    indicesSrc,
                    indicesDst,
                    (int) part.Size,
                    0,
                    1,
                    stream);
                CUDA_SAFE_CALL(error);
            }
        }
    }

//
    void SortWithoutCub(ui32 leafId, const TDataPartition* partsCpu, TSplitPointsContext& context, TCudaStream stream) {
        TDataPartition part = partsCpu[leafId];
        if (part.Size) {

            const bool* flagsSrc = context.TempFlags.Get();
            bool* flagsDst = context.Flags.Get();
            const ui32* indicesSrc = context.TempIndices.Get();
            ui32* indicesDst = context.Indices.Get();
            char* tempStorage = context.TempStorage.Get();

            const ui64 tempOffsetsSize = sizeof(int) * part.Size;
            {
                using TInput = TScanBitIterator<bool>;
                TInput inputIter(context.TempFlags.Get(), 0);

                ui64 tempStorageSize = tempStorage ? context.TempStorageSizes[0] - tempOffsetsSize : 0;
                auto scanTmp = tempStorage ? (void*)(tempStorage + tempOffsetsSize) : nullptr;
                cudaError_t err = cub::DeviceScan::ExclusiveSum < TInput, int*> (scanTmp,
                                                               tempStorageSize,
                                                               inputIter,
                                                               (int*)tempStorage,
                                                               part.Size,
                                                               stream);
                if (!tempStorage) {
                    context.TempStorageSizes[0] = tempStorageSize + tempOffsetsSize;
                }
                CUDA_SAFE_CALL(err);
            }
            if (tempStorage) {

                const int blockSize = 512;
                const int N = 1;
                const int numBlocks = (part.Size + (N * blockSize) - 1) / (N * blockSize);
                ReorderOneBitImpl<bool, ui32, N, blockSize> << < numBlocks, blockSize, 0, stream >> > (
                    flagsSrc,
                        indicesSrc,
                        (int*) tempStorage,
                        0,
                        flagsDst,
                        indicesDst,
                        part.Size);
            }
        }
    }

    ui32 FastSortSize() {
        return 500000;
    }

    void SortByFlagsInLeaf(ui32 leafId,
                           const TDataPartition* partsCpu,
                           TSplitPointsContext& context,
                           TCudaStream stream) {
        /*
         * Sort leaves by flags
         */
        TDataPartition part = partsCpu[leafId];
        if (part.Size > FastSortSize()) {

            const bool* flagsSrc = context.TempFlags.Get();
            bool* flagsDst = context.Flags.Get();
            const ui32* indicesSrc = context.TempIndices.Get();
            ui32* indicesDst = context.Indices.Get();

            cudaError_t error = cub::DeviceRadixSort::SortPairs < bool, ui32 > ((void*) context.TempStorage.Get(),
                context.TempStorageSizes[0],
                flagsSrc,
                flagsDst,
                indicesSrc,
                indicesDst,
                (int) part.Size,
                0,
                1,
                stream);
            CUDA_SAFE_CALL(error);
        } else {
            SortWithoutCub(leafId, partsCpu, context, stream);
        }
    }


    #define TEMPL_INST(Type)\
    template void CopyInLeaves<Type>(const ui32* leaves, const ui32 leavesCount, const TDataPartition* parts, const Type *src, Type *dst, ui32 numCopies, ui32 lineSize, TCudaStream stream);\
    template void GatherInLeaves<Type>(const ui32* leaves, const ui32 leavesCount, const TDataPartition* parts, const Type* src, const ui32* map, Type *dst, ui32 numStats, ui32 lineSize, TCudaStream stream);\
    template void GatherLeaf<Type>(const ui32 leaf, const ui32 size, const TDataPartition* parts, const Type* src, const ui32* map, Type *dst, ui32 numStats, ui32 lineSize, TCudaStream stream);\
    template void CopyLeaf<Type>(const ui32 leaf, const ui32 size, const TDataPartition* parts, const Type *src, Type *dst, ui32 numCopies, ui32 lineSize, TCudaStream stream);


    TEMPL_INST(ui32)
    TEMPL_INST(float)

    #undef TEMPL_INST


    template void GatherInplaceLeqSize<12288>(const ui32* leaf, ui32 leavesCount,
                                     const TDataPartition* parts,
                                     const ui32* map,
                                     float* stats, ui32 statCount,
                                     ui64 lineSize,
                                     ui32* indices,
                                     TCudaStream stream);

    template void GatherInplaceLeqSize<6144>(const ui32* leaf, ui32 leavesCount,
                                    const TDataPartition* parts,
                                    const ui32* map,
                                    float* stats, ui32 statCount,
                                    ui64 lineSize,
                                    ui32* indices,
                                    TCudaStream stream);

    template void GatherInplaceLeqSize<3072>(const ui32* leaf, ui32 leavesCount,
                                    const TDataPartition* parts,
                                    const ui32* map,
                                    float* stats, ui32 statCount,
                                    ui64 lineSize,
                                    ui32* indices,
                                    TCudaStream stream);

    template void GatherInplaceLeqSize<1024>(const ui32* leaf, ui32 leavesCount,
                                             const TDataPartition* parts,
                                             const ui32* map,
                                             float* stats, ui32 statCount,
                                             ui64 lineSize,
                                             ui32* indices,
                                             TCudaStream stream);

    #define INPLACE_SINGLE_LEAF(Size)\
    template void GatherInplaceSingleLeaf<Size>(const ui32 leaf, \
                                                 const TDataPartition* parts,\
                                                 const ui32* map,\
                                                 float* stats, ui32 statCount,\
                                                 ui64 lineSize,\
                                                 ui32* indices,\
                                                 TCudaStream stream);

    INPLACE_SINGLE_LEAF(6144)
    INPLACE_SINGLE_LEAF(12288)
    INPLACE_SINGLE_LEAF(3072)
    INPLACE_SINGLE_LEAF(1024)

}


