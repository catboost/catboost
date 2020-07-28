#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

    struct TSplitPointsContext : public IKernelContext {

        TSplitPointsContext() {
        }

        TDevicePointer<char> TempStorage;
        TVector<size_t> TempStorageSizes;

        TDevicePointer<bool> TempFlags;
        TDevicePointer<bool> Flags;

        TDevicePointer<ui32> TempIndices;
        TDevicePointer<ui32> Indices;

        TDevicePointer<double> UpdatePropsTempBuffer;
        ui32 UpdatePropsTempBufferSize = 0;
    };

    void SortByFlagsInLeaves(const ui32* leavesToSplit, ui32 leafCount,
                             const TDataPartition* partsCpu,
                             TSplitPointsContext& context,
                             TCudaStream stream);


    void SortByFlagsInLeaf(ui32 leafId,const TDataPartition* partsCpu,
                           TSplitPointsContext& context,
                           TCudaStream stream);

    void UpdatePartitionsAfterSplit(const ui32* leftLeafs,
                                    const ui32* rightLeafs,
                                    ui32 leavesCount,
                                    const bool* sortedFlag,
                                    TDataPartition* parts,
                                    TDataPartition* partsCpu,
                                    TCudaStream stream);

    void UpdatePartitionAfterSplit(const ui32 leftLeaf,
                                    const ui32 rightLeaf,
                                    ui32 leafSize,
                                    const bool* sortedFlag,
                                    TDataPartition* parts,
                                    TDataPartition* partsCpu,
                                    TCudaStream stream);

    void SplitAndMakeSequenceInLeaves(const ui32* compressedIndex,
                                      const ui32* loadIndices,
                                      const TDataPartition* parts,
                                      const ui32* leafIds, ui32 leavesCount,
                                      const TCFeature* splitFeatures,
                                      const ui32* splitBins,
                                      bool* splitFlags,
                                      ui32* indices,
                                      TCudaStream stream);

    void SplitAndMakeSequenceInLeaf(const ui32* compressedIndex,
                                    const ui32* loadIndices,
                                    const TDataPartition* parts,
                                    ui32 leafId,
                                    ui32 leafSize,
                                    TCFeature splitFeature,
                                    ui32 splitBin,
                                    bool* splitFlags,
                                    ui32* indices,
                                    TCudaStream stream);



    template <class T>
    void CopyInLeaves(const ui32* leaves,
                      const ui32 leavesCount,
                      const TDataPartition* parts,
                      const T *src,
                      T *dst,
                      ui32 numStats,
                      ui32 lineSize,
                      TCudaStream stream);

    template <class T>
    void GatherInLeaves(const ui32* leaves, const ui32 leavesCount,
                        const TDataPartition* parts,
                        const T *src,
                        const ui32* map,
                        T *dst,
                        ui32 numStats,
                        ui32 lineSize,
                        TCudaStream stream);


    template <int Size>
    void GatherInplaceLeqSize(const ui32* leaf, ui32 leavesCount,
                              const TDataPartition* parts,
                              const ui32* map,
                              float* stats, ui32 statCount,
                              ui64 lineSize,
                              ui32* indices,
                              TCudaStream stream);

    template <int Size>
    void GatherInplaceSingleLeaf(const ui32 leaf,
                                 const TDataPartition* parts,
                                 const ui32* map,
                                 float* stats, ui32 statCount,
                                 ui64 lineSize,
                                 ui32* indices,
                                 TCudaStream stream);

    template <class T>
    void GatherLeaf(const ui32 leafId, const ui32 leafSize,
                    const TDataPartition* parts,
                    const T* src,
                    const ui32* map,
                    T* dst,
                    ui32 numStats,
                    ui32 lineSize,
                    TCudaStream stream);

    template <class T>
    void CopyLeaf(const ui32 leafId, const ui32 leafSize,
                  const TDataPartition* parts,
                  const T* src,
                  T* dst,
                  ui32 numStats,
                  ui32 lineSize,
                  TCudaStream stream);


    ui32 FastSortSize();


}
