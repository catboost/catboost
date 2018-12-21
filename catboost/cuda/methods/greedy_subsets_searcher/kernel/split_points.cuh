#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
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

    void UpdatePartitionsAfterSplit(const ui32* leftLeafs,
                                    const ui32* rightLeafs,
                                    ui32 leavesCount,
                                    const bool* sortedFlag,
                                    TDataPartition* parts,
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

}
