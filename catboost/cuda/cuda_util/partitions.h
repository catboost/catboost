#pragma once

#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/partitions.cuh>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    class TUpdatePartitionDimensionsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> SortedBins;
        TCudaBufferPtr<TDataPartition> Parts;

    public:
        TUpdatePartitionDimensionsKernel() = default;

        TUpdatePartitionDimensionsKernel(TCudaBufferPtr<const ui32> sortedBins,
                                         TCudaBufferPtr<TDataPartition> parts)
            : SortedBins(sortedBins)
            , Parts(parts)
        {
        }

        SAVELOAD(Parts, SortedBins);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Parts.Size() < (1ULL << 32));
            CB_ENSURE(SortedBins.Size() < (1ULL << 32));

            NKernel::UpdatePartitionDimensions(Parts.Get(), (ui32)Parts.Size(), SortedBins.Get(),
                                               (ui32)SortedBins.Size(), stream.GetStream());
        }
    };

    class TUpdatePartitionOffsetsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> SortedBins;
        TCudaBufferPtr<ui32> Offsets;

    public:
        TUpdatePartitionOffsetsKernel() = default;

        TUpdatePartitionOffsetsKernel(TCudaBufferPtr<const ui32> sortedBins,
                                      TCudaBufferPtr<ui32> offsets)
            : SortedBins(sortedBins)
            , Offsets(offsets)
        {
        }

        SAVELOAD(Offsets, SortedBins);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Offsets.Size() < (1ULL << 32));
            CB_ENSURE(SortedBins.Size() < (1ULL << 32));

            NKernel::UpdatePartitionOffsets(Offsets.Get(), (ui32)Offsets.Size(), SortedBins.Get(),
                                            (ui32)SortedBins.Size(), stream.GetStream());
        }
    };
}

template <class TMapping>
inline void UpdatePartitionDimensions(const TCudaBuffer<ui32, TMapping>& sortedBins,
                                      TCudaBuffer<TDataPartition, TMapping>& parts,
                                      ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdatePartitionDimensionsKernel;
    LaunchKernels<TKernel>(sortedBins.NonEmptyDevices(), stream, sortedBins, parts);
}

template <class TMapping>
inline void UpdatePartitionOffsets(const TCudaBuffer<ui32, TMapping>& sortedBins,
                                   TCudaBuffer<ui32, TMapping>& offsets,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdatePartitionOffsetsKernel;
    LaunchKernels<TKernel>(sortedBins.NonEmptyDevices(), stream, sortedBins, offsets);
}

//
//template <class TMapping>
//inline void WriteInitPartitions(const TMapping& objectsMapping, TCudaBuffer<TDataPartition, TMapping>& parts) {
//
//    const int devId = GetSingleDevId(objectsMapping);
//    const TDataPartition initialPart(0, objectsMapping->SizeAt(devId));
//    TVector<TDataPartition> partitioningInitial;
//    partitioningInitial.push_back(initialPart);
//    parts.Write(partitioningInitial);
//}
