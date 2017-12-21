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

        Y_SAVELOAD_DEFINE(Parts, SortedBins);

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

        Y_SAVELOAD_DEFINE(Offsets, SortedBins);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Offsets.Size() < (1ULL << 32));
            CB_ENSURE(SortedBins.Size() < (1ULL << 32));

            NKernel::UpdatePartitionOffsets(Offsets.Get(), (ui32)Offsets.Size(), SortedBins.Get(),
                                            (ui32)SortedBins.Size(), stream.GetStream());
        }
    };

    template <NCudaLib::EPtrType PtrType>
    class TComputeSegmentSizesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Offsets;
        using TDstPtr = TDeviceBuffer<float, TFixedSizesObjectsMeta, PtrType>;
        TDstPtr Dst;

    public:
        TComputeSegmentSizesKernel() = default;

        TComputeSegmentSizesKernel(TCudaBufferPtr<const ui32> offsets,
                                   TDstPtr dst)
            : Offsets(offsets)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Offsets, Dst);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Offsets.Size() < (1ULL << 32));

            NKernel::ComputeSegmentSizes(Offsets.Get(), (ui32)(Dst.Size() + 1), Dst.Get(), stream.GetStream());
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

template <class TMapping, class TUi32, NCudaLib::EPtrType DstPtr>
inline void ComputeSegmentSizes(const TCudaBuffer<TUi32, TMapping>& offsets,
                                TCudaBuffer<float, TMapping, DstPtr>& dst,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeSegmentSizesKernel<DstPtr>;
    LaunchKernels<TKernel>(offsets.NonEmptyDevices(), stream, offsets, dst);
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
