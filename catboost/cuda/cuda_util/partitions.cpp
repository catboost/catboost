#include "partitions.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/partitions.cuh>
#include <catboost/libs/helpers/exception.h>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TStatelessKernel;

// UpdatePartitionDimensions

namespace {
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
            NKernel::UpdatePartitionDimensions(Parts.Get(), SafeIntegerCast<ui32>(Parts.Size()), SortedBins.Get(),
                                               SafeIntegerCast<ui32>(SortedBins.Size()), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void UpdatePartitionDimensionsImpl(
    const TCudaBuffer<ui32, TMapping>& sortedBins,
    TCudaBuffer<TDataPartition, TMapping>& parts,
    ui32 stream) {
    using TKernel = TUpdatePartitionDimensionsKernel;
    LaunchKernels<TKernel>(sortedBins.NonEmptyDevices(), stream, sortedBins, parts);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                            \
    template <>                                                     \
    void UpdatePartitionDimensions<TMapping>(                       \
        const TCudaBuffer<ui32, TMapping>& sortedBins,              \
        TCudaBuffer<TDataPartition, TMapping>& parts,               \
        ui32 stream) {                                              \
        ::UpdatePartitionDimensionsImpl(sortedBins, parts, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// UpdatePartitionOffsets

namespace {
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
}

template <typename TMapping>
static void UpdatePartitionOffsetsImpl(
    const TCudaBuffer<ui32, TMapping>& sortedBins,
    TCudaBuffer<ui32, TMapping>& offsets,
    ui32 stream) {
    using TKernel = TUpdatePartitionOffsetsKernel;
    LaunchKernels<TKernel>(offsets.NonEmptyDevices(), stream, sortedBins, offsets);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                           \
    template <>                                                    \
    void UpdatePartitionOffsets<TMapping>(                         \
        const TCudaBuffer<ui32, TMapping>& sortedBins,             \
        TCudaBuffer<ui32, TMapping>& offsets,                      \
        ui32 stream) {                                             \
        ::UpdatePartitionOffsetsImpl(sortedBins, offsets, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// ComputeSegmentSizes

namespace {
    template <EPtrType PtrType>
    class TComputeSegmentSizesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Offsets;
        using TDstPtr = TDeviceBuffer<float, PtrType>;
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

template <typename TMapping, typename TUi32, EPtrType DstPtr>
static void ComputeSegmentSizesImpl(
    const TCudaBuffer<TUi32, TMapping>& offsets,
    TCudaBuffer<float, TMapping, DstPtr>& dst,
    ui32 stream) {
    using TKernel = TComputeSegmentSizesKernel<DstPtr>;
    LaunchKernels<TKernel>(offsets.NonEmptyDevices(), stream, offsets, dst);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(TMapping, TUi32, DstPtr)  \
    template <>                                          \
    void ComputeSegmentSizes<TMapping, TUi32, DstPtr>(   \
        const TCudaBuffer<TUi32, TMapping>& offsets,     \
        TCudaBuffer<float, TMapping, DstPtr>& dst,       \
        ui32 stream) {                                   \
        ::ComputeSegmentSizesImpl(offsets, dst, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (TMirrorMapping, ui32, EPtrType::CudaHost));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// register kernels

namespace NCudaLib {
    REGISTER_KERNEL(0xAAA001, TUpdatePartitionDimensionsKernel);
    REGISTER_KERNEL(0xAAA002, TUpdatePartitionOffsetsKernel);
    REGISTER_KERNEL_TEMPLATE(0xAAA003, TComputeSegmentSizesKernel, EPtrType::CudaDevice);
    REGISTER_KERNEL_TEMPLATE(0xAAA004, TComputeSegmentSizesKernel, EPtrType::CudaHost);
    REGISTER_KERNEL_TEMPLATE(0xAAA005, TComputeSegmentSizesKernel, EPtrType::Host);
}
