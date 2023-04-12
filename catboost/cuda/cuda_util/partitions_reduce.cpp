#include "partitions_reduce.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>

#include <util/generic/cast.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TKernelBase;
using NKernelHost::TStatelessKernel;

// ComputePartitionStats

namespace {
    class TReducePartitionsKernel: public TKernelBase<NKernel::TPartStatsContext> {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<const TDataPartition> Partitions;
        TCudaBufferPtr<const ui32> PartIds;
        TCudaBufferPtr<double> Output;

    public:
        TReducePartitionsKernel() = default;

        TReducePartitionsKernel(TCudaBufferPtr<const float> input,
                                TCudaBufferPtr<const TDataPartition> partitions,
                                TCudaBufferPtr<const ui32> partIds,
                                TCudaBufferPtr<double> output)
            : Input(input)
            , Partitions(partitions)
            , PartIds(partIds)
            , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Partitions, PartIds, Output);

        using TKernelContext = NKernel::TPartStatsContext;

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->TempVarsCount = NKernel::GetTempVarsCount(Input.GetColumnCount(), SafeIntegerCast<ui32>(PartIds.Size()));
            context->PartResults = memoryManager.Allocate<double>(context->TempVarsCount);
            return context;
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            NKernel::UpdatePartitionsProps(Partitions.Get(),
                                           PartIds.Get(),
                                           SafeIntegerCast<ui32>(PartIds.Size()),
                                           Input.Get(),
                                           Input.GetColumnCount(),
                                           SafeIntegerCast<ui32>(Input.AlignedColumnSize()),
                                           context.TempVarsCount,
                                           context.PartResults.Get(),
                                           Output.Get(),
                                           stream.GetStream());
        }
    };
}

template <typename TMapping>
static void ComputePartitionStatsImpl(
    const TCudaBuffer<float, TMapping>& input,
    const TCudaBuffer<TDataPartition, TMapping>& parts,
    const TCudaBuffer<ui32, TMirrorMapping>& partIds,
    TCudaBuffer<double, TMapping>* output,
    ui32 streamId) {
    using TKernel = TReducePartitionsKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, parts, partIds, output);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                      \
    template <>                                                               \
    void ComputePartitionStats<TMapping>(                                     \
        const TCudaBuffer<float, TMapping>& input,                            \
        const TCudaBuffer<TDataPartition, TMapping>& parts,                   \
        const TCudaBuffer<ui32, TMirrorMapping>& partIds,                     \
        TCudaBuffer<double, TMapping>* output,                                \
        ui32 streamId) {                                                      \
        ::ComputePartitionStatsImpl(input, parts, partIds, output, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ComputePartitionStats

namespace {
    class TReducePartitionsWithOffsetsKernel: public TKernelBase<NKernel::TPartStatsContext> {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<const ui32> Offsets;
        TCudaBufferPtr<double> Output;

    public:
        TReducePartitionsWithOffsetsKernel() = default;

        TReducePartitionsWithOffsetsKernel(TCudaBufferPtr<const float> input,
                                           TCudaBufferPtr<const ui32> offsets,
                                           TCudaBufferPtr<double> output)
            : Input(input)
            , Offsets(offsets)
            , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Offsets, Output);

        using TKernelContext = NKernel::TPartStatsContext;

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->TempVarsCount = NKernel::GetTempVarsCount(Input.GetColumnCount(), SafeIntegerCast<ui32>(Offsets.Size()));
            context->PartResults = memoryManager.Allocate<double>(context->TempVarsCount);
            return context;
        }

        void Run(const TCudaStream& stream, const TKernelContext& context) const {
            CB_ENSURE(Input.GetColumnCount());
            CB_ENSURE(Offsets.Size() > 1);
            NKernel::UpdatePartitionsPropsForOffsets(Offsets.Get(),
                                                     SafeIntegerCast<ui32>(Offsets.Size()) - 1,
                                                     Input.Get(),
                                                     Input.GetColumnCount(),
                                                     SafeIntegerCast<ui32>(Input.AlignedColumnSize()),
                                                     context.TempVarsCount,
                                                     context.PartResults.Get(),
                                                     Output.Get(),
                                                     stream.GetStream());
        }
    };
}

template <typename TMapping, typename TFloat>
static void ComputePartitionStatsImpl(
    const TCudaBuffer<TFloat, TMapping>& input,
    const TCudaBuffer<ui32, TMapping>& offsets,
    TCudaBuffer<double, TMapping>* output,
    ui32 streamId) {
    using TKernel = TReducePartitionsWithOffsetsKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, offsets, output);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(TMapping, TFloat)                       \
    template <>                                                        \
    void ComputePartitionStats<TMapping, TFloat>(                      \
        const TCudaBuffer<TFloat, TMapping>& input,                    \
        const TCudaBuffer<ui32, TMapping>& offsets,                    \
        TCudaBuffer<double, TMapping>* output,                         \
        ui32 streamId) {                                               \
        ::ComputePartitionStatsImpl(input, offsets, output, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (TMirrorMapping, float),
    (TMirrorMapping, const float),
    (TSingleMapping, float),
    (TSingleMapping, const float),
    (TStripeMapping, float),
    (TStripeMapping, const float));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// CastCopy

namespace {
    class TCastCopyKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<double> Output;

    public:
        TCastCopyKernel() = default;

        TCastCopyKernel(TCudaBufferPtr<const float> input,
                        TCudaBufferPtr<double> output)
            : Input(input)
            , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Output);

        void Run(const TCudaStream& stream) const {
            NKernel::CopyFloatToDouble(Input.Get(), Input.Size(), Output.Get(), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void CastCopyImpl(
    const TCudaBuffer<float, TMapping>& input,
    TCudaBuffer<double, TMapping>* output,
    ui32 streamId) {
    using TKernel = TCastCopyKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, output);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)           \
    template <>                                    \
    void CastCopy<TMapping>(                       \
        const TCudaBuffer<float, TMapping>& input, \
        TCudaBuffer<double, TMapping>* output,     \
        ui32 streamId) {                           \
        ::CastCopyImpl(input, output, streamId);   \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

namespace NCudaLib {
    REGISTER_KERNEL(0xAADDD1, TReducePartitionsKernel);
    REGISTER_KERNEL(0xAADDD2, TReducePartitionsWithOffsetsKernel);
    REGISTER_KERNEL(0xAADDD3, TCastCopyKernel);
}
