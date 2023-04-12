#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/scan.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/cast.h>

namespace NKernelHost {
    template <typename T, typename TOut>
    class TScanVectorKernel: public TKernelBase<NKernel::TScanKernelContext<T, TOut>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<TOut> Output;
        bool Inclusive;
        bool IsNonNegativeSegmentedScan;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T, TOut>;
        Y_SAVELOAD_DEFINE(Input, Output, Inclusive, IsNonNegativeSegmentedScan);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->NumParts = NKernel::ScanVectorTempSize<T, TOut>(SafeIntegerCast<ui32>(Input.Size()), Inclusive);
            context->PartResults = memoryManager.Allocate<char>(context->NumParts);
            return context;
        }

        TScanVectorKernel() = default;

        TScanVectorKernel(TCudaBufferPtr<const T> input,
                          TCudaBufferPtr<TOut> output,
                          bool inclusive,
                          bool nonNegativeSegmented)
            : Input(input)
            , Output(output)
            , Inclusive(inclusive)
            , IsNonNegativeSegmentedScan(nonNegativeSegmented)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            if (IsNonNegativeSegmentedScan) {
                CB_ENSURE(Inclusive, "Error: fast exclusive scan currently not working via simple operator transformation");
                CUDA_SAFE_CALL((NKernel::SegmentedScanNonNegativeVector<T, TOut>(Input.Get(), Output.Get(),
                                                                          SafeIntegerCast<ui32>(Input.Size()), Inclusive,
                                                                          context, stream.GetStream())));
            } else {
                //scan is done by cub.
                CUDA_SAFE_CALL((NKernel::ScanVector<T, TOut>(Input.Get(), Output.Get(),
                                                      SafeIntegerCast<ui32>(Input.Size()),
                                                      Inclusive, context,
                                                      stream.GetStream())));
            }
        }
    };

    template <typename T>
    class TNonNegativeSegmentedScanAndScatterVectorKernel: public TKernelBase<NKernel::TScanKernelContext<T, T>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<T> Output;
        bool Inclusive;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T, T>;
        Y_SAVELOAD_DEFINE(Input, Indices, Output, Inclusive);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->NumParts = NKernel::ScanVectorTempSize<T, T>(SafeIntegerCast<ui32>(Input.Size()), Inclusive);
            context->PartResults = memoryManager.Allocate<char>(context->NumParts);
            return context;
        }

        TNonNegativeSegmentedScanAndScatterVectorKernel() = default;

        TNonNegativeSegmentedScanAndScatterVectorKernel(TCudaBufferPtr<const T> input,
                                                        TCudaBufferPtr<const ui32> indices,
                                                        TCudaBufferPtr<T> output,
                                                        bool inclusive)
            : Input(input)
            , Indices(indices)
            , Output(output)
            , Inclusive(inclusive)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            NKernel::SegmentedScanAndScatterNonNegativeVector<T>(Input.Get(), Indices.Get(), Output.Get(),
                                                                 SafeIntegerCast<ui32>(Input.Size()), Inclusive,
                                                                 context, stream.GetStream());
        }
    };
}

template <typename T, typename TOut, class TMapping>
inline void ScanVector(const TCudaBuffer<T, TMapping>& input, TCudaBuffer<TOut, TMapping>& output,
                       bool inclusive = false, ui32 streamId = 0) {
    using TKernel = NKernelHost::TScanVectorKernel<T, TOut>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, output, inclusive, false);
}

//TODO(noxoomo): we should be able to run exclusive also
template <typename T, class TMapping>
inline void InclusiveSegmentedScanNonNegativeVector(const TCudaBuffer<T, TMapping>& input,
                                                    TCudaBuffer<T, TMapping>& output,
                                                    ui32 streamId = 0) {
    using TKernel = NKernelHost::TScanVectorKernel<T, T>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, output, true, true);
}

//Not the safest way…
template <typename T, class TMapping, class TUi32 = ui32>
inline void SegmentedScanAndScatterNonNegativeVector(const TCudaBuffer<T, TMapping>& inputWithSignMasks,
                                                     const TCudaBuffer<TUi32, TMapping>& indices,
                                                     TCudaBuffer<T, TMapping>& output,
                                                     bool inclusive = false,
                                                     ui32 streamId = 0) {
    using TKernel = NKernelHost::TNonNegativeSegmentedScanAndScatterVectorKernel<T>;
    LaunchKernels<TKernel>(inputWithSignMasks.NonEmptyDevices(), streamId, inputWithSignMasks, indices, output, inclusive);
}
