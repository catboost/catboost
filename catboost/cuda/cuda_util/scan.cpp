#include "scan.h"

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/scan.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/va_args.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TKernelBase;

// ScanVector

namespace {
    template <typename T>
    class TScanVectorKernel: public TKernelBase<NKernel::TScanKernelContext<T>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<T> Output;
        bool Inclusive;
        bool IsNonNegativeSegmentedScan;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T>;
        Y_SAVELOAD_DEFINE(Input, Output, Inclusive, IsNonNegativeSegmentedScan);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->NumParts = NKernel::ScanVectorTempSize<T>((ui32)Input.Size(), Inclusive);
            context->PartResults = memoryManager.Allocate<char>(context->NumParts);
            return context;
        }

        TScanVectorKernel() = default;

        TScanVectorKernel(TCudaBufferPtr<const T> input,
                          TCudaBufferPtr<T> output,
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
                CUDA_SAFE_CALL(NKernel::SegmentedScanNonNegativeVector<T>(Input.Get(), Output.Get(),
                                                                          (ui32)Input.Size(), Inclusive,
                                                                          context, stream.GetStream()));
            } else {
                //scan is done by cub.
                CUDA_SAFE_CALL(NKernel::ScanVector<T>(Input.Get(), Output.Get(),
                                                      (ui32)Input.Size(),
                                                      Inclusive, context,
                                                      stream.GetStream()));
            }
        }
    };
}

template <typename T, typename TMapping>
static void ScanVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive,
    ui32 streamId) {
    using TKernel = TScanVectorKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, output, inclusive, false);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                    \
    template <>                                                \
    void ScanVector<T, TMapping>(                              \
        const TCudaBuffer<T, TMapping>& input,                 \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& output, \
        bool inclusive,                                        \
        ui32 streamId) {                                       \
        ::ScanVectorImpl(input, output, inclusive, streamId);  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (double, TMirrorMapping),
    (ui32, TMirrorMapping),
    (int, TMirrorMapping),
    (float, TSingleMapping),
    (double, TSingleMapping),
    (ui32, TSingleMapping),
    (int, TSingleMapping),
    (float, TStripeMapping),
    (double, TStripeMapping),
    (ui32, TStripeMapping),
    (int, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// InclusiveSegmentedScanNonNegativeVector

template <typename T, typename TMapping>
static void InclusiveSegmentedScanNonNegativeVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    ui32 streamId) {
    using TKernel = TScanVectorKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, output, true, true);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                     \
    template <>                                                                 \
    void InclusiveSegmentedScanNonNegativeVector<T, TMapping>(                  \
        const TCudaBuffer<T, TMapping>& input,                                  \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& output,                  \
        ui32 streamId) {                                                        \
        ::InclusiveSegmentedScanNonNegativeVectorImpl(input, output, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (double, TMirrorMapping),
    (ui32, TMirrorMapping),
    (int, TMirrorMapping),
    (float, TSingleMapping),
    (double, TSingleMapping),
    (ui32, TSingleMapping),
    (int, TSingleMapping),
    (float, TStripeMapping),
    (double, TStripeMapping),
    (ui32, TStripeMapping),
    (int, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace {
    template <typename T>
    class TNonNegativeSegmentedScanAndScatterVectorKernel: public TKernelBase<NKernel::TScanKernelContext<T>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<T> Output;
        bool Inclusive;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T>;
        Y_SAVELOAD_DEFINE(Input, Indices, Output, Inclusive);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->NumParts = NKernel::ScanVectorTempSize<T>((ui32)Input.Size(), Inclusive);
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
                                                                 (ui32)Input.Size(), Inclusive,
                                                                 context, stream.GetStream());
        }
    };
}

template <typename T, typename TMapping, typename TUi32>
static void SegmentedScanAndScatterNonNegativeVectorImpl(
    const TCudaBuffer<T, TMapping>& inputWithSignMasks,
    const TCudaBuffer<TUi32, TMapping>& indices,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive,
    ui32 streamId) {
    using TKernel = TNonNegativeSegmentedScanAndScatterVectorKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(inputWithSignMasks.NonEmptyDevices(), streamId, inputWithSignMasks, indices, output, inclusive);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, TUi32)                                                                \
    template <>                                                                                                   \
    void SegmentedScanAndScatterNonNegativeVector<T, TMapping, TUi32>(                                            \
        const TCudaBuffer<T, TMapping>& inputWithSignMasks,                                                       \
        const TCudaBuffer<TUi32, TMapping>& indices,                                                              \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& output,                                                    \
        bool inclusive,                                                                                           \
        ui32 streamId) {                                                                                          \
        ::SegmentedScanAndScatterNonNegativeVectorImpl(inputWithSignMasks, indices, output, inclusive, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (float, TSingleMapping, ui32),
    (float, TStripeMapping, ui32),
    (float, TMirrorMapping, const ui32),
    (float, TSingleMapping, const ui32),
    (float, TStripeMapping, const ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// register kernels

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAD0001, TScanVectorKernel, float);
    REGISTER_KERNEL_TEMPLATE(0xAD0002, TScanVectorKernel, double);
    REGISTER_KERNEL_TEMPLATE(0xAD0003, TScanVectorKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0xAD0004, TScanVectorKernel, int);

    REGISTER_KERNEL_TEMPLATE(0xAD0005, TNonNegativeSegmentedScanAndScatterVectorKernel, float);
}
