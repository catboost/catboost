#include "segmented_scan.h"

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/segmented_scan.cuh>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TKernelBase;

// SegmentedScanVector

namespace {
    template <class T>
    class TSegmentedScanKernel: public TKernelBase<NKernel::TScanKernelContext<T, T>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const ui32> Flags;
        ui32 FlagMask;
        TCudaBufferPtr<T> Output;
        bool Inclusive;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T, T>;

        TSegmentedScanKernel() = default;

        TSegmentedScanKernel(TCudaBufferPtr<const T> input,
                             TCudaBufferPtr<const ui32> flags,
                             ui32 flagMask,
                             TCudaBufferPtr<T> output,
                             bool inclusive)
            : Input(input)
            , Flags(flags)
            , FlagMask(flagMask)
            , Output(output)
            , Inclusive(inclusive)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Flags, Output, FlagMask, Inclusive);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            CB_ENSURE(Input.Size() == Flags.Size(), TStringBuilder() << "Input size #" << Input.Size() << " ≠ flags size #" << Flags.Size());
            CB_ENSURE(Input.Size() == Output.Size());

            auto context = MakeHolder<TKernelContext>();
            context->NumParts = NKernel::SegmentedScanVectorTempSize<T>((ui32)Input.Size(), Inclusive);
            context->PartResults = memoryManager.Allocate<char>(context->NumParts);
            return context;
        }

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            using namespace NKernel;
            CUDA_SAFE_CALL(SegmentedScanCub(Input.Get(), Flags.Get(), FlagMask, Output.Get(), (ui32)Input.Size(), Inclusive, context, stream.GetStream()));
        }
    };
}

template <typename T, typename TMapping, typename TUi32>
static void SegmentedScanVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    const TCudaBuffer<TUi32, TMapping>& flags,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive,
    ui32 flagMask,
    ui32 streamId) {
    using TKernel = TSegmentedScanKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, flags, flagMask, output, inclusive);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, TUi32)                                      \
    template <>                                                                         \
    void SegmentedScanVector<T, TMapping, TUi32>(                                       \
        const TCudaBuffer<T, TMapping>& input,                                          \
        const TCudaBuffer<TUi32, TMapping>& flags,                                      \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& output,                          \
        bool inclusive,                                                                 \
        ui32 flagMask,                                                                  \
        ui32 streamId) {                                                                \
        ::SegmentedScanVectorImpl(input, flags, output, inclusive, flagMask, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, ui32),
    (int, TMirrorMapping, ui32),
    (ui32, TMirrorMapping, ui32),
    (float, TMirrorMapping, const ui32),
    (int, TMirrorMapping, const ui32),
    (ui32, TMirrorMapping, const ui32),
    (float, TSingleMapping, ui32),
    (int, TSingleMapping, ui32),
    (ui32, TSingleMapping, ui32),
    (float, TSingleMapping, const ui32),
    (int, TSingleMapping, const ui32),
    (ui32, TSingleMapping, const ui32),
    (float, TStripeMapping, ui32),
    (int, TStripeMapping, ui32),
    (ui32, TStripeMapping, ui32),
    (float, TStripeMapping, const ui32),
    (int, TStripeMapping, const ui32),
    (ui32, TStripeMapping, const ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAC0001, TSegmentedScanKernel, float);
    REGISTER_KERNEL_TEMPLATE(0xAC0002, TSegmentedScanKernel, int);
    REGISTER_KERNEL_TEMPLATE(0xAC0003, TSegmentedScanKernel, ui32);
}
