#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/segmented_scan.cuh>

namespace NKernelHost {
    template <class T>
    class TSegmentedScanKernel: public TKernelBase<NKernel::TScanKernelContext<T>, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const ui32> Flags;
        ui32 FlagMask;
        TCudaBufferPtr<T> Output;
        bool Inclusive;

    public:
        using TKernelContext = NKernel::TScanKernelContext<T>;

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
            context->PartResults = memoryManager.Allocate<char>(context->NumParts).Get();
            return context;
        }

        void Run(const TCudaStream& stream,
                 TKernelContext& context) const {
            using namespace NKernel;
            CUDA_SAFE_CALL(SegmentedScanCub(Input.Get(), Flags.Get(), FlagMask, Output.Get(), (ui32)Input.Size(), Inclusive, context, stream.GetStream()));
        }
    };
}

template <typename T, class TMapping, class TUi32>
inline void SegmentedScanVector(const TCudaBuffer<T, TMapping>& input,
                                const TCudaBuffer<TUi32, TMapping>& flags,
                                TCudaBuffer<typename std::remove_const<T>::type, TMapping>& output,
                                bool inclusive = false,
                                ui32 flagMask = 1,
                                ui32 streamId = 0) {
    using TNonConstInputType = typename std::remove_const<T>::type;
    using TKernel = NKernelHost::TSegmentedScanKernel<TNonConstInputType>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, flags, flagMask, output, inclusive);
}
