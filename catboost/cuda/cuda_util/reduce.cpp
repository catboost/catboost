#include "reduce.h"

#include "operator.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/cub_storage_context.cuh>
#include <catboost/cuda/cuda_util/kernel/reduce.cuh>

#include <type_traits>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TKernelBase;
using NKernelHost::TStatelessKernel;

// ReduceVector

namespace {
    template <typename T>
    class TReduceKernel: public TKernelBase<NKernel::TCubKernelContext, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<T> Output;
        EOperatorType Type;

    public:
        Y_SAVELOAD_DEFINE(Input, Output, Type);

        using TKernelContext = NKernel::TCubKernelContext;
        TReduceKernel() = default;

        TReduceKernel(TCudaBufferPtr<const T> input,
                      TCudaBufferPtr<T> output,
                      EOperatorType type)
            : Input(input)
            , Output(output)
            , Type(type)
        {
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            CUDA_SAFE_CALL(NKernel::Reduce(Input.Get(), Output.Get(), Input.Size(), Type, *context, 0));
            if (context->TempStorageSize) {
                //TODO(noxoomo): make temp memory more robust
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize);
            }
            context->Initialized = true;
            return context;
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            CUDA_SAFE_CALL(NKernel::Reduce(Input.Get(), Output.Get(), Input.Size(), Type, context, stream.GetStream()));
        }
    };
}

template <typename T, typename TMapping>
static void ReduceVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    EOperatorType type,
    ui32 streamId) {
    using TKernel = TReduceKernel<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(output.NonEmptyDevices(), streamId, input, output, type);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                    \
    template <>                                                \
    void ReduceVector<T, TMapping>(                            \
        const TCudaBuffer<T, TMapping>& input,                 \
        TCudaBuffer<std::remove_const_t<T>, TMapping>& output, \
        EOperatorType type,                                    \
        ui32 streamId) {                                       \
        ::ReduceVectorImpl(input, output, type, streamId);     \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (ui32, TMirrorMapping),
    (int, TMirrorMapping),
    (float, TSingleMapping),
    (ui32, TSingleMapping),
    (int, TSingleMapping),
    (float, TStripeMapping),
    (ui32, TStripeMapping),
    (int, TStripeMapping),
    (ui64, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ReduceToHost

template <typename T, class TMapping>
static std::remove_const_t<T> ReduceToHostImpl(
    const TCudaBuffer<T, TMapping>& input,
    EOperatorType type,
    ui32 streamId) {
    using TKernel = TReduceKernel<std::remove_const_t<T>>;

    auto tmpMapping = input.GetMapping().Transform([&](const TSlice&) { return 1; });
    using TResultBuffer = NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping, NCudaLib::EPtrType::CudaDevice>;
    TResultBuffer tmp;
    tmp.Reset(tmpMapping);
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, tmp, type);
    TVector<std::remove_const_t<T>> result;

    NCudaLib::TCudaBufferReader<TResultBuffer>(tmp)
        .SetFactorSlice(TSlice(0, 1))
        .SetReadSlice(TSlice(0, 1))
        .SetCustomReadingStream(streamId)
        .ReadReduce(result);

    return result[0];
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)             \
    template <>                                         \
    std::remove_const_t<T> ReduceToHost<T, TMapping>(   \
        const TCudaBuffer<T, TMapping>& input,          \
        EOperatorType type,                             \
        ui32 streamId) {                                \
        return ReduceToHostImpl(input, type, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping),
    (const float, TMirrorMapping),
    (ui32, TMirrorMapping),
    (int, TMirrorMapping),
    (float, TSingleMapping),
    (const float, TSingleMapping),
    (ui32, TSingleMapping),
    (int, TSingleMapping),
    (float, TStripeMapping),
    (const float, TStripeMapping),
    (ui32, TStripeMapping),
    (int, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// ReduceByKeyVector

namespace {
    template <typename T, typename K>
    class TReduceByKeyKernel: public TKernelBase<NKernel::TCubKernelContext, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const K> Keys;
        TCudaBufferPtr<T> Output;
        TCudaBufferPtr<K> OutputKeys;
        TCudaBufferPtr<ui32> Sizes;
        EOperatorType Type;

    public:
        using TKernelContext = NKernel::TCubKernelContext;
        TReduceByKeyKernel() = default;

        TReduceByKeyKernel(TCudaBufferPtr<const T> input,
                           TCudaBufferPtr<const K> keys,
                           TCudaBufferPtr<T> output,
                           TCudaBufferPtr<K> outputKeys,
                           TCudaBufferPtr<ui32> sizes,
                           EOperatorType type)
            : Input(input)
            , Keys(keys)
            , Output(output)
            , OutputKeys(outputKeys)
            , Sizes(sizes)
            , Type(type)
        {
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            CUDA_SAFE_CALL(NKernel::ReduceByKey(Input.Get(), Keys.Get(), Input.Size(),
                                                Output.Get(), OutputKeys.Get(),
                                                Sizes.Get(), Type, *context,
                                                0));
            if (context->TempStorageSize) {
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize);
            }
            context->Initialized = true;
            return context;
        }

        Y_SAVELOAD_DEFINE(Input, Keys, Output, OutputKeys, Sizes, Type);

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            CUDA_SAFE_CALL(NKernel::ReduceByKey(Input.Get(), Keys.Get(), Input.Size(),
                                                Output.Get(), OutputKeys.Get(),
                                                Sizes.Get(), Type, context,
                                                stream.GetStream()));
        }
    };
}

template <typename T, typename K, typename TMapping>
static void ReduceByKeyVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    const TCudaBuffer<K, TMapping>& keys,
    TCudaBuffer<std::remove_const_t<K>, TMapping>& outputKeys,
    TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    TCudaBuffer<ui32, TMapping>& outputSizes,
    EOperatorType type,
    ui32 streamId) {
    using TKernel = TReduceByKeyKernel<std::remove_const_t<T>, std::remove_const_t<K>>;
    LaunchKernels<TKernel>(output.NonEmptyDevices(), streamId, input, keys, output, outputKeys, outputSizes, type);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, K, TMapping)                                                 \
    template <>                                                                                \
    void ReduceByKeyVector<T, K, TMapping>(                                                    \
        const TCudaBuffer<T, TMapping>& input,                                                 \
        const TCudaBuffer<K, TMapping>& keys,                                                  \
        TCudaBuffer<K, TMapping>& outputKeys,                                                  \
        TCudaBuffer<T, TMapping>& output,                                                      \
        TCudaBuffer<ui32, TMapping>& outputSizes,                                              \
        EOperatorType type,                                                                    \
        ui32 streamId) {                                                                       \
        ::ReduceByKeyVectorImpl(input, keys, outputKeys, output, outputSizes, type, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32, TMirrorMapping),
    (float, ui32, TSingleMapping),
    (float, ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// SegmentedReduceVector

namespace {
    template <typename T, EPtrType PtrType>
    class TSegmentedReduceKernel: public TKernelBase<NKernel::TCubKernelContext, false> {
    private:
        using TOutputPtr = TDeviceBuffer<T, PtrType>;
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<const ui32> Offsets;
        TOutputPtr Output;
        EOperatorType Type;

    public:
        using TKernelContext = NKernel::TCubKernelContext;

        TSegmentedReduceKernel() = default;

        TSegmentedReduceKernel(TCudaBufferPtr<const T> input,
                               TCudaBufferPtr<const ui32> offsets,
                               TOutputPtr output, EOperatorType type)
            : Input(input)
            , Offsets(offsets)
            , Output(output)
            , Type(type)
        {
        }

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            CUDA_SAFE_CALL(NKernel::SegmentedReduce(Input.Get(), Input.Size(),
                                                    Offsets.Get(), Offsets.Size() - 1,
                                                    Output.Get(), Type, *context, 0));
            if (context->TempStorageSize) {
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize);
            }
            context->Initialized = true;
            return context;
        }

        Y_SAVELOAD_DEFINE(Input, Offsets, Output, Type);

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            CB_ENSURE(Output.Size() + 1 == Offsets.Size(), TStringBuilder() << "Error: outputSize " << Output.Size() << "; Offsets size " << Offsets.Size());
            CUDA_SAFE_CALL(NKernel::SegmentedReduce(Input.Get(), Input.Size(),
                                                    Offsets.Get(), Offsets.Size() - 1,
                                                    Output.Get(), Type, context, stream.GetStream()));
        }
    };
}

template <typename T, typename TMapping, EPtrType OutputPtrType>
static void SegmentedReduceVectorImpl(
    const TCudaBuffer<T, TMapping>& input,
    const TCudaBuffer<ui32, TMapping>& offsets,
    TCudaBuffer<std::remove_const_t<T>, TMapping, OutputPtrType>& output,
    EOperatorType type,
    ui32 streamId) {
    using TKernel = TSegmentedReduceKernel<std::remove_const_t<T>, OutputPtrType>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, offsets, output, type);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, OutputPtrType)                    \
    template <>                                                               \
    void SegmentedReduceVector<T, TMapping, OutputPtrType>(                   \
        const TCudaBuffer<T, TMapping>& input,                                \
        const TCudaBuffer<ui32, TMapping>& offsets,                           \
        TCudaBuffer<std::remove_const_t<T>, TMapping, OutputPtrType>& output, \
        EOperatorType type,                                                   \
        ui32 streamId) {                                                      \
        ::SegmentedReduceVectorImpl(input, offsets, output, type, streamId);  \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TMirrorMapping, EPtrType::CudaDevice),
    (float, TMirrorMapping, EPtrType::CudaHost),
    (float, TSingleMapping, EPtrType::CudaDevice),
    (float, TSingleMapping, EPtrType::CudaHost),
    (float, TStripeMapping, EPtrType::CudaDevice),
    (float, TStripeMapping, EPtrType::CudaHost));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xAE0001, TReduceKernel, float)
    REGISTER_KERNEL_TEMPLATE(0xAE0005, TReduceKernel, ui32)
    REGISTER_KERNEL_TEMPLATE(0xAE0006, TReduceKernel, int)
    REGISTER_KERNEL_TEMPLATE(0xAE0007, TReduceKernel, ui64)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0002, TSegmentedReduceKernel, float, EPtrType::CudaDevice)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0003, TSegmentedReduceKernel, float, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xAE0004, TReduceByKeyKernel, float, ui32)
}
