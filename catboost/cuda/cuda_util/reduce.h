#pragma once

#include "operator.h"

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_util/kernel/cub_storage_context.cuh>
#include <catboost/cuda/cuda_util/kernel/reduce.cuh>
#include <type_traits>

namespace NKernelHost {
    template <typename T>
    class TReduceKernel: public TKernelBase<NKernel::TCubKernelContext, false> {
    private:
        TCudaBufferPtr<const T> Input;
        TCudaBufferPtr<T> Output;
        EOperatorType Type;

    public:
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
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize).Get();
            }
            context->Initialized = true;
            return context;
        }

        Y_SAVELOAD_DEFINE(Input, Output, Type);

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            CUDA_SAFE_CALL(NKernel::Reduce(Input.Get(), Output.Get(), Input.Size(), Type, context, stream.GetStream()));
        }
    };

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
                //TODO(noxoomo): make temp memory more robust
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize).Get();
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

    template <typename T, EPtrType PtrType>
    class TSegmentedReduceKernel: public TKernelBase<NKernel::TCubKernelContext, false> {
    private:
        using TOutputPtr = TDeviceBuffer<T, TFixedSizesObjectsMeta, PtrType>;
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
                //TODO(noxoomo): make temp memory more robust
                context->TempStorage = memoryManager.Allocate<char>(context->TempStorageSize).Get();
            }
            context->Initialized = true;
            return context;
        }

        Y_SAVELOAD_DEFINE(Input, Offsets, Output, Type);

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            Y_ENSURE(Output.Size() + 1 == Offsets.Size(), TStringBuilder() << "Error: outputSize " << Output.Size() << "; Offsets size " << Offsets.Size());
            CUDA_SAFE_CALL(NKernel::SegmentedReduce(Input.Get(), Input.Size(),
                                                    Offsets.Get(), Offsets.Size() - 1,
                                                    Output.Get(), Type, context, stream.GetStream()));
        }
    };
}

template <typename T, class TMapping>
inline void ReduceVector(const TCudaBuffer<T, TMapping>& input, TCudaBuffer<T, TMapping>& output,
                         EOperatorType type = EOperatorType::Sum,
                         ui32 streamId = 0) {
    using TKernel = NKernelHost::TReduceKernel<T>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, output, type);
}

template <typename T, typename K, class TMapping>
inline void ReduceByKeyVector(const TCudaBuffer<T, TMapping>& input,
                              const TCudaBuffer<K, TMapping>& keys,
                              TCudaBuffer<K, TMapping>& outputKeys,
                              TCudaBuffer<T, TMapping>& output,
                              TCudaBuffer<ui32, TMapping>& outputSizes,
                              EOperatorType type = EOperatorType::Sum,
                              ui32 streamId = 0) {
    using TKernel = NKernelHost::TReduceByKeyKernel<T, K>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, keys, outputKeys, output, outputSizes, type);
}

template <typename T, class TMapping, NCudaLib::EPtrType OutputPtrType = NCudaLib::EPtrType::CudaDevice>
inline void SegmentedReduceVector(const TCudaBuffer<T, TMapping>& input,
                                  const TCudaBuffer<ui32, TMapping>& offsets,
                                  TCudaBuffer<typename std::remove_const<T>::type, TMapping, OutputPtrType>& output,
                                  EOperatorType type = EOperatorType::Sum,
                                  ui32 streamId = 0) {
    using TNonConstT = typename std::remove_const<T>::type;
    using TKernel = NKernelHost::TSegmentedReduceKernel<TNonConstT, OutputPtrType>;
    LaunchKernels<TKernel>(input.NonEmptyDevices(), streamId, input, offsets, output, type);
}
