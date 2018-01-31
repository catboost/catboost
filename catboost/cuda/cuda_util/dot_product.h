#pragma once

#include <catboost/cuda/cuda_util/kernel/dot_product.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    template <typename T>
    class TDotProductKernel: public TKernelBase<NKernel::TDotProductContext<T>, true> {
    private:
        TCudaBufferPtr<const T> X;
        TCudaBufferPtr<const T> Y;
        TCudaBufferPtr<const T> Weights;
        TCudaHostBufferPtr<T> Result;

    public:
        Y_SAVELOAD_DEFINE(X, Y, Weights, Result);
        using TKernelContext = NKernel::TDotProductContext<T>;

        TDotProductKernel(TCudaBufferPtr<const T> x,
                          TCudaBufferPtr<const T> y,
                          TCudaHostBufferPtr<T> result)
            : X(x)
            , Y(y)
            , Weights()
            , Result(result)
        {
        }

        TDotProductKernel(TCudaBufferPtr<const T> x,
                          TCudaBufferPtr<const T> y,
                          TCudaBufferPtr<const T> weights,
                          TCudaHostBufferPtr<T> result)
            : X(x)
            , Y(y)
            , Weights(weights)
            , Result(result)
        {
        }

        TDotProductKernel() = default;

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            CB_ENSURE(X.Size() == Y.Size());

            const ui64 size = X.Size();
            auto context = MakeHolder<TKernelContext>();
            const ui64 blockSize = NKernel::GetDotProductBlockSize();

            context->NumBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);
            context->Size = size;
            context->PartResultSize = ((size + 1023) / 1024);
            auto partResultPtr = memoryManager.Allocate<T, EPtrType::CudaHost>(context->PartResultSize);

            context->PartResults = partResultPtr.Get();
            return context;
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            using namespace NKernel;
            if (Weights.Size() == 0) {
                DotProduct<T>(X.Get(), Y.Get(), context, stream.GetStream());
            } else {
                WeightedDotProduct<T>(X.Get(), Weights.Get(), Y.Get(), context, stream.GetStream());
            }
        }

        void Postprocess(const TCudaStream& stream, TKernelContext& context) {
            TVector<T> values;
            values.push_back(0);
            auto& value = values[0];
            for (ui32 i = 0; i < context.PartResultSize; ++i) {
                value += context.PartResults[i];
            }
            Result.Write(values, stream);
        }
    };
}

//
template <typename T1, class T2, typename TMapping, class T3 = T1>
inline float DotProduct(const TCudaBuffer<T1, TMapping>& x,
                        const TCudaBuffer<T2, TMapping>& y,
                        const TCudaBuffer<T3, TMapping>* weights = nullptr,
                        ui64 stream = 0) {
    using T = std::remove_const_t<T1>;
    using TResultBuffer = NCudaLib::TCudaBuffer<T, TMapping, NCudaLib::EPtrType::CudaHost>;
    using TKernel = NKernelHost::TDotProductKernel<T>;

    auto tmp = TResultBuffer::Create(x.GetMapping().RepeatOnAllDevices(1));

    if (weights == nullptr) {
        LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, tmp);
    } else {
        LaunchKernels<TKernel>(x.NonEmptyDevices(), stream, x, y, *weights, tmp);
    }
    TVector<T> result;

    NCudaLib::TCudaBufferReader<TResultBuffer>(tmp)
        .SetFactorSlice(TSlice(0, 1))
        .SetReadSlice(TSlice(0, 1))
        .SetCustomReadingStream(stream)
        .ReadReduce(result);

    return result[0];
}
