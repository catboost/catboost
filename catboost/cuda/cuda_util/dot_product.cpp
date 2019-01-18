#include "dot_product.h"

#include <catboost/cuda/cuda_util/kernel/dot_product.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/exception.h>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaHostBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TKernelBase;
using NKernelHost::TStatelessKernel;

// DotProduct

namespace {
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
            context->PartResults = memoryManager.Allocate<T, EPtrType::CudaHost>(context->PartResultSize);
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
            auto partResultPtr = context.PartResults.Get();

            for (ui32 i = 0; i < context.PartResultSize; ++i) {
                value += partResultPtr[i];
            }
            Result.Write(values, stream);
        }
    };
}

template <typename T1, class T2, typename TMapping, class T3>
static float DotProductImpl(
    const TCudaBuffer<T1, TMapping>& x,
    const TCudaBuffer<T2, TMapping>& y,
    const TCudaBuffer<T3, TMapping>* weights,
    ui64 stream) {
    using T = std::remove_const_t<T1>;
    using TResultBuffer = TCudaBuffer<T, TMapping, EPtrType::CudaHost>;
    using TKernel = TDotProductKernel<T>;

    auto tmp = TResultBuffer::Create(x.GetMapping().RepeatOnAllDevices(1));
    {
        TVector<T> empty(tmp.GetObjectsSlice().Size());
        tmp.Write(empty, stream);
    }

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

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T1, T2, TMapping, T3)  \
    template <>                                       \
    float DotProduct<T1, T2, TMapping, T3>(           \
        const TCudaBuffer<T1, TMapping>& x,           \
        const TCudaBuffer<T2, TMapping>& y,           \
        const TCudaBuffer<T3, TMapping>* weights,     \
        ui64 stream) {                                \
        return DotProductImpl(x, y, weights, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, float, TMirrorMapping, float),
    (float, float, TMirrorMapping, const float),
    (float, const float, TMirrorMapping, float),
    (float, const float, TMirrorMapping, const float),
    (const float, const float, TMirrorMapping, const float),
    (float, float, TSingleMapping, float),
    (float, float, TSingleMapping, const float),
    (float, const float, TSingleMapping, float),
    (float, const float, TSingleMapping, const float),
    (const float, const float, TSingleMapping, const float),
    (float, float, TStripeMapping, float),
    (float, float, TStripeMapping, const float),
    (float, const float, TStripeMapping, float),
    (float, const float, TStripeMapping, const float),
    (const float, const float, TStripeMapping, const float));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xDD0001, TDotProductKernel, float)
}
