#include "helpers.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/hash.h>
#include <catboost/libs/logging/logging.h>

#include <util/stream/file.h>
#include <util/stream/labeled.h>

using NCudaLib::EPtrType;
using NCudaLib::TDistributedObject;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TStatelessKernel;

// DumpPtr

namespace {
    template <class T>
    class TDumpPtrs: public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Buffer;
        TString Message;

    public:
        TDumpPtrs() = default;

        TDumpPtrs(TCudaBufferPtr<const T> buffer,
                  TString message)
            : Buffer(buffer)
            , Message(message)
        {
        }

        Y_SAVELOAD_DEFINE(Buffer, Message);

        void Run(const TCudaStream& stream) const {
            Y_UNUSED(stream);
            CATBOOST_INFO_LOG << Message << " Ptr: " << (ui64)(Buffer.Get()) << " of size " << Buffer.Size() << Endl;
        }
    };
}

template <class T, class TMapping>
static void DumpPtrImpl(
    const TCudaBuffer<T, TMapping>& data,
    const TString& message) {
    using TKernel = ::TDumpPtrs<std::remove_const_t<T>>;
    LaunchKernels<TKernel>(data.NonEmptyDevices(), 0, data, message);
};

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                      \
    template <>                                                                  \
    void DumpPtr(const TCudaBuffer<T, TMapping>& data, const TString& message) { \
        ::DumpPtrImpl(data, message);                                            \
    };

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (ui32, TSingleMapping),
    (ui32, TStripeMapping),
    (float, TMirrorMapping),
    (float, TSingleMapping),
    (float, TStripeMapping),
    (int, TMirrorMapping),
    (int, TSingleMapping),
    (int, TStripeMapping),
    (const ui32, TMirrorMapping),
    (const ui32, TSingleMapping),
    (const ui32, TStripeMapping),
    (const float, TMirrorMapping),
    (const float, TSingleMapping),
    (const float, TStripeMapping),
    (const int, TMirrorMapping),
    (const int, TSingleMapping),
    (const int, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Tail

namespace {
    template <class T, EPtrType PtrType>
    class TTailKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Source;
        TDeviceBuffer<T, PtrType> Dest;

    public:
        TTailKernel() = default;

        TTailKernel(TCudaBufferPtr<const T> source,
                    TDeviceBuffer<T, PtrType> dest)
            : Source(source)
            , Dest(dest)
        {
        }

        Y_SAVELOAD_DEFINE(Source, Dest);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Dest.Size() == 1, LabeledOutput(Dest.Size()));
            CB_ENSURE(Dest.ObjectSize() == Source.ObjectSize(), LabeledOutput(Dest.ObjectSize(), Source.ObjectSize()));
            if (Source.Size()) {
                CopyMemoryAsync(Source.GetForObject(Source.ObjectCount() - 1), Dest.Get(), Dest.ObjectSize(), stream);
            } else {
                NKernel::FillBuffer(Dest.Get(), (T)0, Dest.ObjectSize(), stream.GetStream());
            }
        }
    };
}

template <class T>
static TDistributedObject<std::remove_const_t<T>> TailImpl(const TCudaBuffer<T, TStripeMapping>& data, ui32 stream) {
    Y_ASSERT(data.GetObjectsSlice().Size());
    auto result = TCudaBuffer<std::remove_const_t<T>, TStripeMapping, EPtrType::CudaHost>::Create(data.GetMapping().RepeatOnAllDevices(1, data.GetMapping().SingleObjectSize()));

    using TKernel = TTailKernel<std::remove_const_t<T>, EPtrType::CudaHost>;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), stream, data, result);

    TVector<std::remove_const_t<T>> resultVec;
    result.Read(resultVec, stream);
    auto res = CreateDistributedObject<std::remove_const_t<T>>(0);
    for (ui32 i = 0; i < NCudaLib::GetCudaManager().GetDeviceCount(); ++i) {
        res.Set(i, resultVec[i]);
    }
    return res;
}

#define Y_CATBOOST_CUDA_F_IMPL(T)                                                                                 \
    template <>                                                                                                   \
    TDistributedObject<std::remove_const_t<T>> Tail<T>(const TCudaBuffer<T, TStripeMapping>& data, ui32 stream) { \
        return ::TailImpl(data, stream);                                                                          \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    ui32,
    const ui32,
    ui64,
    const ui64);

#undef Y_CATBOOST_CUDA_F_IMPL

// ReadLast

template <class T, class TMapping>
static std::remove_const_t<T> ReadLastImpl(const TCudaBuffer<T, TMapping>& data, ui32 stream) {
    Y_ASSERT(data.GetObjectsSlice().Size());

    TVector<std::remove_const_t<T>> resVec;
    NCudaLib::TCudaBufferReader<TCudaBuffer<T, TMapping>> reader(data);
    auto dataSlice = data.GetObjectsSlice();
    reader.SetReadSlice(TSlice(dataSlice.Right - 1, dataSlice.Right))
        .SetCustomReadingStream(stream)
        .Read(resVec);

    CB_ENSURE(resVec.size() == 1, LabeledOutput(resVec.size()));
    return resVec[0];
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                                                           \
    template <>                                                                                       \
    std::remove_const_t<T> ReadLast<T, TMapping>(const TCudaBuffer<T, TMapping>& data, ui32 stream) { \
        return ::ReadLastImpl(data, stream);                                                          \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (ui32, TSingleMapping),
    (ui32, TStripeMapping),
    (const ui32, TMirrorMapping),
    (const ui32, TSingleMapping),
    (const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// CreateMappingFromTail

template <class T>
TStripeMapping CreateMappingFromTailImpl(
    const TCudaBuffer<T, TStripeMapping>& data,
    ui32 additionalData,
    ui32 objectSize,
    ui32 stream) {
    auto tailSizes = Tail(data, stream);
    NCudaLib::TMappingBuilder<TStripeMapping> builder;
    for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
        builder.SetSizeAt(dev, tailSizes.At(dev) + (ui64)additionalData);
    }
    return builder.Build(objectSize);
}

#define Y_CATBOOST_CUDA_F_IMPL(T)                                                   \
    template <>                                                                     \
    TStripeMapping CreateMappingFromTail<T>(                                        \
        const TCudaBuffer<T, TStripeMapping>& data,                                 \
        ui32 additionalData,                                                        \
        ui32 objectSize,                                                            \
        ui32 stream) {                                                              \
        return CreateMappingFromTailImpl(data, additionalData, objectSize, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    ui32,
    ui64,
    const ui32,
    const ui64);

#undef Y_CATBOOST_CUDA_F_IMPL

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xFFFF00, TDumpPtrs, float);
    REGISTER_KERNEL_TEMPLATE(0xFFFF01, TDumpPtrs, ui32);
    REGISTER_KERNEL_TEMPLATE(0xFFFF02, TDumpPtrs, int);
    REGISTER_KERNEL_TEMPLATE_2(0xFFFF03, TTailKernel, ui32, EPtrType::CudaHost);
    REGISTER_KERNEL_TEMPLATE_2(0xFFFF04, TTailKernel, ui64, EPtrType::CudaHost);
}
