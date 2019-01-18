#include "compression_helpers_gpu.h"

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/compression.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

#include <catboost/libs/helpers/math_utils.h>

#include <util/system/types.h>

using NCudaLib::EPtrType;
using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TDeviceBuffer;
using NKernelHost::TStatelessKernel;

// Decompress

namespace {
    template <class TStorageType, NCudaLib::EPtrType Type>
    class TDecompressKernel: public TStatelessKernel {
    private:
        using TSrcBufferPtr = TDeviceBuffer<const TStorageType, Type>;
        TSrcBufferPtr Src;
        TCudaBufferPtr<ui32> Dst;
        ui32 BitsPerKey;

    public:
        TDecompressKernel() = default;

        TDecompressKernel(TSrcBufferPtr src,
                          TCudaBufferPtr<ui32> dst,
                          ui32 bitsPerKey)
            : Src(src)
            , Dst(dst)
            , BitsPerKey(bitsPerKey)
        {
        }

        Y_SAVELOAD_DEFINE(Src, Dst, BitsPerKey);

        void Run(const TCudaStream& stream) const {
            NKernel::Decompress(Src.Get(), Dst.Get(), Dst.Size(), BitsPerKey, stream.GetStream());
        }
    };
}

template <typename T, typename TMapping, EPtrType Type>
static void DecompressImpl(
    const TCudaBuffer<T, TMapping, Type>& src,
    TCudaBuffer<ui32, TMapping>& dst,
    ui32 uniqueValues,
    ui32 stream) {
    using TKernel = TDecompressKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, dst, NCB::IntLog2(uniqueValues));
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, Type)         \
    template <>                                           \
    void Decompress<T, TMapping, Type>(                   \
        const TCudaBuffer<T, TMapping, Type>& src,        \
        TCudaBuffer<ui32, TMapping>& dst,                 \
        ui32 uniqueValues,                                \
        ui32 stream) {                                    \
        ::DecompressImpl(src, dst, uniqueValues, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping, EPtrType::CudaHost),
    (ui32, TMirrorMapping, EPtrType::CudaDevice),
    (ui64, TMirrorMapping, EPtrType::CudaHost),
    (ui64, TMirrorMapping, EPtrType::CudaDevice),
    (ui32, TSingleMapping, EPtrType::CudaHost),
    (ui32, TSingleMapping, EPtrType::CudaDevice),
    (ui64, TSingleMapping, EPtrType::CudaHost),
    (ui64, TSingleMapping, EPtrType::CudaDevice),
    (ui32, TStripeMapping, EPtrType::CudaHost),
    (ui32, TStripeMapping, EPtrType::CudaDevice),
    (ui64, TStripeMapping, EPtrType::CudaHost),
    (ui64, TStripeMapping, EPtrType::CudaDevice));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// Compress

namespace {
    template <class TStorageType, EPtrType Type>
    class TCompressKernel: public TStatelessKernel {
    private:
        using TDstBufferPtr = TDeviceBuffer<TStorageType, Type>;
        TCudaBufferPtr<const ui32> Src;
        TDstBufferPtr Dst;
        ui32 BitsPerKey;

    public:
        TCompressKernel() = default;

        TCompressKernel(TCudaBufferPtr<const ui32> src,
                        TDstBufferPtr dst,
                        ui32 bitsPerKey)
            : Src(src)
            , Dst(dst)
            , BitsPerKey(bitsPerKey)
        {
        }

        Y_SAVELOAD_DEFINE(Src, Dst, BitsPerKey);

        void Run(const TCudaStream& stream) const {
            NKernel::Compress(Src.Get(), Dst.Get(), Src.Size(), BitsPerKey, stream.GetStream());
        }
    };
}

template <typename T, typename TMapping, EPtrType Type>
static void CompressImpl(
    const TCudaBuffer<ui32, TMapping>& src,
    TCudaBuffer<T, TMapping, Type>& dst,
    ui32 uniqueValues,
    ui32 stream) {
    using TKernel = TCompressKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, dst, NCB::IntLog2(uniqueValues));
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, Type)       \
    template <>                                         \
    void Compress(                                      \
        const TCudaBuffer<ui32, TMapping>& src,         \
        TCudaBuffer<T, TMapping, Type>& dst,            \
        ui32 uniqueValues,                              \
        ui32 stream) {                                  \
        ::CompressImpl(src, dst, uniqueValues, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping, EPtrType::CudaHost),
    (ui32, TMirrorMapping, EPtrType::CudaDevice),
    (ui64, TMirrorMapping, EPtrType::CudaHost),
    (ui64, TMirrorMapping, EPtrType::CudaDevice),
    (ui32, TSingleMapping, EPtrType::CudaHost),
    (ui32, TSingleMapping, EPtrType::CudaDevice),
    (ui64, TSingleMapping, EPtrType::CudaHost),
    (ui64, TSingleMapping, EPtrType::CudaDevice),
    (ui32, TStripeMapping, EPtrType::CudaHost),
    (ui32, TStripeMapping, EPtrType::CudaDevice),
    (ui64, TStripeMapping, EPtrType::CudaHost),
    (ui64, TStripeMapping, EPtrType::CudaDevice));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// GatherFromCompressed

namespace {
    template <class TStorageType, EPtrType Type>
    class TGatherFromCompressedKernel: public TStatelessKernel {
    private:
        using TSrcBufferPtr = TDeviceBuffer<const TStorageType, Type>;
        TSrcBufferPtr Src;
        TCudaBufferPtr<const ui32> Map;
        TCudaBufferPtr<ui32> Dst;
        ui32 Mask;
        ui32 BitsPerKey;

    public:
        TGatherFromCompressedKernel() = default;

        TGatherFromCompressedKernel(TSrcBufferPtr src,
                                    TCudaBufferPtr<const ui32> map,
                                    ui32 mask,
                                    TCudaBufferPtr<ui32> dst,
                                    ui32 bitsPerKey)
            : Src(src)
            , Map(map)
            , Dst(dst)
            , Mask(mask)
            , BitsPerKey(bitsPerKey)
        {
        }

        Y_SAVELOAD_DEFINE(Src, Map, Dst, Mask, BitsPerKey);

        void Run(const TCudaStream& stream) const {
            NKernel::GatherFromCompressed(Src.Get(), Map.Get(), Mask, Dst.Get(), Map.Size(), BitsPerKey, stream.GetStream());
        }
    };
}

template <typename T, typename TMapping, EPtrType Type, typename TUi32>
static void GatherFromCompressedImpl(
    const TCudaBuffer<T, TMapping, Type>& src,
    const ui32 uniqueValues,
    const TCudaBuffer<TUi32, TMapping>& map,
    const ui32 mask,
    TCudaBuffer<ui32, TMapping>& dst,
    ui32 stream) {
    using TKernel = TGatherFromCompressedKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, map, mask, dst, NCB::IntLog2(uniqueValues));
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping, Type, TUi32)                       \
    template <>                                                                \
    void GatherFromCompressed<T, TMapping, Type, TUi32>(                       \
        const TCudaBuffer<T, TMapping, Type>& src,                             \
        const ui32 uniqueValues,                                               \
        const TCudaBuffer<TUi32, TMapping>& map,                               \
        const ui32 mask,                                                       \
        TCudaBuffer<ui32, TMapping>& dst,                                      \
        ui32 stream) {                                                         \
        ::GatherFromCompressedImpl(src, uniqueValues, map, mask, dst, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping, EPtrType::CudaHost, ui32),
    (ui32, TMirrorMapping, EPtrType::CudaDevice, ui32),
    (ui64, TMirrorMapping, EPtrType::CudaHost, ui32),
    (ui64, TMirrorMapping, EPtrType::CudaDevice, ui32),
    (ui32, TSingleMapping, EPtrType::CudaHost, ui32),
    (ui32, TSingleMapping, EPtrType::CudaDevice, ui32),
    (ui64, TSingleMapping, EPtrType::CudaHost, ui32),
    (ui64, TSingleMapping, EPtrType::CudaDevice, ui32),
    (ui32, TStripeMapping, EPtrType::CudaHost, ui32),
    (ui32, TStripeMapping, EPtrType::CudaDevice, ui32),
    (ui64, TStripeMapping, EPtrType::CudaHost, ui32),
    (ui64, TStripeMapping, EPtrType::CudaDevice, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// CompressedSize

template <typename TStorageType>
static ui32 CompressedSizeImpl(ui32 count, ui32 uniqueValues) {
    const ui32 bitsPerKey = NCB::IntLog2(uniqueValues);
    const ui32 keysPerBlock = NKernel::KeysPerBlock<TStorageType>(bitsPerKey);
    return ::NHelpers::CeilDivide(count, keysPerBlock) * NKernel::CompressCudaBlockSize();
}

#define Y_CATBOOST_CUDA_F_IMPL(TStorageType)                            \
    template <>                                                         \
    ui32 CompressedSize<TStorageType>(ui32 count, ui32 uniqueValues) {  \
        return ::CompressedSizeImpl<TStorageType>(count, uniqueValues); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    ui32,
    ui64);

#undef Y_CATBOOST_CUDA_F_IMPL

// CompressedSize

template <typename TStorageType, typename TMapping>
static TMapping CompressedSizeImpl(const TCudaBuffer<ui32, TMapping>& src, ui32 uniqueValues) {
    const ui32 bitsPerKey = NCB::IntLog2(uniqueValues);
    return src.GetMapping().Transform([&](const TSlice& devSlice) -> ui64 {
        const ui32 keysPerBlock = NKernel::KeysPerBlock<TStorageType>(bitsPerKey);
        return ::NHelpers::CeilDivide((ui32)devSlice.Size(), keysPerBlock) * NKernel::CompressCudaBlockSize();
    });
};

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(TStorageType, TMapping)                                                           \
    template <>                                                                                                  \
    TMapping CompressedSize<TStorageType, TMapping>(const TCudaBuffer<ui32, TMapping>& src, ui32 uniqueValues) { \
        return ::CompressedSizeImpl<TStorageType>(src, uniqueValues);                                            \
    };

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (ui64, TMirrorMapping),
    (ui32, TSingleMapping),
    (ui64, TSingleMapping),
    (ui32, TStripeMapping),
    (ui64, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA01, TDecompressKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA02, TDecompressKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFA03, TCompressKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA04, TCompressKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFA05, TGatherFromCompressedKernel, ui64, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFA06, TGatherFromCompressedKernel, ui64, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB01, TDecompressKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB02, TDecompressKernel, ui32, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB03, TCompressKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB04, TCompressKernel, ui32, EPtrType::CudaDevice)

    REGISTER_KERNEL_TEMPLATE_2(0xFAFB05, TGatherFromCompressedKernel, ui32, EPtrType::CudaHost)
    REGISTER_KERNEL_TEMPLATE_2(0xFAFB06, TGatherFromCompressedKernel, ui32, EPtrType::CudaDevice)
}
