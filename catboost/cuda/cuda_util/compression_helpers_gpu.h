#pragma once

#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/compression.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <util/system/types.h>

namespace NKernelHost {
    template <class TStorageType, NCudaLib::EPtrType Type>
    class TDecompressKernel: public TStatelessKernel {
    private:
        using TSrcBufferPtr = TDeviceBuffer<const TStorageType, TFixedSizesObjectsMeta, Type>;
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

    template <class TStorageType, NCudaLib::EPtrType Type>
    class TCompressKernel: public TStatelessKernel {
    private:
        using TDstBufferPtr = TDeviceBuffer<TStorageType, TFixedSizesObjectsMeta, Type>;
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

    template <class TStorageType, NCudaLib::EPtrType Type>
    class TGatherFromCompressedKernel: public TStatelessKernel {
    private:
        using TSrcBufferPtr = TDeviceBuffer<const TStorageType, TFixedSizesObjectsMeta, Type>;
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

template <typename TStorageType>
inline ui32 CompressedSize(ui32 count, ui32 uniqueValues) {
    const ui32 bitsPerKey = IntLog2(uniqueValues);
    const ui32 keysPerBlock = NKernel::KeysPerBlock<TStorageType>(bitsPerKey);
    return ::NHelpers::CeilDivide(count, keysPerBlock) * NKernel::CompressCudaBlockSize();
}

template <typename TStorageType, typename TMapping>
inline TMapping CompressedSize(const TCudaBuffer<ui32, TMapping>& src, ui32 uniqueValues) {
    const ui32 bitsPerKey = IntLog2(uniqueValues);

    return src.GetMapping().Transform([&](const TSlice& devSlice) -> ui64 {
        const ui32 keysPerBlock = NKernel::KeysPerBlock<TStorageType>(bitsPerKey);
        return ::NHelpers::CeilDivide((ui32)devSlice.Size(), keysPerBlock) * NKernel::CompressCudaBlockSize();
    });
};

//
template <typename T, typename TMapping, NCudaLib::EPtrType Type>
inline void Compress(const TCudaBuffer<ui32, TMapping>& src,
                     TCudaBuffer<T, TMapping, Type>& dst,
                     ui32 uniqueValues,
                     ui32 stream = 0) {
    using TKernel = NKernelHost::TCompressKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, dst, IntLog2(uniqueValues));
}

template <typename T, typename TMapping, NCudaLib::EPtrType Type>
inline void Decompress(const TCudaBuffer<T, TMapping, Type>& src,
                       TCudaBuffer<ui32, TMapping>& dst,
                       ui32 uniqueValues,
                       ui32 stream = 0) {
    using TKernel = NKernelHost::TDecompressKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, dst, IntLog2(uniqueValues));
}

template <typename T, typename TMapping, NCudaLib::EPtrType Type, class TUi32 = ui32>
inline void GatherFromCompressed(const TCudaBuffer<T, TMapping, Type>& src,
                                 const ui32 uniqueValues,
                                 const TCudaBuffer<TUi32, TMapping>& map,
                                 const ui32 mask,
                                 TCudaBuffer<ui32, TMapping>& dst,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TGatherFromCompressedKernel<T, Type>;
    LaunchKernels<TKernel>(src.NonEmptyDevices(), stream, src, map, mask, dst, IntLog2(uniqueValues));
}
