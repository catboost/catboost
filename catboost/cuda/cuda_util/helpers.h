#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/utils/hash_helpers.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <util/stream/file.h>
#include <catboost/cuda/utils/cpu_random.h>

namespace NKernelHost {
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
            MATRIXNET_INFO_LOG << Message << " Ptr: " << (ui64)(Buffer.Get()) << " of size " << Buffer.Size() << Endl;
        }
    };

    template <class T, NCudaLib::EPtrType PtrType>
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
            CB_ENSURE(Dest.Size() == 1);
            CB_ENSURE(Dest.ObjectSize() == Source.ObjectSize());
            if (Source.Size()) {
                CopyMemoryAsync(Source.GetForObject(Source.ObjectCount() - 1), Dest.Get(), Dest.ObjectSize(), stream);
            } else {
                NKernel::FillBuffer(Dest.Get(), (T)0, Dest.ObjectSize(), stream.GetStream());
            }
        }
    };
}
template <class T, class TMapping>
inline T ReadLast(const TCudaBuffer<T, TMapping>& data, ui32 stream = 0) {
    Y_ASSERT(data.GetObjectsSlice().Size());

    TVector<ui32> resVec;
    NCudaLib::TCudaBufferReader<TCudaBuffer<T, TMapping>> reader(data);
    auto dataSlice = data.GetObjectsSlice();
    reader.SetReadSlice(TSlice(dataSlice.Right - 1, dataSlice.Right))
        .SetCustomReadingStream(stream)
        .Read(resVec);

    CB_ENSURE(resVec.size() == 1);
    return resVec[0];
}

template <class T>
inline NCudaLib::TDistributedObject<T> Tail(const TCudaBuffer<T, NCudaLib::TStripeMapping>& data, ui32 stream = 0) {
    Y_ASSERT(data.GetObjectsSlice().Size());
    auto result = TCudaBuffer<T, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost>::Create(data.GetMapping().RepeatOnAllDevices(1, data.GetMapping().SingleObjectSize()));

    using TKernel = NKernelHost::TTailKernel<T, NCudaLib::EPtrType::CudaHost>;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), stream, data, result);

    TVector<ui32> resultVec;
    result.Read(resultVec, stream);
    NCudaLib::TDistributedObject<T> res = CreateDistributedObject<T>(0);
    for (ui32 i = 0; i < NCudaLib::GetCudaManager().GetDeviceCount(); ++i) {
        res.Set(i, resultVec[i]);
    }
    return res;
}

template <class T>
inline NCudaLib::TStripeMapping CreateMappingFromTail(const TCudaBuffer<T, NCudaLib::TStripeMapping>& data,
                                                      ui32 additionalData = 0,
                                                      ui32 objectSize = 1,
                                                      ui32 stream = 0) {
    auto tailSizes = Tail(data, stream);
    NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping> builder;
    for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
        builder.SetSizeAt(dev, tailSizes.At(dev) + additionalData);
    }
    return builder.Build(objectSize);
}

template <class T>
inline TString Printable(T val) {
    return TStringBuilder() << val;
}

template <>
inline TString Printable(ui8 val) {
    return TStringBuilder() << static_cast<ui32>(val);
}

template <class TBuffer>
inline void Dump(const TBuffer& data, TString message, ui32 size = 100) {
    using T = std::remove_const_t<typename TBuffer::TValueType>;
    TVector<T> res;
    data.CreateReader().SetReadSlice(TSlice(0, size)).Read(res);
    Cout << message << Endl;
    Cout << "Size:  " << data.GetMapping().GetObjectsSlice().Size() << Endl;
    Cout << "Hash: " << VecCityHash(res) << Endl;
    for (auto val : res) {
        Cout << Printable(val) << " ";
    }
    Cout << Endl;
};

template <class TBuffer>
inline void DumpToFile(const TBuffer& data, TString file) {
    using T = std::remove_const_t<typename TBuffer::TValueType>;
    TVector<T> res;
    data.CreateReader().Read(res);
    TOFStream out(file);
    for (auto& val : res) {
        out << Printable(val) << Endl;
    }
};

template <class TBuffer>
inline ui64 DumpHash(const TBuffer& data, TString message) {
    using T = std::remove_const_t<typename TBuffer::TValueType>;
    TVector<T> res;
    data.CreateReader().Read(res);
    Cout << message << Endl;
    Cout << "Size:  " << data.GetMapping().GetObjectsSlice().Size() << Endl;
    const ui64 hash = VecCityHash(res);
    Cout << "Hash: " << hash << Endl;
    Cout << Endl;
    return hash;
};

template <class TBuffer>
inline ui64 GetHash(const TBuffer& data) {
    using T = std::remove_const_t<typename TBuffer::TValueType>;
    TVector<T> res;
    data.CreateReader().Read(res);
    return VecCityHash(res);
};

template <class T, class TMapping>
inline void DumpPtr(const TCudaBuffer<T, TMapping>& data,
                    TString message) {
    using TKernel = NKernelHost::TDumpPtrs<T>;
    LaunchKernels<TKernel>(data.NonEmptyDevices(), 0, data, message);
};

template <class TBuffer>
inline void DumpCast(const TBuffer& data, TString message, ui32 size = 16) {
    using T = std::remove_const_t<typename TBuffer::TValueType>;
    TVector<T> res;
    data.Read(res);
    Cout << message << Endl;
    Cout << "Size:  " << data.GetMapping().GetObjectsSlice().Size() << Endl;
    for (ui32 i = 0; i < size; ++i) {
        const ui32 val = res[i];
        Cout << val << " ";
    }
    Cout << Endl;
};

inline NCudaLib::TDistributedObject<ui64> DistributedSeed(TRandom& random) {
    NCudaLib::TDistributedObject<ui64> seed = CreateDistributedObject<ui64>();
    for (ui32 i = 0; i < seed.DeviceCount(); ++i) {
        seed.Set(i, random.NextUniformL());
    }
    return seed;
}
