#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/hash.h>

#include <util/stream/file.h>

#include <type_traits>

template <class T, class TMapping>
std::remove_const_t<T> ReadLast(
    const TCudaBuffer<T, TMapping>& data,
    ui32 stream = 0);

template <class T>
NCudaLib::TDistributedObject<std::remove_const_t<T>> Tail(
    const NCudaLib::TCudaBuffer<T, NCudaLib::TStripeMapping>& data,
    ui32 stream = 0);

template <class T>
NCudaLib::TStripeMapping CreateMappingFromTail(
    const TCudaBuffer<T, NCudaLib::TStripeMapping>& data,
    ui32 additionalData = 0,
    ui32 objectSize = 1,
    ui32 stream = 0);

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
void DumpPtr(const NCudaLib::TCudaBuffer<T, TMapping>& data, const TString& message);

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
