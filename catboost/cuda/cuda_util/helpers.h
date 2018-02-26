#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/utils/hash_helpers.h>
#include <util/stream/file.h>

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
        out << val << Endl;
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
