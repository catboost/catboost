#pragma once

#include <util/generic/buffer.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

namespace NBlockCodecs {
    struct TData: public TStringBuf {
        inline TData() = default;

        Y_HAS_MEMBER(Data);
        Y_HAS_MEMBER(Size);

        template <class T, std::enable_if_t<!THasSize<T>::value || !THasData<T>::value, int> = 0>
        inline TData(const T& t)
            : TStringBuf((const char*)t.data(), t.size())
        {
        }

        template <class T, std::enable_if_t<THasSize<T>::value && THasData<T>::value, int> = 0>
        inline TData(const T& t)
            : TStringBuf((const char*)t.Data(), t.Size())
        {
        }

    };

    template <>
    inline TData::TData(const TString& t)
        : TStringBuf((const char*)t.data(), t.size())
    {
    }

    struct TCodecError: public yexception {
    };

    struct TNotFound: public TCodecError {
    };

    struct TDataError: public TCodecError {
    };

    struct ICodec {
        virtual ~ICodec();

        // main interface
        virtual size_t DecompressedLength(const TData& in) const = 0;
        virtual size_t MaxCompressedLength(const TData& in) const = 0;
        virtual size_t Compress(const TData& in, void* out) const = 0;
        virtual size_t Decompress(const TData& in, void* out) const = 0;

        virtual TStringBuf Name() const noexcept = 0;

        // some useful helpers
        void Encode(const TData& in, TBuffer& out) const;
        void Decode(const TData& in, TBuffer& out) const;

        void Encode(const TData& in, TString& out) const;
        void Decode(const TData& in, TString& out) const;

        inline TString Encode(const TData& in) const {
            TString out;

            Encode(in, out);

            return out;
        }

        inline TString Decode(const TData& in) const {
            TString out;

            Decode(in, out);

            return out;
        }
    private:
        size_t GetDecompressedLength(const TData& in) const;
    };

    using TCodecPtr = THolder<ICodec>;

    const ICodec* Codec(const TStringBuf& name);

    // some aux methods
    typedef TVector<TStringBuf> TCodecList;
    TCodecList ListAllCodecs();
    TString ListAllCodecsAsString();

    // SEARCH-8344: Get the size of max possible decompressed block
    size_t GetMaxPossibleDecompressedLength();
    // SEARCH-8344: Globally set the size of max possible decompressed block
    void SetMaxPossibleDecompressedLength(size_t maxPossibleDecompressedLength);

}
