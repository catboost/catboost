#pragma once

#include <util/generic/buffer.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

namespace NBlockCodecs {
    struct TData: public TStringBuf {
        inline TData() {
        }

        template <class T>
        inline TData(const T& t)
            : TStringBuf(~t, +t)
        {
        }
    };

    struct TCodecError: public yexception {
    };

    struct TNotFound: public TCodecError {
    };

    struct TDataError: public TCodecError {
    };

    struct ICodec {
        virtual ~ICodec();

        //main interface
        virtual size_t DecompressedLength(const TData& in) const = 0;
        virtual size_t MaxCompressedLength(const TData& in) const = 0;
        virtual size_t Compress(const TData& in, void* out) const = 0;
        virtual size_t Decompress(const TData& in, void* out) const = 0;

        virtual TStringBuf Name() const noexcept = 0;

        //some useful helpers
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
    };

    const ICodec* Codec(const TStringBuf& name);

    //some aux methods
    typedef yvector<TStringBuf> TCodecList;
    TCodecList ListAllCodecs();
    TString ListAllCodecsAsString();
}
