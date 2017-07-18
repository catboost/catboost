#pragma once
#include "codecs.h"

#include <util/ysaveload.h>
#include <util/stream/null.h>
#include <util/stream/mem.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/system/align.h>
#include <util/system/unaligned_mem.h>
#include <util/generic/hash.h>
#include <util/generic/cast.h>
#include <util/generic/buffer.h>
#include <util/generic/region.h>
#include <util/generic/singleton.h>
#include <util/generic/algorithm.h>
#include <util/generic/mem_copy.h>

namespace NBlockCodecs {
    struct TDecompressError: public TDataError {
        TDecompressError(int code) {
            *this << "cannot decompress (errcode " << code << ")";
        }

        TDecompressError(size_t exp, size_t real) {
            *this << "broken input (expected len: " << exp << ", got: " << real << ")";
        }
    };

    struct TCompressError: public TDataError {
        TCompressError(int code) {
            *this << "cannot compress (errcode " << code << ")";
        }
    };

    using TCodecPtr = THolder<ICodec>;

    struct TNullCodec: public ICodec {
        size_t DecompressedLength(const TData& in) const override {
            return +in;
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return +in;
        }

        size_t Compress(const TData& in, void* out) const override {
            MemCopy((char*)out, ~in, +in);

            return +in;
        }

        size_t Decompress(const TData& in, void* out) const override {
            MemCopy((char*)out, ~in, +in);

            return +in;
        }

        TStringBuf Name() const noexcept override {
            return STRINGBUF("null");
        }
    };

    template <class T>
    struct TAddLengthCodec: public ICodec {
        static inline void Check(const TData& in) {
            if (+in < sizeof(ui64)) {
                ythrow TDataError() << "too small input";
            }
        }

        size_t DecompressedLength(const TData& in) const override {
            Check(in);

            return *(const ui64*)~in;
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return T::DoMaxCompressedLength(+in) + sizeof(ui64);
        }

        size_t Compress(const TData& in, void* out) const override {
            ui64* ptr = (ui64*)out;

            WriteUnaligned(ptr, (ui64)+in);

            return Base()->DoCompress(!in ? TData(STRINGBUF("")) : in, ptr + 1) + sizeof(*ptr);
        }

        size_t Decompress(const TData& in, void* out) const override {
            Check(in);

            const ui64* ptr = (const ui64*)~in;

            if (!*ptr)
                return 0;

            Base()->DoDecompress(TData(in).Skip(sizeof(*ptr)), out, *ptr);
            return *ptr;
        }

        inline const T* Base() const noexcept {
            return static_cast<const T*>(this);
        }
    };
}
