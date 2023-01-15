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
#include <util/generic/array_ref.h>
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

    struct TNullCodec: public ICodec {
        size_t DecompressedLength(const TData& in) const override {
            return in.size();
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return in.size();
        }

        size_t Compress(const TData& in, void* out) const override {
            MemCopy((char*)out, in.data(), in.size());

            return in.size();
        }

        size_t Decompress(const TData& in, void* out) const override {
            MemCopy((char*)out, in.data(), in.size());

            return in.size();
        }

        TStringBuf Name() const noexcept override {
            return TStringBuf("null");
        }
    };

    template <class T>
    struct TAddLengthCodec: public ICodec {
        static inline void Check(const TData& in) {
            if (in.size() < sizeof(ui64)) {
                ythrow TDataError() << "too small input";
            }
        }

        size_t DecompressedLength(const TData& in) const override {
            Check(in);

            return ReadUnaligned<ui64>(in.data());
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return T::DoMaxCompressedLength(in.size()) + sizeof(ui64);
        }

        size_t Compress(const TData& in, void* out) const override {
            ui64* ptr = (ui64*)out;

            WriteUnaligned<ui64>(ptr, (ui64) in.size());

            return Base()->DoCompress(!in ? TData(TStringBuf("")) : in, ptr + 1) + sizeof(*ptr);
        }

        size_t Decompress(const TData& in, void* out) const override {
            Check(in);

            const auto len = ReadUnaligned<ui64>(in.data());

            if (!len)
                return 0;

            Base()->DoDecompress(TData(in).Skip(sizeof(len)), out, len);
            return len;
        }

        inline const T* Base() const noexcept {
            return static_cast<const T*>(this);
        }
    };
}
