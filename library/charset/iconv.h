#pragma once

#include "codepage.h"

#include <util/generic/noncopyable.h>

// WARNING: Do not use this functions - use functions from wide.h or recyr.hh instead.

namespace NICONVPrivate {
    inline const char* CharsetName(ECharset code) {
        return NameByCharset(code);
    }

    inline const char* CharsetName(const char* code) {
        return code;
    }

    template <int size>
    inline const char* UnicodeNameBySize();

    template <>
    inline const char* UnicodeNameBySize<1>() {
        return "UTF-8";
    }

    template <>
    inline const char* UnicodeNameBySize<2>() {
        return "UTF-16LE";
    }

    template <>
    inline const char* UnicodeNameBySize<4>() {
        return "UCS-4LE";
    }

    template <class C>
    inline const char* UnicodeName() {
        return UnicodeNameBySize<sizeof(C)>();
    }

    class TDescriptor : NNonCopyable::TNonCopyable {
    private:
        void* Descriptor_;
        const char* From_;
        const char* To_;

    public:
        template <class TFrom, class TTo>
        inline TDescriptor(TFrom from, TTo to)
            : TDescriptor(CharsetName(from), CharsetName(to))
        {
        }

        TDescriptor(const char* from, const char* to);

        ~TDescriptor();

        inline void* Get() const {
            return Descriptor_;
        }

        inline bool Invalid() const {
            return Descriptor_ == (void*)(-1);
        }

        inline const char* From() const noexcept {
            return From_;
        }

        inline const char* To() const noexcept {
            return To_;
        }
    };

    template <class TFrom, class TTo>
    inline bool CanConvert(TFrom from, TTo to) {
        TDescriptor descriptor(from, to);

        return !descriptor.Invalid();
    }

    size_t RecodeImpl(const TDescriptor& descriptor, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written);
    void DoRecode(const TDescriptor& descriptor, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written);

    template <class TFrom, class TTo>
    inline void Recode(TFrom from, TTo to, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        TDescriptor descriptor(from, to);

        DoRecode(descriptor, in, out, inSize, outSize, read, written);
    }

    template <class TCharType>
    inline void RecodeToUnicode(ECharset from, const char* in, TCharType* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        const size_t charSize = sizeof(TCharType);

        Recode(from, UnicodeName<TCharType>(), in, reinterpret_cast<char*>(out), inSize, outSize * charSize, read, written);
        written /= charSize;
    }

    template <class TCharType>
    inline void RecodeFromUnicode(ECharset to, const TCharType* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        const size_t charSize = sizeof(TCharType);

        Recode(UnicodeName<TCharType>(), to, reinterpret_cast<const char*>(in), out, inSize * charSize, outSize, read, written);
        read /= charSize;
    }

    RECODE_RESULT DoRecodeNoThrow(const TDescriptor& d, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written);

    template <class TFrom, class TTo>
    inline RECODE_RESULT RecodeNoThrow(TFrom from, TTo to, const char* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        TDescriptor descriptor(from, to);

        return DoRecodeNoThrow(descriptor, in, out, inSize, outSize, read, written);
    }

    template <class TCharType>
    inline RECODE_RESULT RecodeToUnicodeNoThrow(ECharset from, const char* in, TCharType* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        const size_t charSize = sizeof(TCharType);

        RECODE_RESULT res = RecodeNoThrow(from, UnicodeName<TCharType>(), in, reinterpret_cast<char*>(out), inSize, outSize * charSize, read, written);
        written /= charSize;

        return res;
    }

    template <class TCharType>
    inline RECODE_RESULT RecodeFromUnicodeNoThrow(ECharset to, const TCharType* in, char* out, size_t inSize, size_t outSize, size_t& read, size_t& written) {
        const size_t charSize = sizeof(TCharType);

        RECODE_RESULT res = RecodeNoThrow(UnicodeName<TCharType>(), to, reinterpret_cast<const char*>(in), out, inSize * charSize, outSize, read, written);
        read /= charSize;

        return res;
    }
}
