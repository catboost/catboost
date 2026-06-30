#pragma once

#include "case_insensitive_char_traits.h"

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/hash.h>
#include <util/string/split.h>

using TCaseInsensitiveString = TBasicString<char, TCaseInsensitiveCharTraits>;
using TCaseInsensitiveStringBuf = TBasicStringBuf<char, TCaseInsensitiveCharTraits>;

// WARN: comparison works incorrectly if strings contain null bytes (`TCaseInsensitiveAsciiString{"ab\0c", 4} == TCaseInsensitiveAsciiString{"ab\0d", 4}`).
using TCaseInsensitiveAsciiString = TBasicString<char, TCaseInsensitiveAsciiCharTraits>;
// WARN: comparison works incorrectly if strings contain null bytes.
using TCaseInsensitiveAsciiStringBuf = TBasicStringBuf<char, TCaseInsensitiveAsciiCharTraits>;

// Convert chars using std::tolower and hash the resulting string.
// Locale may affect the result.
size_t CaseInsensitiveStringHash(const char* s, size_t n) noexcept;
// Convert chars using AsciiToLower and hash the resulting string.
size_t CaseInsensitiveAsciiStringHash(const char* s, size_t n) noexcept;

template <>
struct THash<TCaseInsensitiveStringBuf> {
    size_t operator()(TCaseInsensitiveStringBuf str) const noexcept {
        return CaseInsensitiveStringHash(str.data(), str.size());
    }
};

template <>
struct THash<TCaseInsensitiveAsciiStringBuf> {
    size_t operator()(TCaseInsensitiveAsciiStringBuf str) const noexcept {
        return CaseInsensitiveAsciiStringHash(str.data(), str.size());
    }
};

template <>
struct THash<TCaseInsensitiveString> : THash<TCaseInsensitiveStringBuf> {};

template <>
struct THash<TCaseInsensitiveAsciiString> : THash<TCaseInsensitiveAsciiStringBuf> {};

namespace NStringSplitPrivate {

    template<>
    struct TStringBufOfImpl<TCaseInsensitiveStringBuf> {
        /*
         * WARN:
         * StringSplitter does not use TCharTraits properly.
         * Splitting such strings is explicitly disabled.
         */
        // using type = TCaseInsensitiveStringBuf;
    };

    template<>
    struct TStringBufOfImpl<TCaseInsensitiveString> : TStringBufOfImpl<TCaseInsensitiveStringBuf> {
    };

} // namespace NStringSplitPrivate

template <>
struct TEqualTo<TCaseInsensitiveString>: public TEqualTo<TCaseInsensitiveStringBuf> {
    using is_transparent = void;
};

template <>
struct TEqualTo<TCaseInsensitiveAsciiString>: public TEqualTo<TCaseInsensitiveAsciiStringBuf> {
    using is_transparent = void;
};
