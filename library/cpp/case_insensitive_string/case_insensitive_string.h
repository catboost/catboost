#pragma once

#include "case_insensitive_char_traits.h"

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/hash.h>
#include <util/string/split.h>

using TCaseInsensitiveString = TBasicString<char, TCaseInsensitiveCharTraits>;
using TCaseInsensitiveStringBuf = TBasicStringBuf<char, TCaseInsensitiveCharTraits>;

template <>
struct THash<TCaseInsensitiveStringBuf> {
    size_t operator()(TCaseInsensitiveStringBuf str) const noexcept;
};

template <>
struct THash<TCaseInsensitiveString> : THash<TCaseInsensitiveStringBuf> {};

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
