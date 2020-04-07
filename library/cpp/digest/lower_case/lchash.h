#pragma once

#include "lciter.h"

#include <util/digest/fnv.h>
#include <util/generic/strbuf.h>

template <class T>
static inline T FnvCaseLess(const char* b, size_t l, T t = 0) noexcept {
    using TIter = TLowerCaseIterator<const char>;

    return FnvHash(TIter(b), TIter(b + l), t);
}

template <class T>
static inline T FnvCaseLess(const TStringBuf& s, T t = 0) noexcept {
    return FnvCaseLess(s.data(), s.size(), t);
}
