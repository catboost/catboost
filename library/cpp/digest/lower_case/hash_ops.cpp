#include "lchash.h"
#include "lciter.h"
#include "hash_ops.h"

#include <util/generic/algorithm.h>

size_t TCIOps::operator()(const TStringBuf& s) const noexcept {
    return FnvCaseLess(s, (size_t)0xBEE);
}

size_t TCIOps::operator()(const char* s) const noexcept {
    return operator()(TStringBuf(s));
}

bool TCIOps::operator()(const TStringBuf& f, const TStringBuf& s) const noexcept {
    using TIter = TLowerCaseIterator<const char>;

    return (f.size() == s.size()) && Equal(TIter(f.begin()), TIter(f.end()), TIter(s.begin()));
}

bool TCIOps::operator()(const char* f, const char* s) const noexcept {
    return operator()(TStringBuf(f), TStringBuf(s));
}
