#include "case_insensitive_string.h"

#include <library/cpp/digest/murmur/murmur.h>

size_t THash<TCaseInsensitiveStringBuf>::operator()(TCaseInsensitiveStringBuf str) const noexcept {
    TMurmurHash2A<size_t> hash;
    for (size_t i = 0; i < str.size(); ++i) {
        char lower = std::tolower(str[i]);
        hash.Update(&lower, 1);
    }
    return hash.Value();
}

template <>
void Out<TCaseInsensitiveString>(IOutputStream& o, const TCaseInsensitiveString& p) {
    o.Write(p.data(), p.size());
}

template <>
void Out<TCaseInsensitiveStringBuf>(IOutputStream& o, const TCaseInsensitiveStringBuf& p) {
    o.Write(p.data(), p.size());
}
