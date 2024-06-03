#include "case_insensitive_string.h"

#include <library/cpp/digest/murmur/murmur.h>

#include <array>

static size_t HashTail(TMurmurHash2A<size_t>& hash, const char* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        char lower = std::tolower(data[i]);
        hash.Update(&lower, 1);
    }
    return hash.Value();
}

size_t THash<TCaseInsensitiveStringBuf>::operator()(TCaseInsensitiveStringBuf str) const noexcept {
    TMurmurHash2A<size_t> hash;
    std::array<char, sizeof(size_t)> buf;
    size_t headSize = str.size() - str.size() % buf.size();
    for (size_t i = 0; i < headSize; i += buf.size()) {
        for (size_t j = 0; j < buf.size(); ++j) {
            buf[j] = std::tolower(str[i + j]);
        }
        hash.Update(buf.data(), buf.size());
    }
    return HashTail(hash, str.data() + headSize, str.size() - headSize);
}

template <>
void Out<TCaseInsensitiveString>(IOutputStream& o, const TCaseInsensitiveString& p) {
    o.Write(p.data(), p.size());
}

template <>
void Out<TCaseInsensitiveStringBuf>(IOutputStream& o, const TCaseInsensitiveStringBuf& p) {
    o.Write(p.data(), p.size());
}
