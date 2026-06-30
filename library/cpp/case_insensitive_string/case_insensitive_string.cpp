#include "case_insensitive_string.h"

#include <library/cpp/digest/murmur/murmur.h>

#include <util/string/escape.h>

#include <array>

namespace {
    template <auto ToLower>
    struct TCaseInsensitiveHash {
        static size_t HashTail(TMurmurHash2A<size_t>& hash, const char* data, size_t size) noexcept {
            for (size_t i = 0; i < size; ++i) {
                char lower = ToLower(data[i]);
                hash.Update(&lower, 1);
            }
            return hash.Value();
        }

        static size_t ComputeHash(const char* s, size_t n) noexcept {
            TMurmurHash2A<size_t> hash;
            std::array<char, sizeof(size_t)> buf;
            size_t headSize = n - n % buf.size();
            for (size_t i = 0; i < headSize; i += buf.size()) {
                for (size_t j = 0; j < buf.size(); ++j) {
                    buf[j] = ToLower(s[i + j]);
                }
                hash.Update(buf.data(), buf.size());
            }
            return HashTail(hash, s + headSize, n - headSize);
        }
    };
}

size_t CaseInsensitiveStringHash(const char* s, size_t n) noexcept {
    return TCaseInsensitiveHash<static_cast<int(*)(int)>(std::tolower)>::ComputeHash(s, n);
}

size_t CaseInsensitiveAsciiStringHash(const char* s, size_t n) noexcept {
    return TCaseInsensitiveHash<static_cast<char(*)(char)>(AsciiToLower)>::ComputeHash(s, n);
}

#define Y_DEFINE_STRING_OUT(type)                       \
    template <>                                         \
    void Out<type>(IOutputStream& o, const type& p) {   \
        o.Write(p.data(), p.size());                    \
    }

Y_DEFINE_STRING_OUT(TCaseInsensitiveString);
Y_DEFINE_STRING_OUT(TCaseInsensitiveStringBuf);
Y_DEFINE_STRING_OUT(TCaseInsensitiveAsciiString);
Y_DEFINE_STRING_OUT(TCaseInsensitiveAsciiStringBuf);

#undef Y_DEFINE_STRING_OUT

#define Y_DEFINE_STRING_ESCAPE(type)                            \
    type EscapeC(const type& str) {                             \
        const auto result = EscapeC(str.data(), str.size());    \
        return {result.data(), result.size()};                  \
    }

Y_DEFINE_STRING_ESCAPE(TCaseInsensitiveString);
Y_DEFINE_STRING_ESCAPE(TCaseInsensitiveAsciiString);

#undef Y_DEFINE_STRING_ESCAPE
