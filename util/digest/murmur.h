#pragma once

#include <util/system/defaults.h>
#include <util/generic/array_ref.h>

/*
 * murmur2 from http://murmurhash.googlepages.com/
 *
 */
namespace NMurmurPrivate {
    Y_PURE_FUNCTION ui32 MurmurHash32(const void* key, size_t len, ui32 seed) noexcept;

    Y_PURE_FUNCTION ui64 MurmurHash64(const void* key, size_t len, ui64 seed) noexcept;

    template <unsigned N>
    struct TMurHelper;

#define DEF_MUR(t)                                                                         \
    template <>                                                                            \
    struct TMurHelper<t> {                                                                 \
        static inline ui##t MurmurHash(const void* buf, size_t len, ui##t init) noexcept { \
            return MurmurHash##t(buf, len, init);                                          \
        }                                                                                  \
    };

    DEF_MUR(32)
    DEF_MUR(64)

#undef DEF_MUR
} // namespace NMurmurPrivate

template <class T>
inline T MurmurHash(const void* buf, size_t len, T init) noexcept {
    return (T)NMurmurPrivate::TMurHelper<8 * sizeof(T)>::MurmurHash(buf, len, init);
}

template <class T>
inline T MurmurHash(const void* buf, size_t len) noexcept {
    return MurmurHash<T>(buf, len, (T)0);
}

template <typename TOut = size_t>
struct TMurmurHash {
    TOut operator()(const void* buf, size_t len) const noexcept {
        return MurmurHash<TOut>(buf, len);
    }

    template <typename ElementType>
    TOut operator()(const TArrayRef<ElementType>& data) const noexcept {
        return operator()(data.data(), data.size() * sizeof(ElementType));
    }
};
