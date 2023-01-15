#pragma once

#include <util/system/defaults.h>
#include <util/generic/array_ref.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MurmurHash was written by Austin Appleby.
 * Forward declared here to avoid inclusions of contrib/restricted
 */
Y_PURE_FUNCTION
uint32_t MurmurHash2(const void * key, size_t len, uint32_t seed) noexcept;

Y_PURE_FUNCTION
uint64_t MurmurHash64A(const void* key, size_t len, uint64_t seed) noexcept;

#ifdef __cplusplus
}
#endif

template <class T>
static inline std::enable_if_t<sizeof(T) == sizeof(uint32_t), T>
MurmurHash(const void* buf, size_t len, T init) noexcept {
    return MurmurHash2(buf, len, init);
}

template <class T>
static inline std::enable_if_t<sizeof(T) == sizeof(uint64_t), T>
MurmurHash(const void* buf, size_t len, T init) noexcept {
    return MurmurHash64A(buf, len, init);
}

template <class T>
static inline T MurmurHash(const void* buf, size_t len) noexcept {
    return MurmurHash<T>(buf, len, (T)0);
}

//non-inline version
size_t MurmurHashSizeT(const char* buf, size_t len) noexcept;

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
