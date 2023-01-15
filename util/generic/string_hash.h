#pragma once

#include <cstddef>

// reduce code bloat and cycled includes, declare functions here
#if defined(_64_) && !defined(NO_CITYHASH)
ui64 CityHash64(const char* buf, size_t len) noexcept;
#else
size_t MurmurHashSizeT(const char* buf, size_t len) noexcept;
#endif

namespace NHashPrivate {
    template <typename C>
    size_t ComputeStringHash(const C* ptr, size_t size) noexcept {
#if defined(_64_) && !defined(NO_CITYHASH)
        return CityHash64((const char *)ptr, size * sizeof(C));
#else
        return MurmurHashSizeT((const char *)ptr, size * sizeof(C));
#endif
    }
}
