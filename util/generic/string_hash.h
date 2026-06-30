#pragma once

#include <cstddef>

// reduce code bloat and cycled includes, declare functions here
#if defined(_64_)
ui64 CityHash64(const char* buf, size_t len) noexcept;
#else
template <typename T>
T MurmurHash(const void* buf, size_t len) noexcept;
#endif

namespace NHashPrivate {
    template <typename C>
    size_t ComputeStringHash(const C* ptr, size_t size) noexcept {
#if defined(_64_)
        return CityHash64((const char*)ptr, size * sizeof(C));
#else
        return MurmurHash<size_t>(ptr, size * sizeof(C));
#endif
    }
} // namespace NHashPrivate
