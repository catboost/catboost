#pragma once

#include <contrib/libs/farmhash/farmhash.h>

namespace NYT {

using TFingerprint = ui64;

////////////////////////////////////////////////////////////////////////////////

static inline TFingerprint FarmHash(ui64 value)
{
    return ::util::Fingerprint(value);
}

static inline TFingerprint FarmHash(const void* buf, size_t len)
{
    return ::util::Hash64(static_cast<const char*>(buf), len);
}

static inline TFingerprint FarmHash(const void* buf, size_t len, ui64 seed)
{
    return ::util::Hash64WithSeed(static_cast<const char*>(buf), len, seed);
}

static inline TFingerprint FarmFingerprint(ui64 value)
{
    return ::util::Fingerprint(value);
}

static inline TFingerprint FarmFingerprint(const void* buf, size_t len)
{
    return ::util::Fingerprint64(static_cast<const char*>(buf), len);
}

static inline TFingerprint FarmFingerprint(ui64 first, ui64 second)
{
    return ::util::Fingerprint(::util::Uint128(first, second));
}

// Forever-fixed Google FarmHash fingerprint.
template <class T>
TFingerprint FarmFingerprint(const T* begin, const T* end)
{
    ui64 result = 0xdeadc0de;
    for (const auto* value = begin; value < end; ++value) {
        result = FarmFingerprint(result, FarmFingerprint(*value));
    }
    return result ^ (end - begin);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
