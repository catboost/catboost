#ifndef HASH_INL_H_
#error "Direct inclusion of this file is not allowed, include hash.h"
// For the sake of sane code completion.
#include "hash.h"
#endif

#include <cmath>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline size_t SplitMix64(size_t value)
{
    static_assert(sizeof(size_t) == 8, "size_t must be 64 bit.");

    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

////////////////////////////////////////////////////////////////////////////////

inline void HashCombine(size_t& h, size_t k)
{
    static_assert(sizeof(size_t) == 8, "size_t must be 64 bit.");

    const size_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
}

template <class T>
void HashCombine(size_t& h, const T& k)
{
    HashCombine(h, THash<T>()(k));
}

template <class T>
Y_FORCE_INLINE size_t NaNSafeHash(const T& value)
{
    return ::THash<T>()(value);
}

template <class T>
    requires std::is_floating_point_v<T>
Y_FORCE_INLINE size_t NaNSafeHash(const T& value)
{
    return std::isnan(value) ? 0 : ::THash<T>()(value);
}

////////////////////////////////////////////////////////////////////////////////

template <class TElement, class TUnderlying>
TRandomizedHash<TElement, TUnderlying>::TRandomizedHash()
    : Seed_(RandomNumber<size_t>())
{ }

template <class TElement, class TUnderlying>
TRandomizedHash<TElement, TUnderlying>::TRandomizedHash(size_t seed)
    : Seed_(seed)
{ }

template <class TElement, class TUnderlying>
size_t TRandomizedHash<TElement, TUnderlying>::operator ()(const TElement& element) const
{
    auto result = Seed_;
    HashCombine(result, Underlying_(element));
    return result;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
