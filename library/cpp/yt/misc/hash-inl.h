#ifndef HASH_INL_H_
#error "Direct inclusion of this file is not allowed, include hash.h"
// For the sake of sane code completion.
#include "hash.h"
#endif

#include <cmath>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

inline size_t SplitMix(size_t value)
{
    // GOLDEN_GAMMA is the odd integer closest to (2 ^ {bit_depth}) / phi,
    // where phi = (1 + sqrt(5))/2 is the golden ratio.
    // Defined in https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf
    //
    // The second part of algorithm is finalization mix and uses some magic constants
    // For 32bit platforms it taken from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp#L70-L74
    // For 64bit from http://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
#ifdef _64_
    constexpr size_t GOLDEN_GAMMA = 0x9e3779b97f4a7c15ULL;
    constexpr size_t C1 = 0xbf58476d1ce4e5b9ULL;
    constexpr size_t C2 = 0x94d049bb133111ebULL;
    constexpr size_t S0 = 30;
    constexpr size_t S1 = 27;
    constexpr size_t S2 = 31;
#else
    constexpr size_t GOLDEN_GAMMA = 0x9e3779b9UL;
    constexpr size_t C1 = 0x85ebca6bUL;
    constexpr size_t C2 = 0xc2b2ae35UL;
    constexpr size_t S0 = 16;
    constexpr size_t S1 = 13;
    constexpr size_t S2 = 16;
#endif

    value += GOLDEN_GAMMA;
    value = (value ^ (value >> S0)) * C1;
    value = (value ^ (value >> S1)) * C2;
    return value ^ (value >> S2);
}

////////////////////////////////////////////////////////////////////////////////

inline void HashCombine(size_t& h, size_t k)
{
    // Combine step from MurmurHash https://github.com/abrandoned/murmur2/blob/master/MurmurHash2.c
    // 64bit constants are taken from MurmurHash64A, 32bit — MurmurHash2
#ifdef _64_
    constexpr size_t m = 0xc6a4a7935bd1e995ULL;
    constexpr size_t r = 47;
#else
    constexpr size_t m = 0x5bd1e995;
    constexpr size_t r = 24;
#endif

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
