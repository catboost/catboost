#ifndef HASH_INL_H_
#error "Direct inclusion of this file is not allowed, include hash.h"
// For the sake of sane code completion.
#include "hash.h"
#endif

namespace NYT {

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

////////////////////////////////////////////////////////////////////////////////

template <class TElement, class TUnderlying>
TRandomizedHash<TElement, TUnderlying>::TRandomizedHash()
    : Seed_(RandomNumber<size_t>())
{ }

template <class TElement, class TUnderlying>
size_t TRandomizedHash<TElement, TUnderlying>::operator ()(const TElement& element) const
{
    return Underlying_(element) + Seed_;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
