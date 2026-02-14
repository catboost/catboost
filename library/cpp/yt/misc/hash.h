#pragma once

#include <util/generic/hash.h>

#include <util/random/random.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Finalization function that makes each bit of the output depend on each bit of the input.
//! Needed to achieve unbiased distribution for bit-sensitive application like HLL,
//! as opposed to raw collision minimization.
//! This is also SplitMix64 PRNG.
//! Cf. |http://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html|, |boost::random::splitmix64|.
size_t SplitMix64(size_t value);

////////////////////////////////////////////////////////////////////////////////

//! Updates #h with #k.
//! Cf. |boost::hash_combine|.
void HashCombine(size_t& h, size_t k);

//! Updates #h with the hash of #k.
//! Cf. |boost::hash_combine|.
template <class T>
void HashCombine(size_t& h, const T& k);

//! Computes the hash of #value handling NaN values gracefully
//! (returning the same constant for all NaNs).
//! If |T| is not a floating-point type, #NaNSafeHash is equivalent to #THash.
template <class T>
size_t NaNSafeHash(const T& value);

////////////////////////////////////////////////////////////////////////////////

//! Provides a hasher that randomizes the results of another one.
//! \note In case seed value is 0, the hash is just the underlying hash.
template <class TElement, class TUnderlying = ::THash<TElement>>
class TRandomizedHash
{
public:
    TRandomizedHash();
    explicit TRandomizedHash(size_t seed);

    size_t operator()(const TElement& element) const;

private:
    size_t Seed_;
    TUnderlying Underlying_;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define HASH_INL_H_
#include "hash-inl.h"
#undef HASH_INL_H_
