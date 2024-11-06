#pragma once

#include <util/generic/hash.h>

#include <util/random/random.h>

namespace NYT {

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
template <class TElement, class TUnderlying = ::THash<TElement>>
class TRandomizedHash
{
public:
    TRandomizedHash();
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
