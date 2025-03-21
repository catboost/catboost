#pragma once

#include "restorable_rng.h"

#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <numeric>
#include <type_traits>


template <typename DstIndexType, typename SrcIndexType>
TVector<DstIndexType> CreateOrderByKey(const TConstArrayRef<SrcIndexType> key) {
    TVector<DstIndexType> indices(key.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&key](SrcIndexType i1, SrcIndexType i2) {
            return key[i1] < key[i2];
        }
    );

    return indices;
}

template <typename IndexType>
bool IsPermutation(TVector<IndexType>&& indicesSubset) {
    static_assert(std::is_integral<IndexType>::value);

    const size_t size = indicesSubset.size();
    if (size == (size_t)0) {
        return true;
    }
    if (size > (size_t)Max<IndexType>()) {
        return false;
    }
    const IndexType sizeAsIndexType = (IndexType)size;

    IndexType i = (IndexType)0;
    while (true) {
        IndexType dstI = indicesSubset[i];
        if (dstI < (IndexType)i) {
            return false;
        }
        if (dstI >= sizeAsIndexType) {
            return false;
        }
        if (dstI == (IndexType)i) {
            ++i;
            if (i == sizeAsIndexType) {
                return true;
            }
        } else {
            if (dstI == indicesSubset[dstI]) {
                return false;
            }

            indicesSubset[i] = indicesSubset[dstI];
            indicesSubset[dstI] = dstI;
        }
    }
    Y_UNREACHABLE();
    return true; // make compiler happy
}

template <typename IndexType>
TVector<IndexType> InvertPermutation(const TVector<IndexType>& permutation) {
    TVector<IndexType> result(permutation.size());
    for (ui64 i = 0; i < permutation.size(); ++i) {
        result[permutation[i]] = i;
    }
    return result;
}


template <class T>
void CreateShuffledIndices(size_t size, TRestorableFastRng64* rand, TVector<T>* result) {
    static_assert(std::is_integral<T>::value);
    result->yresize(size);
    std::iota(result->begin(), result->end(), T(0));
    Shuffle(result->begin(), result->end(), *rand);
}
