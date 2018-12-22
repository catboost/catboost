#pragma once

#include "restorable_rng.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/random/shuffle.h>
#include <util/system/types.h>

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
