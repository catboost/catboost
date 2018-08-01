#include "permutation.h"

#include <util/random/shuffle.h>

TVector<ui64> CreateOrderByKey(const TVector<ui64>& key) {
    TVector<ui64> indices(key.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(
        indices.begin(),
        indices.end(),
        [&key](ui64 i1, ui64 i2) {
            return key[i1] < key[i2];
        }
    );

    return indices;
}
