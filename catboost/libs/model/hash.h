#pragma once

#include <util/generic/vector.h>

/*
* Specially designed hash functions for low
* collision rate.
*/

const ui64 MAGIC_MULT = 0x4906ba494954cb65ull;
inline ui64 CalcHash(ui64 a, ui64 b) {
    return MAGIC_MULT * (a + MAGIC_MULT * b);
}

template <typename T>
struct TVecHash {
    int operator()(const yvector<T>& a) const {
        int res = 1988712;
        for (int i = 0; i < a.ysize(); ++i)
            res = 984121 * res + a[i].GetHash();
        return res;
    }
};

template <>
struct TVecHash<int> {
    int operator()(const yvector<int>& a) const {
        int res = 1988712;
        for (int i = 0; i < a.ysize(); ++i)
            res = 984121 * res + a[i];
        return res;
    }
};
