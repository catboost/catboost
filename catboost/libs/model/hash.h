#pragma once

#include <util/generic/vector.h>
#include <util/system/types.h>

/*
* Specially designed hash functions for low
* collision rate.
*/

inline ui64 CalcHash(ui64 a, ui64 b) {
    const static constexpr ui64 MAGIC_MULT = 0x4906ba494954cb65ull;
    return MAGIC_MULT * (a + MAGIC_MULT * b);
}

template <typename T>
struct TVecHash {
    int operator()(const TVector<T>& a) const {
        ui32 res = 1988712;
        for (int i = 0; i < a.ysize(); ++i)
            res = 984121 * res + a[i].GetHash();
        return static_cast<int>(res);
    }
};

template <>
struct TVecHash<int> {
    int operator()(const TVector<int>& a) const {
        ui32 res = 1988712;
        for (int i = 0; i < a.ysize(); ++i)
            res = 984121 * res + static_cast<size_t>(a[i]);
        return static_cast<int>(res);
    }
};
