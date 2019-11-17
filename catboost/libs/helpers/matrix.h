#pragma once

#include <util/generic/vector.h>

template <class T>
inline void MakeZeroAverage(TVector<T>* res) {
    T average = 0;
    for (int i = 0; i < res->ysize(); ++i) {
        average += (*res)[i];
    }
    average /= res->ysize();
    for (int i = 0; i < res->ysize(); ++i) {
        (*res)[i] -= average;
    }
}

template <typename T, typename T2DArrayLike>
inline static TVector<TConstArrayRef<T>> To2DConstArrayRef(const T2DArrayLike& array) {
    auto arrayView = TVector<TConstArrayRef<T>>();
    for (const auto& subArray : array) {
        arrayView.emplace_back(subArray);
    }
    return arrayView;
}
