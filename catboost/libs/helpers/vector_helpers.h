#pragma once

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>

template<typename T>
static TVector<const T*> GetConstPointers(const TVector<THolder<T>>& pointers) {
    TVector<const T*> result(pointers.ysize());
    for (int i = 0; i < pointers.ysize(); ++i) {
        result[i] = pointers[i].Get();
    }
    return result;
}

inline bool IsConst(const TVector<float>& values) {
    if (values.empty()) {
        return true;
    }
    float minValue = *MinElement(values.begin(), values.end());
    float maxValue = *MaxElement(values.begin(), values.end());
    return minValue == maxValue;
}
