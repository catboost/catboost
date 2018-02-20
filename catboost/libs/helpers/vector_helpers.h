#pragma once

#include <util/generic/vector.h>

template<typename T>
static TVector<const T*> GetConstPointers(const TVector<THolder<T>>& pointers) {
    TVector<const T*> result(pointers.ysize());
    for (int i = 0; i < pointers.ysize(); ++i) {
        result[i] = pointers[i].Get();
    }
    return result;
}

