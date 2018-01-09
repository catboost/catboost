#pragma once

#include <util/system/types.h>

namespace NReducers {
    template <class T>
    class TSumReducer {
    public:
        inline static void Reduce(T* left, const T* right, ui64 size) {
            for (ui64 i = 0; i < size; ++i) {
                left[i] += right[i];
            }
        }
    };
}
