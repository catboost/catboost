#pragma once

#include <util/generic/hash.h>
#include <util/generic/vector.h>

template <class T>
class TSocketMap {
public:
    T& Get(size_t idx) {
        if (idx < 128000) {
            if (V_.size() <= idx) {
                V_.resize(idx + 1);
            }

            return V_[idx];
        }

        return H_[idx];
    }

private:
    TVector<T> V_;
    THashMap<size_t, T> H_;
};
