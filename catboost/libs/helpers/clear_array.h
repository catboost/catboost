#pragma once

#include <util/generic/vector.h>

template <class T>
inline void Clear(TVector<T>* res, size_t size) {
    static_assert(std::is_pod<T>::value, "trying to memset non pod type");
    res->yresize(size);
    if (!res->empty()) {
        Fill(res->begin(), res->end(), 0);
    }
}
