#pragma once

#include <util/generic/vector.h>

template <class T>
void Clear(yvector<T>* res, size_t size) {
    res->yresize(size);
    if (res->empty())
        return;
    static_assert(std::is_pod<T>::value, "trying to memset non pod type");
    memset(&(*res)[0], 0, sizeof(T) * res->size());
}
