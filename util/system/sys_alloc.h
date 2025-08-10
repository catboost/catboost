#pragma once

#include <util/system/compiler.h>

#include <cstdlib>
#include <new>

inline void* y_allocate(size_t n) {
    void* r = malloc(n);

    if (r == nullptr) {
        throw std::bad_alloc();
    }

    return r;
}

inline void y_deallocate(void* p) {
    free(p);
}

/**
 * Behavior of 'realloc' changed from C++98 to C++11 (http://www.cplusplus.com/reference/cstdlib/realloc/).
 *
 * Our implementation works as in C++98: if new_sz == 0 'free' will be called on 'p' and 'nullptr' returned.
 */
inline void* y_reallocate(void* p, size_t new_sz) {
    if (!new_sz) {
        if (p) {
            free(p);
        }

        return nullptr;
    }

    void* r = realloc(p, new_sz);

    if (r == nullptr) {
        throw std::bad_alloc();
    }

    return r;
}
