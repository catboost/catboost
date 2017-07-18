#pragma once

#include <cstdlib>

extern void ThrowBadAlloc();

inline void* y_allocate(size_t n) {
    void* r = malloc(n);

    if (r == nullptr) {
        ThrowBadAlloc();
    }

    return r;
}

inline void y_deallocate(void* p) {
    free(p);
}

/**
 * Behavior of realloc from C++99 to C++11 changed (http://www.cplusplus.com/reference/cstdlib/realloc/).
 *
 * Our implementation work as C++99: if new_sz == 0 free will be called on 'p' and nullptr returned.
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
        ThrowBadAlloc();
    }

    return r;
}
