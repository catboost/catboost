#include "cxxabi.h"

#if defined(__GNUC__)

#include <new>

extern "C" {
    void __cxa_throw_bad_array_length() {
        throw std::bad_alloc();
    }

    void __cxa_throw_bad_array_new_length() {
        throw std::bad_alloc();
    }
}

#endif
