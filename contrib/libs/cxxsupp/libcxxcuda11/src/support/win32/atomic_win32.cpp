#include <intrin.h>
#include <cstdint>


_LIBCPP_BEGIN_NAMESPACE_STD

namespace {
static const int __msvc_locks_size = 1024;
volatile long __msvc_locks[__msvc_locks_size];

size_t __msvc_lock_hash(void* __p) {
    uintptr_t __num = reinterpret_cast<uintptr_t>(__p);
    return (__num ^ (__num >> 10)) & (__msvc_locks_size - 1);
}
}

void __msvc_lock(void* __p) {
    volatile long& __lock = __msvc_locks[__msvc_lock_hash(__p)];
    while (_InterlockedExchange(&__lock, 1) == 0) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __yield();
#endif
    }
}

void __msvc_unlock(void* __p) {
    volatile long& __lock = __msvc_locks[__msvc_lock_hash(__p)];
    _InterlockedExchange(&__lock, 0);
}

_LIBCPP_END_NAMESPACE_STD