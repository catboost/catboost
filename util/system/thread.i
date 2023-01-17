//do not use directly
#pragma once
#include "platform.h"

#if defined(_win_)
    #include "winint.h"
    #include <process.h>

    typedef HANDLE THREADHANDLE;
#else
    #include <pthread.h>

    typedef pthread_t THREADHANDLE;
#endif

#if defined(_freebsd_)
    #include <pthread_np.h>
#endif

#include <util/digest/numeric.h>

static inline size_t SystemCurrentThreadIdImpl() noexcept {
    #if defined(_unix_)
        return (size_t)pthread_self();
    #elif defined(_win_)
        return (size_t)GetCurrentThreadId();
    #else
        #error todo
    #endif
}

template <class T>
static inline T ThreadIdHashFunction(T t) noexcept {
    /*
     * we must permute threadid bits, because some strange platforms(such Linux)
     * have strange threadid numeric properties
     *
     * Because they are alligned pointers to pthread_t rather that tid.
     * Currently there is no way to get tid without syscall (slightly slower)
     * (pthread_getthreadid_np is not implemeted in glibc/musl for some reason).
     */
    return IntHash(t);
}

static inline size_t SystemCurrentThreadId() noexcept {
    return ThreadIdHashFunction(SystemCurrentThreadIdImpl());
}
