#pragma once

// Functions for manipulating signal sets

#include "compat.h"

#if defined _unix_
    #include <pthread.h>
#elif defined _win_
    // Flags for sigprocmask:
    #define SIG_BLOCK 1
    #define SIG_UNBLOCK 2
    #define SIG_SETMASK 3

using sigset_t = ui32;

#else
    #error not supported yet
#endif

inline int SigEmptySet(sigset_t* set) {
#if defined _unix_
    return sigemptyset(set);
#else
    Y_UNUSED(set);
    return 0;
#endif
}

inline int SigFillSet(sigset_t* set) {
#if defined _unix_
    return sigfillset(set);
#else
    Y_UNUSED(set);
    return 0;
#endif
}

inline int SigAddSet(sigset_t* set, int signo) {
#if defined _unix_
    return sigaddset(set, signo);
#else
    Y_UNUSED(set);
    Y_UNUSED(signo);
    return 0;
#endif
}

inline int SigDelSet(sigset_t* set, int signo) {
#if defined _unix_
    return sigdelset(set, signo);
#else
    Y_UNUSED(set);
    Y_UNUSED(signo);
    return 0;
#endif
}

inline int SigIsMember(const sigset_t* set, int signo) {
#if defined _unix_
    return sigismember(const_cast<sigset_t*>(set), signo);
#else
    Y_UNUSED(set);
    Y_UNUSED(signo);
    return 0;
#endif
}

inline int SigProcMask(int how, const sigset_t* set, sigset_t* oset) {
#if defined _unix_
    return pthread_sigmask(how, set, oset);
#else
    Y_UNUSED(set);
    Y_UNUSED(oset);
    Y_UNUSED(how);
    return 0;
#endif
}
