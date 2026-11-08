#pragma once

#include "platform.h"

#if defined(_bionic_) || defined(__IOS__)
    #define USE_GENERIC_CONT
#elif defined(_cygwin_)
    #define USE_UCONTEXT_CONT
#elif defined(_win_)
    #define USE_FIBER_CONT
#elif (defined(_x86_64_) || defined(_arm64_)) && !defined(_k1om_)
    #define USE_JUMP_CONT
#else
    #define USE_UCONTEXT_CONT
#endif

#if (defined(USE_GENERIC_CONT) + defined(USE_UCONTEXT_CONT) + defined(USE_FIBER_CONT) + defined(USE_JUMP_CONT)) != 1
    #error "exactly one context backend must be selected"
#endif
