#pragma once

#include <inttypes.h>

#if !defined(__WORDSIZE)
    #define __WORDSIZE (sizeof(unsigned long) * 8)
#endif

#define DISALLOW_COPY_AND_ASSIGN(x)
#define RunningOnValgrind() false
#define COMPILE_ASSERT(x, y)
