#pragma once

#include "defaults.h"

using WIN_HANDLE = void*;
#define INVALID_WIN_HANDLE ((WIN_HANDLE)(long)-1)

using UNIX_HANDLE = int;
#define INVALID_UNIX_HANDLE -1

#if defined(_win_)
using FHANDLE = WIN_HANDLE;
    #define INVALID_FHANDLE INVALID_WIN_HANDLE
#elif defined(_unix_)
using FHANDLE = UNIX_HANDLE;
    #define INVALID_FHANDLE INVALID_UNIX_HANDLE
#else
    #error
#endif

#if defined(_cygwin_)
using OS_HANDLE = WIN_HANDLE;
    #define INVALID_OS_HANDLE INVALID_WIN_HANDLE
#else
using OS_HANDLE = FHANDLE;
    #define INVALID_OS_HANDLE INVALID_FHANDLE
#endif
