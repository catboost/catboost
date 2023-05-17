#pragma once

#if defined(__APPLE__)
#   include "jemalloc-osx.h"
#elif defined(_MSC_VER)
#   include "jemalloc-win.h"
#elif defined(__linux__) && defined(__arm__)
#   include "jemalloc-linux-arm.h"
#else
#   include "jemalloc-linux.h"
#endif
