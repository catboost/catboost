#pragma once

#if defined(__APPLE__)
#   include "jemalloc-osx.h"
#elif defined(_MSC_VER)
#   include "jemalloc-win.h"
#else
#   include "jemalloc-linux.h"
#endif
