#pragma once

#if defined(__APPLE__) && defined(__arm64__)
#   include "jemalloc_internal_defs-osx-arm64.h"
#elif defined(__APPLE__)
#   include "jemalloc_internal_defs-osx.h"
#elif defined(_MSC_VER)
#   include "jemalloc_internal_defs-win.h"
#else
#   include "jemalloc_internal_defs-linux.h"
#endif
