#pragma once

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "jemalloc_internal_defs-osx-arm64.h"
#elif defined(__APPLE__)
#   include "jemalloc_internal_defs-osx.h"
#elif defined(_MSC_VER)
#   include "jemalloc_internal_defs-win.h"
#elif defined(__linux__) && defined(__arm__)
#   include "jemalloc_internal_defs-linux-arm.h"
#else
#   include "jemalloc_internal_defs-linux.h"
#endif
