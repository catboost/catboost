#pragma once

#include <util/system/platform.h>

#if defined(_linux_) || defined(__CYGWIN__)
#   include "jemalloc_internal_defs-linux.h"
#elif defined(_freebsd_)
#   include "jemalloc_internal_defs-freebsd.h"
#elif defined(_windows_)
#   include "jemalloc_internal_defs-windows.h"
#elif defined(_darwin_)
#   include "jemalloc_internal_defs-darwin.h"
#else
#   error There is no jemalloc_internal_defs-PLATFORM.h for this platform
#endif
