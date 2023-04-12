#pragma once

#if defined(_MSC_VER)
#   include "config-win.h"
#else
#   include "config-linux.h"
#endif

#if defined(__arm__) || defined(__ARM__)
#   include "config-armv7a.h"
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#   include "config-armv8a.h"
#endif
