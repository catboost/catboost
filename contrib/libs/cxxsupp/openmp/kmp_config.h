#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)
#   include "kmp_config-armv8a.h"
#else
#   include "kmp_config-linux.h"
#endif
