#pragma once

#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#   include "ares_build-win-x86_64.h"
#else
#   include "ares_build-linux.h"
#endif
