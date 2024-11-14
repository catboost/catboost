#pragma once

#if defined(__APPLE__)
#   include "llvm-config-osx.h"
#elif defined(_MSC_VER)
#   include "llvm-config-win.h"
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "llvm-config-linux-aarch64.h"
#else
#   include "llvm-config-linux.h"
#endif
