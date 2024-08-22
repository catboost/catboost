#pragma once

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "pyconfig-osx-arm64.h"
#elif defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
#   include "pyconfig-osx-x86_64.h"
#elif defined(_MSC_VER)
#   include "pyconfig-win.h"
#else
#   include "pyconfig-linux.h"
#endif

#if defined(_musl_)
#   include "pyconfig-musl.h"
#endif
