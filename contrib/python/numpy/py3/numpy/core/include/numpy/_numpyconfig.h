#pragma once

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "_numpyconfig-osx-arm64.h"
#elif defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
#   include "_numpyconfig-osx-x86_64.h"
#elif defined(_MSC_VER)
#   include "_numpyconfig-win.h"
#else
#   include "_numpyconfig-linux.h"
#endif
