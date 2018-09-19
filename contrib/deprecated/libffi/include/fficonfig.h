#pragma once

#if defined(__linux__)
#include "fficonfig-linux.h"
#endif

#if defined(__APPLE__)
#include "fficonfig-osx.h"
#endif

#if defined(_MSC_VER)
#include "fficonfig-win.h"
#endif
