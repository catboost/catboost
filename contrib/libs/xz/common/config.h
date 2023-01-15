#pragma once

#include <util/system/platform.h>

#if defined(_android_)
#   if defined(_i386_)
#       include "config-android-i386.h"
#   elif defined(_x86_64_)
#       include "config-linux.h"
#   elif defined(_arm32_)
#       include "config-android-arm32.h"
#   elif defined(_arm64_)
#       include "config-android-arm64.h"
#   endif
#elif defined(_linux_)
#   include "config-linux.h"
#elif defined(_darwin_)
#   include "config-darwin.h"
#elif defined(_windows_)
#   include "config-windows.h"
#else
#   error "Unsupported platform"
#endif


