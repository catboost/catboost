#pragma once

#if defined(__APPLE__)
#   include "config-osx.h"
#elif defined(_MSC_VER)
#   include "config-win.h"
#else
#   include "config-linux.h"
#endif

#if defined(__ANDROID__) && defined(__arm__)
#   include "config-android-arm.h"
#endif

#if defined(__ANDROID__) && defined(__aarch64__)
#   include "config-android-arm64.h"
#endif

#if defined(__ANDROID__) && defined(__i686__)
#   include "config-android-i686.h"
#endif
