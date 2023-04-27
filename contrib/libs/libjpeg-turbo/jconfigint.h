#pragma once

#if defined(__ANDROID__)
#   include "jconfigint-android.h"
#elif defined(__IOS__)
#   include "jconfigint-ios.h"
#elif defined(__APPLE__)
#   include "jconfigint-osx.h"
#elif defined(_MSC_VER)
#   include "jconfigint-win.h"
#else
#   include "jconfigint-linux.h"
#endif

#if defined(__arm__) || defined(__ARM__)
#   include "jconfigint-armv7a.h"
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#   include "jconfigint-armv8a.h"
#endif

#if defined(__i686__) || defined(_M_IX86)
#   include "jconfigint-x86.h"
#endif
