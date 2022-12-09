#pragma once

#if defined(__ANDROID__) && defined(__arm__)
#   include "buildinf-android-arm.h"
#elif defined(__ANDROID__) && defined(__aarch64__)
#   include "buildinf-android-arm64.h"
#elif defined(__ANDROID__) && defined(__i686__)
#   include "buildinf-android-i686.h"
#elif defined(__ANDROID__) && defined(__x86_64__)
#   include "buildinf-android-x86_64.h"
#elif defined(__IOS__) && defined(__aarch64__)
#   include "buildinf-ios-arm64.h"
#elif defined(__IOS__) && defined(__x86_64__)
#   include "buildinf-ios-x86_64.h"
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "buildinf-osx-arm64.h"
#elif defined(__APPLE__)
#   include "buildinf-osx.h"
#elif defined(_MSC_VER)
#   include "buildinf-win.h"
#else
#   include "buildinf-linux.h"
#endif
