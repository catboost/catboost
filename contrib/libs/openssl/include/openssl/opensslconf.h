#pragma once

#if defined(__ANDROID__) && defined(__arm__)
#   include "opensslconf-android-arm.h"
#elif defined(__ANDROID__) && defined(__aarch64__)
#   include "opensslconf-android-arm64.h"
#elif defined(__ANDROID__) && defined(__i686__)
#   include "opensslconf-android-i686.h"
#elif defined(__ANDROID__) && defined(__x86_64__)
#   include "opensslconf-android-x86_64.h"
#elif defined(__IOS__) && defined(__aarch64__)
#   include "opensslconf-ios-arm64.h"
#elif defined(__IOS__) && defined(__x86_64__)
#   include "opensslconf-ios-x86_64.h"
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "opensslconf-osx-arm64.h"
#elif defined(__APPLE__)
#   include "opensslconf-osx.h"
#elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#   include "opensslconf-win-x86_64.h"
#elif defined(_MSC_VER) && (defined(__i686__) || defined(_M_IX86))
#   include "opensslconf-win-i686.h"
#elif defined(__linux__) && defined(__arm__)
#   include "opensslconf-linux-arm.h"
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "opensslconf-linux-aarch64.h"
#else
#   include "opensslconf-linux.h"
#endif
