#pragma once

#if defined(__ANDROID__) && defined(__arm__)
#   include "bn_conf-android-arm.h"
#elif defined(__ANDROID__) && defined(__aarch64__)
#   include "bn_conf-android-arm64.h"
#elif defined(__ANDROID__) && defined(__i686__)
#   include "bn_conf-android-i686.h"
#elif defined(__ANDROID__) && defined(__x86_64__)
#   include "bn_conf-android-x86_64.h"
#elif defined(__IOS__) && defined(__aarch64__)
#   include "bn_conf-ios-arm64.h"
#elif defined(__IOS__) && defined(__x86_64__)
#   include "bn_conf-ios-x86_64.h"
#elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#   include "bn_conf-win-x86_64.h"
#elif defined(_MSC_VER) && (defined(__i686__) || defined(_M_IX86))
#   include "bn_conf-win-i686.h"
#elif defined(__linux__) && defined(__arm__)
#   include "bn_conf-linux-arm.h"
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "bn_conf-linux-aarch64.h"
#else
#   include "bn_conf-linux.h"
#endif
