#pragma once

#if defined(__ANDROID__)
#   include "dso_conf-android.h"
#elif defined(__IOS__) && defined(__aarch64__)
#   include "dso_conf-ios-arm64.h"
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "dso_conf-osx-arm64.h"
#elif defined(_MSC_VER)
#   include "dso_conf-win.h"
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "dso_conf-linux-aarch64.h"
#else
#   include "dso_conf-linux.h"
#endif
