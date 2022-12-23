#if defined(__APPLE__) && defined(__IOS__) && defined(__i386__)
#   include "buildinf-ios-i386.h"
#elif defined(__APPLE__) && defined(__IOS__) && defined(__x86_64__)
#   include "buildinf-ios-x86_64.h"
#elif defined(__APPLE__) && defined(__IOS__) && defined(__aarch64__)
#   include "buildinf-ios-arm64.h"
#elif defined(__APPLE__) && defined(__IOS__) && defined(__arm__)
#   include "buildinf-ios-arm.h"
#elif defined(__APPLE__) && defined(__x86_64__)
#   include "buildinf-osx.h"
#elif defined(__APPLE__) && defined(__aarch64__)
#   include "buildinf-osx_arm64.h"
#elif defined(__linux__) && defined(__aarch64__)
#   include "buildinf-linux_aarch64.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "buildinf-win.h"
#elif defined(__ANDROID__) && defined(__x86_64__)
#   include "buildinf-android_x86_64.h"
#elif defined(__ANDROID__) && defined(__i686__)
#   include "buildinf-android_x86.h"
#elif defined(__ANDROID__) && defined(__aarch64__)
#   include "buildinf-android_arm64.h"
#elif defined(__ANDROID__) && defined(__arm__)
#   include "buildinf-android_arm.h"
#else
#   include "buildinf-linux.h"
#endif
