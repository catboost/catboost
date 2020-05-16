#pragma once

#if defined(__linux__)
#include "fficonfig-linux.h"
#endif

#if defined(__APPLE__)
#if defined(__IOS__)

#ifdef __arm64__
#include <ios/fficonfig_arm64.h>
#endif
#ifdef __i386__
#include <ios/fficonfig_i386.h>
#endif
#ifdef __arm__
#include <ios/fficonfig_armv7.h>
#endif
#ifdef __x86_64__
#include <ios/fficonfig_x86_64.h>
#endif

#else
#include "fficonfig-osx.h"

#endif // __IOS__
#endif // __APPLE__

#if defined(_MSC_VER)
#   if defined(_M_IX86)
#       include "fficonfig-win32.h"
#   else
#       include "fficonfig-win.h"
#   endif
#endif
