#pragma once
#if defined(__IOS__)
#ifdef __arm64__
#include <ios/ffitarget_arm64.h>
#endif
#ifdef __i386__
#include <ios/ffitarget_i386.h>
#endif
#ifdef __arm__
#include <ios/ffitarget_armv7.h>
#endif
#ifdef __x86_64__
#include <ios/ffitarget_x86_64.h>
#endif

#else

#if defined(__linux__)
#define X86_64
#endif

#if defined(__APPLE__)
#define X86_DARWIN
#endif

#if defined(_MSC_VER)
#define X86_WIN64
#endif

#include "../src/x86/ffitarget.h"
#endif
