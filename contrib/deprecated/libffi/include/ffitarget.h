#pragma once

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
