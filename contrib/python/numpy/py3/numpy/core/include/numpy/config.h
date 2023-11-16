#pragma once

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "config-osx-arm64.h"
#elif defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
#   include "config-osx-x86_64.h"
#elif defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#   include "config-win-x86_64.h"
#elif defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
#   include "config-linux-x86_64.h"
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
#    include "config-linux-aarch64.h"
#elif defined(__linux__) && defined(__powerpc__)
#    include "config-linux-ppc64.h"
#else
#    error "We do not have numpy for your OS / Arch combination"
#endif
