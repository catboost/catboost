#pragma once

#if defined(Py_BUILD_CORE) || defined(Py_BUILD_CORE_BUILTIN) || defined(Py_BUILD_CORE_MODULE)

#define ABIFLAGS ""

// Need for Modules/getpath.c
#define PREFIX "/var/empty"
#define EXEC_PREFIX "/var/empty"
#define VERSION "3.12"
#define VPATH ""
#define PLATLIBDIR "lib"

#define USE_ZLIB_CRC32

#if defined(__linux__)
#   define PLATFORM "linux"
#   define MULTIARCH "x86_64-linux-gnu"
#   define SOABI "cpython-312-x86_64-linux-gnu"
#elif defined(__APPLE__)
#   define PLATFORM "darwin"
#   define MULTIARCH "darwin"
#   define SOABI "cpython-312-darwin"
#endif

#endif

#if !defined(NDEBUG) && !defined(Py_DEBUG) && !defined(Py_LIMITED_API) && !defined(DISABLE_PYDEBUG)
#define Py_DEBUG
#define GC_NDEBUG
#endif

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#   include "pyconfig-osx-arm64.h"
#elif defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
#   include "pyconfig-osx-x86_64.h"
#elif defined(_MSC_VER)
#   include "pyconfig-win.h"
#else
#   include "pyconfig-linux.h"
#endif

#if defined(_musl_)
#   include "pyconfig-musl.h"
#endif
