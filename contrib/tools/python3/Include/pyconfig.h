#pragma once

#if defined(Py_BUILD_CORE) || defined(Py_BUILD_CORE_BUILTIN) || defined(Py_BUILD_CORE_MODULE)
#define ABIFLAGS ""
#define PREFIX "/var/empty"
#define EXEC_PREFIX "/var/empty"
#define VERSION "3.10"
#define VPATH ""
#define BLAKE2_USE_SSE
#define USE_ZLIB_CRC32
#if defined(__linux__)
#define PLATFORM "linux"
#define MULTIARCH "x86_64-linux-gnu"
#define SOABI "cpython-310-x86_64-linux-gnu"
#elif defined(__APPLE__)
#define PLATFORM "darwin"
#define MULTIARCH "darwin"
#define SOABI "cpython-310-darwin"
#endif
#endif

#define PLATLIBDIR "lib"

#if defined(__linux__)
#include "pyconfig-linux.h"
#endif

#if defined(__APPLE__)
#  if defined(__arm64__)
#    include "pyconfig-osx-arm64.h"
#  else
#    include "pyconfig-osx.h"
#  endif
#endif

#if defined(_MSC_VER)
#define NTDDI_VERSION 0x06020000
#define _WIN32_WINNT 0x0602
#define Py_NO_ENABLE_SHARED
#include "../PC/pyconfig.h"
#endif

#if defined(_musl_)
#include "pyconfig-musl.h"
#endif

#if !defined(NDEBUG) && !defined(Py_DEBUG) && !defined(Py_LIMITED_API) && !defined(DISABLE_PYDEBUG)
#define Py_DEBUG
#define GC_NDEBUG
#endif
