#include <util/system/platform.h>
#include <util/system/defaults.h>

#ifndef VERSION
#define VERSION Y_STRINGIZE(UNQUOTED_VERSION)
#endif
#define PYTHONPATH Y_STRINGIZE(UNQUOTED_PYTHONPATH)

#if !defined(PLATFORM) && defined(UNQUOTED_PLATFORM)
#   define PLATFORM Y_STRINGIZE(UNQUOTED_PLATFORM)
#endif

#ifdef _win_
#   include "pyconfig.win32.h"
#endif

#ifdef _linux_
#   include "pyconfig.linux.h"
#endif

#ifdef _cygwin_
#   include "pyconfig.cygwin.h"
#endif

#ifdef _freebsd_
#   include "pyconfig.freebsd.h"
#endif

#ifdef _darwin_
#   ifdef _ios_
#      if defined(_arm64_)
#         include "pyconfig.ios.arm64.h"
#      elif defined(_arm32_)
#         include "pyconfig.ios.armv7.h"
#      elif defined(_x86_64_)
#         include "pyconfig.ios.x86_64.h"
#      elif defined(_i386_)
#         include "pyconfig.ios.i386.h"
#      else
#         error "Unsupported architecture for ios"
#      endif // defined(_arm64_) || defined(_arm32_) || defined(_x86_64_) || defined(_i386_)
#   else
#       if defined(_arm64_)
#           include "pyconfig.darwin.arm64.h"
#       else 
#           include "pyconfig.darwin.h"
#       endif // _arm64_
#   endif // __IOS__
#endif
