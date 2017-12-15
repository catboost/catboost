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
#   include "pyconfig.darwin.h"
#endif
