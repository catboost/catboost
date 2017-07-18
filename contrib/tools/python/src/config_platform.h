#include <util/system/platform.h>

#ifdef _win_
#   define _WIN32_
#endif

#ifdef _linux_
#   define _UNIX_
#   define _LINUX_
#endif

#ifdef _freebsd_
#   define _UNIX_
#   define _FREEBSD_
#endif

#ifdef _darwin_
#   define _UNIX_
#   define _DARWIN_
#endif

#ifdef _cygwin_
#   define _UNIX_
#   define _CYGWIN_
#endif

#ifdef _arm_
#   define _ARM_
#endif
