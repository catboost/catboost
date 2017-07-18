#if defined(__APPLE__) && defined(__x86_64__)
#   include "opensslconf-osx.h"
#elif defined(__FreeBSD__) && defined(__x86_64__)
#   include "opensslconf-freebsd-linux.h"
#elif defined(__linux__) && defined(__aarch64__)
#   include "opensslconf-linux_aarch64.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "opensslconf-win.h"
#else
#   include "opensslconf-freebsd-linux.h"
#endif
