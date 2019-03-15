#if defined(__APPLE__) && defined(__x86_64__)
#   include "buildinf-osx.h"
#elif defined(__linux__) && defined(__aarch64__)
#   include "buildinf-linux_aarch64.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "buildinf-win.h"
#else
#   include "buildinf-linux.h"
#endif
