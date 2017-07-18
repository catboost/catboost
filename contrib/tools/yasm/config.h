#if defined(__APPLE__) && defined(__x86_64__)
#   include "config-osx-linux.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "config-win.h"
#else
#   include "config-osx-linux.h"
#endif
