#if defined(__APPLE__) && defined(__x86_64__)
#   include "dso_conf-osx-linux_aarch64-linux.h"
#elif defined(__linux__) && defined(__aarch64__)
#   include "dso_conf-osx-linux_aarch64-linux.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "dso_conf-win.h"
#else
#   include "dso_conf-osx-linux_aarch64-linux.h"
#endif
