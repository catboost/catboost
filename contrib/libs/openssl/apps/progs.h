#if defined(__linux__) && defined(__aarch64__)
#   include "progs-linux_aarch64-linux.h"
#elif defined(_MSC_VER) && defined(_M_X64)
#   include "progs-win.h"
#else
#   include "progs-linux_aarch64-linux.h"
#endif
