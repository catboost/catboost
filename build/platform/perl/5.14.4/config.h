#if defined(__linux__)
#   include "config.linux.h"
#elif defined(__FreeBSD__)
#   include "config.freebsd.h"
#elif defined(__APPLE__)
#   include "config.darwin.h"
#elif defined(_WIN64)
//  No arcadia perl for Windows now
#endif
