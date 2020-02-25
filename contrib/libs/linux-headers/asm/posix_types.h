#if defined(__arm__)
#include "posix_types_arm.h"
#elif defined(__aarch64__)
#include "posix_types_arm64.h"
#elif defined(__powerpc__)
#include "posix_types_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "posix_types_x86.h"
#else
#error unexpected
#endif
