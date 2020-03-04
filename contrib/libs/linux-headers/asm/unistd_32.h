#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#error unavailable for arm64
#elif defined(__powerpc__)
#include "unistd_32_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "unistd_32_x86.h"
#else
#error unexpected
#endif
