#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#include "unistd_64_arm64.h"
#elif defined(__powerpc__)
#include "unistd_64_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "unistd_64_x86.h"
#else
#error unexpected
#endif
