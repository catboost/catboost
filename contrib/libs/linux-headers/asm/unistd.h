#if defined(__arm__)
#include "unistd_arm.h"
#elif defined(__aarch64__)
#include "unistd_arm64.h"
#elif defined(__powerpc__)
#include "unistd_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "unistd_x86.h"
#else
#error unexpected
#endif
