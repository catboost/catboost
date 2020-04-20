#if defined(__arm__)
#include "stat_arm.h"
#elif defined(__aarch64__)
#include "stat_arm64.h"
#elif defined(__powerpc__)
#include "stat_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "stat_x86.h"
#else
#error unexpected
#endif
