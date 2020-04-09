#if defined(__arm__)
#include "errno_arm.h"
#elif defined(__aarch64__)
#include "errno_arm64.h"
#elif defined(__powerpc__)
#include "errno_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "errno_x86.h"
#else
#error unexpected
#endif
