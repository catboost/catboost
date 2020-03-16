#if defined(__arm__)
#include "mman_arm.h"
#elif defined(__aarch64__)
#include "mman_arm64.h"
#elif defined(__powerpc__)
#include "mman_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "mman_x86.h"
#else
#error unexpected
#endif
