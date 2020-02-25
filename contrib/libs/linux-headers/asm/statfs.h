#if defined(__arm__)
#include "statfs_arm.h"
#elif defined(__aarch64__)
#include "statfs_arm64.h"
#elif defined(__powerpc__)
#include "statfs_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "statfs_x86.h"
#else
#error unexpected
#endif
