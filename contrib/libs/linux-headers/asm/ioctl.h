#if defined(__arm__)
#include "ioctl_arm.h"
#elif defined(__aarch64__)
#include "ioctl_arm64.h"
#elif defined(__powerpc__)
#include "ioctl_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "ioctl_x86.h"
#else
#error unexpected
#endif
