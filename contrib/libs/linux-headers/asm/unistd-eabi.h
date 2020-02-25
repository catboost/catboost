#if defined(__arm__)
#include "unistd-eabi_arm.h"
#elif defined(__aarch64__)
#error unavailable for arm64
#elif defined(__powerpc__)
#error unavailable for powerpc
#elif defined(__i386__) || defined(__x86_64__)
#error unavailable for x86
#else
#error unexpected
#endif
