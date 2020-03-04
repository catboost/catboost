#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#error unavailable for arm64
#elif defined(__powerpc__)
#error unavailable for powerpc
#elif defined(__i386__) || defined(__x86_64__)
#include "ist_x86.h"
#else
#error unexpected
#endif
