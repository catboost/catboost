#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#include "sve_context_arm64.h"
#elif defined(__powerpc__)
#error unavailable for powerpc
#elif defined(__i386__) || defined(__x86_64__)
#error unavailable for x86
#else
#error unexpected
#endif
