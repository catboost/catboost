#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#include "ucontext_arm64.h"
#elif defined(__powerpc__)
#include "ucontext_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "ucontext_x86.h"
#else
#error unexpected
#endif
