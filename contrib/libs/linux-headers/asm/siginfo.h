#if defined(__arm__)
#include "siginfo_arm.h"
#elif defined(__aarch64__)
#include "siginfo_arm64.h"
#elif defined(__powerpc__)
#include "siginfo_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "siginfo_x86.h"
#else
#error unexpected
#endif
