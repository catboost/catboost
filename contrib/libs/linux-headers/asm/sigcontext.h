#if defined(__arm__)
#include "sigcontext_arm.h"
#elif defined(__aarch64__)
#include "sigcontext_arm64.h"
#elif defined(__powerpc__)
#include "sigcontext_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "sigcontext_x86.h"
#else
#error unexpected
#endif
