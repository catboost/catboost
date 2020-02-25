#if defined(__arm__)
#include "perf_regs_arm.h"
#elif defined(__aarch64__)
#include "perf_regs_arm64.h"
#elif defined(__powerpc__)
#include "perf_regs_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "perf_regs_x86.h"
#else
#error unexpected
#endif
