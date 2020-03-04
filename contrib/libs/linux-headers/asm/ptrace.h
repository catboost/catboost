#if defined(__arm__)
#include "ptrace_arm.h"
#elif defined(__aarch64__)
#include "ptrace_arm64.h"
#elif defined(__powerpc__)
#include "ptrace_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "ptrace_x86.h"
#else
#error unexpected
#endif
