#if defined(__arm__)
#include "param_arm.h"
#elif defined(__aarch64__)
#include "param_arm64.h"
#elif defined(__powerpc__)
#include "param_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "param_x86.h"
#else
#error unexpected
#endif
