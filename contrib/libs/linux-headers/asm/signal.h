#if defined(__arm__)
#include "signal_arm.h"
#elif defined(__aarch64__)
#include "signal_arm64.h"
#elif defined(__powerpc__)
#include "signal_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "signal_x86.h"
#else
#error unexpected
#endif
