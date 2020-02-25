#if defined(__arm__)
#include "setup_arm.h"
#elif defined(__aarch64__)
#include "setup_arm64.h"
#elif defined(__powerpc__)
#include "setup_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "setup_x86.h"
#else
#error unexpected
#endif
