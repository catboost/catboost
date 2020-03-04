#if defined(__arm__)
#include "types_arm.h"
#elif defined(__aarch64__)
#include "types_arm64.h"
#elif defined(__powerpc__)
#include "types_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "types_x86.h"
#else
#error unexpected
#endif
