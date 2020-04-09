#if defined(__arm__)
#include "swab_arm.h"
#elif defined(__aarch64__)
#include "swab_arm64.h"
#elif defined(__powerpc__)
#include "swab_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "swab_x86.h"
#else
#error unexpected
#endif
