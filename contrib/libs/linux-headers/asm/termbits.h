#if defined(__arm__)
#include "termbits_arm.h"
#elif defined(__aarch64__)
#include "termbits_arm64.h"
#elif defined(__powerpc__)
#include "termbits_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "termbits_x86.h"
#else
#error unexpected
#endif
