#if defined(__arm__)
#include "bitsperlong_arm.h"
#elif defined(__aarch64__)
#include "bitsperlong_arm64.h"
#elif defined(__powerpc__)
#include "bitsperlong_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "bitsperlong_x86.h"
#else
#error unexpected
#endif
