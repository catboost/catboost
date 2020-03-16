#if defined(__arm__)
#include "fcntl_arm.h"
#elif defined(__aarch64__)
#include "fcntl_arm64.h"
#elif defined(__powerpc__)
#include "fcntl_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "fcntl_x86.h"
#else
#error unexpected
#endif
