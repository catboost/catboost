#if defined(__arm__)
#include "ioctls_arm.h"
#elif defined(__aarch64__)
#include "ioctls_arm64.h"
#elif defined(__powerpc__)
#include "ioctls_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "ioctls_x86.h"
#else
#error unexpected
#endif
