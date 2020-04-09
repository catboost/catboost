#if defined(__arm__)
#include "socket_arm.h"
#elif defined(__aarch64__)
#include "socket_arm64.h"
#elif defined(__powerpc__)
#include "socket_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "socket_x86.h"
#else
#error unexpected
#endif
