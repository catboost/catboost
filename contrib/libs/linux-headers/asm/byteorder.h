#if defined(__arm__)
#include "byteorder_arm.h"
#elif defined(__aarch64__)
#include "byteorder_arm64.h"
#elif defined(__powerpc__)
#include "byteorder_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "byteorder_x86.h"
#else
#error unexpected
#endif
