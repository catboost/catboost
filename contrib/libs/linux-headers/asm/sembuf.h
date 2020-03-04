#if defined(__arm__)
#include "sembuf_arm.h"
#elif defined(__aarch64__)
#include "sembuf_arm64.h"
#elif defined(__powerpc__)
#include "sembuf_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "sembuf_x86.h"
#else
#error unexpected
#endif
