#if defined(__arm__)
#include "shmbuf_arm.h"
#elif defined(__aarch64__)
#include "shmbuf_arm64.h"
#elif defined(__powerpc__)
#include "shmbuf_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "shmbuf_x86.h"
#else
#error unexpected
#endif
