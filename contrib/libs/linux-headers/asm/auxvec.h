#if defined(__arm__)
#include "auxvec_arm.h"
#elif defined(__aarch64__)
#include "auxvec_arm64.h"
#elif defined(__powerpc__)
#include "auxvec_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "auxvec_x86.h"
#else
#error unexpected
#endif
