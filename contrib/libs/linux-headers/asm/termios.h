#if defined(__arm__)
#include "termios_arm.h"
#elif defined(__aarch64__)
#include "termios_arm64.h"
#elif defined(__powerpc__)
#include "termios_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "termios_x86.h"
#else
#error unexpected
#endif
