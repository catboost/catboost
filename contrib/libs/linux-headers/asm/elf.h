#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#error unavailable for arm64
#elif defined(__powerpc__)
#include "elf_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#error unavailable for x86
#else
#error unexpected
#endif
