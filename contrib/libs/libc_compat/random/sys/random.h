#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

#if !defined(SYS_getrandom)
#if defined(__x86_64__)
    #define SYS_getrandom 318
#elif defined(__i386__)
    #define SYS_getrandom 355
#elif defined(__aarch64__)
    #define SYS_getrandom 278
#elif defined(__arm__)
    #define SYS_getrandom 384
#elif defined(__powerpc__)
    #define SYS_getrandom 359
#else
#error Unsupported platform
#endif
#endif

#define GRND_NONBLOCK	0x0001
#define GRND_RANDOM		0x0002
#define GRND_INSECURE	0x0004

ssize_t getrandom(void* buf, size_t buflen, unsigned int flags);

#ifdef __cplusplus
} // extern "C"
#endif
