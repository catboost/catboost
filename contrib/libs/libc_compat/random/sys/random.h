#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

#define SYS_getrandom   318

#define GRND_NONBLOCK	0x0001
#define GRND_RANDOM		0x0002
#define GRND_INSECURE	0x0004

ssize_t getrandom(void* buf, size_t buflen, unsigned int flags);

#ifdef __cplusplus
} // extern "C"
#endif
