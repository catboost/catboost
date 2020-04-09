#if defined(__arm__)
#include "ipcbuf_arm.h"
#elif defined(__aarch64__)
#include "ipcbuf_arm64.h"
#elif defined(__powerpc__)
#include "ipcbuf_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "ipcbuf_x86.h"
#else
#error unexpected
#endif
