#if defined(__arm__)
#include "msgbuf_arm.h"
#elif defined(__aarch64__)
#include "msgbuf_arm64.h"
#elif defined(__powerpc__)
#include "msgbuf_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "msgbuf_x86.h"
#else
#error unexpected
#endif
