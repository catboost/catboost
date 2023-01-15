#if defined(__arm64__)
#include <ffi_arm64.h>
#elif defined(__i386__)
#include <ffi_i386.h>
#elif defined(__arm__)
#include <ffi_armv7.h>
#else
#include <ffi_x86_64.h>
#endif
