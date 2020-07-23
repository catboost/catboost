#if defined(__arm__)
#error unavailable for arm
#elif defined(__aarch64__)
#include "kvm_arm64.h"
#elif defined(__powerpc__)
#include "kvm_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "kvm_x86.h"
#else
#error unexpected
#endif
