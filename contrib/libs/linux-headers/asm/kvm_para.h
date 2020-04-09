#if defined(__arm__)
#include "kvm_para_arm.h"
#elif defined(__aarch64__)
#include "kvm_para_arm64.h"
#elif defined(__powerpc__)
#include "kvm_para_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "kvm_para_x86.h"
#else
#error unexpected
#endif
