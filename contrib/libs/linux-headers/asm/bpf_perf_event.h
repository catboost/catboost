#if defined(__arm__)
#include "bpf_perf_event_arm.h"
#elif defined(__aarch64__)
#include "bpf_perf_event_arm64.h"
#elif defined(__powerpc__)
#include "bpf_perf_event_powerpc.h"
#elif defined(__i386__) || defined(__x86_64__)
#include "bpf_perf_event_x86.h"
#else
#error unexpected
#endif
