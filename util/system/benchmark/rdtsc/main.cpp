#include <library/testing/benchmark/bench.h>

#include <util/system/datetime.h>
#include <util/generic/xrange.h>

Y_FORCE_INLINE ui64 GetCycleCountLinux() {
    unsigned hi, lo;
    __asm__ __volatile__("lfence\n"
                         "rdtsc"
                         : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

Y_FORCE_INLINE ui64 GetCycleCountAgri1() {
    unsigned hi, lo;

    __asm__ __volatile__("rdtscp\n"
                         : "=a"(lo), "=d"(hi)::"%rbx", "%rcx");

    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

Y_FORCE_INLINE ui64 GetCycleCountAgri2() {
    unsigned hi, lo;
    __asm__ __volatile__("rdtscp\n"
                         : "=a"(lo), "=d"(hi)::"%rbx", "%rcx");
    /* call cpuid to prevent out of order execution */
    __asm__ __volatile__("mov $0, %%eax\n"
                         "cpuid\n" ::
                             : "%eax");

    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

Y_CPU_BENCHMARK(RdtscUtil, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(GetCycleCount());
    }
}

Y_CPU_BENCHMARK(RdtscLinux, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(GetCycleCountLinux());
    }
}

Y_CPU_BENCHMARK(RdtscAgri1, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(GetCycleCountAgri1());
    }
}

Y_CPU_BENCHMARK(RdtscAgri2, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(GetCycleCountAgri2());
    }
}
