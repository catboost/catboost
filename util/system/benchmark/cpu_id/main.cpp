#include <library/testing/benchmark/bench.h>

#include <util/system/cpu_id.h>

#include <util/generic/xrange.h>

#define DEFINE_BENCHMARK_PAIR(name)                           \
    Y_CPU_BENCHMARK(Have##name, iface) {                      \
        for (const auto i : xrange(iface.Iterations())) {     \
            Y_UNUSED(i);                                      \
            Y_DO_NOT_OPTIMIZE_AWAY(NX86::Have##name());       \
        }                                                     \
    }                                                         \
                                                              \
    Y_CPU_BENCHMARK(CachedHave##name, iface) {                \
        for (const auto i : xrange(iface.Iterations())) {     \
            Y_UNUSED(i);                                      \
            Y_DO_NOT_OPTIMIZE_AWAY(NX86::CachedHave##name()); \
        }                                                     \
    }

DEFINE_BENCHMARK_PAIR(SSE)
DEFINE_BENCHMARK_PAIR(SSE2)
DEFINE_BENCHMARK_PAIR(SSE3)
DEFINE_BENCHMARK_PAIR(SSSE3)
DEFINE_BENCHMARK_PAIR(SSE41)
DEFINE_BENCHMARK_PAIR(SSE42)
DEFINE_BENCHMARK_PAIR(POPCNT)
DEFINE_BENCHMARK_PAIR(BMI1)
DEFINE_BENCHMARK_PAIR(AES)
DEFINE_BENCHMARK_PAIR(AVX)
DEFINE_BENCHMARK_PAIR(AVX2)
DEFINE_BENCHMARK_PAIR(AVX512F)
DEFINE_BENCHMARK_PAIR(AVX512DQ)
DEFINE_BENCHMARK_PAIR(AVX512IFMA)
DEFINE_BENCHMARK_PAIR(AVX512PF)
DEFINE_BENCHMARK_PAIR(AVX512ER)
DEFINE_BENCHMARK_PAIR(AVX512CD)
DEFINE_BENCHMARK_PAIR(AVX512BW)
DEFINE_BENCHMARK_PAIR(AVX512VL)
DEFINE_BENCHMARK_PAIR(AVX512VBMI)
DEFINE_BENCHMARK_PAIR(PREFETCHWT1)
DEFINE_BENCHMARK_PAIR(SHA)
DEFINE_BENCHMARK_PAIR(ADX)
DEFINE_BENCHMARK_PAIR(RDRAND)
DEFINE_BENCHMARK_PAIR(RDSEED)
DEFINE_BENCHMARK_PAIR(PCOMMIT)
DEFINE_BENCHMARK_PAIR(CLFLUSHOPT)
DEFINE_BENCHMARK_PAIR(CLWB)

#undef DEFINE_BENCHMARK_PAIR
