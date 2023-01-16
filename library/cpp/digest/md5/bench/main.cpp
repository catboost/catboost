#include <benchmark/benchmark.h>

#include <library/cpp/digest/md5/md5.h>

#define MD5_DEF(N)                                                  \
    static void MD5Benchmark_##N(benchmark::State& st) {            \
        char buf[N];                                                \
        for (auto _ : st) {                                         \
            Y_DO_NOT_OPTIMIZE_AWAY(MD5().Update(buf, sizeof(buf))); \
        }                                                           \
    }                                                               \
    BENCHMARK(MD5Benchmark_##N);

MD5_DEF(32)
MD5_DEF(64)
MD5_DEF(128)

MD5_DEF(1024)
MD5_DEF(2048)
