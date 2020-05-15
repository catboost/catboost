#include <library/testing/benchmark/bench.h>

#include <library/cpp/fast_log/fast_log.h>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/generic/xrange.h>

#include <cmath>
namespace {
    template <typename T, size_t N>
    struct TExamplesHolder {
        TVector<T> Examples;

        TExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{N * 42};
            for (auto& x : Examples) {
                x = prng.GenRandReal4() + prng.Uniform(1932); // 1934 is just a random number
            }
        }
    };
}

#define DEFINE_BENCHMARK(type, count)                                            \
    Y_CPU_BENCHMARK(libm_log2f_##type##_##count, iface) {                        \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(log2f(e));                                \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(libm_logf_##type##_##count, iface) {                         \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(logf(e));                                 \
            }                                                                    \
        }                                                                        \
    }                                                                            \
    Y_CPU_BENCHMARK(STL_Log2_##type##_##count, iface) {                          \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(std::log2(e));                            \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(STL_Log_##type##_##count, iface) {                           \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(std::log(e));                             \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(Fast_Log2_##type##_##count, iface) {                         \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FastLog2f(e));                            \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(FastLogf##type##_##count, iface) {                           \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FastLogf(e));                             \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(Faster_Log2_##type##_##count, iface) {                       \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FasterLog2f(e));                          \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(Faster_Log_##type##_##count, iface) {                        \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FasterLogf(e));                           \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(Fastest_Log2f_##type##_##count, iface) {                     \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FastestLog2f(e));                         \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(Fastest_Log_##type##_##count, iface) {                       \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FastestLogf(e));                          \
            }                                                                    \
        }                                                                        \
    }

DEFINE_BENCHMARK(float, 1)
DEFINE_BENCHMARK(float, 2)
DEFINE_BENCHMARK(float, 4)
DEFINE_BENCHMARK(float, 8)
DEFINE_BENCHMARK(float, 16)
DEFINE_BENCHMARK(float, 32)
DEFINE_BENCHMARK(float, 64)
DEFINE_BENCHMARK(float, 128)
DEFINE_BENCHMARK(float, 256)
DEFINE_BENCHMARK(float, 1024)
DEFINE_BENCHMARK(float, 2048)
DEFINE_BENCHMARK(float, 4096)

#undef DEFINE_BENCHMARK
