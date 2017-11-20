#include <library/testing/benchmark/bench.h>

#include <util/generic/bitops.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/singleton.h>

#include <util/random/fast.h>

namespace {
    template <typename T, size_t N>
    struct TExamplesHolder {
        TExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{42u * sizeof(T) * N};
            for (auto& x : Examples) {
                x = prng();
            }
        }

        TVector<T> Examples;
    };
}

#define DEFINE_BENCHMARK(type, count)                                            \
    Y_CPU_BENCHMARK(FastClp2_##type##_##count, iface) {                          \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(FastClp2(e));                             \
            }                                                                    \
        }                                                                        \
    }

DEFINE_BENCHMARK(ui8, 1)
DEFINE_BENCHMARK(ui8, 10)
DEFINE_BENCHMARK(ui8, 100)
DEFINE_BENCHMARK(ui16, 1)
DEFINE_BENCHMARK(ui16, 10)
DEFINE_BENCHMARK(ui16, 100)
DEFINE_BENCHMARK(ui32, 1)
DEFINE_BENCHMARK(ui32, 10)
DEFINE_BENCHMARK(ui32, 100)
DEFINE_BENCHMARK(ui64, 1)
DEFINE_BENCHMARK(ui64, 10)
DEFINE_BENCHMARK(ui64, 100)

#undef DEFINE_BENCHMARKS
