#include <library/testing/benchmark/bench.h>

#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/singleton.h>

#include <util/random/fast.h>

namespace {
    template <typename T>
    struct TExample {
        T Value;
        ui8 Shift;
    };

    template <typename T, size_t N>
    struct TExamplesHolder {
        TExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{42u * sizeof(T) * N};
            for (auto& e : Examples) {
                e.Value = prng();
                e.Shift = prng() % (8 * sizeof(T));
            }
        }

        TVector<TExample<T>> Examples;
    };
}

#define DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(type, count)                        \
    Y_CPU_BENCHMARK(LeftRotate_##type##_##count, iface) {                        \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(RotateBitsLeft(e.Value, e.Shift));        \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    Y_CPU_BENCHMARK(RightRotate_##type##_##count, iface) {                       \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                        \
            Y_UNUSED(i);                                                         \
            for (const auto e : examples) {                                      \
                Y_DO_NOT_OPTIMIZE_AWAY(RotateBitsRight(e.Value, e.Shift));       \
            }                                                                    \
        }                                                                        \
    }

DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui8, 1)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui8, 10)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui8, 100)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui16, 1)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui16, 10)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui16, 100)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui32, 1)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui32, 10)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui32, 100)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui64, 1)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui64, 10)
DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES(ui64, 100)

#undef DEFINE_BENCHMARKS_FOR_UNSIGNED_TYPES
