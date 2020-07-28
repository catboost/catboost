#include <library/cpp/accurate_accumulate/accurate_accumulate.h>
#include <library/cpp/testing/benchmark/bench.h>

#include <util/generic/algorithm.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>

namespace {
    template <typename T, size_t N>
    struct TNormalizedExamplesHolder {
        TVector<T> Examples;
        TNormalizedExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{sizeof(T) * N * 42u};
            for (auto& x : Examples) {
                x = prng.GenRandReal4();
            }
        }
    };

    template <typename T, size_t N>
    struct TExamplesHolder {
        TVector<T> Examples;
        TExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{sizeof(T) * N * 42u + 100500u};
            for (auto& x : Examples) {
                // operations with non-normalized floating point numbers are rumored to work slower
                x = prng.GenRandReal4() + prng.Uniform(1024u);
            }
        }
    };
}

#define DEFINE_BENCHMARK(type, count)                                                                     \
    Y_CPU_BENCHMARK(SimpleNorm_##type##_##count, iface) {                                                 \
        const auto& examples = Default<TNormalizedExamplesHolder<type, count>>().Examples;                \
        for (const auto i : xrange(iface.Iterations())) {                                                 \
            Y_UNUSED(i);                                                                                  \
            Y_DO_NOT_OPTIMIZE_AWAY(                                                                       \
                (type)Accumulate(std::cbegin(examples), std::cend(examples), type{}));                    \
        }                                                                                                 \
    }                                                                                                     \
                                                                                                          \
    Y_CPU_BENCHMARK(KahanNorm_##type##_##count, iface) {                                                  \
        const auto& examples = Default<TNormalizedExamplesHolder<type, count>>().Examples;                \
        for (const auto i : xrange(iface.Iterations())) {                                                 \
            Y_UNUSED(i);                                                                                  \
            Y_DO_NOT_OPTIMIZE_AWAY(                                                                       \
                (type)Accumulate(std::cbegin(examples), std::cend(examples), TKahanAccumulator<type>{})); \
        }                                                                                                 \
    }                                                                                                     \
                                                                                                          \
    Y_CPU_BENCHMARK(Simple_##type##_##count, iface) {                                                     \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                          \
        for (const auto i : xrange(iface.Iterations())) {                                                 \
            Y_UNUSED(i);                                                                                  \
            Y_DO_NOT_OPTIMIZE_AWAY(                                                                       \
                (type)Accumulate(std::cbegin(examples), std::cend(examples), type{}));                    \
        }                                                                                                 \
    }                                                                                                     \
                                                                                                          \
    Y_CPU_BENCHMARK(Kahan_##type##_##count, iface) {                                                      \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                          \
        for (const auto i : xrange(iface.Iterations())) {                                                 \
            Y_UNUSED(i);                                                                                  \
            Y_DO_NOT_OPTIMIZE_AWAY(                                                                       \
                (type)Accumulate(std::cbegin(examples), std::cend(examples), TKahanAccumulator<type>{})); \
        }                                                                                                 \
    }

DEFINE_BENCHMARK(float, 2)
DEFINE_BENCHMARK(float, 4)
DEFINE_BENCHMARK(float, 8)
DEFINE_BENCHMARK(float, 16)
DEFINE_BENCHMARK(float, 32)
DEFINE_BENCHMARK(float, 64)
DEFINE_BENCHMARK(float, 128)
DEFINE_BENCHMARK(float, 256)
DEFINE_BENCHMARK(float, 512)
DEFINE_BENCHMARK(float, 1024)
DEFINE_BENCHMARK(double, 2)
DEFINE_BENCHMARK(double, 4)
DEFINE_BENCHMARK(double, 8)
DEFINE_BENCHMARK(double, 16)
DEFINE_BENCHMARK(double, 32)
DEFINE_BENCHMARK(double, 64)
DEFINE_BENCHMARK(double, 128)
DEFINE_BENCHMARK(double, 256)
DEFINE_BENCHMARK(double, 512)
DEFINE_BENCHMARK(double, 1024)

#undef DEFINE_BENCHMARK
