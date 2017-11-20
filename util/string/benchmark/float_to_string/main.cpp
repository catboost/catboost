#include <library/testing/benchmark/bench.h>

#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>
#include <util/string/cast.h>
#include <util/string/printf.h>

#include <limits>

#include <cmath>

/* Please be careful before making any decisions based on this benchmark.
 *
 * Only `Sprintf("%.<decimals>f", x)` and `FloatToString(x, PREC_POINT_DIGITS, decimals` produce
 * equal results in general case. However, results for cases when x \in [0, 1) must be equal for
 * both `Sprintf` and `FloatToString`.
 *
 * Read more about formatting in STL [1, 2] and Yandex Util formatting [3]
 *
 * [1] http://www.cplusplus.com/reference/cstdio/printf/
 * [2] http://en.cppreference.com/w/c/io/fprintf
 * [3] https://a.yandex-team.ru/arc/trunk/arcadia/util/string/cast.h?rev=2432660#L29
 */

namespace {
    template <typename T>
    struct TExample {
        T Value{};
        int DigitsCount{};
    };

    template <typename T, size_t N>
    struct TExamplesHolder {
        TVector<TExample<T>> Examples;

        TExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{N * sizeof(T) * 42};
            for (auto& x : Examples) {
                x.Value = prng.GenRandReal4() + prng.Uniform(Max<ui16>());
                x.DigitsCount = prng.Uniform(std::numeric_limits<T>::max_digits10 + 1);
            }
        }
    };

    template <typename T, size_t N>
    struct TNearZeroExamplesHolder {
        TVector<TExample<T>> Examples;

        TNearZeroExamplesHolder()
            : Examples(N)
        {
            TFastRng<ui64> prng{N * sizeof(T) * 42};
            for (auto& x : Examples) {
                x.Value = prng.GenRandReal4();
                x.DigitsCount = prng.Uniform(std::numeric_limits<T>::max_digits10 + 1);
            }
        }
    };
}

static const char* FORMAT_FIXED[] = {
    "%.0f",
    "%.1f",
    "%.2f",
    "%.3f",
    "%.4f",
    "%.5f",
    "%.6f",
    "%.7f",
    "%.8f",
    "%.9f",
    "%.10f",
    "%.11f",
    "%.12f",
    "%.13f",
    "%.14f",
    "%.15f",
    "%.16f",
    "%.17f",
};

static const char* FORMAT_SIGNIFICANT[] = {
    "%.0g",
    "%.1g",
    "%.2g",
    "%.3g",
    "%.4g",
    "%.5g",
    "%.6g",
    "%.7g",
    "%.8g",
    "%.9g",
    "%.10g",
    "%.11g",
    "%.12g",
    "%.13g",
    "%.14g",
    "%.15g",
    "%.16g",
    "%.17g",
};

#define DEFINE_BENCHMARK(type, count)                                                                \
    Y_CPU_BENCHMARK(SprintfAuto_##type##_##count, iface) {                                           \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                /* this is in fact equal to Sprintf("%.6f", e.Value) and that is why it is faster */ \
                /* than FloatToString(e.Value) */                                                    \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf("%f", e.Value));                                      \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(FloatToStringAuto_##type##_##count, iface) {                                     \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value));                                      \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(SprintfFixed_##type##_##count, iface) {                                          \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf(FORMAT_FIXED[e.DigitsCount], e.Value));               \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(FloatToStringFixed_##type##_##count, iface) {                                    \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value, PREC_NDIGITS, e.DigitsCount));         \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(SprintfSignificant_##type##_##count, iface) {                                    \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf(FORMAT_SIGNIFICANT[e.DigitsCount], e.Value));         \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(FloatToStringSignificant_##type##_##count, iface) {                              \
        const auto& examples = Default<TExamplesHolder<type, count>>().Examples;                     \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value, PREC_POINT_DIGITS, e.DigitsCount));    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroSprintfAuto_##type##_##count, iface) {                                   \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                /* this is in fact equal to Sprintf("%.6f", e.Value) and that is why it is faster */ \
                /* than FloatToString(e.Value) */                                                    \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf("%f", e.Value));                                      \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroFloatToStringAuto_##type##_##count, iface) {                             \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value));                                      \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroSprintfFixed_##type##_##count, iface) {                                  \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf(FORMAT_FIXED[e.DigitsCount], e.Value));               \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroFloatToStringFixed_##type##_##count, iface) {                            \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value, PREC_NDIGITS, e.DigitsCount));         \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroSprintfSignificant_##type##_##count, iface) {                            \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(Sprintf(FORMAT_SIGNIFICANT[e.DigitsCount], e.Value));         \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    Y_CPU_BENCHMARK(NearZeroFloatToStringSignificant_##type##_##count, iface) {                      \
        const auto& examples = Default<TNearZeroExamplesHolder<type, count>>().Examples;             \
        for (const auto i : xrange(iface.Iterations())) {                                            \
            Y_UNUSED(i);                                                                             \
            for (const auto e : examples) {                                                          \
                Y_DO_NOT_OPTIMIZE_AWAY(FloatToString(e.Value, PREC_POINT_DIGITS, e.DigitsCount));    \
            }                                                                                        \
        }                                                                                            \
    }

DEFINE_BENCHMARK(float, 1);
DEFINE_BENCHMARK(float, 2);
DEFINE_BENCHMARK(float, 4);
DEFINE_BENCHMARK(float, 8);
DEFINE_BENCHMARK(float, 16);
DEFINE_BENCHMARK(float, 32);
DEFINE_BENCHMARK(float, 64);
DEFINE_BENCHMARK(float, 128);
DEFINE_BENCHMARK(float, 256);

DEFINE_BENCHMARK(double, 1);
DEFINE_BENCHMARK(double, 2);
DEFINE_BENCHMARK(double, 4);
DEFINE_BENCHMARK(double, 8);
DEFINE_BENCHMARK(double, 16);
DEFINE_BENCHMARK(double, 32);
DEFINE_BENCHMARK(double, 64);
DEFINE_BENCHMARK(double, 128);
DEFINE_BENCHMARK(double, 256);

#undef DEFINE_BENCHMARK
