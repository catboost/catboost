#include <library/testing/benchmark/bench.h>

#include <util/generic/cast.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>
#include <util/string/cast.h>
#include <util/string/subst.h>

namespace {
    template <size_t N, char What, char With>
    struct TNoMatches {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TNoMatches() {
            for (const auto dummy : xrange(N)) {
                Y_UNUSED(dummy);
                Str += WHAT + 1;
            }
        }
    };

    template <size_t N, char What, char With>
    struct TOneMatchInTheBeginning {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TOneMatchInTheBeginning() {
            if (!N) {
                return;
            }

            Str += WHAT;
            if (N > 1) {
                for (const auto dummy : xrange(N - 1)) {
                    Y_UNUSED(dummy);
                    Str += WHAT + 1;
                }
            }
        }
    };

    template <size_t N, char What, char With>
    struct TOneMatchInTheEnd {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TOneMatchInTheEnd() {
            if (!N) {
                return;
            }

            if (N > 1) {
                for (const auto dummy : xrange(N - 1)) {
                    Y_UNUSED(dummy);
                    Str += WHAT + 1;
                }
            }
            Str += WHAT;
        }
    };

    template <size_t N, char What, char With>
    struct TOneMatchInTheMiddle {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TOneMatchInTheMiddle() {
            if (!N) {
                return;
            }

            for (size_t i = 0; i < N / 2; ++i) {
                Str += WHAT + 1;
            }
            Str += WHAT;
            for (; Str.size() < N;) {
                Str += WHAT + 1;
            }
        }
    };

    template <size_t N, char What, char With>
    struct TFirstHalfMatches {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TFirstHalfMatches() {
            for (size_t i = 0; i < N / 2; ++i) {
                Str += WHAT;
            }
            for (; Str.size() != N;) {
                Str += WHAT + 1;
            }
        }
    };

    template <size_t N, char What, char With>
    struct TSecondHalfMatches {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TSecondHalfMatches() {
            for (size_t i = 0; i < N / 2; ++i) {
                Str += WHAT + 1;
            }
            for (; Str.size() != N;) {
                Str += WHAT;
            }
        }
    };

    template <size_t N, size_t K, char What, char With>
    struct TEveryKth {
        enum : char {
            WHAT = What,
            WITH = With
        };
        TString Str;

        TEveryKth() {
            TFastRng<ui64> prng{N * K * 101};
            for (size_t i = 0; i < N; ++i) {
                Str += (prng() % K) ? (WHAT + 1) : WHAT;
            }
        }
    };
}

#define DEFINE_BENCHMARK(type, N)                                                                 \
    Y_CPU_BENCHMARK(type##_##N, i) {                                                              \
        using D = T##type<N, 'a', 'z'>;                                                           \
        const auto& str = Default<D>().Str;                                                       \
        for (const auto dummy : xrange(i.Iterations())) {                                         \
            Y_UNUSED(dummy);                                                                      \
            auto s = str;                                                                         \
            NBench::Escape(s.data());                                                                   \
            Y_DO_NOT_OPTIMIZE_AWAY(SubstGlobal(s, ToUnderlying(D::WHAT), ToUnderlying(D::WITH))); \
            NBench::Clobber();                                                                    \
        }                                                                                         \
    }

#define DEFINE_RNG_BENCHMARK(N, K)                                                                \
    Y_CPU_BENCHMARK(Random_##N##_##K, i) {                                                        \
        using D = TEveryKth<N, K, 'a', 'z'>;                                                      \
        const auto& str = Default<D>().Str;                                                       \
        for (const auto dummy : xrange(i.Iterations())) {                                         \
            Y_UNUSED(dummy);                                                                      \
            auto s = str;                                                                         \
            NBench::Escape(s.data());                                                                   \
            Y_DO_NOT_OPTIMIZE_AWAY(SubstGlobal(s, ToUnderlying(D::WHAT), ToUnderlying(D::WITH))); \
            NBench::Clobber();                                                                    \
        }                                                                                         \
    }

DEFINE_BENCHMARK(NoMatches, 0)
DEFINE_BENCHMARK(NoMatches, 1)
DEFINE_BENCHMARK(NoMatches, 128)
DEFINE_BENCHMARK(NoMatches, 4096)
DEFINE_BENCHMARK(OneMatchInTheBeginning, 1)
DEFINE_BENCHMARK(OneMatchInTheBeginning, 16)
DEFINE_BENCHMARK(OneMatchInTheBeginning, 128)
DEFINE_BENCHMARK(OneMatchInTheBeginning, 4096)
DEFINE_BENCHMARK(OneMatchInTheEnd, 16)
DEFINE_BENCHMARK(OneMatchInTheEnd, 128)
DEFINE_BENCHMARK(OneMatchInTheEnd, 4096)
DEFINE_BENCHMARK(OneMatchInTheMiddle, 16)
DEFINE_BENCHMARK(OneMatchInTheMiddle, 128)
DEFINE_BENCHMARK(OneMatchInTheMiddle, 4096)
DEFINE_BENCHMARK(FirstHalfMatches, 16)
DEFINE_BENCHMARK(FirstHalfMatches, 128)
DEFINE_BENCHMARK(FirstHalfMatches, 4096)
DEFINE_BENCHMARK(SecondHalfMatches, 16)
DEFINE_BENCHMARK(SecondHalfMatches, 128)
DEFINE_BENCHMARK(SecondHalfMatches, 4096)

DEFINE_RNG_BENCHMARK(4096, 1)
DEFINE_RNG_BENCHMARK(4096, 2)
DEFINE_RNG_BENCHMARK(4096, 3)
DEFINE_RNG_BENCHMARK(4096, 4)
DEFINE_RNG_BENCHMARK(4096, 10)
DEFINE_RNG_BENCHMARK(4096, 32)
DEFINE_RNG_BENCHMARK(4096, 100)
