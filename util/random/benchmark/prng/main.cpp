#include <library/testing/benchmark/bench.h>

#include <util/random/entropy.h>
#include <util/random/fast.h>
#include <util/random/normal.h>
#include <util/random/mersenne.h>
#include <util/system/compiler.h>
#include <util/generic/xrange.h>

#include <random>

// double part
Y_CPU_BENCHMARK(Mersenne32_Double, p) {
    TMersenne<ui32> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRandReal1());
    }
}

Y_CPU_BENCHMARK(Mersenne64_Double, p) {
    TMersenne<ui64> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRandReal1());
    }
}

Y_CPU_BENCHMARK(Fast32_Double, p) {
    TFastRng<ui32> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRandReal1());
    }
}

Y_CPU_BENCHMARK(Fast64_Double, p) {
    TFastRng<ui64> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRandReal1());
    }
}

// integer part
Y_CPU_BENCHMARK(mt19937_32, p) {
    std::mt19937 mt;

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;

        Y_DO_NOT_OPTIMIZE_AWAY(mt());
    }
}

Y_CPU_BENCHMARK(mt19937_64, p) {
    std::mt19937_64 mt;

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;

        Y_DO_NOT_OPTIMIZE_AWAY(mt());
    }
}

Y_CPU_BENCHMARK(Mersenne32_GenRand, p) {
    TMersenne<ui32> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRand());
    }
}

Y_CPU_BENCHMARK(Mersenne64_GenRand, p) {
    TMersenne<ui64> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRand());
    }
}

Y_CPU_BENCHMARK(Fast32_GenRand, p) {
    TFastRng<ui32> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRand());
    }
}

Y_CPU_BENCHMARK(Fast64_GenRand, p) {
    TFastRng<ui64> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(rng.GenRand());
    }
}

Y_CPU_BENCHMARK(StlNormal, p) {
    TFastRng<ui64> rng(Seed());
    std::normal_distribution<double> d(1.0, 0.0);

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(d(rng));
    }
}

Y_CPU_BENCHMARK(UtilNormal, p) {
    TFastRng<ui64> rng(Seed());

    for (auto i : xrange<size_t>(0, p.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(NormalDistribution<double>(rng, 1.0, 0.0));
    }
}
