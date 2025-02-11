#include <library/cpp/testing/gbenchmark/benchmark.h>

#include <util/datetime/base.h>
#include <util/random/fast.h>

void BM_GmTimeR(benchmark::State& state) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    for (auto _ : state) {
        Y_DO_NOT_OPTIMIZE_AWAY(GmTimeR(&now, &buf));
    }
}

void BM_gmtime_r(benchmark::State& state) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    for (auto _ : state) {
        Y_DO_NOT_OPTIMIZE_AWAY(gmtime_r(&now, &buf));
    }
}

void BM_GmTimeRRandom(benchmark::State& state, TDuration window) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    TFastRng<ui32> rng(2);
    const size_t range = window.Seconds();
    for (auto _ : state) {
        size_t offset = rng.GenRand() % range;
        time_t v = now - offset;
        Y_DO_NOT_OPTIMIZE_AWAY(GmTimeR(&v, &buf));
    }
}

void BM_gmtime_r_Random(benchmark::State& state, TDuration window) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    TFastRng<ui32> rng(2);
    const size_t range = window.Seconds();
    for (auto _ : state) {
        size_t offset = rng.GenRand() % range;
        time_t v = now - offset;
        Y_DO_NOT_OPTIMIZE_AWAY(gmtime_r(&v, &buf));
    }
}

BENCHMARK(BM_GmTimeR);
BENCHMARK(BM_gmtime_r);
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_hour, TDuration::Hours(1));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_hour, TDuration::Hours(1));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_day, TDuration::Days(1));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_day, TDuration::Days(1));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_month, TDuration::Days(31));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_month, TDuration::Days(31));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_year, TDuration::Days(365));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_year, TDuration::Days(365));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_decade, TDuration::Days(3653));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_decade, TDuration::Days(3653));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_half_century, TDuration::Days(18262));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_half_century, TDuration::Days(18262));
BENCHMARK_CAPTURE(BM_GmTimeRRandom, last_century, TDuration::Days(36525));
BENCHMARK_CAPTURE(BM_gmtime_r_Random, last_century, TDuration::Days(36525));
