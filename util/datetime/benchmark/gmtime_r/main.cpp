#include <library/cpp/testing/gbenchmark/benchmark.h>

#include <util/datetime/base.h>
#include <util/random/fast.h>

using TRandomEngine = TFastRng<ui64>;

namespace {

    class TTimeRange {
    public:
        TTimeRange(TInstant base, TDuration spread)
            : TTimeRange(base.Seconds(), spread.Seconds())
        {
        }

        TTimeRange(time_t base, time_t spread)
            : Base_(base)
            , Spread_(spread)
        {
            Y_ASSERT(Spread_ >= 0);
        }

        time_t Next(TRandomEngine& gen) const {
            if (Spread_ <= 1) {
                return Base_;
            }
            time_t offset = gen.GenRand() % Spread_;
            return Base_ + offset;
        }

    private:
        time_t Base_;
        time_t Spread_;
    };

    // change base date after `BATCH_SIZE` iterations to reduce the effect of changing the benchmark's speed depending on the day it was launched
    const TDuration FOUR_YEARS = TDuration::Days(1461); // the interval length should be at least four years so that the probability of choosing a leap day/year is not greatly skewed
    const TTimeRange BASE_TIME{TInstant::Now(), FOUR_YEARS};
    const size_t BATCH_SIZE = 5'000;

} // namespace

void BM_GmTimeR(benchmark::State& state, TDuration window) {
    struct tm buf {};
    TRandomEngine rng(2);
    const time_t range = window.Seconds();
    while (state.KeepRunningBatch(BATCH_SIZE)) {
        const TTimeRange time{BASE_TIME.Next(rng) - range, range};
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            time_t v = time.Next(rng);
            Y_DO_NOT_OPTIMIZE_AWAY(GmTimeR(&v, &buf));
        }
    }
}

void BM_gmtime_r(benchmark::State& state, TDuration window) {
    struct tm buf {};
    TRandomEngine rng(2);
    const time_t range = window.Seconds();
    while (state.KeepRunningBatch(BATCH_SIZE)) {
        const TTimeRange time{BASE_TIME.Next(rng) - range, range};
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            time_t v = time.Next(rng);
            Y_DO_NOT_OPTIMIZE_AWAY(gmtime_r(&v, &buf));
        }
    }
}

BENCHMARK_CAPTURE(BM_GmTimeR, now, TDuration::Seconds(0));
BENCHMARK_CAPTURE(BM_GmTimeR, last_hour, TDuration::Hours(1));
BENCHMARK_CAPTURE(BM_GmTimeR, last_day, TDuration::Days(1));
BENCHMARK_CAPTURE(BM_GmTimeR, last_month, TDuration::Days(31));
BENCHMARK_CAPTURE(BM_GmTimeR, last_year, TDuration::Days(365));
BENCHMARK_CAPTURE(BM_GmTimeR, last_decade, TDuration::Days(3653));
BENCHMARK_CAPTURE(BM_GmTimeR, last_half_century, TDuration::Days(18262));
BENCHMARK_CAPTURE(BM_GmTimeR, last_century, TDuration::Days(36525));

BENCHMARK_CAPTURE(BM_gmtime_r, now, TDuration::Seconds(0));
BENCHMARK_CAPTURE(BM_gmtime_r, last_hour, TDuration::Hours(1));
BENCHMARK_CAPTURE(BM_gmtime_r, last_day, TDuration::Days(1));
BENCHMARK_CAPTURE(BM_gmtime_r, last_month, TDuration::Days(31));
BENCHMARK_CAPTURE(BM_gmtime_r, last_year, TDuration::Days(365));
BENCHMARK_CAPTURE(BM_gmtime_r, last_decade, TDuration::Days(3653));
BENCHMARK_CAPTURE(BM_gmtime_r, last_half_century, TDuration::Days(18262));
BENCHMARK_CAPTURE(BM_gmtime_r, last_century, TDuration::Days(36525));
