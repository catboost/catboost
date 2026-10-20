#include <library/cpp/yt/system/access_spreader.h>

#include <benchmark/benchmark.h>

#include <array>
#include <atomic>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

constexpr int StripeCount = 64;

struct alignas(64) TStripe
{
    std::atomic<ui64> Value = 0;
};

static_assert(sizeof(TStripe) == 64);

template <bool UseStripe>
void RunContentionBenchmark(benchmark::State& state)
{
    static std::array<TStripe, StripeCount> Stripes;

    i64 casFailures = 0;
    for (auto _ : state) {
        int stripe = GetCurrentAccessStripe(StripeCount);
        int stripeIndex = UseStripe ? stripe : 0;
        auto& value = Stripes[stripeIndex].Value;
        ui64 oldValue = value.load(std::memory_order::relaxed);
        while (!value.compare_exchange_weak(
            oldValue,
            oldValue + 1,
            std::memory_order::relaxed))
        {
            ++casFailures;
        }
    }

    state.counters["cas_failures_per_op"] = benchmark::Counter(
        casFailures,
        benchmark::Counter::kAvgIterations);
}

void BM_GetCurrentAccessStripe(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCurrentAccessStripe(StripeCount));
    }
}
BENCHMARK(BM_GetCurrentAccessStripe);

void BM_ForcedAccessStripeContention(benchmark::State& state)
{
    RunContentionBenchmark<false>(state);
}
BENCHMARK(BM_ForcedAccessStripeContention)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

void BM_CurrentAccessStripeContention(benchmark::State& state)
{
    RunContentionBenchmark<true>(state);
}
BENCHMARK(BM_CurrentAccessStripeContention)
    ->Threads(2)
    ->Unit(benchmark::kNanosecond);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
