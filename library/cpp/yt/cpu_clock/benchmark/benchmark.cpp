#include "benchmark/benchmark.h"
#include <benchmark/benchmark.h>

#include <library/cpp/yt/cpu_clock/clock.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

void BM_GetCpuInstant(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCpuInstant());
    }
}

BENCHMARK(BM_GetCpuInstant);

void BM_GetCpuApproximateInstant(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetApproximateCpuInstant());
    }
}

BENCHMARK(BM_GetCpuApproximateInstant);

void BM_GetInstant(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetInstant());
    }
}

BENCHMARK(BM_GetInstant);

void BM_InstantNow(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(TInstant::Now());
    }
}

BENCHMARK(BM_InstantNow);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
