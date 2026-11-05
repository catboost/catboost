#include <benchmark/benchmark.h>

#include <library/cpp/yt/system/tscp.h>

namespace NYT::NProfiling {
namespace {

////////////////////////////////////////////////////////////////////////////////

// Timestamp counter and processor id via a single serializing rdtscp.
void BM_TscpGet(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(TTscp::Get());
    }
}
BENCHMARK(BM_TscpGet);

// Cheaper, lower-precision counterpart: rseq fast path + non-serializing rdtsc.
void BM_TscpGetApproximate(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(TTscp::GetApproximate());
    }
}
BENCHMARK(BM_TscpGetApproximate);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NProfiling
