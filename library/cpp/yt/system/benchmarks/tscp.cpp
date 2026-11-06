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

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT::NProfiling
