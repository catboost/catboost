#include <benchmark/benchmark.h>

#include <library/cpp/yt/system/process_id.h>

#include <util/system/getpid.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

void BM_GetProcessId(benchmark::State& state)
{
    // Cached getpid: only the first call hits the kernel.
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetProcessId());
    }
}

BENCHMARK(BM_GetProcessId);

void BM_RawGetPid(benchmark::State& state)
{
    // Uncached getpid syscall (uncached on glibc >= 2.25), for comparison.
    for (auto _ : state) {
        benchmark::DoNotOptimize(::GetPID());
    }
}

BENCHMARK(BM_RawGetPid);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
