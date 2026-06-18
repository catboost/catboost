#include <library/cpp/yt/system/cpu_id.h>

#include <benchmark/benchmark.h>

#include <sched.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

// rseq: a single thread-local read.
void BM_GetCurrentCpuId(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCurrentCpuId());
    }
}
BENCHMARK(BM_GetCurrentCpuId);

// The fallback path: sched_getcpu() (vDSO getcpu).
void BM_SchedGetCpu(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(sched_getcpu());
    }
}
BENCHMARK(BM_SchedGetCpu);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
