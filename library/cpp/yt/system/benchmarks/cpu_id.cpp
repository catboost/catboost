#include <library/cpp/yt/system/cpu_id.h>

#include <benchmark/benchmark.h>

#include <sched.h>

#include <util/system/types.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

// The mechanism GetCurrentCpuId replaces in the per-CPU profiling sensors:
// NProfiling::TTscp::Get() obtains the processor id via a serializing rdtscp.
// Replicated here (instead of depending on yt/yt/core) to compare the bare cost.
Y_FORCE_INLINE int GetCpuIdViaRdtscp()
{
#if defined(__x86_64__)
    ui64 rax, rcx, rdx;
    asm volatile ("rdtscp\n" : "=a" (rax), "=c" (rcx), "=d" (rdx) : : );
    return static_cast<int>(rcx) & 63;
#else
    return sched_getcpu();
#endif
}

////////////////////////////////////////////////////////////////////////////////

// rseq: a single thread-local read.
void BM_GetCurrentCpuId(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCurrentCpuId());
    }
}
BENCHMARK(BM_GetCurrentCpuId);

// rdtscp, as in TTscp::Get().
void BM_Rdtscp(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCpuIdViaRdtscp());
    }
}
BENCHMARK(BM_Rdtscp);

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
