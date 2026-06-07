#include <benchmark/benchmark.h>

#include <library/cpp/yt/system/thread_id.h>
#include <library/cpp/yt/system/thread_name.h>

#include <util/system/thread.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

void BM_GetSystemThreadId(benchmark::State& state)
{
    // Cached gettid: only the first call per thread hits the kernel.
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetSystemThreadId());
    }
}

BENCHMARK(BM_GetSystemThreadId);

void BM_GetSequentialThreadId(benchmark::State& state)
{
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetSequentialThreadId());
    }
}

BENCHMARK(BM_GetSequentialThreadId);

void BM_RawGetTid(benchmark::State& state)
{
    // Uncached gettid syscall, for comparison.
    for (auto _ : state) {
        benchmark::DoNotOptimize(::TThread::CurrentThreadNumericId());
    }
}

BENCHMARK(BM_RawGetTid);

void BM_GetCurrentThreadName(benchmark::State& state)
{
    // TLS-cached thread name (also read by TOriginAttributes::Capture).
    for (auto _ : state) {
        benchmark::DoNotOptimize(GetCurrentThreadName());
    }
}

BENCHMARK(BM_GetCurrentThreadName);

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
