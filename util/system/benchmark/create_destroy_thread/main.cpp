#include <library/testing/benchmark/bench.h>

#include <util/system/thread.h>

static void* DoNothing(void*) noexcept {
    return nullptr;
}

Y_CPU_BENCHMARK(CreateDestroyThread, iface) {
    for (size_t i = 0, iEnd = iface.Iterations(); i < iEnd; ++i) {
        NBench::Clobber();
        TThread t(&DoNothing, nullptr);
        Y_DO_NOT_OPTIMIZE_AWAY(t);
        NBench::Clobber();
    }
}

Y_CPU_BENCHMARK(CreateRunDestroyThread, iface) {
    for (size_t i = 0, iEnd = iface.Iterations(); i < iEnd; ++i) {
        NBench::Clobber();
        TThread t(&DoNothing, nullptr);
        t.Start();
        NBench::Escape(t.Join());
        NBench::Clobber();
    }
}
