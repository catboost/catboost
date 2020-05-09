#include <library/chromium_trace/interface.h>
#include <library/chromium_trace/json.h>
#include <library/chromium_trace/sync.h>
#include <library/chromium_trace/yson.h>
#include <library/chromium_trace/saveload.h>

#include <library/cpp/json/json_writer.h>
#include <library/testing/benchmark/bench.h>

#include <util/generic/xrange.h>
#include <util/stream/null.h>
#include <util/system/guard.h>
#include <util/system/mutex.h>

using namespace NChromiumTrace;
using namespace NYT;

namespace {
    // Benchmark involve configuring singleton state, thus multi-threaded benchmarks
    // must be forbidden
    //
    // FIXME: avoids crashing, but benchmark results are crap
    TMutex SingletonBenchmarkLock;

    Y_NO_INLINE void FnEmpty(size_t i) {
        Y_DO_NOT_OPTIMIZE_AWAY(i);
    }

    Y_NO_INLINE void FnEmptyTraced(size_t i) {
        CHROMIUM_TRACE_FUNCTION();

        Y_DO_NOT_OPTIMIZE_AWAY(i);
    }

}

Y_CPU_BENCHMARK(EmptyFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    for (size_t i : xrange(iface.Iterations())) {
        FnEmpty(i);
    }
}

Y_CPU_BENCHMARK(EmptyNullTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(JsonEmptyTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TJsonTraceConsumer>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(JsonEmptySyncTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TSyncTraceConsumer<TJsonTraceConsumer>>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(YsonEmptyTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TYsonTraceConsumer>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(YsonEmptySyncTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TSyncTraceConsumer<TYsonTraceConsumer>>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(SaveLoadEmptyTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TSaveLoadTraceConsumer>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}

Y_CPU_BENCHMARK(SaveLoadEmptySyncTracedFunction, iface) {
    auto singletonGuard = Guard(SingletonBenchmarkLock);

    TGlobalTraceConsumerGuard guard(
        MakeHolder<TSyncTraceConsumer<TSaveLoadTraceConsumer>>(&Cnull));

    for (size_t i : xrange(iface.Iterations())) {
        FnEmptyTraced(i);
    }
}
