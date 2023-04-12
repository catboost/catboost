#include <library/cpp/testing/benchmark/bench.h>

#include <util/draft/datetime.h>

Y_CPU_BENCHMARK(GmTimeR, iface) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        Y_DO_NOT_OPTIMIZE_AWAY(GmTimeR(&now, &buf));
    }
}

Y_CPU_BENCHMARK(gmtime_r, iface) {
    time_t now = TInstant::Now().TimeT();
    struct tm buf {};

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        Y_DO_NOT_OPTIMIZE_AWAY(gmtime_r(&now, &buf));
    }
}
