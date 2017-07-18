#include <library/testing/benchmark/bench.h>

#include <util/string/cast.h>
#include <util/generic/xrange.h>

Y_CPU_BENCHMARK(Parse_1, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>("1", 1));
    }
}

Y_CPU_BENCHMARK(Parse_12, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>("12", 2));
    }
}

Y_CPU_BENCHMARK(Parse_1234, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>("1234", 4));
    }
}

Y_CPU_BENCHMARK(Parse_12345678, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>("12345678", 8));
    }
}

//atoi
Y_CPU_BENCHMARK(Atoi_1, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi("1"));
    }
}

Y_CPU_BENCHMARK(Atoi_12, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi("12"));
    }
}

Y_CPU_BENCHMARK(Atoi_1234, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi("1234"));
    }
}

Y_CPU_BENCHMARK(Atoi_12345678, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi("12345678"));
    }
}
