#include <library/testing/benchmark/bench.h>

#include <util/string/cast.h>
#include <util/generic/xrange.h>

char str1[] = "1";
char str12[] = "12";
char str1234[] = "1234";
char str12345678[] = "12345678";

Y_CPU_BENCHMARK(Parse_1, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>(str1, 1));
    }
}

Y_CPU_BENCHMARK(Parse_12, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>(str12, 2));
    }
}

Y_CPU_BENCHMARK(Parse_1234, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>(str1234, 4));
    }
}

Y_CPU_BENCHMARK(Parse_12345678, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FromString<ui32>(str12345678, 8));
    }
}

//atoi
Y_CPU_BENCHMARK(Atoi_1, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi(str1));
    }
}

Y_CPU_BENCHMARK(Atoi_12, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi(str12));
    }
}

Y_CPU_BENCHMARK(Atoi_1234, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi(str1234));
    }
}

Y_CPU_BENCHMARK(Atoi_12345678, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(atoi(str12345678));
    }
}
