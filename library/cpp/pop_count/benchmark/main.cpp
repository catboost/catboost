#include <util/stream/output.h>
#include <util/datetime/cputimer.h>
#include <util/system/type_name.h>

#include <library/cpp/pop_count/popcount.h>
#include <library/cpp/testing/benchmark/bench.h>

template <class F, class I>
inline void DoRun(F&& f, I&& i) {
    const ui64 n = i.Iterations();

    for (ui64 j = 0; j < n; ++j) {
        Y_DO_NOT_OPTIMIZE_AWAY(f(j * (ui64)123456 + (ui64)1));
    }
}

Y_CPU_BENCHMARK(PopCount_8, iface) {
    DoRun([](ui8 x) {
        return PopCount<ui8>(x);
    },
          iface);
}

Y_CPU_BENCHMARK(PopCount_16, iface) {
    DoRun([](ui16 x) {
        return PopCount<ui16>(x);
    },
          iface);
}

Y_CPU_BENCHMARK(PopCount_32, iface) {
    DoRun([](ui32 x) {
        return PopCount<ui32>(x);
    },
          iface);
}

Y_CPU_BENCHMARK(PopCount_64, iface) {
    DoRun([](ui64 x) {
        return PopCount<ui64>(x);
    },
          iface);
}

#if !defined(_MSC_VER)
Y_CPU_BENCHMARK(BUILTIN_64, iface) {
    DoRun([](ui64 x) {
        return __builtin_popcountll(x);
    },
          iface);
}
#endif
