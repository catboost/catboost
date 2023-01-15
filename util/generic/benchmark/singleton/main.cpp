#include <library/testing/benchmark/bench.h>

#include <util/generic/singleton.h>
#include <util/generic/xrange.h>

char& FF1() noexcept;
char& FF2() noexcept;

namespace {
    struct X {
        inline X() {
        }

        char Buf[100];
    };

    inline X& F1() noexcept {
        static X x;

        return x;
    }

    inline X& F2() noexcept {
        return *Singleton<X>();
    }
}

Y_CPU_BENCHMARK(MagicStatic, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(F1().Buf);
    }
}

Y_CPU_BENCHMARK(Singleton, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(F2().Buf);
    }
}

Y_CPU_BENCHMARK(MagicStaticNI, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FF1());
    }
}

Y_CPU_BENCHMARK(SingletonNI, iface) {
    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Y_DO_NOT_OPTIMIZE_AWAY(FF2());
    }
}
