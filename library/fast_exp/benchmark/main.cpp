#include <library/fast_exp/fast_exp.h>
#include <library/testing/benchmark/bench.h>

#include <util/random/fast.h>
#include <util/random/random.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>

#include <cmath>

namespace {
    struct TData: public TVector<double> {
        inline TData() {
            for (size_t i = 0; i < 10000000; ++i) {
                push_back(RandomNumber<double>() * 10);
            }
        }
    };
}

Y_CPU_BENCHMARK(FastExp, iface) {
    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(fast_exp(1.0 / (1.0 + x)));
    }
}

Y_CPU_BENCHMARK(FastExpOld, iface) {
    const auto data = -RandomNumber<double>();

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(fast_exp(data));
    }
}

Y_CPU_BENCHMARK(FastExpNegative, iface) {
    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(fast_exp(-1.0 / (1.0 + x)));
    }
}

Y_CPU_BENCHMARK(FastExpRnd, iface) {
    TReallyFastRng32 rng(0);

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(fast_exp(rng.GenRandReal1()));
    }
}

Y_CPU_BENCHMARK(FastExpPure, iface) {
    const auto& data = *Singleton<TData>();

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(fast_exp(data[x % data.size()]));
    }
}

Y_CPU_BENCHMARK(Libc, iface) {
    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(std::exp(1.0 / (1.0 + x)));
    }
}

Y_CPU_BENCHMARK(LibcOld, iface) {
    const auto data = -RandomNumber<double>();

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(std::exp(data));
    }
}

Y_CPU_BENCHMARK(LibcNegative, iface) {
    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(std::exp(-1.0 / (1.0 + x)));
    }
}

Y_CPU_BENCHMARK(LibcRnd, iface) {
    TReallyFastRng32 rng(0);

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(std::exp(rng.GenRandReal1()));
    }
}

Y_CPU_BENCHMARK(LibcPure, iface) {
    const auto& data = *Singleton<TData>();

    for (size_t x = 0; x < iface.Iterations(); ++x) {
        Y_DO_NOT_OPTIMIZE_AWAY(std::exp(data[x % data.size()]));
    }
}
