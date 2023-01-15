#include <library/testing/benchmark/bench.h>

#include <util/generic/function.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>
#include <util/string/cast.h>
#include <util/string/join.h>

namespace {
    // This class assigns random values to variadic lists of variables of different types.
    // It can be used to randomize a tuple via Apply() (arcadia version of std::apply).
    class TRandomizer {
    public:
        TRandomizer(ui64 seed)
            : Prng(seed)
        {
        }

        void Randomize(ui16& i) {
            i = static_cast<ui16>(Prng.GenRand());
        }

        void Randomize(ui32& i) {
            i = static_cast<ui32>(Prng.GenRand());
        }

        void Randomize(double& d) {
            d = Prng.GenRandReal4() + Prng.Uniform(Max<ui16>());
        }

        void Randomize(TString& s) {
            s = ::ToString(Prng.GenRand());
        }

        template <typename T, typename... TArgs>
        void Randomize(T& t, TArgs&... args) {
            Randomize(t);
            Randomize(args...);
        }

    private:
        TFastRng<ui64> Prng;
    };

    template <size_t N, typename... T>
    struct TExamplesHolder {
        using TExamples = TVector<std::tuple<T...>>;
        TExamples Examples;

        TExamplesHolder()
            : Examples(N)
        {
            TRandomizer r{N * sizeof(typename TExamples::value_type) * 42};
            for (auto& x : Examples) {
                Apply([&r](T&... t) { r.Randomize(t...); }, x);
            }
        }
    };

    template <typename... TArgs>
    TString JoinTuple(std::tuple<TArgs...> t) {
        return Apply([](TArgs... x) -> TString { return Join("-", x...); }, t);
    }
}

#define DEFINE_BENCHMARK(count, types, ...)                                             \
    Y_CPU_BENCHMARK(Join_##count##_##types, iface) {                                    \
        const auto& examples = Default<TExamplesHolder<count, __VA_ARGS__>>().Examples; \
        for (const auto i : xrange(iface.Iterations())) {                               \
            Y_UNUSED(i);                                                                \
            for (const auto e : examples) {                                             \
                Y_DO_NOT_OPTIMIZE_AWAY(JoinTuple(e));                                   \
            }                                                                           \
        }                                                                               \
    }

DEFINE_BENCHMARK(100, SS, TString, TString);
DEFINE_BENCHMARK(100, SSS, TString, TString, TString);
DEFINE_BENCHMARK(100, SSSSS, TString, TString, TString, TString, TString);

DEFINE_BENCHMARK(100, ss, ui16, ui16);
DEFINE_BENCHMARK(100, SsS, TString, ui16, TString);
DEFINE_BENCHMARK(100, SsSsS, TString, ui16, TString, ui16, TString);

DEFINE_BENCHMARK(100, ii, ui32, ui32);
DEFINE_BENCHMARK(100, SiS, TString, ui32, TString);
DEFINE_BENCHMARK(100, SiSiS, TString, ui32, TString, ui32, TString);

DEFINE_BENCHMARK(100, dd, double, double);
DEFINE_BENCHMARK(100, SdS, TString, double, TString);
DEFINE_BENCHMARK(100, SdSdS, TString, double, TString, double, TString);

#undef DEFINE_BENCHMARK
