#include <library/cpp/case_insensitive_string/ut_gtest/util/locale_guard.h>
#include <library/cpp/case_insensitive_string/case_insensitive_string.h>

#include <benchmark/benchmark.h>

#include <util/generic/string.h>
#include <util/random/random.h>

#include <array>

#include <cerrno>
#include <clocale>
#include <cstring>

namespace {
    template <typename TStrBuf>
    void BenchmarkCompare(benchmark::State& state, TStrBuf s1, TStrBuf s2) {
        for (auto _ : state) {
            benchmark::DoNotOptimize(s1);
            benchmark::DoNotOptimize(s2);
            auto cmp = s1.compare(s2);
            benchmark::DoNotOptimize(cmp);
        }
    }

    char RandomPrintableChar() {
        while (true) {
            unsigned char c = RandomNumber(127u);
            if (std::isprint(c)) {
                return c;
            }
        }
    }

    const std::array<const char*, 2> Locales = {
        "C",
        "ru_RU.CP1251",
    };
}

template <typename TStrBuf>
void CompareEqualStrings(benchmark::State& state) {
    SetRandomSeed(123);

    size_t n = state.range(0);
    size_t locIndex = state.range(1);

    TLocaleGuard loc(Locales[locIndex]);
    if (loc.Error()) {
        state.SkipWithMessage(TString::Join(Locales[locIndex], " locale is not available: ", loc.Error()));
        return;
    }
    TString s1(Reserve(n)), s2(Reserve(n));
    for (size_t i = 0; i < n; ++i) {
        auto c = RandomPrintableChar();
        s1.push_back(std::toupper(c));
        s2.push_back(std::tolower(c));
    }

    BenchmarkCompare(state, TStrBuf(s1.data(), s1.size()), TStrBuf(s2.data(), s2.size()));
}

#define BENCH_ARGS ArgNames({"strlen", "locale"})->ArgsProduct({{2, 4, 8, 16, 32, 64}, {0, 1}})

BENCHMARK(CompareEqualStrings<TCaseInsensitiveStringBuf>)->BENCH_ARGS;
BENCHMARK(CompareEqualStrings<TCaseInsensitiveAsciiStringBuf>)->BENCH_ARGS;

#undef BENCH_ARGS
