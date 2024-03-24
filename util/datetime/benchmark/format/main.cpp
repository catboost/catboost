#include <library/cpp/testing/gbenchmark/benchmark.h>

#include <util/datetime/base.h>
#include <util/stream/str.h>

class TTimestampGenerator {
public:
    TInstant operator()() {
        TInstant result = TInstant::MicroSeconds(Base_ + Current_);
        Current_ = (Current_ + Step_) % Range_;
        return result;
    }

private:
    static constexpr ui64 Step_ = TDuration::MicroSeconds(1234567891011).MicroSeconds();
    static constexpr ui64 Range_ = 1ull << 45; // ~ year
    static constexpr ui64 Base_ = TInstant::Seconds(1605320321).MicroSeconds();
    ui64 Current_ = 0;
};

Y_FORCE_INLINE static void BenchFormatStream(auto&& formatFn, benchmark::State& state) {
    TTimestampGenerator gen;
    TStringStream ss;
    for (auto _ : state) {
        ss << formatFn(gen());
        Y_DO_NOT_OPTIMIZE_AWAY(ss.Str());
        ss.clear();
    }
}

Y_FORCE_INLINE static void BenchToString(auto&& toStringFn, benchmark::State& state) {
    TTimestampGenerator gen;
    TString s;
    for (auto _ : state) {
        s = toStringFn(gen());
        Y_DO_NOT_OPTIMIZE_AWAY(s);
    }
}

void BM_FormatIsoLocal(benchmark::State& state) {
    BenchFormatStream(FormatIsoLocal, state);
}

void BM_FormatLocal(benchmark::State& state) {
    BenchFormatStream(FormatLocal, state);
}

void BM_ToStringIsoLocal(benchmark::State& state) {
    BenchToString(std::mem_fn(&TInstant::ToIsoStringLocal), state);
}

void BM_ToStringLocal(benchmark::State& state) {
    BenchToString(std::mem_fn(&TInstant::ToIsoStringLocal), state);
}

BENCHMARK(BM_FormatIsoLocal);
BENCHMARK(BM_FormatLocal);
BENCHMARK(BM_ToStringIsoLocal);
BENCHMARK(BM_ToStringLocal);
