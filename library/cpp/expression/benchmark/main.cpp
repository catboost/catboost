#include <library/cpp/testing/gbenchmark/benchmark.h>
#include <library/cpp/expression/expression.h>
#include <library/cpp/resource/resource.h>
#include <util/stream/file.h>
#include <util/stream/input.h>

class TSimpleBench : public benchmark::Fixture {
public:
    TSimpleBench() {}

    void SetUp(benchmark::State&) override {
        // setup test data
        TFileInput dataFile("./expression_formulas.txt");

        TString line;
        while (dataFile.ReadLine(line)) {
            ExpressionFormulas_.push_back(line);
        }
    }

    void TearDown(benchmark::State&) override {
        ExpressionFormulas_.clear();
    }

    // benchmarks
    void Initialize() {
        for (const auto& expressionFormula : ExpressionFormulas_) {
            TExpression expression(expressionFormula);
        }
    }

private:
    TVector<TString> ExpressionFormulas_;
};

BENCHMARK_DEFINE_F(TSimpleBench, BMSimpleBench)(benchmark::State& state) {
    for (auto _ : state) {
        Initialize();
    }
}

BENCHMARK_REGISTER_F(TSimpleBench, BMSimpleBench)
    ->Iterations(10)->Unit(benchmark::kSecond)->UseRealTime();
