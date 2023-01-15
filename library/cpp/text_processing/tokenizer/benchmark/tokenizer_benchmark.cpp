#include <library/cpp/text_processing/tokenizer/tokenizer.h>
#include <util/generic/size_literals.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>

#include <contrib/libs/benchmark/include/benchmark/benchmark.h>
#include <library/cpp/testing/common/env.h>


using namespace NTextProcessing::NTokenizer;

class TTokenizerFixture: public ::benchmark::Fixture {
public:
    TTokenizerFixture()
    {
        TFileInput file(SRC_("texts.txt"));
        while (true) {
            if (0 == file.ReadLine(Queries.emplace_back())) {
                Queries.pop_back();
                break;
            }
        }
    }

protected:
    TVector<TString> Queries;
};

static void BMTokenize(benchmark::State& st, const TVector<TString>& queries, bool useCache) {
    TTokenizerOptions options;
    options.SeparatorType = ESeparatorType::BySense;
    options.TokenTypes = {ETokenType::Word, ETokenType::Number, ETokenType::Punctuation};
    options.Lowercasing = true;
    options.Lemmatizing = true;
    options.LemmerCacheSize = useCache ? 1_MB : 0;
    options.SeparatorType = ESeparatorType::BySense;
    TTokenizer tokenizer(options);

    for (auto _ : st) {
        for (auto& q : queries) {
            benchmark::DoNotOptimize(tokenizer.Tokenize(q));
        }
    }
}

BENCHMARK_DEFINE_F(TTokenizerFixture, TokenizeWithCache)(benchmark::State& st) {
    BMTokenize(st, Queries, true);
}

BENCHMARK_DEFINE_F(TTokenizerFixture, TokenizeWithoutCache)(benchmark::State& st) {
    BMTokenize(st, Queries, false);
}

BENCHMARK_REGISTER_F(TTokenizerFixture, TokenizeWithCache)->Threads(5)->Threads(25);
BENCHMARK_REGISTER_F(TTokenizerFixture, TokenizeWithoutCache)->Threads(5)->Threads(25);
