#include <catboost/libs/text_features/dictionary.h>
#include <catboost/libs/text_features/embedding_loader.h>
#include <catboost/libs/text_features/embedding.h>
#include <catboost/libs/text_features/embedding_online_features.h>
#include <catboost/libs/text_features/text_column_builder.h>
#include <library/text_processing/dictionary/dictionary_builder.h>
#include <library/unittest/registar.h>
#include <util/generic/guid.h>
#include <util/random/fast.h>
#include <util/stream/file.h>
using namespace std;
using namespace NCB;


Y_UNIT_TEST_SUITE(EmbeddingTest) {
    using TDictionaryBuilder = NTextProcessing::NDictionary::TDictionaryBuilder;

    TVector<float> RandomVec(ui32 dim, TFastRng64* rand) {
        TVector<float> res;
        res.reserve(dim);
        for (ui32 i = 0; i < dim; ++i) {
            const auto val = static_cast<int>(100000 * rand->GenRandReal1()) / 100000.;
            res.push_back(val);
        }
        return res;
    }

    THashMap<TString, TVector<float>> GenerateEmbedding(ui32 tokenCount, ui32 dim, TFastRng64* rand, TDictionaryBuilder* builder) {
        THashMap<TString, TVector<float>> result;
        for (ui32 i = 0; i < tokenCount; ++i) {
            auto word = TStringBuilder() << "word_" << i;
            result[word] = RandomVec(dim, rand);
            builder->Add(word);
        }
        return result;
    }

    void DumpEmbedding(const THashMap<TString, TVector<float>>& embedding, TString file) {
        TOFStream out(file);
        for (const auto& [key, vec] : embedding) {
            out << key;
            for (auto val : vec) {
                out << "\t" << val;
            }
            out << Endl;
        }
    }


    void EnsureVecEqual(TConstArrayRef<float> left, TConstArrayRef<float> right) {
        UNIT_ASSERT_VALUES_EQUAL_C(left.size(), right.size(), TStringBuilder() << left.size() << "≠" << right.size());
        for (ui32 i = 0; i < left.size(); ++i) {
            UNIT_ASSERT(std::isfinite(left[i]) == std::isfinite(right[i]));
            UNIT_ASSERT_DOUBLES_EQUAL(left[i], right[i], 1e-7);
        }
    }

    Y_UNIT_TEST(TestEmbeddingLoad) {

        TFastRng64 rng(0);
        TDictionaryBuilder builder{NTextProcessing::NDictionary::TDictionaryBuilderOptions{1, -1},  NTextProcessing::NDictionary::TDictionaryOptions()};
        auto embedding = GenerateEmbedding(1000, 100, &rng, &builder);
        DumpEmbedding(embedding, "embedding.txt");

        auto dict = builder.FinishBuilding();
        auto embeddingFromFile = LoadEmbedding("embedding.txt", *dict);

        TVector<float> testVec;

        TEmbeddingOnlineFeatures calcer(2, embeddingFromFile);

        int i = 0;

        for (const auto& [word, vec] : embedding) {
            UNIT_ASSERT(dict->Apply(word) != dict->GetUnknownTokenId());
            const TText& text = TokenToText(*dict, word);
            auto id = text.begin()->first;
            UNIT_ASSERT(dict->Apply(word) == id);

            embeddingFromFile->Apply(text, &testVec);
            TVector<float> ref;
            for (auto val : vec) {
                ref.push_back(val / 1.5);
            }
            EnsureVecEqual(ref, testVec);
            auto features = calcer.CalcFeaturesAndAddEmbedding(rng.GenRandReal1() > 0.5, testVec);
            int j = 0;
            for (auto f : features) {
                if (!std::isfinite(f)) {
                    Cout << i << " " << j << Endl;
                }
                UNIT_ASSERT(std::isfinite(f));
                ++j;
            }
            ++i;
        }
    }

}
