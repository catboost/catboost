#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/text_features/embedding_online_features.h>
#include <catboost/private/libs/text_processing/dictionary.h>
#include <catboost/private/libs/text_processing/embedding_loader.h>
#include <catboost/private/libs/text_processing/embedding.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>
#include <library/cpp/text_processing/dictionary/dictionary_builder.h>
#include <library/cpp/unittest/registar.h>
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
        using namespace NCatboostOptions;

        TFastRng64 rng(0);
        auto dictionaryBuilderOptions = TDictionaryBuilderOptions{1, -1};
        NTextProcessing::NDictionary::TDictionaryOptions dictionaryOptions;

        TDictionaryBuilder builder{dictionaryBuilderOptions, dictionaryOptions};
        NCatboostOptions::TTextColumnDictionaryOptions textColumnDictionaryOptions(
            "default",
            dictionaryOptions,
            dictionaryBuilderOptions
        );
        auto embedding = GenerateEmbedding(1000, 100, &rng, &builder);
        DumpEmbedding(embedding, "embedding.txt");

        auto dict = TDictionaryProxy(builder.FinishBuilding());
        auto embeddingFromFile = LoadEmbedding("embedding.txt", dict);

        TVector<float> testVec;

        const ui32 numClasses = 2;
        TEmbeddingOnlineFeatures calcer(CreateGuid(), numClasses, embeddingFromFile);

        int i = 0;

        for (const auto& [word, vec] : embedding) {
            UNIT_ASSERT(dict.Apply(word) != dict.GetUnknownTokenId());
            TText text;
            dict.Apply({word}, &text);
            const auto id = text.begin()->Token();
            UNIT_ASSERT(dict.Apply(word) == id);

            embeddingFromFile->Apply(text, &testVec);
            TVector<float> ref;
            for (auto val : vec) {
                ref.push_back(val / 1.5);
            }
            EnsureVecEqual(ref, testVec);
            {
                TEmbeddingFeaturesVisitor visitor(numClasses, embeddingFromFile->Dim());
                visitor.UpdateEmbedding(rng.GenRandReal1() > 0.5, testVec, &calcer);
            }
            TVector<float> features;
            features.resize(calcer.FeatureCount());
            calcer.Compute(testVec, TOutputFloatIterator(features.begin(), features.size()));
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
