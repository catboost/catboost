#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/options/runtime_text_options.h>

#include <library/unittest/registar.h>
#include <library/json/json_reader.h>

#include <util/generic/xrange.h>


using namespace NCatboostOptions;
using namespace NTextProcessing::NDictionary;

Y_UNIT_TEST_SUITE(TTextOptionsTest) {
    Y_UNIT_TEST(TestPlainOptions) {
        TString stringJson =
            "{\n"
            "   \"dictionaries\": ["
            "      \"Letter:token_level_type=Letter\","
            "      \"Word:token_level_type=Word\","
            "      \"UniGram:token_level_type=Letter,gram_order=1\", "
            "      \"BiGram:token_level_type=Letter,gram_order=2\", "
            "      \"TriGram:token_level_type=Letter,gram_order=3\",\n"
            "      \"WordDictOccur2:min_token_occurrence=2,token_level_type=Word\",\n"
            "      \"WordDictOccur5:min_token_occurrence=5,token_level_type=Word\",\n"
            "      \"WordDictOccur10:min_token_occurrence=10,token_level_type=Word\"],\n"
            "   \"text_processing\": ["
            "      \"BoW:UniGram,BiGram,TriGram|NaiveBayes:UniGram,BiGram,TriGram\",\n"
            "      \"0~BoW:Letter,Word|NaiveBayes:Word\",\n"
            "      \"1~NaiveBayes:WordDictOccur2,WordDictOccur5|BM25:WordDictOccur2,WordDictOccur10\",\n"
            "   ]\n"
            "}";

        NJson::TJsonValue plainOptions;
        NJson::ReadJsonTree(stringJson, &plainOptions);
        NJson::TJsonValue optionsJson;

        TSet<TString> seenKeys;
        ParseTextProcessingOptionsFromPlainJson(plainOptions, &optionsJson, &seenKeys);

        TTextProcessingOptions textProcessingOptions;
        textProcessingOptions.Load(optionsJson);

        TVector<TTextColumnDictionaryOptions> dictionaries = textProcessingOptions.GetDictionaries();
        for (const auto& dictionary : dictionaries) {
            const TString dictionaryName = dictionary.DictionaryId.Get();
            const ETokenLevelType tokenLevelType = dictionary.DictionaryOptions->TokenLevelType;
            const ui32 gramOrder = dictionary.DictionaryOptions->GramOrder;

            if (dictionaryName == "Word") {
                UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Word);
                UNIT_ASSERT_EQUAL(gramOrder, 1);
            } else if (dictionaryName == "Letter" || dictionaryName == "UniGram") {
                UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Letter);
                UNIT_ASSERT_EQUAL(gramOrder, 1);
            } else if (dictionaryName == "BiGram") {
                UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Letter);
                UNIT_ASSERT_EQUAL(gramOrder, 2);
            } else if (dictionaryName == "TriGram") {
                UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Letter);
                UNIT_ASSERT_EQUAL(gramOrder, 3);
            } else {
                ui64 minTokenOccurrence = dictionary.DictionaryBuilderOptions->OccurrenceLowerBound;
                if (dictionaryName == "WordDictOccur2") {
                    UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Word);
                    UNIT_ASSERT_EQUAL(minTokenOccurrence, 2);
                } else if (dictionaryName == "WordDictOccur5") {
                    UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Word);
                    UNIT_ASSERT_EQUAL(minTokenOccurrence, 5);
                } else if (dictionaryName == "WordDictOccur10") {
                    UNIT_ASSERT_EQUAL(tokenLevelType, ETokenLevelType::Word);
                    UNIT_ASSERT_EQUAL(minTokenOccurrence, 10);
                }
            }
        }

        {
            const auto& defaultFeatureProcessing = textProcessingOptions.GetFeatureProcessing(10);
            for (const auto& processingUnit: defaultFeatureProcessing) {
                EFeatureCalcerType calcerType = processingUnit.FeatureCalcer->CalcerType.Get();
                UNIT_ASSERT(
                    calcerType == EFeatureCalcerType::BoW ||
                    calcerType == EFeatureCalcerType::NaiveBayes
                );
                TVector<TString> expected = {"BiGram", "TriGram", "UniGram"};
                TVector<TString> dictionaries = processingUnit.DictionariesNames.Get();
                Sort(dictionaries);
                UNIT_ASSERT_VALUES_EQUAL(expected, dictionaries);
            }

            const auto& feature0Processing = textProcessingOptions.GetFeatureProcessing(0);
            for (const auto& processingUnit: feature0Processing) {
                EFeatureCalcerType featureCalcer = processingUnit.FeatureCalcer->CalcerType.Get();
                TVector<TString> dictionaries = processingUnit.DictionariesNames.Get();
                UNIT_ASSERT(
                    featureCalcer == EFeatureCalcerType::BoW ||
                    featureCalcer == EFeatureCalcerType::NaiveBayes
                );
                if (featureCalcer == EFeatureCalcerType::BoW) {
                    TVector<TString> expected = {"Letter", "Word"};
                    Sort(dictionaries);
                    UNIT_ASSERT_VALUES_EQUAL(expected, dictionaries);
                } else {
                    UNIT_ASSERT_EQUAL(dictionaries.size(), 1);
                    UNIT_ASSERT_EQUAL(dictionaries[0], "Word");
                }
            }

            const auto& feature1Processing = textProcessingOptions.GetFeatureProcessing(1);
            for (const auto& processingUnit: feature1Processing) {
                EFeatureCalcerType featureCalcer = processingUnit.FeatureCalcer->CalcerType.Get();
                TVector<TString> dictionaries = processingUnit.DictionariesNames.Get();
                Sort(dictionaries);
                UNIT_ASSERT(
                    featureCalcer == EFeatureCalcerType::BM25 ||
                    featureCalcer == EFeatureCalcerType::NaiveBayes
                );
                if (featureCalcer == EFeatureCalcerType::NaiveBayes) {
                    TVector<TString> expected = {"WordDictOccur2", "WordDictOccur5"};
                    UNIT_ASSERT_VALUES_EQUAL(expected, dictionaries);
                } else {
                    TVector<TString> expected = {"WordDictOccur10", "WordDictOccur2"};
                    UNIT_ASSERT_EQUAL(expected, dictionaries);
                }
            }
        }

        using TTestTextProcessingDescription = std::tuple<TString, ui32, TSet<EFeatureCalcerType>>;
        TVector<TTestTextProcessingDescription> expectedDescriptions = {
            {"Letter", 0, {EFeatureCalcerType::BoW}},
            {"Word", 0, {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes}},
            {"WordDictOccur2", 1, {EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BM25}},
            {"WordDictOccur5", 1, {EFeatureCalcerType::NaiveBayes}},
            {"WordDictOccur10", 1, {EFeatureCalcerType::BM25}},
            {"UniGram", 2, {EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BoW}},
            {"BiGram", 2, {EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BoW}},
            {"TriGram", 2, {EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BoW}}
        };

        TRuntimeTextOptions runtimeTextOptions(xrange(3), textProcessingOptions);
        UNIT_ASSERT_EQUAL(runtimeTextOptions.TokenizedFeatureCount(), 8);
        for (ui32 tokenizedFeatureIdx : xrange(runtimeTextOptions.TokenizedFeatureCount())) {
            const auto& featureDescription = runtimeTextOptions.GetTokenizedFeatureDescription(tokenizedFeatureIdx);
            const auto& expectedDescription = expectedDescriptions[tokenizedFeatureIdx];

            TString dictionaryId = std::get<0>(expectedDescription);
            ui32 textFeatureId = std::get<1>(expectedDescription);
            TSet<EFeatureCalcerType> expectedCalcers = std::get<2>(expectedDescription);

            TSet<EFeatureCalcerType> calcers;
            for (const auto& calcerDescription: featureDescription.FeatureEstimators.Get()) {
                calcers.insert(calcerDescription.CalcerType);
            }
            UNIT_ASSERT_EQUAL(dictionaryId, featureDescription.DictionaryId.Get());
            UNIT_ASSERT_EQUAL(textFeatureId, featureDescription.TextFeatureId.Get());
            UNIT_ASSERT_VALUES_EQUAL(expectedCalcers, calcers);
        }
    }
}
