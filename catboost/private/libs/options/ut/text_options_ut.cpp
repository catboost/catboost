#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/options/runtime_text_options.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/json/json_reader.h>

#include <util/generic/xrange.h>


using namespace NCatboostOptions;
using namespace NTextProcessing::NDictionary;

Y_UNIT_TEST_SUITE(TTextOptionsTest) {
    Y_UNIT_TEST(TestPlainOptions) {
        const TString stringJson =
            "{\n"
            "    \"text_processing\": {\n"
            "        \"tokenizers\": [\n"
            "            {\n"
            "                \"tokenizer_id\": \"Space\"\n"
            "            },\n"
            "            {\n"
            "                \"tokenizer_id\": \"Comma\",\n"
            "                \"delimiter\": \"#\"\n"
            "            }\n"
            "        ],\n"
            "        \"dictionaries\": [\n"
            "            {\n"
            "                \"dictionary_id\": \"Letter\",\n"
            "                \"token_level_type\": \"Letter\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"Word\",\n"
            "                \"token_level_type\": \"Word\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"UniGram\",\n"
            "                \"gram_order\": \"1\",\n"
            "                \"token_level_type\": \"Letter\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"BiGram\",\n"
            "                \"gram_order\": \"2\",\n"
            "                \"token_level_type\": \"Letter\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"TriGram\",\n"
            "                \"gram_order\": \"3\",\n"
            "                \"token_level_type\": \"Letter\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"WordDictOccur2\",\n"
            "                \"occurrence_lower_bound\": \"2\",\n"
            "                \"token_level_type\": \"Word\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"WordDictOccur5\",\n"
            "                \"occurrence_lower_bound\": \"5\",\n"
            "                \"token_level_type\": \"Word\"\n"
            "            },\n"
            "            {\n"
            "                \"dictionary_id\": \"WordDictOccur10\",\n"
            "                \"occurrence_lower_bound\": \"10\",\n"
            "                \"token_level_type\": \"Word\"\n"
            "            }\n"
            "        ],\n"
            "        \"feature_processing\": {\n"
            "            \"default\": [\n"
            "                {\n"
            "                    \"tokenizers_names\": [\"Space\"],\n"
            "                    \"dictionaries_names\": [\"UniGram\", \"BiGram\", \"TriGram\"],\n"
            "                    \"feature_calcers\": [\"BoW\", \"NaiveBayes\"]\n"
            "                }\n"
            "            ],\n"
            "            \"0\": [\n"
            "                {\n"
            "                    \"tokenizers_names\": [\"Space\"],\n"
            "                    \"dictionaries_names\": [\"Letter\", \"Word\"],\n"
            "                    \"feature_calcers\": [\"BoW\"]\n"
            "                },\n"
            "                {\n"
            "                    \"tokenizers_names\": [\"Space\"],\n"
            "                    \"dictionaries_names\": [\"Word\"],\n"
            "                    \"feature_calcers\": [\"NaiveBayes\"]\n"
            "                }\n"
            "            ],\n"
            "            \"1\": [\n"
            "                {\n"
            "                    \"tokenizers_names\": [\"Space\"],\n"
            "                    \"dictionaries_names\": [\"WordDictOccur2\", \"WordDictOccur5\"],\n"
            "                    \"feature_calcers\": [\"NaiveBayes\"]\n"
            "                },\n"
            "                {\n"
            "                    \"tokenizers_names\": [\"Space\"],\n"
            "                    \"dictionaries_names\": [\"WordDictOccur2\", \"WordDictOccur10\"],\n"
            "                    \"feature_calcers\": [\"BM25\"]\n"
            "                }\n"
            "            ]\n"
            "        }\n"
            "    }\n"
            "}";

        NJson::TJsonValue plainOptions;
        NJson::ReadJsonTree(stringJson, &plainOptions);
        NJson::TJsonValue optionsJson;

        TSet<TString> seenKeys;
        ParseTextProcessingOptionsFromPlainJson(plainOptions, &optionsJson, &seenKeys);

        TTextProcessingOptions textProcessingOptions;
        textProcessingOptions.Load(optionsJson);

        TVector<TTextColumnTokenizerOptions> tokenizers = textProcessingOptions.GetTokenizers();
        for (const auto& tokenizer : tokenizers) {
            const TString tokenizerName = tokenizer.TokenizerId.Get();
            const TString delimiter = tokenizer.TokenizerOptions->Delimiter;

            if (tokenizerName == "Space") {
                UNIT_ASSERT_EQUAL(delimiter, " ");
            } else if (tokenizerName == "Comma") {
                UNIT_ASSERT_EQUAL(delimiter, "#");
            }
        }

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
                EFeatureCalcerType calcerType = processingUnit.FeatureCalcers.Get()[0].CalcerType.Get();
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
                EFeatureCalcerType featureCalcer = processingUnit.FeatureCalcers.Get()[0].CalcerType.Get();
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
                EFeatureCalcerType featureCalcer = processingUnit.FeatureCalcers.Get()[0].CalcerType.Get();
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
