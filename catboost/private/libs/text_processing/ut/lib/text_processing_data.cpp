#include "text_processing_data.h"

#include <catboost/private/libs/options/text_processing_options.h>

#include <util/generic/set.h>
#include <util/generic/xrange.h>

using namespace NCB;
using namespace NCatboostOptions;
using namespace NTextProcessing::NDictionary;

namespace {
    struct TTextProcessingDescription {
    public:
        ui32 TextFeatureIdx;
        TString DictionaryName;
        TSet<EFeatureCalcerType> FeatureCalcers;
    };
}

void NCBTest::CreateTextDataForTest(
    TVector<TTextFeature>* features,
    TMap<ui32, TTokenizedTextFeature>* tokenizedFeatures,
    TVector<ui32>* target,
    TTextDigitizers* textDigitizers,
    TTextProcessingOptions* textProcessingOptions
) {
    {
        TVector<TString> feature0 = {
            "a b a c a b a",
            "a a",
            "b b",
            "a a a c c"
        };
        TVector<TString> feature1 = {
            "e e f f g g h h",
            "f f",
            "g g",
            "h h"
        };
        TVector<TString> feature2 = {
            "i j k l m n",
            "m",
            "n n n",
            "l i i"
        };
        TVector<TString> feature3 = {
            "aabbaaccaba",
            "aa bb",
            "bb",
            "aaacc"
        };
        features->insert(features->end(), {feature0, feature1, feature2, feature3});
    }

    *target = {1, 1, 0, 1};

    TVector<TTextProcessingDescription> textProcessingDescriptions = {
        {
            0,
            "default_dictionary",
            {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BM25}
        },
        {
            1,
            "default_dictionary",
            {EFeatureCalcerType::NaiveBayes}
        },
        {
            2,
            "default_dictionary",
            {EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BM25}
        },
        {
            3,
            "letter_unigram_dictionary",
            {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes}
        },
        {
            3,
            "letter_bigram_dictionary",
            {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BM25}
        },
        {
            3,
            "word_bigram_dictionary",
            {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes, EFeatureCalcerType::BM25}
        }
    };

    TMap<TString, TDictionaryOptions> dictionariesOptions;
    {
        NTextProcessing::NDictionary::TDictionaryOptions letterGramDictionaryOptions;
        letterGramDictionaryOptions.TokenLevelType = NTextProcessing::NDictionary::ETokenLevelType::Letter;
        letterGramDictionaryOptions.GramOrder = 1;

        NTextProcessing::NDictionary::TDictionaryOptions letterBiGramDictionaryOptions;
        letterBiGramDictionaryOptions.TokenLevelType = NTextProcessing::NDictionary::ETokenLevelType::Letter;
        letterBiGramDictionaryOptions.GramOrder = 2;

        NTextProcessing::NDictionary::TDictionaryOptions wordBiGramDictionaryOptions;
        letterBiGramDictionaryOptions.TokenLevelType = NTextProcessing::NDictionary::ETokenLevelType::Word;
        letterBiGramDictionaryOptions.GramOrder = 2;

        dictionariesOptions = {
            {"default_dictionary", TDictionaryOptions{}},
            {"letter_unigram_dictionary", letterGramDictionaryOptions},
            {"letter_bigram_dictionary", letterBiGramDictionaryOptions},
            {"word_bigram_dictionary", wordBiGramDictionaryOptions}
        };
    }

    {
        TVector<TTextColumnDictionaryOptions> textColumnDictionariesOptions;
        TMap<TString, TVector<TTextFeatureProcessing>> textFeatureProcessings;

        {
            ui32 tokenizedFeatureIdx = features->size();
            for (const auto& textProcessingDescription: textProcessingDescriptions) {
                const TString& dictionaryName = textProcessingDescription.DictionaryName;
                const ui32 textFeatureId = textProcessingDescription.TextFeatureIdx;
                const TDictionaryOptions& dictionaryOptions = dictionariesOptions[dictionaryName];

                TDictionaryBuilderOptions dictionaryBuilderOptions{1, -1};
                TTextColumnDictionaryOptions textColumnDictionaryOptions{
                    dictionaryName,
                    dictionaryOptions,
                    dictionaryBuilderOptions
                };
                textColumnDictionariesOptions.push_back(textColumnDictionaryOptions);

                TTokenizerPtr tokenizer = CreateTokenizer();

                TDictionaryPtr dictionary = CreateDictionary(
                    TIterableTextFeature(features->at(textFeatureId)),
                    textColumnDictionaryOptions,
                    tokenizer
                );

                textDigitizers->AddDigitizer(textFeatureId, tokenizedFeatureIdx, {tokenizer, dictionary});
                tokenizedFeatureIdx++;

                const TString stringFeatureId = ToString(textFeatureId);
                if (!textFeatureProcessings.contains(stringFeatureId)) {
                    textFeatureProcessings[stringFeatureId] = {};
                }

                for (EFeatureCalcerType featureCalcer : textProcessingDescription.FeatureCalcers) {
                    TTextFeatureProcessing featureProcessing{
                        {TFeatureCalcerDescription(featureCalcer)},
                        {"SomeTokenizerName"},
                        {dictionaryName}
                    };
                    textFeatureProcessings.at(stringFeatureId).push_back(featureProcessing);
                }
            }
        }

        NPar::TLocalExecutor localExecutor;
        textDigitizers->Apply(
            [&](ui32 textFeatureIdx) {
                return TIterableTextFeature(features->at(textFeatureIdx));
            },
            [&](ui32 tokenizedFeatureIdx, const TVector<TText>& tokenizedFeature) {
                tokenizedFeatures->operator[](tokenizedFeatureIdx) = tokenizedFeature;
            },
            &localExecutor
        );

        *textProcessingOptions = TTextProcessingOptions(
            {TTextColumnTokenizerOptions()},
            std::move(textColumnDictionariesOptions),
            std::move(textFeatureProcessings)
        );
    }
}
