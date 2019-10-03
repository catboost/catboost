#include "text_processing_data.h"

#include <catboost/private/libs/options/runtime_text_options.h>
#include <catboost/private/libs/options/text_processing_options.h>
#include <catboost/private/libs/text_processing/embedding.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <util/generic/xrange.h>

using namespace NCB;
using namespace NCatboostOptions;

static TVector<TText> Preprocess(
    const TVector<TString>& feature,
    const TTokenizerPtr& tokenizer,
    const TDictionaryPtr& dictionary
) {
    TTextColumnBuilder textColumnBuilder(tokenizer, dictionary, feature.size());
    for (ui32 index: xrange(feature.size())) {
        textColumnBuilder.AddText(index, feature[index]);
    }

    return textColumnBuilder.Build();
}

static TVector<TFeatureCalcerDescription> CreateDescriptionFromCalcerType(
    TConstArrayRef<EFeatureCalcerType> featureCalcers
) {
    TVector<TFeatureCalcerDescription> calcerDescriptions;
    calcerDescriptions.resize(featureCalcers.size());
    Copy(featureCalcers.begin(), featureCalcers.end(), calcerDescriptions.begin());
    return calcerDescriptions;
}

void NCBTest::CreateTextDataForTest(
    TVector<TTextFeature>* features,
    TVector<TTokenizedTextFeature>* tokenizedFeatures,
    TVector<TDictionaryPtr>* dictionaries,
    TTokenizerPtr* tokenizer,
    TVector <ui32>* target,
    TRuntimeTextOptions* textProcessingOptions
) {
    *tokenizer = CreateTokenizer();
    const TEmbeddingPtr embeddingPtr;
    const TTextColumnDictionaryOptions defaultDictionaryOptions = TTextColumnDictionaryOptions();

    TVector<TTextColumnDictionaryOptions> dictionariesOptions;
    TVector<TTokenizedFeatureDescription> tokenizedFeaturesOptions;

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
            "aa",
            "bb",
            "aaacc"
        };

        features->insert(features->end(), {feature0, feature1, feature2, feature3});
    }

    *target = {1, 1, 0, 1};

    {
        TDictionaryPtr dictionary0 = CreateDictionary<TVector<TString>>(
            TIterableTextFeature((*features)[0]),
            defaultDictionaryOptions,
            *tokenizer
        );

        TDictionaryPtr dictionary1 = CreateDictionary(
            TIterableTextFeature((*features)[1]),
            defaultDictionaryOptions,
            *tokenizer
        );

        TDictionaryPtr dictionary2 = CreateDictionary<TVector<TString>>(
            TIterableTextFeature((*features)[2]),
            defaultDictionaryOptions,
            *tokenizer
        );

        dictionariesOptions.push_back(defaultDictionaryOptions);


        NTextProcessing::NDictionary::TDictionaryOptions letterGramDictionaryOptions;
        letterGramDictionaryOptions.TokenLevelType = NTextProcessing::NDictionary::ETokenLevelType::Letter;
        letterGramDictionaryOptions.GramOrder = 1;

        TTextColumnDictionaryOptions letterGramOptions;
        letterGramOptions.DictionaryId = "letter_gram_dictionary";
        letterGramOptions.DictionaryOptions.Set(letterGramDictionaryOptions);
        dictionariesOptions.push_back(letterGramOptions);


        TDictionaryPtr dictionary3 = CreateDictionary(
            TIterableTextFeature((*features)[3]),
            letterGramOptions,
            *tokenizer
        );


        NTextProcessing::NDictionary::TDictionaryOptions letterBiGramDictionaryOptions;
        letterBiGramDictionaryOptions.TokenLevelType = NTextProcessing::NDictionary::ETokenLevelType::Letter;
        letterBiGramDictionaryOptions.GramOrder = 2;

        TTextColumnDictionaryOptions letterBiGramOptions;
        letterBiGramOptions.DictionaryId.Set("letter_bigram_dictionary");
        letterBiGramOptions.DictionaryOptions.Set(letterBiGramDictionaryOptions);
        dictionariesOptions.push_back(letterBiGramOptions);

        TDictionaryPtr dictionary4 = CreateDictionary(
            TIterableTextFeature((*features)[3]),
            letterBiGramOptions,
            *tokenizer
        );

        *dictionaries = {dictionary0, dictionary1, dictionary2, dictionary3, dictionary4};
    }

    {
        tokenizedFeaturesOptions.push_back(
            TTokenizedFeatureDescription(
                dictionariesOptions[0].DictionaryId.Get(),
                /* textFeature */0,
                CreateDescriptionFromCalcerType(
                    {EFeatureCalcerType::BoW, EFeatureCalcerType::BM25, EFeatureCalcerType::NaiveBayes}
                )
            )
        );

        tokenizedFeaturesOptions.push_back(
            TTokenizedFeatureDescription(
                dictionariesOptions[0].DictionaryId.Get(),
                /* textFeature */1,
                CreateDescriptionFromCalcerType({EFeatureCalcerType::NaiveBayes})
            )
        );

        tokenizedFeaturesOptions.push_back(
            TTokenizedFeatureDescription(
                dictionariesOptions[0].DictionaryId.Get(),
                /* textFeature */2,
                CreateDescriptionFromCalcerType(
                    {EFeatureCalcerType::BM25, EFeatureCalcerType::NaiveBayes}
                )
            )
        );

        tokenizedFeaturesOptions.push_back(
            TTokenizedFeatureDescription(
                dictionariesOptions[1].DictionaryId.Get(),
                /* textFeature */3,
                CreateDescriptionFromCalcerType(
                    {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes}
                )
            )
        );

        tokenizedFeaturesOptions.push_back(
            TTokenizedFeatureDescription(
                dictionariesOptions[2].DictionaryId.Get(),
                /* textFeature */3,
                CreateDescriptionFromCalcerType(
                    {EFeatureCalcerType::BoW, EFeatureCalcerType::BM25, EFeatureCalcerType::NaiveBayes}
                )
            )
        );

        for (ui32 tokenizedFeatureIdx: xrange(tokenizedFeaturesOptions.size())) {
            const auto& featureOptions = tokenizedFeaturesOptions[tokenizedFeatureIdx];
            tokenizedFeatures->emplace_back(
                Preprocess(
                    (*features)[featureOptions.TextFeatureId],
                    *tokenizer,
                    (*dictionaries)[tokenizedFeatureIdx]
                )
            );
        }
    }

    *textProcessingOptions = TRuntimeTextOptions(
        MakeConstArrayRef(dictionariesOptions),
        MakeConstArrayRef(tokenizedFeaturesOptions)
    );
}
