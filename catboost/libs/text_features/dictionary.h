#pragma once

#include "tokenizer.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/options/text_feature_options.h>

#include <library/text_processing/dictionary/dictionary.h>
#include <library/text_processing/dictionary/dictionary_builder.h>


namespace NCB {
    using IDictionary = NTextProcessing::NDictionary::IDictionary;
    using TDictionaryPtr = TIntrusivePtr<IDictionary>;

    template <class TTextFeature>
    class TIterableTextFeature {
    public:
        TIterableTextFeature(const TTextFeature& textFeature)
        : TextFeature(textFeature)
        {}

        template <class F>
        void ForEach(F&& visitor) const {
            std::for_each(TextFeature.begin(), TextFeature.end(), visitor);
        }

    private:
        const TTextFeature& TextFeature;
    };

    template <>
    class TIterableTextFeature<TMaybeOwningConstArraySubset<TString, ui32>> {
    public:
        TIterableTextFeature(const TMaybeOwningConstArraySubset<TString, ui32>& textFeature)
        : TextFeature(textFeature)
        {}

        template <class F>
        void ForEach(F&& visitor) const {
            TextFeature.ForEach([&visitor](ui32 /*index*/, TStringBuf phrase){visitor(phrase);});
        }
    private:
        const TMaybeOwningConstArraySubset<TString, ui32>& TextFeature;
    };

    template <class TTextFeatureType>
    inline TDictionaryPtr CreateDictionary(
        const TIterableTextFeature<TTextFeatureType>& textFeature,
        const NCatboostOptions::TTextProcessingOptions& textProcessingOptions,
        TTokenizerPtr tokenizer) {
        NTextProcessing::NDictionary::TDictionaryBuilder dictionaryBuilder(
            textProcessingOptions.DictionaryBuilderOptions,
            textProcessingOptions.DictionaryOptions
        );

        TVector<TStringBuf> tokens;
        /* the order of tokens doesn't matter for dictionary builder */
        const auto& tokenize = [&tokenizer, &tokens](TStringBuf phrase) {
            TVector<TStringBuf> phraseTokens;
            tokenizer->Tokenize(phrase, &phraseTokens);
            tokens.insert(tokens.end(), phraseTokens.begin(), phraseTokens.end());
        };
        textFeature.ForEach(tokenize);
        dictionaryBuilder.Add(tokens);

        return dictionaryBuilder.FinishBuilding().Release();
    }
}
