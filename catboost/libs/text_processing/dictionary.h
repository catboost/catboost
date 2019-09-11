#pragma once

#include "tokenizer.h"

#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/options/text_feature_options.h>

#include <library/text_processing/dictionary/dictionary.h>
#include <library/text_processing/dictionary/dictionary_builder.h>
#include <library/text_processing/dictionary/mmap_frequency_based_dictionary.h>

namespace NCB {
    using IDictionary = NTextProcessing::NDictionary::IDictionary;
    using TDictionary = NTextProcessing::NDictionary::TDictionary;
    using TDictionaryPtr = TIntrusivePtr<IDictionary>;
    using TMMapDictionary = NTextProcessing::NDictionary::TMMapDictionary;

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
    class TIterableTextFeature<ITypedArraySubsetPtr<TString>> {
    public:
        TIterableTextFeature(ITypedArraySubsetPtr<TString> textFeature)
        : TextFeature(std::move(textFeature))
        {}

        template <class F>
        void ForEach(F&& visitor) const {
            TextFeature->ForEach([&visitor](ui32 /*index*/, TStringBuf phrase){visitor(phrase);});
        }
    private:
        ITypedArraySubsetPtr<TString> TextFeature;
    };

    template <class TTextFeatureType>
    inline TDictionaryPtr CreateDictionary(
        TConstArrayRef<TIterableTextFeature<TTextFeatureType>> textFeatures,
        const NCatboostOptions::TTextProcessingOptions& textProcessingOptions,
        const TTokenizerPtr& tokenizer) {

        NTextProcessing::NDictionary::TDictionaryBuilder dictionaryBuilder(
            textProcessingOptions.DictionaryBuilderOptions,
            textProcessingOptions.DictionaryOptions
        );

        for (const auto& textFeature: textFeatures) {
            TVector<TStringBuf> tokens;
            const auto& tokenize = [&tokenizer, &tokens](TStringBuf phrase) {
                TVector<TStringBuf> phraseTokens;
                tokenizer->Tokenize(phrase, &phraseTokens);
                tokens.insert(tokens.end(), phraseTokens.begin(), phraseTokens.end());
            };
            textFeature.ForEach(tokenize);
            dictionaryBuilder.Add(tokens);
        }

        return dictionaryBuilder.FinishBuilding();
    }

    template <class TTextFeatureType>
    inline TDictionaryPtr CreateDictionary(
        const TIterableTextFeature<TTextFeatureType>& textFeature,
        const NCatboostOptions::TTextProcessingOptions& textProcessingOptions,
        const TTokenizerPtr& tokenizer) {
        return CreateDictionary<TTextFeatureType>(
            TVector<TIterableTextFeature<TTextFeatureType>>({textFeature}),
            textProcessingOptions,
            tokenizer
        );
    }
}
