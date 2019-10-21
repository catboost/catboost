#pragma once

#include "tokenizer.h"

#include <catboost/libs/helpers/guid.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/private/libs/data_types/text.h>
#include <catboost/private/libs/options/text_processing_options.h>

#include <library/text_processing/dictionary/dictionary.h>
#include <library/text_processing/dictionary/dictionary_builder.h>
#include <library/text_processing/dictionary/mmap_frequency_based_dictionary.h>

#include <util/stream/input.h>
#include <util/stream/output.h>

namespace NCB {
    class TDictionaryProxy : public TThrRefBase {
    public:
        using IDictionary = NTextProcessing::NDictionary::IDictionary;
        using TDictionary = NTextProcessing::NDictionary::TDictionary;
        using TDictionaryPtr = TIntrusivePtr<IDictionary>;
        using TMMapDictionary = NTextProcessing::NDictionary::TMMapDictionary;
        using TDictionaryOptions = NCatboostOptions::TTextColumnDictionaryOptions;

        TDictionaryProxy() = default;
        explicit TDictionaryProxy(TDictionaryPtr dictionaryImpl);

        TGuid Id() const;

        TTokenId Apply(TStringBuf token) const;
        TText Apply(TConstArrayRef<TStringBuf> tokens) const;
        void Apply(TConstArrayRef<TStringBuf> tokens, TText* text) const;

        ui32 Size() const;

        TTokenId GetUnknownTokenId() const;

        void Save(IOutputStream* stream) const;
        void Load(IInputStream* stream);

    private:
        TDictionaryPtr DictionaryImpl;
        TGuid Guid;

        static constexpr std::array<char, 13> DictionaryMagic = {"DictionaryV1"};
        static constexpr ui32 MagicSize = DictionaryMagic.size();
        static constexpr ui32 Alignment = 16;
    };

    using TDictionaryPtr = TIntrusivePtr<TDictionaryProxy>;

    template <class TTextFeature>
    class TIterableTextFeature {
    public:
        TIterableTextFeature(const TTextFeature& textFeature)
        : TextFeature(textFeature)
        {}

        template <class F>
        void ForEach(F&& visitor) const {
            for (ui32 i: xrange(Size())) {
                visitor(i, TextFeature[i]);
            }
        }

        ui32 Size() const {
            return TextFeature.size();
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
            TextFeature->ForEach(
                [&visitor](ui32 index, TStringBuf phrase){visitor(index, phrase);}
            );
        }

        ui32 Size() const {
            return TextFeature->GetSize();
        }
    private:
        ITypedArraySubsetPtr<TString> TextFeature;
    };

    template <class TTextFeatureType>
    inline TDictionaryPtr CreateDictionary(
        TIterableTextFeature<TTextFeatureType> textFeature,
        const NCatboostOptions::TTextColumnDictionaryOptions& dictionaryOptions,
        const TTokenizerPtr& tokenizer) {

        NTextProcessing::NDictionary::TDictionaryBuilder dictionaryBuilder(
            dictionaryOptions.DictionaryBuilderOptions,
            dictionaryOptions.DictionaryOptions
        );

        TVector<TStringBuf> tokens;
        const auto& tokenize = [&tokenizer, &tokens](ui32 /*index*/, TStringBuf phrase) {
            TVector<TStringBuf> phraseTokens;
            tokenizer->Tokenize(phrase, &phraseTokens);
            tokens.insert(tokens.end(), phraseTokens.begin(), phraseTokens.end());
        };
        textFeature.ForEach(tokenize);
        dictionaryBuilder.Add(tokens);

        return new TDictionaryProxy(dictionaryBuilder.FinishBuilding());
    }
}
