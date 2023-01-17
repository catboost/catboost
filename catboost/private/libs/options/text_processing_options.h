#pragma once

#include "enums.h"
#include "option.h"

#include <library/cpp/json/json_value.h>
#include <library/cpp/text_processing/dictionary/options.h>
#include <library/cpp/text_processing/tokenizer/options.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/strbuf.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <utility>


namespace NCatboostOptions {
    using TTokenizerOptions = NTextProcessing::NTokenizer::TTokenizerOptions;
    using TDictionaryOptions = NTextProcessing::NDictionary::TDictionaryOptions;
    using TDictionaryBuilderOptions = NTextProcessing::NDictionary::TDictionaryBuilderOptions;

    class TTextColumnTokenizerOptions {
    public:
        TTextColumnTokenizerOptions();
        TTextColumnTokenizerOptions(TString tokenizerId, TTokenizerOptions tokenizerOptions);

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextColumnTokenizerOptions& rhs) const;
        bool operator!=(const TTextColumnTokenizerOptions& rhs) const;

    public:
        TOption<TString> TokenizerId;
        TOption<TTokenizerOptions> TokenizerOptions;
    };

    constexpr TDictionaryOptions DEFAULT_DICTIONARY_OPTIONS = {
        /*TokenLevelType=*/NTextProcessing::NDictionary::ETokenLevelType::Word,
        /*GramOrder=*/1,
        /*SkipStep=*/0,
        /*StartTokenId=*/0,
        /*EndOfWordTokenPolicy=*/NTextProcessing::NDictionary::EEndOfWordTokenPolicy::Insert,
        /*EndOfSentenceTokenPolicy=*/NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::Skip
    };

    constexpr TDictionaryBuilderOptions DEFAULT_DICTIONARY_BUILDER_OPTIONS = {
       /*OccurrenceLowerBound=*/3,
       /*MaxDictionarySize=*/50000
    };

    class TTextColumnDictionaryOptions {
    public:
        TTextColumnDictionaryOptions();
        TTextColumnDictionaryOptions(
            TString dictionaryId,
            TDictionaryOptions dictionaryOptions,
            TDictionaryBuilderOptions dictionaryBuilderOptions = DEFAULT_DICTIONARY_BUILDER_OPTIONS
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextColumnDictionaryOptions& rhs) const;
        bool operator!=(const TTextColumnDictionaryOptions& rhs) const;

    public:
        TOption<TString> DictionaryId;
        TOption<TDictionaryOptions> DictionaryOptions;
        TOption<TDictionaryBuilderOptions> DictionaryBuilderOptions;
    };

    struct TFeatureCalcerDescription {
    public:
        TFeatureCalcerDescription();
        explicit TFeatureCalcerDescription(
            EFeatureCalcerType featureCalcerType,
            NJson::TJsonValue calcerOptions = NJson::TJsonValue()
        );
        TFeatureCalcerDescription& operator=(EFeatureCalcerType featureCalcerType);
        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TFeatureCalcerDescription& rhs) const;
        bool operator!=(const TFeatureCalcerDescription& rhs) const;

    public:
        TOption<EFeatureCalcerType> CalcerType;
        TOption<NJson::TJsonValue> CalcerOptions;
    };

    struct TTextFeatureProcessing {
    public:
        TTextFeatureProcessing();
        TTextFeatureProcessing(
            TVector<TFeatureCalcerDescription>&& featureCalcers,
            TVector<TString>&& tokenizersNames,
            TVector<TString>&& dictionariesNames
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextFeatureProcessing& rhs) const;
        bool operator!=(const TTextFeatureProcessing& rhs) const;

    public:
        TOption<TVector<TFeatureCalcerDescription>> FeatureCalcers;
        TOption<TVector<TString>> TokenizersNames;
        TOption<TVector<TString>> DictionariesNames;
    };

    class TTextProcessingOptions {
    public:
        TTextProcessingOptions();
        TTextProcessingOptions(
            TVector<TTextColumnTokenizerOptions>&& tokenizers,
            TVector<TTextColumnDictionaryOptions>&& dictionaries,
            TMap<TString, TVector<TTextFeatureProcessing>>&& textFeatureProcessing
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextProcessingOptions& rhs) const;
        bool operator!=(const TTextProcessingOptions& rhs) const;

        void Validate(bool forClassification) const;

        const TVector<TTextColumnTokenizerOptions>& GetTokenizers() const;
        const TVector<TTextColumnDictionaryOptions>& GetDictionaries() const;
        const TVector<TTextFeatureProcessing>& GetFeatureProcessing(ui32 textFeatureIdx) const;
        void SetDefault(bool forClassification = false);
        void SetDefaultMinTokenOccurrence(ui64 minTokenOccurrence);
        void SetDefaultMaxDictionarySize(ui32 maxDictionarySize);

        static TString DefaultProcessingName() {
            static TString name("default");
            return name;
        }

    private:
        void SetNotSpecifiedOptionsToDefaults();

        TOption<TVector<TTextColumnTokenizerOptions>> Tokenizers;
        TOption<TVector<TTextColumnDictionaryOptions>> Dictionaries;
        TOption<TMap<TString, TVector<TTextFeatureProcessing>>> TextFeatureProcessing;
    };

    void ParseTextProcessingOptionsFromPlainJson(
        const NJson::TJsonValue& plainOptions,
        NJson::TJsonValue* textProcessingOptions,
        TSet<TString>* seenKeys
    );

    void SaveTextProcessingOptionsToPlainJson(
        const NJson::TJsonValue& textProcessingOptions,
        NJson::TJsonValue* plainOptions
    );
}
