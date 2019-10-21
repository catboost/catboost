#pragma once

#include "enums.h"
#include "option.h"

#include <library/json/json_value.h>
#include <library/text_processing/dictionary/options.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/strbuf.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <utility>


namespace NCatboostOptions {
    using TDictionaryOptions = NTextProcessing::NDictionary::TDictionaryOptions;
    using TDictionaryBuilderOptions = NTextProcessing::NDictionary::TDictionaryBuilderOptions;

    class TTextColumnDictionaryOptions {
    public:
        TTextColumnDictionaryOptions();
        TTextColumnDictionaryOptions(
            TString dictionaryId,
            TDictionaryOptions dictionaryOptions,
            TMaybe<TDictionaryBuilderOptions> dictionaryBuilderOptions = Nothing()
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
        explicit TFeatureCalcerDescription(EFeatureCalcerType featureCalcerType);
        TFeatureCalcerDescription& operator=(EFeatureCalcerType featureCalcerType);
        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TFeatureCalcerDescription& rhs) const;
        bool operator!=(const TFeatureCalcerDescription& rhs) const;

    public:
        TOption<EFeatureCalcerType> CalcerType;
    };

    struct TTextFeatureProcessing {
    public:
        TTextFeatureProcessing();
        TTextFeatureProcessing(
            TFeatureCalcerDescription&& featureCalcer,
            TVector<TString>&& dictionariesNames
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextFeatureProcessing& rhs) const;
        bool operator!=(const TTextFeatureProcessing& rhs) const;

    public:
        TOption<TFeatureCalcerDescription> FeatureCalcer;
        TOption<TVector<TString>> DictionariesNames;
    };

    class TTextProcessingOptions {
    public:
        TTextProcessingOptions();
        TTextProcessingOptions(
            TVector<TTextColumnDictionaryOptions>&& dictionaries,
            TMap<TString, TVector<TTextFeatureProcessing>>&& textFeatureProcessing
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTextProcessingOptions& rhs) const;
        bool operator!=(const TTextProcessingOptions& rhs) const;

        bool HasOnlyDefaultDictionaries() const;
        const TVector<TTextColumnDictionaryOptions>& GetDictionaries() const;
        const TVector<TTextFeatureProcessing>& GetFeatureProcessing(ui32 textFeatureIdx) const;
        void SetDefaultMinTokenOccurrence(ui64 minTokenOccurrence);

        static TVector<TTextColumnDictionaryOptions> GetDefaultDictionaries();
        static TMap<EFeatureCalcerType, TVector<TString>> GetDefaultCalcerDictionaries();
        static TVector<TString> GetDefaultCalcerDictionaries(EFeatureCalcerType calcerType);
        static TString DefaultProcessingName() {
            static TString name("default");
            return name;
        }

    private:
        static TTextColumnDictionaryOptions BiGramDictionaryOptions();
        static TTextColumnDictionaryOptions WordDictionaryOptions();

    private:
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
