#pragma once

#include "text_processing_options.h"

namespace NCatboostOptions {
    class TTokenizedFeatureDescription {
    public:
        TTokenizedFeatureDescription();

        TTokenizedFeatureDescription(
            TString tokenizerId,
            TString dictionaryId,
            ui32 textFeatureId,
            TConstArrayRef<TFeatureCalcerDescription> featureEstimators
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TTokenizedFeatureDescription& rhs) const;
        bool operator!=(const TTokenizedFeatureDescription& rhs) const;

    public:
        TOption<TString> FeatureId;
        TOption<TString> TokenizerId;
        TOption<TString> DictionaryId;
        TOption<ui32> TextFeatureId;
        TOption<TVector<TFeatureCalcerDescription>> FeatureEstimators;
    };

    class TRuntimeTextOptions {
    public:
        TRuntimeTextOptions();
        TRuntimeTextOptions(
            const TVector<ui32>& textFeatureIndices,
            const TTextProcessingOptions& textOptions
        );
        TRuntimeTextOptions(
            TConstArrayRef<TTextColumnTokenizerOptions> tokenizers,
            TConstArrayRef<TTextColumnDictionaryOptions> dictionaries,
            TConstArrayRef<TTokenizedFeatureDescription> features
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TRuntimeTextOptions& rhs) const;
        bool operator!=(const TRuntimeTextOptions& rhs) const;

        const TTextColumnTokenizerOptions& GetTokenizerOptions(const TString& tokenizerId) const;
        const TTextColumnDictionaryOptions& GetDictionaryOptions(const TString& dictionaryId) const;
        const TTokenizedFeatureDescription& GetTokenizedFeatureDescription(ui32 tokenizedFeatureIdx) const;
        const TVector<TTokenizedFeatureDescription>& GetTokenizedFeatureDescriptions() const;

        ui32 TokenizedFeatureCount() const;
        void UpdateDefaultProcessing(ui32 textFeatureId);

    private:
        bool IsUnusedTextFeature(ui32 textFeature) const;
        void CheckUniqueDictionaryIds() const;

        TOption<TMap<TString, TTextColumnTokenizerOptions>> Tokenizers;
        TOption<TMap<TString, TTextColumnDictionaryOptions>> Dictionaries;
        TOption<TVector<TTokenizedFeatureDescription>> TokenizedFeatures;
    };

}
