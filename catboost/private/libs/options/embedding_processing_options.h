#pragma once

#include "text_processing_options.h"

namespace NCatboostOptions {

    class TEmbeddingFeatureDescription {
    public:
        TEmbeddingFeatureDescription();

        TEmbeddingFeatureDescription(
            ui32 embeddingFeatureIdx,
            TConstArrayRef<TFeatureCalcerDescription> featureEstimators
        );

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TEmbeddingFeatureDescription& rhs) const;
        bool operator!=(const TEmbeddingFeatureDescription& rhs) const;

    public:
        TOption<ui32> EmbeddingFeatureId;
        TOption<TVector<TFeatureCalcerDescription>> FeatureEstimators;
    };

    class TEmbeddingProcessingOptions {
    public:
        TEmbeddingProcessingOptions();

        const TVector<TFeatureCalcerDescription>& GetCalcersDescriptions(ui32 embeddingFeatureIdx) const;

        static TString DefaultProcessingName() {
            static TString name("default");
            return name;
        }

        static TVector<TFeatureCalcerDescription> DefaultEmbeddingCalcers() {
            return {{
                TFeatureCalcerDescription{EFeatureCalcerType::LDA},
                TFeatureCalcerDescription{EFeatureCalcerType::KNN}
            }};
        };

        void Save(NJson::TJsonValue* optionsJson) const;
        void Load(const NJson::TJsonValue& options);
        bool operator==(const TEmbeddingProcessingOptions& rhs) const;
        bool operator!=(const TEmbeddingProcessingOptions& rhs) const;

    public:
        void SetNotSpecifiedOptionsToDefaults();

        TOption<TMap<TString, TVector<TFeatureCalcerDescription>>> EmbeddingFeatureProcessing;
    };

    void ParseEmbeddingProcessingOptionsFromPlainJson(
        const NJson::TJsonValue& plainOptions,
        NJson::TJsonValue* embeddingProcessingOptions,
        TSet<TString>* seenKeys
    );

    void SaveEmbeddingProcessingOptionsToPlainJson(
        const NJson::TJsonValue& embeddingProcessingOptions,
        NJson::TJsonValue* plainOptions
    );
}

