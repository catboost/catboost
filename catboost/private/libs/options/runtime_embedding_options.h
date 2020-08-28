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

    public:
        TOption<TMap<TString, TVector<TFeatureCalcerDescription>>> EmbeddingFeatureProcessing;
    };

    class TRuntimeEmbeddingOptions {
    public:
        TRuntimeEmbeddingOptions()
            : EmbeddingFeatures("embedding_features", TVector<TEmbeddingFeatureDescription>{})
        {}

        TRuntimeEmbeddingOptions(
            const TVector<ui32>& embeddingFeatureIndices,
            const TEmbeddingProcessingOptions& embeddingOptions
        );

        const TVector<NCatboostOptions::TEmbeddingFeatureDescription>& GetFeatureDescriptions() const {
            return EmbeddingFeatures;
        }
    private:
        TOption<TVector<TEmbeddingFeatureDescription>> EmbeddingFeatures;
    };

};
