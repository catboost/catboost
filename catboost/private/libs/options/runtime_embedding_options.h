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

};
