#pragma once

#include "embedding_processing_options.h"

namespace NCatboostOptions {
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
