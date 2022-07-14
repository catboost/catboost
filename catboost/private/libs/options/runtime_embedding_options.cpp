#include "runtime_embedding_options.h"
#include "json_helper.h"

#include <util/string/builder.h>

namespace NCatboostOptions {

    TRuntimeEmbeddingOptions::TRuntimeEmbeddingOptions(
        const TVector<ui32>& embeddingFeatureIndices,
        const TEmbeddingProcessingOptions& embeddingOptions
    )
        : TRuntimeEmbeddingOptions()
    {

        TVector<TEmbeddingFeatureDescription> embeddingFeatures;
        for (ui32 embeddingFeatureIdx: embeddingFeatureIndices) {
            embeddingFeatures.emplace_back(embeddingFeatureIdx, embeddingOptions.GetCalcersDescriptions(embeddingFeatureIdx));
        }
        EmbeddingFeatures.Set(embeddingFeatures);
    }
};
