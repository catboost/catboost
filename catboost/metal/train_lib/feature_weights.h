#pragma once

#include "feature_metadata.h"

#include <catboost/private/libs/options/feature_penalties_options.h>

namespace NCB {
    inline TVector<float> MakeMetalFeatureWeights(
            const NCatboostOptions::TFeaturePenaltiesOptions& options,
            const TMetalStaticFeatureMetadata& metadata) {
        // Like CUDA, option keys address the feature-manager registry. This
        // usually matches original flat IDs, but estimated columns and sliced
        // floats change that registry. A category weight does not implicitly
        // propagate to its independently registered CTR columns.
        const auto sourceWeights = NCatboostOptions::ExpandFeatureWeights(options, metadata.ManagerCount);
        TVector<float> result;
        result.reserve(metadata.DenseToManagerIds.size());
        for (ui32 source : metadata.DenseToManagerIds) {
            CB_ENSURE(source < sourceWeights.size(), "Metal feature weight source ID is out of range");
            result.push_back(sourceWeights[source]);
        }
        return result;
    }
}
