#pragma once

// Model export: converts MLX training results to CatBoost's TFullModel.
// Maps GPU-internal split representation (quantized bin indices) back to
// CatBoost's TModelSplit (float feature + border value) using the original
// quantization info.

#include <catboost/mlx/methods/mlx_boosting.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/private/libs/options/catboost_options.h>

namespace NCatboostMlx {

    // Convert MLX boosting results to a CatBoost TFullModel.
    //
    // The conversion maps each MLX split (FeatureId + BinId from TBestSplitProperties)
    // back to a TModelSplit with the original float border value from quantization info.
    //
    // Parameters:
    //   boostingResult        - tree structures and leaf values from RunBoosting()
    //   quantizedFeaturesInfo - quantization borders for each feature
    //   featuresLayout        - feature layout (internal/external index mapping)
    //   gpuFeatures           - GPU feature metadata from compressed index
    //   externalFeatureIndices - maps GPU local feature index → CatBoost external feature index
    //   approxDimension       - number of output dimensions (1 for RMSE)
    //   catboostOptions       - training options (serialized into model info)
    TFullModel ConvertToFullModel(
        const TBoostingResult& boostingResult,
        const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const NCB::TFeaturesLayout& featuresLayout,
        const TVector<TCFeature>& gpuFeatures,
        const TVector<ui32>& externalFeatureIndices,
        ui32 approxDimension,
        const NCatboostOptions::TCatBoostOptions& catboostOptions
    );

}  // namespace NCatboostMlx
