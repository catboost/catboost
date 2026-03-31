#pragma once

// Builds a TMLXDataSet from CatBoost's TTrainingDataProvider.
// Extracts quantized features, targets, and weights, transfers them to GPU.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/libs/data/data_provider.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostMlx {

    // Build a GPU-resident dataset from CatBoost's training data.
    // This extracts the quantized features into a compressed uint32 index,
    // and copies targets/weights to GPU arrays.
    //
    // For now, supports only one-byte quantized features (the common case).
    // Half-byte and binary features will be added in Phase 7.
    TMLXDataSet BuildMLXDataSet(
        const NCB::TTrainingDataProvider& dataProvider,
        ui32 approxDimension,
        NPar::ILocalExecutor* localExecutor
    );

    // Build compressed index from quantized feature columns.
    // Packs one-byte features 4 per uint32 word, matching CatBoost's CUDA layout.
    TMLXCompressedIndex BuildCompressedIndex(
        const NCB::TQuantizedObjectsDataProvider& objectsData,
        NPar::ILocalExecutor* localExecutor
    );

}  // namespace NCatboostMlx
