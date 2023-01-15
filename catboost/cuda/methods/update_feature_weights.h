#pragma once

#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>


void UpdateFeatureWeightsForBestSplits(
    const NCatboostCuda::TBinarizedFeaturesManager& featuresManager,
    double modelSizeReg,
    TMirrorBuffer<float>& featureWeights,
    ui32 maxUniqueValues = 1);
