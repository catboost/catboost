#pragma once

#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
namespace NCatboostCuda {
    ui32 EstimatePinnedMemorySizeInBytesPerDevice(const TDataProvider& dataProvider,
                                                  const TDataProvider* testProvider,
                                                  const TBinarizedFeaturesManager& featuresManager,
                                                  ui32 deviceCount);
}
