#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
namespace NCatboostCuda {
    ui32 EstimatePinnedMemorySizeInBytesPerDevice(const NCB::TTrainingDataProvider& dataProvider,
                                                  const NCB::TTrainingDataProvider* testProvider,
                                                  const TBinarizedFeaturesManager& featuresManager,
                                                  ui32 deviceCount);
}
