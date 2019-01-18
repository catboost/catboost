#include "pinned_memory_estimation.h"

#include <catboost/cuda/cuda_lib/helpers.h>

#include <catboost/libs/helpers/math_utils.h>

ui32 NCatboostCuda::EstimatePinnedMemorySizeInBytesPerDevice(const NCB::TTrainingDataProvider& dataProvider,
                                                             const NCB::TTrainingDataProvider* testProvider,
                                                             const TBinarizedFeaturesManager& featuresManager,
                                                             ui32 deviceCount) {
    CB_ENSURE(deviceCount, "Need at least one device for pinned memory size estimation");
    const ui64 sampleCount = dataProvider.GetObjectCount() + (testProvider ? testProvider->GetObjectCount() : 0);
    ui32 maxBitsPerFeature = 0;

    ui32 treeCtrFeaturesCount = 0;
    for (const auto& feature : featuresManager.GetCatFeatureIds()) {
        if (featuresManager.UseForTreeCtr(feature)) {
            const ui32 reserveBias = 4;
            maxBitsPerFeature = Max<ui32>(maxBitsPerFeature, NCB::IntLog2(featuresManager.GetBinCount(feature) + reserveBias));
            ++treeCtrFeaturesCount;
        }
    }
    //conservative estimation
    return NHelpers::CeilDivide(maxBitsPerFeature * sampleCount * treeCtrFeaturesCount, 8 * deviceCount);
}
