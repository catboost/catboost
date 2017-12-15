#include "pinned_memory_estimation.h"

ui32 NCatboostCuda::EstimatePinnedMemorySizeInBytesPerDevice(const TDataProvider& dataProvider,
                                                             const TDataProvider* testProvider,
                                                             const TBinarizedFeaturesManager& featuresManager,
                                                             ui32 deviceCount) {
    CB_ENSURE(deviceCount, "Need at least one device for pinned memory size estimation");
    const ui64 sampleCount = dataProvider.GetSampleCount() + (testProvider ? testProvider->GetSampleCount() : 0);
    ui32 maxBitsPerFeature = 0;

    ui32 treeCtrFeaturesCount = 0;
    for (const auto& feature : featuresManager.GetCatFeatureIds()) {
        if (featuresManager.UseForTreeCtr(feature)) {
            const ui32 reserveBias = 4;
            maxBitsPerFeature = Max<ui32>(maxBitsPerFeature, IntLog2(featuresManager.GetBinCount(feature) + reserveBias));
            ++treeCtrFeaturesCount;
        }
    }
    //conservative estimation
    return NHelpers::CeilDivide(maxBitsPerFeature * sampleCount * treeCtrFeaturesCount, 8 * deviceCount);
}
