#pragma once

#include <util/system/spinlock.h>
#include <util/generic/vector.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/data/binarizations_manager.h>

namespace NCatboostCuda {

    class TGpuBordersBuilder {
    public:
        TGpuBordersBuilder(TBinarizedFeaturesManager& manager)
        : FeaturesManager(manager) {

        }

        TVector<float> GetOrComputeBorders(ui32 featureId,
                                           const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                           TSingleBuffer<const float> floatFeature,
                                           ui32 stream = 0);

        TVector<float> GetOrComputeBorders(ui32 featureId,
                                           const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                           TConstArrayRef<float> floatFeature,
                                           ui32 stream = 0);
    private:
        TAdaptiveLock Lock;
        TBinarizedFeaturesManager& FeaturesManager;
    };

}
