#include "gpu_binarization_helpers.h"
#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>


inline TVector<float> ComputeBorders(const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                     const TSingleBuffer<const float>& floatFeature, ui32 stream) {
    TVector<float> borders;
    NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
    TSingleBuffer<float> sortedFeature = TSingleBuffer<float>::CopyMapping(floatFeature);
    sortedFeature.Copy(floatFeature, stream);
    RadixSort(sortedFeature, false, stream);

    TVector<float> sortedFeatureCpu;
    sortedFeature.Read(sortedFeatureCpu,
                       stream);

    borders = gridBuilderFactory
        .Create(binarizationDescription.BorderSelectionType)
        ->BuildBorders(sortedFeatureCpu,
                       binarizationDescription.BorderCount);

    //hack to work with constant features
    if (borders.size() == 0) {
        borders.push_back(0.5);
    }
    return borders;
}


TVector<float> NCatboostCuda::TGpuBordersBuilder::GetOrComputeBorders(
    ui32 featureId,
    const NCatboostOptions::TBinarizationOptions& binarizationDescription,
    TSingleBuffer<const float> floatFeature,
    ui32 stream) {
        TVector<float> borders;
        bool hasBorders = false;
        with_lock (Lock) {
            if (FeaturesManager.HasBorders(featureId)) {
                borders = FeaturesManager.GetBorders(featureId);
                hasBorders = true;
            }
        }
        if (!hasBorders) {
            borders = ComputeBorders(binarizationDescription, floatFeature, stream);
            with_lock (Lock) {
                if (FeaturesManager.HasBorders(featureId)) {
                    borders = FeaturesManager.GetBorders(featureId);
                } else {
                    FeaturesManager.SetBorders(featureId, borders);
                }
            }
        }
        return borders;
}


TVector<float> NCatboostCuda::TGpuBordersBuilder::GetOrComputeBorders(
    ui32 featureId,
    const NCatboostOptions::TBinarizationOptions& binarizationDescription,
    TConstArrayRef<float> floatFeature,
    ui32 stream) {
    TVector<float> borders;
    bool hasBorders = false;
    with_lock (Lock) {
        if (FeaturesManager.HasBorders(featureId)) {
            borders = FeaturesManager.GetBorders(featureId);
            hasBorders = true;
        }
    }
    if (hasBorders) {
        return borders;
    } else {
        TSingleBuffer<float> floatFeatureGpu = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(0, floatFeature.size()));
        floatFeatureGpu.Write(floatFeature, stream);
        return GetOrComputeBorders(featureId, binarizationDescription, floatFeatureGpu.AsConstBuf(), stream);
    }
}

